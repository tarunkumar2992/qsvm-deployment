"""
QSVM Brain Tumour Classification - Production API
"""
import time
import logging
import os
from contextlib import asynccontextmanager
from typing import List, Optional

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from prometheus_client import (
    Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
)
from starlette.responses import Response

from app.model import QSVMModel
from app.logger import setup_logger

# ── Logging ────────────────────────────────────────────────────────────────────
logger = setup_logger(__name__)

# ── Prometheus Metrics ─────────────────────────────────────────────────────────
REQUEST_COUNT = Counter(
    "qsvm_requests_total", "Total prediction requests", ["method", "endpoint", "status"]
)
REQUEST_LATENCY = Histogram(
    "qsvm_request_duration_seconds", "Request latency", ["endpoint"],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
)
PREDICTION_COUNT = Counter(
    "qsvm_predictions_total", "Total predictions made", ["predicted_class"]
)
MODEL_ACCURACY = Gauge("qsvm_model_accuracy", "Latest model accuracy score")
ACTIVE_REQUESTS = Gauge("qsvm_active_requests", "Active requests in flight")
MODEL_LOAD_TIME = Gauge("qsvm_model_load_duration_seconds", "Time taken to load model")

# ── Lifespan ───────────────────────────────────────────────────────────────────
model_instance: Optional[QSVMModel] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_instance
    start = time.time()
    logger.info("Loading QSVM model...")
    model_instance = QSVMModel()
    model_instance.load()
    duration = time.time() - start
    MODEL_LOAD_TIME.set(duration)
    logger.info(f"Model loaded in {duration:.2f}s")
    yield
    logger.info("Shutting down QSVM API...")

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="QSVM Brain Tumour Classifier",
    description="Production API for Quantum Support Vector Machine brain tumour classification.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Middleware: timing & metrics ───────────────────────────────────────────────
@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    ACTIVE_REQUESTS.inc()
    start = time.time()
    try:
        response = await call_next(request)
        duration = time.time() - start
        REQUEST_LATENCY.labels(endpoint=request.url.path).observe(duration)
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=request.url.path,
            status=response.status_code,
        ).inc()
        return response
    except Exception as exc:
        REQUEST_COUNT.labels(method=request.method, endpoint=request.url.path, status=500).inc()
        raise exc
    finally:
        ACTIVE_REQUESTS.dec()

# ── Schemas ────────────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    features: List[float] = Field(
        ...,
        min_items=12,
        max_items=12,
        description="12 MRI texture/intensity features: "
                    "[Mean, Variance, StandardDeviation, Entropy, Skewness, "
                    "Kurtosis, Contrast, Energy, ASM, Homogeneity, Dissimilarity, Correlation]",
        example=[0.45, 0.02, 0.14, 3.21, 0.88, 4.5, 0.3, 0.012, 0.0001, 0.78, 0.22, 0.91],
    )

    @validator("features")
    def validate_features(cls, v):
        if any(not np.isfinite(x) for x in v):
            raise ValueError("All feature values must be finite numbers.")
        return v

class PredictResponse(BaseModel):
    prediction: int = Field(..., description="0 = Benign, 1 = Malignant")
    label: str
    confidence: Optional[float] = None
    inference_time_ms: float

class BatchPredictRequest(BaseModel):
    samples: List[PredictRequest] = Field(..., min_items=1, max_items=100)

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str

# ── Endpoints ──────────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health():
    return {
        "status": "ok",
        "model_loaded": model_instance is not None and model_instance.is_loaded,
        "version": app.version,
    }

@app.get("/ready", tags=["ops"])
def readiness():
    if model_instance is None or not model_instance.is_loaded:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model not ready")
    return {"status": "ready"}

@app.post("/predict", response_model=PredictResponse, tags=["inference"])
def predict(request: PredictRequest):
    if model_instance is None or not model_instance.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    t0 = time.time()
    try:
        result = model_instance.predict(np.array(request.features).reshape(1, -1))
    except Exception as exc:
        logger.error(f"Prediction error: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))

    inference_ms = (time.time() - t0) * 1000
    label = "Malignant" if result == 1 else "Benign"
    PREDICTION_COUNT.labels(predicted_class=label).inc()
    logger.info(f"Prediction: {label} | latency={inference_ms:.1f}ms")

    return PredictResponse(
        prediction=int(result),
        label=label,
        inference_time_ms=round(inference_ms, 2),
    )

@app.post("/predict/batch", response_model=List[PredictResponse], tags=["inference"])
def predict_batch(request: BatchPredictRequest):
    if model_instance is None or not model_instance.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    results = []
    for item in request.samples:
        t0 = time.time()
        result = model_instance.predict(np.array(item.features).reshape(1, -1))
        inference_ms = (time.time() - t0) * 1000
        label = "Malignant" if result == 1 else "Benign"
        PREDICTION_COUNT.labels(predicted_class=label).inc()
        results.append(PredictResponse(prediction=int(result), label=label, inference_time_ms=round(inference_ms, 2)))
    return results

@app.get("/metrics", tags=["ops"])
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/model/info", tags=["model"])
def model_info():
    if model_instance is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return model_instance.info()

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)
