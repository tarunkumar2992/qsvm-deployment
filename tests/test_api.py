"""
Tests for QSVM API
Run: pytest tests/ -v --cov=app
"""
import pytest
import numpy as np
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch


# ── Fixtures ───────────────────────────────────────────────────────────────────
VALID_FEATURES = [0.45, 0.02, 0.14, 3.21, 0.88, 4.5, 0.3, 0.012, 0.0001, 0.78, 0.22, 0.91]


@pytest.fixture(scope="module")
def mock_model():
    m = MagicMock()
    m.is_loaded = True
    m.predict.return_value = 1
    m.info.return_value = {
        "model_type": "QSVC",
        "feature_map": "ZFeatureMap(reps=2)",
        "kernel": "FidelityStatevectorKernel",
        "num_qubits": 3,
        "accuracy": 0.92,
        "n_train": 2000,
        "n_test": 500,
        "train_time_seconds": 45.2,
    }
    return m


@pytest.fixture(scope="module")
def client(mock_model):
    with patch("app.main.model_instance", mock_model):
        from app.main import app
        with TestClient(app) as c:
            yield c


# ── Health ─────────────────────────────────────────────────────────────────────
def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True


def test_readiness(client):
    r = client.get("/ready")
    assert r.status_code == 200


# ── Predict ────────────────────────────────────────────────────────────────────
def test_predict_malignant(client):
    r = client.post("/predict", json={"features": VALID_FEATURES})
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] in [0, 1]
    assert body["label"] in ["Benign", "Malignant"]
    assert body["inference_time_ms"] > 0


def test_predict_wrong_feature_count(client):
    r = client.post("/predict", json={"features": [0.1, 0.2, 0.3]})
    assert r.status_code == 422


def test_predict_nan_features(client):
    bad = VALID_FEATURES.copy()
    bad[0] = float("nan")
    r = client.post("/predict", json={"features": bad})
    assert r.status_code == 422


def test_predict_inf_features(client):
    bad = VALID_FEATURES.copy()
    bad[1] = float("inf")
    r = client.post("/predict", json={"features": bad})
    assert r.status_code == 422


# ── Batch predict ──────────────────────────────────────────────────────────────
def test_batch_predict(client):
    payload = {"samples": [{"features": VALID_FEATURES}, {"features": VALID_FEATURES}]}
    r = client.post("/predict/batch", json=payload)
    assert r.status_code == 200
    assert len(r.json()) == 2


def test_batch_predict_empty(client):
    r = client.post("/predict/batch", json={"samples": []})
    assert r.status_code == 422


# ── Model info ─────────────────────────────────────────────────────────────────
def test_model_info(client):
    r = client.get("/model/info")
    assert r.status_code == 200
    body = r.json()
    assert "model_type" in body
    assert "accuracy" in body


# ── Metrics endpoint ───────────────────────────────────────────────────────────
def test_metrics_endpoint(client):
    r = client.get("/metrics")
    assert r.status_code == 200
    assert b"qsvm_" in r.content
