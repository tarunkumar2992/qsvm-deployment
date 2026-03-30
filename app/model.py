"""
QSVM Model: training pipeline + artifact serialization
"""
import os
import time
import pickle
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report

from qiskit.circuit.library import ZFeatureMap
from qiskit_machine_learning.kernels import FidelityStatevectorKernel
from qiskit_machine_learning.algorithms import QSVC

logger = logging.getLogger(__name__)

MODEL_DIR = Path(os.getenv("MODEL_DIR", "artifacts"))
MODEL_PATH = MODEL_DIR / "qsvm_model.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"
PCA_PATH = MODEL_DIR / "pca.pkl"
META_PATH = MODEL_DIR / "meta.pkl"

NUM_QUBITS = 3
FEATURE_COLS = [
    "Mean", "Variance", "StandardDeviation", "Entropy",
    "Skewness", "Kurtosis", "Contrast", "Energy",
    "ASM", "Homogeneity", "Dissimilarity", "Correlation",
]


class QSVMModel:
    def __init__(self):
        self.model: Optional[QSVC] = None
        self.scaler: Optional[MinMaxScaler] = None
        self.pca: Optional[PCA] = None
        self.meta: dict = {}
        self.is_loaded: bool = False

    # ── Training ───────────────────────────────────────────────────────────────
    def train(self, data_path: str) -> dict:
        logger.info(f"Training QSVM from dataset: {data_path}")
        dataset = pd.read_csv(data_path)

        X = dataset.iloc[:, 2:14]
        y = dataset.iloc[:, 1]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # MinMax scaling
        self.scaler = MinMaxScaler((0, 1)).fit(X_train)
        X_train_s = self.scaler.transform(X_train)
        X_test_s = self.scaler.transform(X_test)

        # PCA → 3 components (= num qubits)
        self.pca = PCA(n_components=NUM_QUBITS).fit(X_train_s)
        X_train_p = self.pca.transform(X_train_s)
        X_test_p = self.pca.transform(X_test_s)

        # Quantum kernel
        feature_map = ZFeatureMap(feature_dimension=NUM_QUBITS, reps=2)
        qkernel = FidelityStatevectorKernel(feature_map=feature_map)

        # Fit QSVC
        t0 = time.time()
        self.model = QSVC(quantum_kernel=qkernel)
        self.model.fit(X_train_p, y_train)
        train_time = time.time() - t0

        # Evaluate
        y_pred = self.model.predict(X_test_p)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        self.meta = {
            "accuracy": acc,
            "classification_report": report,
            "train_time_seconds": train_time,
            "num_qubits": NUM_QUBITS,
            "feature_map": "ZFeatureMap(reps=2)",
            "kernel": "FidelityStatevectorKernel",
            "n_train": len(X_train),
            "n_test": len(X_test),
        }

        logger.info(f"Training complete — accuracy={acc:.4f}, time={train_time:.1f}s")
        self.is_loaded = True
        return self.meta

    # ── Persistence ────────────────────────────────────────────────────────────
    def save(self):
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(self.model, f)
        with open(SCALER_PATH, "wb") as f:
            pickle.dump(self.scaler, f)
        with open(PCA_PATH, "wb") as f:
            pickle.dump(self.pca, f)
        with open(META_PATH, "wb") as f:
            pickle.dump(self.meta, f)
        logger.info(f"Model artifacts saved to {MODEL_DIR}")

    def load(self):
        if not MODEL_PATH.exists():
            logger.warning("No saved model found — you must train first.")
            return
        with open(MODEL_PATH, "rb") as f:
            self.model = pickle.load(f)
        with open(SCALER_PATH, "rb") as f:
            self.scaler = pickle.load(f)
        with open(PCA_PATH, "rb") as f:
            self.pca = pickle.load(f)
        with open(META_PATH, "rb") as f:
            self.meta = pickle.load(f)
        self.is_loaded = True
        logger.info("Model artifacts loaded successfully.")

    # ── Inference ──────────────────────────────────────────────────────────────
    def predict(self, features: np.ndarray) -> int:
        """
        features: shape (1, 12) — raw MRI features, unscaled.
        Returns 0 (Benign) or 1 (Malignant).
        """
        if not self.is_loaded:
            raise RuntimeError("Model is not loaded.")
        scaled = self.scaler.transform(features)
        reduced = self.pca.transform(scaled)
        return int(self.model.predict(reduced)[0])

    # ── Metadata ───────────────────────────────────────────────────────────────
    def info(self) -> dict:
        return {
            "model_type": "QSVC",
            "feature_map": self.meta.get("feature_map"),
            "kernel": self.meta.get("kernel"),
            "num_qubits": self.meta.get("num_qubits"),
            "accuracy": self.meta.get("accuracy"),
            "n_train": self.meta.get("n_train"),
            "n_test": self.meta.get("n_test"),
            "train_time_seconds": self.meta.get("train_time_seconds"),
        }
