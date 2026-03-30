"""
Unit tests for QSVMModel preprocessing pipeline
"""
import numpy as np
import pytest
from unittest.mock import MagicMock, patch, mock_open
import pickle


def test_model_predict_calls_pipeline():
    """Ensure predict applies scaler → pca → model in order."""
    from app.model import QSVMModel
    m = QSVMModel()
    m.is_loaded = True

    mock_scaler = MagicMock()
    mock_scaler.transform.return_value = np.zeros((1, 12))
    mock_pca = MagicMock()
    mock_pca.transform.return_value = np.zeros((1, 3))
    mock_clf = MagicMock()
    mock_clf.predict.return_value = np.array([1])

    m.scaler = mock_scaler
    m.pca = mock_pca
    m.model = mock_clf

    result = m.predict(np.ones((1, 12)))
    assert result == 1
    mock_scaler.transform.assert_called_once()
    mock_pca.transform.assert_called_once()
    mock_clf.predict.assert_called_once()


def test_model_not_loaded_raises():
    from app.model import QSVMModel
    m = QSVMModel()
    with pytest.raises(RuntimeError, match="not loaded"):
        m.predict(np.ones((1, 12)))


def test_model_info_returns_dict():
    from app.model import QSVMModel
    m = QSVMModel()
    m.meta = {"accuracy": 0.9, "num_qubits": 3}
    info = m.info()
    assert isinstance(info, dict)
    assert "accuracy" in info
