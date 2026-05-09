from __future__ import annotations

import numpy as np


def binary_log_loss(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-7) -> float:
    """Kaggle-compatible binary log loss."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    y_pred = np.clip(y_pred, eps, 1.0 - eps)
    loss = -(y_true * np.log(y_pred) + (1.0 - y_true) * np.log(1.0 - y_pred))
    return float(np.mean(loss))

