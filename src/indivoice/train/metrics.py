"""Binary anti-spoofing metrics used for early stopping."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_curve


def equal_error_rate(labels: np.ndarray, scores: np.ndarray) -> float:
    """Approximate EER by the ROC point with the smallest FPR/FNR gap."""
    labels, scores = np.asarray(labels), np.asarray(scores)
    if set(np.unique(labels)) != {0, 1}:
        return float("inf")
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1.0 - tpr
    return float((fpr[np.argmin(np.abs(fpr - fnr))] + fnr[np.argmin(np.abs(fpr - fnr))]) / 2)
