"""Evaluation metrics with explicit dev/test threshold semantics."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def roc_auc(labels: Iterable[int], scores: Iterable[float]) -> float:
    """Compute ROC-AUC, returning NaN when only one class is present."""
    y, s = np.asarray(list(labels)), np.asarray(list(scores))
    return float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else float("nan")


def eer_with_threshold(labels: Iterable[int], scores: Iterable[float]) -> tuple[float, float]:
    """Return EER and its score threshold using the closest ROC operating point."""
    y, s = np.asarray(list(labels)), np.asarray(list(scores))
    if len(np.unique(y)) != 2:
        return float("nan"), float("nan")
    fpr, tpr, thresholds = roc_curve(y, s)
    fnr = 1.0 - tpr
    index = int(np.argmin(np.abs(fpr - fnr)))
    return float((fpr[index] + fnr[index]) / 2.0), float(thresholds[index])


def select_dev_threshold(labels: Iterable[int], scores: Iterable[float]) -> float:
    """Select one fixed threshold on dev by maximizing Youden's J statistic."""
    y, s = np.asarray(list(labels)), np.asarray(list(scores))
    fpr, tpr, thresholds = roc_curve(y, s)
    return float(thresholds[int(np.argmax(tpr - fpr))])


def fpr_fnr_at_threshold(
    labels: Iterable[int], scores: Iterable[float], threshold: float
) -> tuple[float, float]:
    """Compute false-positive and false-negative rates at a fixed score threshold."""
    y, s = np.asarray(list(labels)), np.asarray(list(scores))
    predicted = s >= threshold
    genuine, synthetic = y == 0, y == 1
    fpr = float(np.mean(predicted[genuine])) if genuine.any() else float("nan")
    fnr = float(np.mean(~predicted[synthetic])) if synthetic.any() else float("nan")
    return fpr, fnr


def expected_calibration_error(
    labels: Iterable[int], probabilities: Iterable[float], bins: int = 10
) -> float:
    """Compute equal-width expected calibration error."""
    y, p = np.asarray(list(labels)), np.asarray(list(probabilities))
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for left, right in zip(edges[:-1], edges[1:], strict=False):
        mask = (p >= left) & (p <= right if right == 1 else p < right)
        if mask.any():
            total += float(mask.mean()) * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return total


def min_dcf(
    labels: Iterable[int],
    scores: Iterable[float],
    p_target: float = 0.5,
    c_miss: float = 1.0,
    c_fa: float = 1.0,
) -> float:
    """Compute minimum normalized detection cost over all observed thresholds."""
    y, s = np.asarray(list(labels)), np.asarray(list(scores))
    thresholds = np.unique(s)
    costs = []
    for threshold in thresholds:
        fpr, fnr = fpr_fnr_at_threshold(y, s, float(threshold))
        costs.append(c_miss * p_target * fnr + c_fa * (1 - p_target) * fpr)
    default = c_miss * p_target + c_fa * (1 - p_target)
    return float(min(costs) / default) if costs and default else float("nan")
