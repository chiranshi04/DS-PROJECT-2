"""Speaker-clustered bootstrap intervals."""

from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np

from .metrics import eer_with_threshold, roc_auc


def speaker_bootstrap(
    labels: Iterable[int],
    scores: Iterable[float],
    speaker_ids: Iterable[str],
    metric: str = "auc",
    resamples: int = 1000,
    seed: int = 2025,
    confidence: float = 0.95,
) -> dict[str, float | int]:
    """Resample speakers with replacement, retaining all their utterances."""
    y, s, speakers = (
        np.asarray(list(labels)),
        np.asarray(list(scores)),
        np.asarray(list(speaker_ids)),
    )
    unique = np.unique(speakers)
    if len(unique) < 2:
        return {
            "estimate": float("nan"),
            "lower": float("nan"),
            "upper": float("nan"),
            "resamples": 0,
            "speakers": int(len(unique)),
        }
    function: Callable[[np.ndarray, np.ndarray], float]
    if metric == "auc":

        def function(a: np.ndarray, b: np.ndarray) -> float:
            return roc_auc(a, b)
    elif metric == "eer":

        def function(a: np.ndarray, b: np.ndarray) -> float:
            return eer_with_threshold(a, b)[0]
    else:
        raise ValueError("metric must be 'auc' or 'eer'")
    estimate = function(y, s)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(resamples):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([np.flatnonzero(speakers == speaker) for speaker in sampled])
        value = function(y[indices], s[indices])
        if np.isfinite(value):
            values.append(value)
    alpha = (1.0 - confidence) / 2.0
    return {
        "estimate": float(estimate),
        "lower": float(np.quantile(values, alpha)) if values else float("nan"),
        "upper": float(np.quantile(values, 1.0 - alpha)) if values else float("nan"),
        "resamples": int(len(values)),
        "speakers": int(len(unique)),
    }
