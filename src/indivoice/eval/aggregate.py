"""Window-to-utterance score aggregation."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

import numpy as np


def aggregate_window_scores(
    utterance_ids: Iterable[str], logits: Iterable[float], method: str = "mean"
) -> dict[str, float]:
    """Aggregate multiple window logits per utterance using mean or median."""
    if method not in {"mean", "median"}:
        raise ValueError("method must be 'mean' or 'median'")
    grouped: dict[str, list[float]] = defaultdict(list)
    for utterance_id, logit in zip(utterance_ids, logits, strict=False):
        grouped[str(utterance_id)].append(float(logit))
    reducer = np.mean if method == "mean" else np.median
    return {key: float(reducer(values)) for key, values in grouped.items()}
