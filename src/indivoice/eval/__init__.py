"""Evaluation metrics and protocol runner."""

from .aggregate import aggregate_window_scores
from .bootstrap import speaker_bootstrap
from .calibration import TemperatureScaler
from .metrics import eer_with_threshold, expected_calibration_error, roc_auc

__all__ = [
    "TemperatureScaler",
    "aggregate_window_scores",
    "eer_with_threshold",
    "expected_calibration_error",
    "roc_auc",
    "speaker_bootstrap",
]
