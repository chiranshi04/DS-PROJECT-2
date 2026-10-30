import numpy as np

from indivoice.eval.aggregate import aggregate_window_scores
from indivoice.eval.bootstrap import speaker_bootstrap
from indivoice.eval.calibration import TemperatureScaler
from indivoice.eval.metrics import (
    eer_with_threshold,
    fpr_fnr_at_threshold,
    roc_auc,
    select_dev_threshold,
)


def test_known_auc_and_eer() -> None:
    labels = [0, 0, 1, 1]
    scores = [0.1, 0.2, 0.8, 0.9]
    eer, threshold = eer_with_threshold(labels, scores)
    assert roc_auc(labels, scores) == 1.0
    assert eer == 0.0
    assert 0.2 < threshold <= 0.9


def test_fixed_threshold_is_selected_from_dev_only() -> None:
    dev_labels, dev_scores = [0, 0, 1, 1], [0.2, 0.4, 0.6, 0.8]
    threshold = select_dev_threshold(dev_labels, dev_scores)
    test_fpr, test_fnr = fpr_fnr_at_threshold([0, 1], [0.55, 0.65], threshold)
    assert threshold == 0.6
    assert test_fpr == 0.0 and test_fnr == 0.0


def test_bootstrap_reports_speaker_clusters() -> None:
    labels = [0, 1, 0, 1, 0, 1, 0, 1]
    scores = [0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6]
    speakers = ["a", "a", "b", "b", "c", "c", "d", "d"]
    interval = speaker_bootstrap(labels, scores, speakers, metric="auc", resamples=20, seed=4)
    assert interval["speakers"] == 4
    assert interval["resamples"] == 20
    assert interval["lower"] <= interval["estimate"] <= interval["upper"]


def test_window_aggregation_mean_and_median() -> None:
    ids = ["u1", "u1", "u2"]
    assert aggregate_window_scores(ids, [1, 3, 2], "mean")["u1"] == 2.0
    assert aggregate_window_scores(ids, [1, 3, 2], "median")["u1"] == 2.0


def test_temperature_scaler_is_serializable(tmp_path) -> None:
    scaler = TemperatureScaler().fit(np.array([-2, 2, -1, 1]), np.array([0, 1, 0, 1]))
    path = tmp_path / "temperature.json"
    scaler.save(path)
    assert TemperatureScaler.load(path).temperature > 0
