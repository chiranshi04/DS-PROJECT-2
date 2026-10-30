"""Prediction-CSV evaluation runner for protocols P1-P5."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .bootstrap import speaker_bootstrap
from .calibration import TemperatureScaler, reliability_bins
from .metrics import (
    eer_with_threshold,
    expected_calibration_error,
    fpr_fnr_at_threshold,
    min_dcf,
    roc_auc,
    select_dev_threshold,
)

REQUIRED = {"logit", "label", "speaker_id", "language", "source_corpus"}


def _metrics(
    frame: pd.DataFrame, threshold: float, eer_threshold: float, bootstrap_resamples: int, seed: int
) -> dict[str, object]:
    labels, logits = frame["label"].astype(int).to_numpy(), frame["logit"].astype(float).to_numpy()
    eer, observed_eer_threshold = eer_with_threshold(labels, logits)
    fixed_fpr, fixed_fnr = fpr_fnr_at_threshold(labels, logits, threshold)
    eer_fpr, eer_fnr = fpr_fnr_at_threshold(labels, logits, eer_threshold)
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -60, 60)))
    result = {
        "n": int(len(frame)),
        "speakers": int(frame["speaker_id"].nunique()),
        "auc": roc_auc(labels, logits),
        "eer": eer,
        "eer_threshold": float(observed_eer_threshold),
        "comparison_eer_threshold": float(eer_threshold),
        "fixed_threshold": float(threshold),
        "fixed_fpr": fixed_fpr,
        "fixed_fnr": fixed_fnr,
        "eer_threshold_fpr": eer_fpr,
        "eer_threshold_fnr": eer_fnr,
        "threshold_gap": float(abs(threshold - eer_threshold)),
        "fpr_gap": float(abs(fixed_fpr - eer_fpr)),
        "fnr_gap": float(abs(fixed_fnr - eer_fnr)),
        "min_dcf": min_dcf(labels, logits),
        "ece": expected_calibration_error(labels, probabilities),
        "auc_ci": speaker_bootstrap(
            labels, logits, frame["speaker_id"], "auc", bootstrap_resamples, seed
        ),
        "eer_ci": speaker_bootstrap(
            labels, logits, frame["speaker_id"], "eer", bootstrap_resamples, seed
        ),
    }
    return result


def _breakdowns(
    frame: pd.DataFrame, threshold: float, eer_threshold: float, bootstrap_resamples: int, seed: int
) -> dict[str, dict[str, object]]:
    output: dict[str, dict[str, object]] = {}
    for column in ("language", "generator_id", "gender", "source_corpus"):
        if column in frame:
            output[column] = {
                str(key): _metrics(group, threshold, eer_threshold, bootstrap_resamples, seed)
                for key, group in frame.groupby(column, dropna=False)
            }
    return output


def _save_heatmap(frame: pd.DataFrame, output: Path, metric: str = "auc") -> None:
    if "train_language" in frame and "language" in frame:
        pivot = (
            frame.groupby(["train_language", "language"])
            .apply(lambda x: roc_auc(x.label, x.logit), include_groups=False)
            .unstack()
        )
    else:
        pivot = (
            frame.groupby("language")
            .apply(lambda x: roc_auc(x.label, x.logit), include_groups=False)
            .to_frame("test")
        )
    if pivot.empty:
        return
    figure, axis = plt.subplots(
        figsize=(max(4, pivot.shape[1] * 1.2), max(3, pivot.shape[0] * 0.8))
    )
    image = axis.imshow(pivot.to_numpy(dtype=float), vmin=0, vmax=1, cmap="viridis")
    axis.set_xticks(range(pivot.shape[1]), pivot.columns.astype(str), rotation=45, ha="right")
    axis.set_yticks(range(pivot.shape[0]), pivot.index.astype(str))
    axis.set_xlabel("test language")
    axis.set_ylabel("train language" if "train_language" in frame else "language")
    axis.set_title(f"{metric.upper()} language-transfer matrix")
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def _save_reliability(labels: np.ndarray, probabilities: np.ndarray, output: Path) -> None:
    """Save a reliability diagram using probabilities calibrated on dev."""
    bins = reliability_bins(labels, probabilities)
    confidence = [item["confidence"] for item in bins if item["count"]]
    accuracy = [item["accuracy"] for item in bins if item["count"]]
    if not confidence:
        return
    figure, axis = plt.subplots(figsize=(4, 4))
    axis.plot([0, 1], [0, 1], "--", color="gray", label="perfect")
    axis.plot(confidence, accuracy, "o-", label="model")
    axis.set(
        xlim=(0, 1),
        ylim=(0, 1),
        xlabel="confidence",
        ylabel="accuracy",
        title="Reliability diagram",
    )
    axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def evaluate(
    run_dir: str | Path,
    protocol: str,
    split: str,
    output_dir: str | Path = "results",
    bootstrap_resamples: int = 1000,
    aggregation: str = "mean",
) -> Path:
    """Evaluate a prediction CSV; calibration and fixed threshold use dev only."""
    run_path, output_path = Path(run_dir), Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    test_file = run_path / f"predictions_{protocol}_{split}.csv"
    dev_file = run_path / f"predictions_{protocol}_dev.csv"
    if not test_file.exists() or not dev_file.exists():
        raise FileNotFoundError(
            f"Expected {test_file} and {dev_file}; no placeholder evaluation was created."
        )
    dev, test = pd.read_csv(dev_file), pd.read_csv(test_file)
    missing = REQUIRED - set(dev.columns) - set(test.columns)
    if missing:
        raise ValueError(f"Prediction CSVs are missing columns: {sorted(missing)}")
    if aggregation in {"mean", "median"} and "utterance_id" in dev and "utterance_id" in test:
        reducer = "median" if aggregation == "median" else "mean"

        # Aggregate only logits; retain the first metadata row per utterance.
        def aggregate(frame: pd.DataFrame) -> pd.DataFrame:
            grouped = frame.groupby("utterance_id", as_index=False)
            logits = grouped.logit.median() if reducer == "median" else grouped.logit.mean()
            meta = frame.drop_duplicates("utterance_id").drop(columns=["logit"])
            return meta.merge(logits, on="utterance_id")

        dev, test = aggregate(dev), aggregate(test)
    scaler = TemperatureScaler().fit(dev.logit.to_numpy(), dev.label.to_numpy())
    dev = dev.copy()
    test = test.copy()
    dev["logit"] = scaler.transform_logits(dev.logit.to_numpy())
    test["logit"] = scaler.transform_logits(test.logit.to_numpy())
    scaler.save(run_path / "temperature.json")
    threshold = select_dev_threshold(dev.label, dev.logit)
    eer_dev, eer_threshold = eer_with_threshold(dev.label, dev.logit)
    result = {
        "protocol": protocol,
        "split": split,
        "aggregation": aggregation,
        "temperature": scaler.temperature,
        "dev_eer": eer_dev,
        "threshold_source": "dev only",
        "metrics": _metrics(test, threshold, eer_threshold, bootstrap_resamples, 2025),
        "breakdowns": _breakdowns(test, threshold, eer_threshold, bootstrap_resamples, 2025),
    }
    if protocol in {"P3", "P4"}:
        _save_heatmap(test, output_path / f"{run_path.name}_{protocol}_language_matrix.png")
    independent = {}
    for key, group in test.groupby(["source_corpus", "language"]):
        if (group.label == 0).all():
            fpr, _ = fpr_fnr_at_threshold(group.label, group.logit, threshold)
            independent[f"{key[0]}::{key[1]}"] = {
                "fpr_fixed_dev_threshold": fpr,
                "n": int(len(group)),
                "speakers": int(group.speaker_id.nunique()),
            }
    result["independent_genuine_fpr"] = independent
    _save_reliability(
        test.label.to_numpy(),
        1.0 / (1.0 + np.exp(-np.clip(test.logit.to_numpy(), -60, 60))),
        output_path / f"{run_path.name}_{protocol}_{split}_reliability.png",
    )
    json_path = output_path / f"{run_path.name}_{protocol}_{split}.json"
    json_path.write_text(
        json.dumps(result, indent=2, allow_nan=True, default=str) + "\n", encoding="utf-8"
    )
    rows = []
    for group_type, groups in {
        "overall": {"all": result["metrics"]},
        **result["breakdowns"],
    }.items():
        for group_name, values in groups.items():
            rows.append(
                {
                    "group_type": group_type,
                    "group": group_name,
                    "auc": values.get("auc"),
                    "eer": values.get("eer"),
                    "fixed_fpr": values.get("fixed_fpr"),
                    "fixed_fnr": values.get("fixed_fnr"),
                    "ece": values.get("ece"),
                }
            )
    pd.DataFrame(rows).to_csv(output_path / f"{run_path.name}_{protocol}_{split}.csv", index=False)
    return json_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--protocol", required=True, choices=["P1", "P2", "P3", "P4", "P5"])
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--bootstrap-resamples", type=int, default=1000)
    parser.add_argument("--aggregation", choices=["mean", "median"], default="mean")
    args = parser.parse_args()
    result_path = evaluate(
        args.run,
        args.protocol,
        args.split,
        args.output_dir,
        args.bootstrap_resamples,
        args.aggregation,
    )
    print(f"Wrote {result_path}")
    return 0
