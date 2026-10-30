"""Run robustness metrics from scored degraded examples."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from indivoice.eval.metrics import eer_with_threshold, fpr_fnr_at_threshold, roc_auc


def evaluate_scored_degradations(
    predictions: pd.DataFrame,
    fixed_threshold: float,
    output_dir: str | Path,
    run_name: str = "robustness",
) -> Path:
    """Write AUC/EER/FPR/FNR by degradation and level from a scored CSV."""
    required = {"degradation", "level", "label", "logit"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Robustness scores missing columns: {sorted(missing)}")
    rows, payload = [], {"fixed_threshold": fixed_threshold, "degradations": {}}
    for (degradation, level), group in predictions.groupby(["degradation", "level"]):
        eer, eer_threshold = eer_with_threshold(group.label, group.logit)
        fpr, fnr = fpr_fnr_at_threshold(group.label, group.logit, fixed_threshold)
        row = {
            "degradation": str(degradation),
            "level": str(level),
            "auc": roc_auc(group.label, group.logit),
            "eer": eer,
            "eer_threshold": eer_threshold,
            "fpr_fixed": fpr,
            "fnr_fixed": fnr,
            "n": int(len(group)),
        }
        if "training_augmentation" in group:
            row["training_augmentation"] = str(group["training_augmentation"].iloc[0])
        rows.append(row)
        payload["degradations"].setdefault(str(degradation), []).append(row)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / f"{run_name}.json"
    json_path.write_text(json.dumps(payload, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    pd.DataFrame(rows).to_csv(output / f"{run_name}.csv", index=False)
    figure, axis = plt.subplots(figsize=(8, 4))
    result_frame = pd.DataFrame(rows)
    grouping = (
        ["degradation", "training_augmentation"]
        if "training_augmentation" in result_frame
        else ["degradation"]
    )
    for keys, group in result_frame.groupby(grouping):
        group = group.sort_values("level")
        label = " / ".join(keys) if isinstance(keys, tuple) else str(keys)
        axis.plot(group["level"], group["eer"], marker="o", label=label)
    axis.set(xlabel="degradation level", ylabel="EER", title="Robustness EER")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / f"{run_name}_eer.png", dpi=160)
    plt.close(figure)
    return json_path
