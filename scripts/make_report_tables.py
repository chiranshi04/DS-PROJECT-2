"""Build Markdown tables and plots from actual evaluation JSON results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, default=Path("results/evaluation_tables.md"))
    args = parser.parse_args()
    files = sorted(
        path for path in args.results_dir.glob("*.json") if path.name != "data_audit.json"
    )
    if not files:
        raise SystemExit(
            f"No evaluation JSON files found in {args.results_dir}; "
            "refusing to create placeholder tables."
        )
    rows = []
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if "metrics" not in payload or "protocol" not in payload:
            continue
        metrics = payload["metrics"]
        rows.append(
            (
                path.name,
                payload["protocol"],
                payload.get("split", ""),
                metrics.get("auc"),
                metrics.get("eer"),
                metrics.get("fixed_fpr"),
                metrics.get("fixed_fnr"),
                metrics.get("ece"),
            )
        )
    if not rows:
        raise SystemExit(
            "Evaluation JSON files were found, but none contain actual metrics; "
            "refusing to create placeholders."
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Evaluation results",
        "",
        "| File | Protocol | Split | ROC-AUC | EER | Fixed-threshold FPR | "
        "Fixed-threshold FNR | ECE |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                str(value) if not isinstance(value, float) else f"{value:.6f}" for value in row
            )
            + " |"
        )
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    figure, axis = plt.subplots(figsize=(max(5, len(rows) * 1.3), 4))
    axis.bar([row[0][:16] for row in rows], [row[3] for row in rows])
    axis.set_ylim(0, 1)
    axis.set_ylabel("ROC-AUC")
    axis.set_title("Evaluation ROC-AUC")
    axis.tick_params(axis="x", rotation=45)
    figure.tight_layout()
    figure.savefig(args.results_dir / "evaluation_auc.png", dpi=160)
    plt.close(figure)
    print(f"Wrote {args.output} and {args.results_dir / 'evaluation_auc.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
