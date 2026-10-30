"""Run robustness metrics from a scored degraded-example CSV."""

from __future__ import annotations

import argparse

import pandas as pd

from .runner import evaluate_scored_degradations


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scores", required=True, help="CSV with degradation, level, label, and logit columns"
    )
    parser.add_argument("--fixed-threshold", required=True, type=float)
    parser.add_argument("--output-dir", default="results/robustness")
    parser.add_argument("--run-name", default="robustness")
    args = parser.parse_args()
    result_path = evaluate_scored_degradations(
        pd.read_csv(args.scores), args.fixed_threshold, args.output_dir, args.run_name
    )
    print(f"Wrote {result_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
