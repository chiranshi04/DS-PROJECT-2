"""Audit local data availability without downloading or fabricating facts."""
from pathlib import Path


def main() -> int:
    data_dir = Path("data")
    print(f"Data directory exists: {data_dir.exists()}")
    print("No dataset audit is performed in Phase 1; see docs/DATASETS.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
