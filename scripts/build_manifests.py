"""Build and validate leakage-safe protocol manifests from a verified source CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

from indivoice.config import load_config
from indivoice.data.manifests import (
    assign_protocol,
    limit_balanced_records,
    read_manifest,
    validate_records,
    write_manifest,
)


def build(records, output_dir: Path, max_hours: float | None) -> int:
    records = limit_balanced_records(records, max_hours)
    write_manifest(records, output_dir / "P1.csv")
    validate_records(records)

    generators = sorted(
        {record.generator_id for record in records if record.generator_id != "human"}
    )
    for generator in generators:
        variant = assign_protocol(records, "P2", generator)
        validate_records(variant, held_out_generators={generator})
        write_manifest(variant, output_dir / f"P2-generator-{generator}.csv")
    if generators:
        grouped = assign_protocol(records, "P2", "__all_generators__")
        grouped = [
            record
            for record in grouped
            if record.generator_id == "human" or record.split != "train"
        ]
        validate_records(grouped)
        write_manifest(grouped, output_dir / "P2-grouped.csv")

    for language in sorted({record.language for record in records}):
        variant = assign_protocol(records, "P3", language)
        validate_records(variant, held_out_languages={language})
        write_manifest(variant, output_dir / f"P3-language-{language}.csv")
        pairwise = assign_protocol(records, "P4", language)
        validate_records(pairwise)
        write_manifest(pairwise, output_dir / f"P4-test-{language}.csv")

    external = assign_protocol(records, "P5")
    write_manifest(external, output_dir / "P5-external-test-only.csv")
    return len(records)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=None, help="Verified source-record CSV")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    config = load_config("data")
    source = args.source or Path(str(config.source_manifest))
    output_dir = args.output_dir or Path(str(config.manifest_dir))
    if not source.exists():
        print(f"No verified source manifest found at {source}. Run the Phase 2 data audit first.")
        return 0
    count = build(read_manifest(source), output_dir, config.max_hours_per_language_per_class)
    print(f"Wrote validated P1-P5 protocol manifests for {count} source records to {output_dir}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
