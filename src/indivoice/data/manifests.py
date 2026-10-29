"""Deterministic manifest construction and leakage checks for Phase 2."""

from __future__ import annotations

import csv
import hashlib
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

COLUMNS = [
    "path_or_uri", "sha256", "label", "language", "speaker_id", "source_corpus",
    "generator_id", "duration_s", "sample_rate", "split",
]


@dataclass(frozen=True)
class ManifestRecord:
    """One audio item and its provenance/split metadata."""

    path_or_uri: str
    sha256: str
    label: int
    language: str
    speaker_id: str
    source_corpus: str
    generator_id: str
    duration_s: float
    sample_rate: int
    split: str

    def as_dict(self) -> dict[str, str | int | float]:
        return asdict(self)


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Hash a local file without loading it entirely into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest(path: str | Path) -> list[ManifestRecord]:
    """Read a CSV manifest and coerce the typed fields."""
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = csv.DictReader(handle)
        missing = set(COLUMNS) - set(rows.fieldnames or [])
        if missing:
            raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
        return [
            ManifestRecord(
                path_or_uri=row["path_or_uri"], sha256=row["sha256"], label=int(row["label"]),
                language=row["language"], speaker_id=row["speaker_id"],
                source_corpus=row["source_corpus"], generator_id=row["generator_id"],
                duration_s=float(row["duration_s"]), sample_rate=int(row["sample_rate"]),
                split=row["split"],
            )
            for row in rows
        ]


def write_manifest(records: Iterable[ManifestRecord], path: str | Path) -> None:
    """Write records with a stable header and row ordering."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(records, key=lambda record: (record.split, record.path_or_uri))
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(record.as_dict() for record in ordered)


def _groups(records: Iterable[ManifestRecord], key: str) -> dict[str, set[str]]:
    grouped: dict[str, set[str]] = defaultdict(set)
    for record in records:
        grouped[getattr(record, key)].add(record.split)
    return grouped


def validate_speaker_disjoint(records: Iterable[ManifestRecord]) -> None:
    """Fail when a speaker appears in more than one split."""
    violations = {key: splits for key, splits in _groups(records, "speaker_id").items() if len(splits) > 1}
    if violations:
        raise ValueError(f"Speaker leakage across splits: {violations}")


def validate_generator_disjoint(records: Iterable[ManifestRecord], held_out: set[str]) -> None:
    """Fail when any held-out generator occurs in a training row."""
    leaked = sorted({record.generator_id for record in records if record.split == "train"} & held_out)
    if leaked:
        raise ValueError(f"Held-out generators in training: {leaked}")


def validate_language_disjoint(records: Iterable[ManifestRecord], held_out: set[str]) -> None:
    """Fail when a held-out language occurs in a training row."""
    leaked = sorted({record.language for record in records if record.split == "train"} & held_out)
    if leaked:
        raise ValueError(f"Held-out languages in training: {leaked}")


def validate_hash_disjoint(records: Iterable[ManifestRecord]) -> None:
    """Fail when the same non-empty file hash occurs across splits."""
    seen: dict[str, str] = {}
    violations: set[str] = set()
    for record in records:
        if not record.sha256:
            continue
        previous = seen.setdefault(record.sha256, record.split)
        if previous != record.split:
            violations.add(record.sha256)
    if violations:
        raise ValueError(f"Duplicate file hashes across splits: {sorted(violations)}")


def validate_records(records: Iterable[ManifestRecord], *, held_out_generators: set[str] | None = None,
                     held_out_languages: set[str] | None = None) -> None:
    """Apply all checks relevant to one protocol manifest."""
    materialized = list(records)
    validate_speaker_disjoint(materialized)
    validate_hash_disjoint(materialized)
    if held_out_generators:
        validate_generator_disjoint(materialized, held_out_generators)
    if held_out_languages:
        validate_language_disjoint(materialized, held_out_languages)


def limit_balanced_records(
    records: Iterable[ManifestRecord],
    max_hours_per_language_per_class: float | None = None,
) -> list[ManifestRecord]:
    """Apply a deterministic per-language/class duration cap."""
    materialized = sorted(records, key=lambda record: (record.language, record.label, record.path_or_uri))
    if max_hours_per_language_per_class is None:
        return materialized
    cap_seconds = max_hours_per_language_per_class * 3600
    selected: list[ManifestRecord] = []
    for language in sorted({record.language for record in materialized}):
        for label in (0, 1):
            elapsed = 0.0
            candidates = [
                record for record in materialized
                if record.language == language and record.label == label
            ]
            for record in candidates:
                if elapsed + record.duration_s > cap_seconds:
                    break
                selected.append(record)
                elapsed += record.duration_s
    return selected


def _replace_split(record: ManifestRecord, split: str) -> ManifestRecord:
    """Return a record with only its split changed."""
    values = record.as_dict()
    values["split"] = split
    return ManifestRecord(**values)


def assign_protocol(records: Iterable[ManifestRecord], protocol: str, held_out: str | None = None) -> list[ManifestRecord]:
    """Assign a source manifest to one deterministic protocol variant."""
    materialized = list(records)
    if protocol == "P1":
        return materialized
    if protocol == "P2":
        if held_out is None:
            raise ValueError("P2 requires held_out generator")
        return [_replace_split(record, "test" if record.generator_id == held_out else record.split) for record in materialized]
    if protocol == "P3":
        if held_out is None:
            raise ValueError("P3 requires held_out language")
        return [_replace_split(record, "test" if record.language == held_out else record.split) for record in materialized]
    if protocol == "P4":
        if held_out is None:
            raise ValueError("P4 requires test language")
        return [_replace_split(record, "test" if record.language == held_out else "train") for record in materialized]
    if protocol == "P5":
        return [_replace_split(record, "test") for record in materialized]
    raise ValueError(f"Unknown protocol: {protocol}")


def compare_metadata_overlap(left: Iterable[ManifestRecord], right: Iterable[ManifestRecord]) -> dict[str, int]:
    """Compare speaker IDs and source paths between two metadata manifests."""
    left_records, right_records = list(left), list(right)
    left_speakers = {record.speaker_id for record in left_records if record.speaker_id}
    right_speakers = {record.speaker_id for record in right_records if record.speaker_id}
    left_paths = {record.path_or_uri for record in left_records if record.path_or_uri}
    right_paths = {record.path_or_uri for record in right_records if record.path_or_uri}
    return {
        "shared_speakers": len(left_speakers & right_speakers),
        "shared_source_utterances": len(left_paths & right_paths),
    }
