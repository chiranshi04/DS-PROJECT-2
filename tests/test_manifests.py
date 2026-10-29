"""Leakage checks run on tiny synthetic fixture manifests only."""

import pytest

from indivoice.data.manifests import ManifestRecord, compare_metadata_overlap, validate_records


def record(
    path: str,
    split: str,
    speaker: str = "s1",
    generator: str = "human",
    language: str = "hi",
    digest: str = "",
) -> ManifestRecord:
    return ManifestRecord(
        path, digest, 0, language, speaker, "fixture", generator, 1.0, 16000, split
    )


def test_clean_fixture_passes() -> None:
    validate_records([record("a.wav", "train", "s1"), record("b.wav", "test", "s2")])


@pytest.mark.parametrize(
    "records",
    [
        [record("a.wav", "train", "s1"), record("b.wav", "test", "s1")],
        [
            record("a.wav", "train", "s1", digest="same"),
            record("b.wav", "test", "s2", digest="same"),
        ],
    ],
)
def test_speaker_or_hash_leakage_fails(records: list[ManifestRecord]) -> None:
    with pytest.raises(ValueError):
        validate_records(records)


def test_held_out_generator_in_training_fails() -> None:
    with pytest.raises(ValueError):
        validate_records(
            [record("a.wav", "train", "s1", generator="g2")], held_out_generators={"g2"}
        )


def test_held_out_language_in_training_fails() -> None:
    with pytest.raises(ValueError):
        validate_records([record("a.wav", "train", "s1", language="ta")], held_out_languages={"ta"})


def test_metadata_overlap_is_counted() -> None:
    left = [record("shared.wav", "test", "same-speaker")]
    right = [record("shared.wav", "test", "same-speaker"), record("other.wav", "test", "new")]
    assert compare_metadata_overlap(left, right) == {
        "shared_speakers": 1,
        "shared_source_utterances": 1,
    }
