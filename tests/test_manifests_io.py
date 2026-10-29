from pathlib import Path

from indivoice.data.manifests import ManifestRecord, read_manifest, write_manifest


def test_manifest_round_trip(tmp_path: Path) -> None:
    records = [
        ManifestRecord("b.wav", "hash-b", 1, "ta", "s2", "fixture", "g1", 2.0, 16000, "test")
    ]
    output = tmp_path / "manifest.csv"
    write_manifest(records, output)
    assert read_manifest(output) == records
    assert output.read_text(encoding="utf-8").splitlines()[0].startswith("path_or_uri,sha256,label")
