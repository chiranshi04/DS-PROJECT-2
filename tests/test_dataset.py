from pathlib import Path

import numpy as np
import soundfile as sf

from indivoice.data.dataset import ManifestAudioDataset, make_dataloader
from indivoice.data.manifests import ManifestRecord, write_manifest
from indivoice.features.mel import LogMelExtractor


def test_manifest_dataset_cpu(tmp_path: Path) -> None:
    audio = tmp_path / "example.wav"
    sf.write(audio, np.sin(np.linspace(0, 1000, 16000)).astype("float32"), 16000)
    manifest = tmp_path / "manifest.csv"
    write_manifest(
        [
            ManifestRecord(
                str(audio), "", 0, "hi", "speaker", "fixture", "human", 1.0, 16000, "test"
            )
        ],
        manifest,
    )
    dataset = ManifestAudioDataset(
        manifest, LogMelExtractor(), training=False, cache_dir=tmp_path / "cache"
    )
    batch = next(iter(make_dataloader(dataset, batch_size=1)))
    assert tuple(batch["features"].shape) == (1, 80, 401)
    assert batch["label"].item() == 0
    assert list((tmp_path / "cache").glob("*.pt"))
