"""PyTorch datasets for manifest-backed audio features."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from indivoice.data.audio import load_audio, make_windows
from indivoice.data.manifests import ManifestRecord, read_manifest


class ManifestAudioDataset(Dataset[dict[str, object]]):
    """Extract one feature tensor per manifest row, with optional test windows and caching."""

    def __init__(
        self,
        manifest: str | Path | list[ManifestRecord],
        extractor: Callable[[torch.Tensor], torch.Tensor],
        training: bool = False,
        sample_rate: int = 16000,
        crop_seconds: float = 4.0,
        short_mode: str = "zero",
        trim_silence: bool = False,
        silence_threshold: float = 1e-4,
        cache_dir: str | Path | None = None,
        augmentation: Callable[[torch.Tensor], torch.Tensor] | None = None,
        seed: int = 2025,
    ) -> None:
        self.records = (
            read_manifest(manifest) if isinstance(manifest, str | Path) else list(manifest)
        )
        self.extractor, self.training, self.sample_rate = extractor, training, sample_rate
        self.crop_seconds, self.short_mode = crop_seconds, short_mode
        self.trim_silence, self.silence_threshold = trim_silence, silence_threshold
        self.augmentation, self.seed = augmentation, seed
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._index = [(i, 0) for i in range(len(self.records))]
        if not training:
            self._index = []
            for i, record in enumerate(self.records):
                clip = load_audio(record.path_or_uri, sample_rate, trim_silence, silence_threshold)
                count = len(
                    make_windows(
                        clip.waveform, sample_rate, crop_seconds, False, short_mode=short_mode
                    )
                )
                self._index.extend((i, window) for window in range(count))

    def __len__(self) -> int:
        return len(self._index)

    def _cache_path(self, record: ManifestRecord, window: int) -> Path | None:
        if self.cache_dir is None:
            return None
        key = hashlib.sha256(
            f"{record.path_or_uri}|{window}|{self.extractor!r}".encode()
        ).hexdigest()
        return self.cache_dir / f"{key}.pt"

    def __getitem__(self, index: int) -> dict[str, object]:
        record_index, window_index = self._index[index]
        record = self.records[record_index]
        cache_path = self._cache_path(record, window_index)
        if cache_path and cache_path.exists():
            features = torch.load(cache_path, weights_only=True)
        else:
            clip = load_audio(
                record.path_or_uri,
                self.sample_rate,
                self.trim_silence,
                self.silence_threshold,
            )
            rng = np.random.default_rng(self.seed + record_index)
            windows = make_windows(
                clip.waveform,
                self.sample_rate,
                self.crop_seconds,
                self.training,
                rng,
                short_mode=self.short_mode,
            )
            window = windows[0] if self.training else windows[window_index]
            waveform = self.augmentation(window.waveform) if self.augmentation else window.waveform
            features = self.extractor(waveform)
            if cache_path:
                torch.save(features, cache_path)
        return {
            "features": features,
            "label": record.label,
            "language": record.language,
            "speaker_id": record.speaker_id,
            "path_or_uri": record.path_or_uri,
            "original_duration_s": record.duration_s,
        }


def make_dataloader(
    dataset: ManifestAudioDataset,
    batch_size: int = 32,
    num_workers: int = 0,
    shuffle: bool | None = None,
) -> DataLoader[dict[str, object]]:
    """Create a CPU-friendly DataLoader; shuffle defaults to training mode."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=dataset.training if shuffle is None else shuffle,
        num_workers=num_workers,
    )
