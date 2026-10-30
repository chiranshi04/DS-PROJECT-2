"""CPU fixtures for audio loading, windowing, and feature determinism."""

from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from indivoice.data.audio import AudioDecodeError, load_audio, make_windows
from indivoice.features.lfcc import LFCCExtractor
from indivoice.features.mel import LogMelExtractor

SR = 16000


def write_fixture(path: Path, seconds: float = 1.0, sample_rate: int = SR) -> None:
    time = np.arange(int(seconds * sample_rate), dtype=np.float32) / sample_rate
    signal = 0.4 * np.sin(2 * np.pi * 440 * time)
    sf.write(path, signal, sample_rate)


def test_loader_resamples_mono_and_peak_normalizes(tmp_path: Path) -> None:
    path = tmp_path / "stereo.wav"
    time = np.arange(8000, dtype=np.float32) / 8000
    sf.write(
        path, np.stack([0.2 * np.sin(2 * np.pi * 220 * time), np.zeros_like(time)], axis=1), 8000
    )
    clip = load_audio(path)
    assert clip.sample_rate == SR
    assert clip.waveform.ndim == 1
    assert clip.waveform.numel() == SR
    assert torch.isclose(clip.waveform.abs().max(), torch.tensor(1.0))
    assert clip.original_duration_s == 1.0


def test_corrupt_and_silent_files_raise(tmp_path: Path) -> None:
    corrupt = tmp_path / "bad.wav"
    corrupt.write_bytes(b"not audio")
    with np.testing.assert_raises(AudioDecodeError):
        load_audio(corrupt)
    silent = tmp_path / "silent.wav"
    sf.write(silent, np.zeros(SR, dtype=np.float32), SR)
    with np.testing.assert_raises(AudioDecodeError):
        load_audio(silent)


def test_silence_trimming_is_optional(tmp_path: Path) -> None:
    path = tmp_path / "padded.wav"
    signal = np.concatenate([np.zeros(4000), np.ones(8000) * 0.2, np.zeros(4000)]).astype("float32")
    sf.write(path, signal, SR)
    untouched = load_audio(path)
    trimmed = load_audio(path, trim_silence=True)
    assert untouched.waveform.numel() == SR
    assert trimmed.waveform.numel() < untouched.waveform.numel()
    assert trimmed.original_duration_s == untouched.original_duration_s


def test_short_clip_padding_and_tiling() -> None:
    waveform = torch.ones(8000)
    padded = make_windows(waveform, SR, short_mode="zero")[0].waveform
    tiled = make_windows(waveform, SR, short_mode="tile")[0].waveform
    assert padded.shape == tiled.shape == (4 * SR,)
    assert torch.count_nonzero(padded[8000:]) == 0
    assert torch.allclose(tiled[:8000], tiled[8000:16000])


def test_random_crop_seed_and_deterministic_test_windows() -> None:
    waveform = torch.arange(6 * SR, dtype=torch.float32)
    first = make_windows(waveform, SR, training=True, rng=np.random.default_rng(7))[0]
    second = make_windows(waveform, SR, training=True, rng=np.random.default_rng(7))[0]
    assert torch.equal(first.waveform, second.waveform)
    windows = make_windows(waveform, SR, training=False)
    assert len(windows) == 2
    assert windows[1].start_s - windows[0].start_s == 2.0


def test_logmel_shape_normalization_and_determinism() -> None:
    waveform = torch.sin(torch.linspace(0, 2 * torch.pi * 440 * 4, 4 * SR))
    extractor = LogMelExtractor()
    first, second = extractor(waveform), extractor(waveform)
    assert first.shape == (80, 401)
    assert torch.equal(first, second)
    assert abs(float(first.mean())) < 1e-5
    assert abs(float(first.std(unbiased=False)) - 1.0) < 1e-5


def test_lfcc_shape() -> None:
    waveform = torch.randn(4 * SR, generator=torch.Generator().manual_seed(5))
    features = LFCCExtractor()(waveform)
    assert features.shape == (40, 401)
    assert torch.isfinite(features).all()
