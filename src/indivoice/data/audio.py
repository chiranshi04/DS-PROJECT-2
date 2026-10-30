"""Audio loading and deterministic fixed-window preprocessing."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import soundfile as sf
import torch
import torchaudio


class AudioDecodeError(RuntimeError):
    """Raised when an audio file is empty, corrupt, or cannot be decoded."""


@dataclass(frozen=True)
class AudioClip:
    """A normalized mono waveform and its original duration."""

    waveform: torch.Tensor
    sample_rate: int
    original_duration_s: float
    path: str


@dataclass(frozen=True)
class AudioWindow:
    """One fixed-size waveform crop with source timing metadata."""

    waveform: torch.Tensor
    start_s: float
    original_duration_s: float


def _validate_waveform(waveform: torch.Tensor, sample_rate: int, path: str) -> torch.Tensor:
    if waveform.numel() == 0 or sample_rate <= 0:
        raise AudioDecodeError(f"Audio is empty or has an invalid sample rate: {path}")
    waveform = waveform.detach().to(dtype=torch.float32)
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.ndim != 2 or not torch.isfinite(waveform).all():
        raise AudioDecodeError(f"Audio contains invalid samples or dimensions: {path}")
    waveform = waveform.mean(dim=0)
    peak = waveform.abs().max()
    if not torch.isfinite(peak) or float(peak) == 0.0:
        raise AudioDecodeError(f"Audio contains only silence or invalid samples: {path}")
    return waveform / peak


def _ffmpeg_load(path: str) -> tuple[torch.Tensor, int]:
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        path,
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "-ac",
        "1",
        "-ar",
        "16000",
        "pipe:1",
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True)
    except (OSError, subprocess.CalledProcessError) as error:
        raise AudioDecodeError(f"Could not decode audio with ffmpeg: {path}") from error
    return torch.from_numpy(np.frombuffer(result.stdout, dtype=np.float32).copy()), 16000


def load_audio(
    path: str | Path,
    target_sample_rate: int = 16000,
    trim_silence: bool = False,
    silence_threshold: float = 1e-4,
) -> AudioClip:
    """Decode common audio formats, convert to mono, resample, and peak-normalize."""
    source = str(path)
    if not Path(source).is_file():
        raise AudioDecodeError(f"Audio file does not exist: {source}")
    waveform: torch.Tensor
    sample_rate: int
    try:
        array, sample_rate = sf.read(source, always_2d=True, dtype="float32")
        waveform = torch.from_numpy(array.T.copy())
    except Exception:
        try:
            waveform, sample_rate = torchaudio.load(source)
        except Exception:
            waveform, sample_rate = _ffmpeg_load(source)
    waveform = _validate_waveform(waveform, sample_rate, source)
    duration = waveform.numel() / sample_rate
    if sample_rate != target_sample_rate:
        waveform = torchaudio.functional.resample(
            waveform.unsqueeze(0), sample_rate, target_sample_rate
        ).squeeze(0)
        waveform = waveform / waveform.abs().max().clamp_min(torch.finfo(waveform.dtype).eps)
    if trim_silence:
        waveform = _trim_silence(waveform, silence_threshold)
    return AudioClip(waveform, target_sample_rate, duration, source)


def _pad_or_tile(
    waveform: torch.Tensor, length: int, mode: Literal["zero", "tile"]
) -> torch.Tensor:
    if waveform.numel() >= length:
        return waveform[:length]
    if mode == "tile":
        repeats = (length + waveform.numel() - 1) // waveform.numel()
        return waveform.repeat(repeats)[:length]
    if mode != "zero":
        raise ValueError(f"Unknown short-clip mode: {mode}")
    return torch.nn.functional.pad(waveform, (0, length - waveform.numel()))


def _trim_silence(waveform: torch.Tensor, threshold: float) -> torch.Tensor:
    """Trim leading/trailing samples below an absolute-amplitude threshold."""
    active = torch.nonzero(waveform.abs() > threshold, as_tuple=False).flatten()
    if active.numel() == 0:
        raise AudioDecodeError("Audio becomes empty after silence trimming")
    return waveform[int(active[0]) : int(active[-1]) + 1]


def make_windows(
    waveform: torch.Tensor,
    sample_rate: int,
    crop_seconds: float = 4.0,
    training: bool = False,
    rng: np.random.Generator | None = None,
    overlap: float = 0.5,
    short_mode: Literal["zero", "tile"] = "zero",
) -> list[AudioWindow]:
    """Create one random training crop or deterministic 50%-overlap test windows."""
    if waveform.ndim != 1 or waveform.numel() == 0:
        raise ValueError("waveform must be a non-empty mono tensor")
    if not 0 <= overlap < 1:
        raise ValueError("overlap must be in [0, 1)")
    length = int(round(crop_seconds * sample_rate))
    if length <= 0:
        raise ValueError("crop_seconds must be positive")
    original_duration = waveform.numel() / sample_rate
    if waveform.numel() <= length:
        return [AudioWindow(_pad_or_tile(waveform, length, short_mode), 0.0, original_duration)]
    if training:
        generator = rng or np.random.default_rng()
        start = int(generator.integers(0, waveform.numel() - length + 1))
        return [
            AudioWindow(waveform[start : start + length], start / sample_rate, original_duration)
        ]
    step = max(1, int(round(length * (1.0 - overlap))))
    starts = list(range(0, waveform.numel() - length + 1, step))
    final_start = waveform.numel() - length
    if starts[-1] != final_start:
        starts.append(final_start)
    return [
        AudioWindow(waveform[start : start + length], start / sample_rate, original_duration)
        for start in starts
    ]
