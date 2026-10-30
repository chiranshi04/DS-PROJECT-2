"""Reusable deterministic robustness degradations at a fixed sample rate."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch
import torchaudio
from scipy import signal


class DegradationError(RuntimeError):
    """Raised when a requested degradation cannot be produced."""


def _validate(waveform: np.ndarray, sample_rate: int) -> np.ndarray:
    array = np.asarray(waveform, dtype=np.float32)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise ValueError("waveform must be a finite, non-empty mono array")
    return array


def additive_noise(
    waveform: np.ndarray, snr_db: float, seed: int = 2025, color: str = "white"
) -> np.ndarray:
    """Add deterministic white, pink, or brown noise at the requested SNR."""
    clean = _validate(waveform, 1)
    rng = np.random.default_rng(seed)
    noise = rng.normal(size=clean.size).astype(np.float32)
    if color == "pink":
        frequencies = np.fft.rfftfreq(clean.size)
        spectrum = np.fft.rfft(noise)
        spectrum /= np.sqrt(np.maximum(frequencies, 1.0 / clean.size))
        noise = np.fft.irfft(spectrum, clean.size).astype(np.float32)
    elif color == "brown":
        noise = np.cumsum(noise)
        noise /= max(float(np.max(np.abs(noise))), 1e-8)
    elif color != "white":
        raise ValueError("color must be white, pink, or brown")
    clean_power, noise_power = np.mean(clean**2), np.mean(noise**2)
    desired_noise_power = clean_power / (10.0 ** (snr_db / 10.0))
    return clean + noise * np.sqrt(desired_noise_power / max(noise_power, 1e-12))


def _ffmpeg_roundtrip(
    waveform: np.ndarray, sample_rate: int, codec: str, bitrate: str, seed: int = 2025
) -> np.ndarray:
    del seed
    if shutil.which("ffmpeg") is None:
        raise DegradationError("ffmpeg is required for codec degradation")
    waveform = _validate(waveform, sample_rate)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "input.wav"
        target = root / "output.wav"
        torchaudio.save(str(source), torch.from_numpy(waveform).unsqueeze(0), sample_rate)
        codec_args = {
            "mp3": ["-c:a", "libmp3lame"],
            "aac": ["-c:a", "aac"],
            "opus": ["-c:a", "libopus"],
            "g711": ["-c:a", "pcm_mulaw"],
        }.get(codec)
        if codec_args is None:
            raise ValueError(f"Unsupported codec: {codec}")
        output_rate = 8000 if codec == "g711" else sample_rate
        command = [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-i",
            str(source),
            *codec_args,
            "-b:a",
            bitrate,
            "-ar",
            str(output_rate),
            str(target),
        ]
        try:
            subprocess.run(command, check=True, capture_output=True)
            decoded, rate = torchaudio.load(str(target))
        except (OSError, subprocess.CalledProcessError, RuntimeError) as error:
            raise DegradationError(
                f"ffmpeg codec round-trip failed for {codec} {bitrate}"
            ) from error
    decoded = decoded.mean(dim=0)
    if rate != sample_rate:
        decoded = torchaudio.functional.resample(decoded.unsqueeze(0), rate, sample_rate).squeeze(0)
    return decoded.numpy().astype(np.float32)


def codec_degrade(waveform: np.ndarray, sample_rate: int, codec: str, bitrate: str) -> np.ndarray:
    """Encode/decode through ffmpeg and return mono audio at the input rate."""
    return _ffmpeg_roundtrip(waveform, sample_rate, codec, bitrate)


def reverb(
    waveform: np.ndarray, sample_rate: int, seed: int = 2025, rt60: float = 0.35
) -> np.ndarray:
    """Convolve with a deterministic simulated exponentially decaying room impulse response."""
    clean = _validate(waveform, sample_rate)
    rng = np.random.default_rng(seed)
    length = max(2, int(sample_rate * rt60))
    impulse = np.zeros(length, dtype=np.float32)
    impulse[0] = 1.0
    taps = rng.choice(np.arange(1, length), size=min(24, length - 1), replace=False)
    impulse[taps] = rng.normal(0, 0.15, len(taps)) * np.exp(-taps / (sample_rate * rt60))
    result = signal.fftconvolve(clean, impulse, mode="full")[: clean.size]
    return (result / max(float(np.max(np.abs(result))), 1e-8)).astype(np.float32)


def resample_roundtrip(
    waveform: np.ndarray, sample_rate: int, intermediate_rate: int = 8000
) -> np.ndarray:
    """Resample down and back up, preserving the requested output rate."""
    clean = _validate(waveform, sample_rate)
    tensor = torch.from_numpy(clean).unsqueeze(0)
    down = torchaudio.functional.resample(tensor, sample_rate, intermediate_rate)
    up = torchaudio.functional.resample(down, intermediate_rate, sample_rate)
    return up.squeeze(0).numpy().astype(np.float32)


def lowpass(waveform: np.ndarray, sample_rate: int, cutoff_hz: float = 3500.0) -> np.ndarray:
    """Apply a deterministic fourth-order low-pass filter."""
    clean = _validate(waveform, sample_rate)
    sos = signal.butter(4, cutoff_hz, btype="low", fs=sample_rate, output="sos")
    return signal.sosfiltfilt(sos, clean).astype(np.float32)


def duration_sweep(waveform: np.ndarray, sample_rate: int, seconds: float) -> np.ndarray:
    """Return a deterministic crop or zero-padded clip of the requested duration."""
    clean = _validate(waveform, sample_rate)
    length = int(round(seconds * sample_rate))
    if length <= 0:
        raise ValueError("seconds must be positive")
    if clean.size >= length:
        return clean[:length].copy()
    return np.pad(clean, (0, length - clean.size)).astype(np.float32)
