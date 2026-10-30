import shutil

import numpy as np
import pandas as pd
import pytest

from indivoice.robustness.augment import RandomAugment
from indivoice.robustness.degradations import (
    additive_noise,
    codec_degrade,
    duration_sweep,
    lowpass,
    resample_roundtrip,
    reverb,
)
from indivoice.robustness.runner import evaluate_scored_degradations

SR = 16000


def fixture_waveform() -> np.ndarray:
    time = np.arange(SR, dtype=np.float32) / SR
    return (0.4 * np.sin(2 * np.pi * 440 * time) + 0.02 * np.sin(2 * np.pi * 2200 * time)).astype(
        np.float32
    )


def test_deterministic_degradations_preserve_rate_and_change_audio() -> None:
    waveform = fixture_waveform()
    cases = [
        additive_noise(waveform, 10, seed=7),
        reverb(waveform, SR, seed=7),
        resample_roundtrip(waveform, SR),
        lowpass(waveform, SR),
    ]
    for degraded in cases:
        assert degraded.dtype == np.float32
        assert degraded.shape == waveform.shape
        assert np.isfinite(degraded).all()
        assert not np.allclose(degraded, waveform)
    assert np.array_equal(
        additive_noise(waveform, 10, seed=7), additive_noise(waveform, 10, seed=7)
    )
    assert np.array_equal(reverb(waveform, SR, seed=7), reverb(waveform, SR, seed=7))


def test_duration_sweep_is_deterministic_and_rate_aware() -> None:
    waveform = fixture_waveform()
    assert len(duration_sweep(waveform, SR, 1)) == SR
    assert len(duration_sweep(waveform, SR, 2)) == 2 * SR
    assert np.count_nonzero(duration_sweep(waveform[: SR // 2], SR, 1)[SR // 2 :]) == 0


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None,
    reason="ffmpeg is not installed; skipping codec degradation test",
)
def test_mp3_codec_roundtrip_preserves_rate() -> None:
    waveform = fixture_waveform()
    degraded = codec_degrade(waveform, SR, "mp3", "64k")
    assert degraded.dtype == np.float32
    assert degraded.size > 0
    assert not np.allclose(
        degraded[: min(len(degraded), len(waveform))], waveform[: min(len(degraded), len(waveform))]
    )


def test_random_augmentation_seed_and_off_switch() -> None:
    waveform = fixture_waveform()
    first = RandomAugment(SR, enabled=True, seed=11, operations=("noise",))(waveform)
    second = RandomAugment(SR, enabled=True, seed=11, operations=("noise",))(waveform)
    assert np.array_equal(first, second)
    assert not np.allclose(first, waveform)
    assert np.array_equal(RandomAugment(SR, enabled=False)(waveform), waveform)


def test_scored_robustness_runner_outputs_metrics(tmp_path) -> None:
    rows = []
    for degradation, offset in [("clean", 0.0), ("noise", 0.2)]:
        for label, speaker in [(0, "a"), (0, "b"), (1, "c"), (1, "d")]:
            rows.append(
                {
                    "degradation": degradation,
                    "level": "10dB",
                    "label": label,
                    "logit": (-1 if label == 0 else 1) + offset,
                    "speaker_id": speaker,
                }
            )
    path = evaluate_scored_degradations(pd.DataFrame(rows), 0.0, tmp_path)
    assert path.exists()
    assert (tmp_path / "robustness.csv").exists()
    assert (tmp_path / "robustness_eer.png").exists()
