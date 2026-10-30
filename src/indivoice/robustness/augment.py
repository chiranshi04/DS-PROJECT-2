"""Config-controlled training-time robustness augmentation."""

from __future__ import annotations

import numpy as np

from .degradations import additive_noise, lowpass, resample_roundtrip, reverb


class RandomAugment:
    """Sample one configured degradation per call from a seeded generator."""

    def __init__(
        self,
        sample_rate: int = 16000,
        enabled: bool = False,
        seed: int = 2025,
        operations: tuple[str, ...] = ("noise", "reverb", "resample", "lowpass"),
    ) -> None:
        self.sample_rate, self.enabled, self.rng = sample_rate, enabled, np.random.default_rng(seed)
        self.operations = operations

    def __call__(self, waveform: np.ndarray) -> np.ndarray:
        """Return an augmented copy or the input copy when disabled."""
        clean = np.asarray(waveform, dtype=np.float32).copy()
        if not self.enabled or not self.operations:
            return clean
        operation = self.operations[int(self.rng.integers(0, len(self.operations)))]
        if operation == "noise":
            return additive_noise(
                clean, float(self.rng.choice([20, 10, 5, 0])), int(self.rng.integers(0, 2**31))
            )
        if operation == "reverb":
            return reverb(clean, self.sample_rate, int(self.rng.integers(0, 2**31)))
        if operation == "resample":
            return resample_roundtrip(clean, self.sample_rate)
        if operation == "lowpass":
            return lowpass(clean, self.sample_rate)
        raise ValueError(f"Unknown augmentation operation: {operation}")
