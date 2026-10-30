"""Linear-frequency cepstral coefficient extraction."""

from __future__ import annotations

import torch
import torchaudio


class LFCCExtractor:
    """Extract normalized LFCC features from a mono waveform."""

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 512,
        hop_length: int = 160,
        win_length: int = 400,
        n_lfcc: int = 40,
        n_filter: int = 128,
        eps: float = 1e-6,
    ) -> None:
        self.n_fft, self.hop_length, self.win_length = n_fft, hop_length, win_length
        self.n_lfcc, self.eps = n_lfcc, eps
        self.window = torch.hann_window(win_length)
        self.filter = torchaudio.functional.create_dct(n_lfcc, n_filter, norm="ortho").T

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """Return a tensor shaped ``(n_lfcc, frames)`` with mean zero and std one."""
        spectrum = (
            torch.stft(
                waveform,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=self.window,
                center=True,
                return_complex=True,
            )
            .abs()
            .square()
        )
        # DCT on log linear-frequency filter energies is a simple, reproducible LFCC baseline.
        log_power = torch.log(spectrum.clamp_min(self.eps)).transpose(0, 1)
        if log_power.shape[1] < self.filter.shape[1]:
            log_power = torch.nn.functional.pad(
                log_power, (0, self.filter.shape[1] - log_power.shape[1])
            )
        cepstra = torch.matmul(self.filter, log_power[:, : self.filter.shape[1]].T)
        return (cepstra - cepstra.mean()) / cepstra.std(unbiased=False).clamp_min(self.eps)
