"""CPU log-Mel feature extraction."""

from __future__ import annotations

import torch
import torchaudio


class LogMelExtractor:
    """Extract normalized 80-bin log-Mel spectrograms from fixed waveform windows."""

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 512,
        hop_length: int = 160,
        win_length: int = 400,
        n_mels: int = 80,
        f_min: float = 20.0,
        f_max: float | None = 7600.0,
        eps: float = 1e-6,
    ) -> None:
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max or sample_rate / 2
        self.eps = eps
        self.window = torch.hann_window(win_length)
        self.mel_filter = torchaudio.functional.melscale_fbanks(
            n_freqs=n_fft // 2 + 1,
            f_min=f_min,
            f_max=self.f_max,
            n_mels=n_mels,
            sample_rate=sample_rate,
            norm="slaney",
            mel_scale="htk",
        ).T

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """Return a tensor shaped ``(n_mels, frames)`` with mean zero and std one."""
        if waveform.ndim != 1:
            raise ValueError("waveform must be mono with shape (samples,)")
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
        mel = torch.matmul(self.mel_filter, spectrum)
        log_mel = torch.log(mel.clamp_min(self.eps))
        return (log_mel - log_mel.mean()) / log_mel.std(unbiased=False).clamp_min(self.eps)
