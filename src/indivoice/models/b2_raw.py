"""Tiny raw-waveform anti-spoofing model."""

from __future__ import annotations

import torch
from torch import nn


class RawWaveTiny(nn.Module):
    """A compact strided-convolution waveform encoder with global pooling."""

    def __init__(self, channels: tuple[int, ...] = (8, 16, 32, 64), dropout: float = 0.2) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_channels = 1
        for channel in channels:
            layers.extend(
                [
                    nn.Conv1d(
                        in_channels, channel, kernel_size=11, stride=4, padding=5, bias=False
                    ),
                    nn.BatchNorm1d(channel),
                    nn.ReLU(inplace=True),
                ]
            )
            in_channels = channel
        self.encoder = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(channels[-1], 1)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.ndim == 2:
            waveform = waveform.unsqueeze(1)
        return self.classifier(self.dropout(self.pool(self.encoder(waveform)).squeeze(-1))).squeeze(
            -1
        )
