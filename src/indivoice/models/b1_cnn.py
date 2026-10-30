"""Lightweight ResNet-style log-Mel classifiers."""

from __future__ import annotations

import torch
from torch import nn


class ResidualBlock(nn.Module):
    """Two-convolution residual block for spectrogram inputs."""

    def __init__(self, channels: int, stride: int = 1) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.skip = (
            nn.Identity()
            if stride == 1
            else nn.Sequential(
                nn.Conv2d(channels, channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(channels),
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.body(x) + self.skip(x))


class LogMelCNN(nn.Module):
    """Single-logit CPU classifier with tiny/small channel variants."""

    CHANNELS = {"tiny": (16, 32, 64), "small": (24, 48, 96)}

    def __init__(self, variant: str = "small", dropout: float = 0.2) -> None:
        super().__init__()
        if variant not in self.CHANNELS:
            raise ValueError(f"Unknown B1 variant: {variant}")
        channels = self.CHANNELS[variant]
        layers: list[nn.Module] = [
            nn.Conv2d(1, channels[0], 3, padding=1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
        ]
        for index, channel in enumerate(channels):
            if index:
                layers.extend(
                    [
                        nn.Conv2d(channels[index - 1], channel, 3, stride=2, padding=1, bias=False),
                        nn.BatchNorm2d(channel),
                        nn.ReLU(inplace=True),
                    ]
                )
            layers.append(ResidualBlock(channel))
        self.encoder = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(channels[-1], 1)
        self.variant = variant

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = x.unsqueeze(1)
        return self.classifier(self.dropout(self.pool(self.encoder(x)).flatten(1))).squeeze(-1)


def count_parameters(model: nn.Module) -> int:
    """Count trainable and frozen parameters."""
    return sum(parameter.numel() for parameter in model.parameters())
