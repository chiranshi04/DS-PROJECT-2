"""Temperature scaling fitted only on development logits."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn


class TemperatureScaler:
    """One-parameter temperature model for binary logits."""

    def __init__(self, temperature: float = 1.0) -> None:
        self.temperature = float(temperature)

    def fit(self, logits: np.ndarray, labels: np.ndarray, max_iter: int = 50) -> TemperatureScaler:
        tensor_logits = torch.tensor(logits, dtype=torch.float32)
        tensor_labels = torch.tensor(labels, dtype=torch.float32)
        parameter = torch.nn.Parameter(torch.tensor(np.log(max(self.temperature, 1e-3))))
        optimizer = torch.optim.LBFGS([parameter], lr=0.1, max_iter=max_iter)
        criterion = nn.BCEWithLogitsLoss()

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            loss = criterion(tensor_logits / parameter.exp(), tensor_labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        self.temperature = float(parameter.detach().exp().clamp_min(1e-3))
        return self

    def transform_logits(self, logits: np.ndarray) -> np.ndarray:
        return np.asarray(logits, dtype=float) / self.temperature

    def probabilities(self, logits: np.ndarray) -> np.ndarray:
        scaled = self.transform_logits(logits)
        return 1.0 / (1.0 + np.exp(-np.clip(scaled, -60, 60)))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps({"temperature": self.temperature}, indent=2) + "\n", encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str | Path) -> TemperatureScaler:
        return cls(json.loads(Path(path).read_text(encoding="utf-8"))["temperature"])


def reliability_bins(
    labels: np.ndarray, probabilities: np.ndarray, bins: int = 10
) -> list[dict[str, float | int]]:
    """Return expected/observed accuracy data for a reliability diagram."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    output = []
    for left, right in zip(edges[:-1], edges[1:], strict=False):
        mask = (probabilities >= left) & (
            probabilities <= right if right == 1 else probabilities < right
        )
        output.append(
            {
                "confidence": float(probabilities[mask].mean())
                if mask.any()
                else float((left + right) / 2),
                "accuracy": float(labels[mask].mean()) if mask.any() else float("nan"),
                "count": int(mask.sum()),
            }
        )
    return output
