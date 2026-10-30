"""Optional frozen SSL reference model; disabled by default."""

from __future__ import annotations

import torch
from torch import nn


class FrozenSSLHead(nn.Module):
    """Frozen WavLM/XLS-R backbone plus a trainable binary head.

    The optional ``transformers`` dependency is intentionally not in the default
    lightweight requirements. Enable this reference only in a separate environment.
    """

    def __init__(self, model_name: str = "microsoft/wavlm-base") -> None:
        super().__init__()
        try:
            from transformers import AutoModel
        except ImportError as error:
            raise ImportError(
                "B3 requires optional dependency transformers; it is off by default"
            ) from error
        self.backbone = AutoModel.from_pretrained(model_name)
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        hidden_size = int(self.backbone.config.hidden_size)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        output = self.backbone(input_values=waveform)
        pooled = output.last_hidden_state.mean(dim=1)
        return self.head(pooled).squeeze(-1)
