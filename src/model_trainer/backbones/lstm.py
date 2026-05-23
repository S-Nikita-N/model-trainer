"""Minimal LSTM with a built-in classification head.

Returns ``{"logits": (B, num_labels)}`` — pair with ``IdentityHead`` for
simple classification (the default ``head=identity`` setup).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class LSTMBackbone(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_layers: int,
        num_labels: int,
        dropout: float = 0.2,
        input_keys: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, num_labels)
        self.input_keys: list[str] = list(input_keys or ["input_ids"])

    def forward(
        self,
        input_ids: torch.Tensor,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        emb = self.embedding(input_ids)
        out, _ = self.lstm(emb)
        last = out[:, -1, :]
        logits = self.fc(last)
        return {"logits": logits}
