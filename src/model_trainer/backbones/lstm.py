"""Minimal LSTM classifier.

Contract: ``forward(input_ids, attention_mask=None, **kwargs) -> logits``.
``attention_mask`` is accepted (for a uniform batch contract with
transformers) but ignored.
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

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        emb = self.embedding(input_ids)
        out, _ = self.lstm(emb)
        last = out[:, -1, :]
        return self.fc(last)
