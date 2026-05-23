"""Sentence-level head: pool by token offsets, then small MLP.

Reads from the batch:
  - ``offsets_key`` (``(B, S, 2)``)

Pad sentences are encoded as ``[0, 0]`` offsets in collate, so no separate
mask key is needed — see ``mean_pool_by_offsets``.

Uses ``nn.LazyLinear`` for the first projection so the head doesn't have to
know the encoder's ``hidden_size`` at config time — it's inferred on the
first forward.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from model_trainer.heads.base import Head
from model_trainer.heads.pooling import mean_pool_by_offsets


class SentenceMLPHead(Head):
    def __init__(
        self,
        offsets_key: str = "sentence_offsets",
        hidden_dim: int = 256,
        num_outputs: int = 1,
        dropout: float = 0.1,
        backbone_input_key: str = "hidden_states",
    ) -> None:
        super().__init__(
            input_keys=[offsets_key],
            backbone_input_key=backbone_input_key,
        )
        self.offsets_key = offsets_key
        self.mlp = nn.Sequential(
            nn.LazyLinear(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_outputs),
        )

    def forward(self, backbone_output: dict[str, Any], **head_inputs: Any) -> torch.Tensor:
        h = backbone_output[self.backbone_input_key]
        offsets = head_inputs[self.offsets_key]
        s = mean_pool_by_offsets(h, offsets)
        return self.mlp(s)
