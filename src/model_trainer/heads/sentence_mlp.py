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


class SentenceMLPHead(Head):
    def __init__(
        self,
        hidden_dim: int = 256,
        num_outputs: int = 1,
        dropout: float = 0.1,
        offsets_key: str = "sentence_offsets",
        backbone_output_key: str = "hidden_states",
        backbone_output_dim: int | None = None,
    ) -> None:
        super().__init__(
            input_keys=[offsets_key],
            backbone_output_key=backbone_output_key,
        )
        self.offsets_key = offsets_key
        self.mlp = nn.Sequential(
            nn.Linear(backbone_output_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_outputs),
        )

    def mean_pool_by_offsets(
        self,
        hidden: torch.Tensor,    # (B, T, D)
        offsets: torch.Tensor,   # (B, S, 2) — (start, end), end exclusive
    ) -> torch.Tensor:
        b, t, _ = hidden.shape
        starts = offsets[..., 0].unsqueeze(-1)            # (B, S, 1)
        ends = offsets[..., 1].unsqueeze(-1)              # (B, S, 1)
        positions = torch.arange(t, device=hidden.device).view(1, 1, t)
        token_mask = (positions >= starts) & (positions < ends)  # (B, S, T)
        weights = token_mask.to(hidden.dtype)
        counts = weights.sum(dim=-1, keepdim=True).clamp(min=1.0)
        return torch.einsum("bst,btd->bsd", weights, hidden) / counts

    def forward(self, backbone_output: dict[str, Any], **head_inputs: Any) -> torch.Tensor:
        h = backbone_output[self.backbone_output_key]
        offsets = head_inputs[self.offsets_key]
        s = self.mean_pool_by_offsets(h, offsets)
        return self.mlp(s)
