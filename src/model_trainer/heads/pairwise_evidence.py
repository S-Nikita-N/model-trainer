"""Pairwise evidence head: per (response_i, context_j), produce a score.

Pools both response and context sentences from offsets, builds NLI-style pair
features ``[h_r ; h_c ; h_r * h_c ; |h_r - h_c|]``, and runs them through an
MLP. ``num_outputs`` controls how many independent labels per pair
(e.g. ``2`` for ``{support_evidence, contradict_evidence}``).

Pad sentences are encoded as ``[0, 0]`` offsets, so no separate mask keys are
needed — see ``mean_pool_by_offsets``.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from model_trainer.heads.base import Head


class PairwiseEvidenceHead(Head):
    def __init__(
        self,
        hidden_dim: int = 256,
        num_outputs: int = 2,
        dropout: float = 0.1,
        response_offsets_key: str = "response_fact_offsets",
        context_offsets_key: str = "context_fact_offsets",
        backbone_input_key: str = "hidden_states",
    ) -> None:
        super().__init__(
            input_keys=[response_offsets_key, context_offsets_key],
            backbone_input_key=backbone_input_key,
        )
        self.response_offsets_key = response_offsets_key
        self.context_offsets_key = context_offsets_key
        self.mlp = nn.Sequential(
            nn.LazyLinear(hidden_dim),
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
        h = backbone_output[self.backbone_input_key]
        s_resp = self.mean_pool_by_offsets(h, head_inputs[self.response_offsets_key])
        s_ctx = self.mean_pool_by_offsets(h, head_inputs[self.context_offsets_key])

        m, n = s_resp.shape[1], s_ctx.shape[1]
        h_r = s_resp.unsqueeze(2).expand(-1, -1, n, -1)
        h_c = s_ctx.unsqueeze(1).expand(-1, m, -1, -1)
        pair_feat = torch.cat([h_r, h_c, h_r * h_c, (h_r - h_c).abs()], dim=-1)

        return self.mlp(pair_feat)
