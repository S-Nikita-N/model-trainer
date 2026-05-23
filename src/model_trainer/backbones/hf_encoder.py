"""Pure HuggingFace encoder (no classification head).

Returns ``{"hidden_states": ..., "attention_mask": ...}``. Pair this with a
head from ``model_trainer.heads`` (``SentenceMLPHead``, ``PairwiseEvidenceHead``,
or whatever you implement) that picks ``hidden_states`` and applies pooling.

For HuggingFace models that already produce ``logits`` (such as
``AutoModelForSequenceClassification``), use a plain ``_target_:
transformers.AutoModelForSequenceClassification.from_pretrained`` config
together with an ``IdentityHead`` instead.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from transformers import AutoModel


class HFEncoder(nn.Module):
    def __init__(
        self,
        encoder_name: str,
        local_files_only: bool = False,
        trust_remote_code: bool = False,
        input_keys: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(
            encoder_name,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
        )
        self.input_keys: list[str] = list(input_keys)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **_: Any,
    ) -> dict[str, torch.Tensor]:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return {
            "hidden_states": out.last_hidden_state,
            "attention_mask": attention_mask,
        }
