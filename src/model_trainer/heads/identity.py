"""Identity head: pass-through of one key from backbone output.

Use this when the backbone already produces logits (LSTMBackbone with its
own ``fc`` layer, or ``AutoModelForSequenceClassification``). Default reads
``backbone_output["logits"]``.
"""

from __future__ import annotations

from typing import Any

import torch

from model_trainer.heads.base import Head


class IdentityHead(Head):
    def __init__(self, backbone_output_key: str = "logits") -> None:
        super().__init__(input_keys=[], backbone_output_key=backbone_output_key)

    def forward(self, backbone_output: dict[str, Any], **_: Any) -> torch.Tensor:
        return backbone_output[self.backbone_output_key]
