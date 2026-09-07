"""HuggingFace sequence-classification model as a backbone.

Unlike :class:`HFEncoder`, this one already owns a classification head, so it
returns ``{"logits": ...}`` and pairs with :class:`~model_trainer.heads.IdentityHead`.
Pooling, dropout and the classifier come from ``transformers`` itself, which
means the resulting checkpoint loads back with a plain
``AutoModelForSequenceClassification.from_pretrained`` — handy for serving.

``num_labels`` drives the head width: 2 for binary, ``C`` for multiclass,
``L`` for multilabel. Set ``problem_type`` only if you want the HF model to
compute its own loss; the trainer feeds logits to ``task.loss`` instead, so
leaving it ``null`` is the normal case.

``config_overrides`` is splatted into ``from_pretrained`` and reaches the
model's config, e.g.::

    +model.config_overrides.classifier_dropout=0.1
    +model.config_overrides.mlp_dropout=0.1
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification


class HFSeqClsBackbone(nn.Module):
    def __init__(
        self,
        pretrained_model_name_or_path: str,
        num_labels: int = 2,
        problem_type: str | None = None,
        config_overrides: dict[str, Any] | None = None,
        local_files_only: bool = False,
        trust_remote_code: bool = False,
        input_keys: list[str] | None = None,
    ) -> None:
        super().__init__()
        # Anything accepted by the model's ``*Config`` can be overridden here —
        # ModernBERT ships every dropout at 0.0, so e.g. ``classifier_dropout``
        # has to be set explicitly if you want any regularisation.
        extra: dict[str, Any] = dict(config_overrides or {})
        if problem_type is not None:
            extra["problem_type"] = problem_type

        self.model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path,
            num_labels=num_labels,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
            **extra,
        )
        # ``from_pretrained`` hands the model back in eval mode, and Lightning
        # only calls ``.train()`` when *returning* from a validation loop — so
        # without this the first training epoch runs with dropout disabled.
        self.model.train()
        self.input_keys: list[str] = list(input_keys or ["input_ids", "attention_mask"])

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **_: Any,
    ) -> dict[str, torch.Tensor]:
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return {"logits": out.logits}
