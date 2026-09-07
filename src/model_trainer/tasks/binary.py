"""Single-logit binary classification with BCE.

Use this when the head produces an unnormalized score (``(..., 1)`` or
``(...,)``) and you want ``BCEWithLogitsLoss``. Metric input = ``sigmoid``.

Element-wise ``ignore_index`` masking is applied to both loss and metrics so
the same task works on padded structured batches (per-sentence, per-pair, …)
without polluting either with ``-100`` positions.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from model_trainer.tasks.base import Task


class BinaryClassificationTask(Task):
    def __init__(
        self,
        head: nn.Module,
        loss: nn.Module,
        label_key: str = "labels",
        metrics: Any = None,
        ignore_index: int | None = -100,
    ) -> None:
        super().__init__(head=head, loss=loss, label_key=label_key, metrics=metrics)
        self.ignore_index = ignore_index

    @staticmethod
    def _squeeze_singleton(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if logits.ndim == labels.ndim + 1 and logits.shape[-1] == 1:
            return logits.squeeze(-1)
        return logits

    def _mask(self, logits: torch.Tensor, labels: torch.Tensor):
        if self.ignore_index is None:
            return logits, labels
        keep = labels != self.ignore_index
        if not keep.any():
            return None
        return logits[keep], labels[keep]

    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        logits = self._squeeze_singleton(logits, labels)
        filtered = self._mask(logits, labels)
        if filtered is None:
            return torch.zeros((), device=logits.device, dtype=logits.dtype)
        flogits, flabels = filtered
        return self.loss(flogits, flabels.to(flogits.dtype))

    def update_metrics(
        self,
        info: dict[str, Any],
        batch: dict[str, Any],
        group: nn.ModuleDict,
    ) -> None:
        logits = info["logits"]
        labels = batch[self.label_key]
        logits = self._squeeze_singleton(logits, labels)
        filtered = self._mask(logits, labels)
        if filtered is None:
            return
        flogits, flabels = filtered
        x = torch.sigmoid(flogits)
        for m in group.values():
            m.update(x, flabels)

    def postprocess_for_metrics(self, logits: torch.Tensor) -> torch.Tensor:
        # Kept for symmetry; ``update_metrics`` does its own masking + sigmoid.
        if logits.ndim > 1 and logits.shape[-1] == 1:
            logits = logits.squeeze(-1)
        return torch.sigmoid(logits)
