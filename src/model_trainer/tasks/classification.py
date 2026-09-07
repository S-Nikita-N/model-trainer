"""Softmax-based classification tasks.

- ``ClassificationTask``: multiclass (or 2-class softmax-style binary).
  Default loss = ``CrossEntropyLoss``. Metric input = ``softmax(logits)``,
  with the positive-class column extracted in the binary case so that
  ``torchmetrics.<metric>(task="binary")`` receives a 1-D score.

Loss-side ``ignore_index`` is configured on the loss itself (``CrossEntropyLoss``
has native support — see ``configs/loss/crossentropy.yaml``). The task carries
its own ``ignore_index`` for **metric-side** masking, because torchmetrics
have no ignore semantics — ``-100`` rows would otherwise pollute the score.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from model_trainer.tasks.base import Task


class ClassificationTask(Task):
    def __init__(
        self,
        head: nn.Module,
        loss: nn.Module,
        label_key: str = "labels",
        metrics: Any = None,
        num_classes: int = 2,
        positive_class: int = 1,
        ignore_index: int | None = -100,
    ) -> None:
        if num_classes < 2:
            raise ValueError(f"num_classes must be >= 2, got {num_classes}")
        if not 0 <= positive_class < num_classes:
            raise ValueError(f"positive_class must be in [0, {num_classes}), got {positive_class}")
        super().__init__(head=head, loss=loss, label_key=label_key, metrics=metrics)
        self.num_classes = num_classes
        self.positive_class = positive_class
        self.ignore_index = ignore_index

    def postprocess_for_metrics(self, logits: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=-1)
        # Binary special case: torchmetrics ``task="binary"`` expects a 1-D
        # score vector — feed the positive-class column.
        if self.num_classes == 2 and probs.shape[-1] == 2:
            return probs[..., self.positive_class]
        return probs

    def update_metrics(
        self,
        info: dict[str, Any],
        batch: dict[str, Any],
        group: nn.ModuleDict,
    ) -> None:
        logits = info["logits"]
        labels = batch[self.label_key]
        if self.ignore_index is not None:
            # Row-wise mask: labels shape is (B,) or (..., ), logits is
            # (..., C). Indexing logits[keep] flattens the leading dims and
            # keeps the class axis intact.
            keep = labels != self.ignore_index
            if not keep.any():
                return
            logits = logits[keep]
            labels = labels[keep]
        x = self.postprocess_for_metrics(logits)
        for metric in group.values():
            metric.update(x, labels)
