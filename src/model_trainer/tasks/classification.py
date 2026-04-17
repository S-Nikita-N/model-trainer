"""Classification task adapter.

Handles binary and multiclass classification uniformly:

- ``preds``: argmax over the class dimension.
- ``probs``: softmax; in the binary case returns the positive-class column so
  metrics like ``AUROC`` / ``AveragePrecision`` (``task="binary"``) work
  out of the box.
- ``logits``: raw logits, unchanged.
"""

from __future__ import annotations

import torch

from model_trainer.tasks.base import Task


class ClassificationTask(Task):
    def __init__(self, num_classes: int = 2, positive_class: int = 1) -> None:
        if num_classes < 2:
            raise ValueError(f"num_classes must be >= 2, got {num_classes}")
        if not 0 <= positive_class < num_classes:
            raise ValueError(f"positive_class must be in [0, {num_classes}), got {positive_class}")
        self.num_classes = num_classes
        self.positive_class = positive_class

    def prepare_metric_input(self, logits: torch.Tensor, kind: str) -> torch.Tensor:
        if kind == "preds":
            return logits.argmax(dim=-1)
        if kind == "logits":
            return logits
        if kind == "probs":
            probs = torch.softmax(logits, dim=-1)
            if self.num_classes == 2:
                return probs[..., self.positive_class]
            return probs
        raise ValueError(
            f"Unknown metric input kind: {kind!r}. Expected one of: 'preds', 'probs', 'logits'."
        )
