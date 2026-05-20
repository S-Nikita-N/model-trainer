"""Multilabel classification task adapter.

Each label is an independent binary decision, so outputs go through
``sigmoid`` (not ``softmax``) and ``preds`` are threshold-based.
"""

from __future__ import annotations

import torch

from model_trainer.tasks.base import Task


class MultilabelClassificationTask(Task):
    def __init__(self, num_labels: int, threshold: float = 0.5) -> None:
        if num_labels < 1:
            raise ValueError(f"num_labels must be >= 1, got {num_labels}")
        if not 0.0 < threshold < 1.0:
            raise ValueError(f"threshold must be in (0, 1), got {threshold}")
        self.num_labels = num_labels
        self.threshold = threshold

    def prepare_metric_input(self, logits: torch.Tensor, kind: str) -> torch.Tensor:
        if kind == "logits":
            return logits
        probs = torch.sigmoid(logits)
        if kind == "probs":
            return probs
        if kind == "preds":
            return (probs >= self.threshold).long()
        raise ValueError(
            f"Unknown metric input kind: {kind!r}. Expected one of: 'preds', 'probs', 'logits'."
        )

    def prepare_metric_target(self, labels: torch.Tensor, kind: str) -> torch.Tensor:
        # BCEWithLogitsLoss wants float targets; torchmetrics multilabel wants
        # int/long. We keep labels as float in the batch for the loss and cast
        # here on the metric side.
        if labels.dtype.is_floating_point:
            return labels.long()
        return labels
