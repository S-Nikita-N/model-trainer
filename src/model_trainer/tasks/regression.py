"""Regression task.

Identity post-process — loss (``MSE`` / ``L1``) and metrics (``MAE``,
``R2``, ...) all take raw outputs directly. Trailing singleton is squeezed
in ``compute_loss`` / ``postprocess`` to keep ``(B,)`` aligned shapes.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from model_trainer.tasks.base import Task


class RegressionTask(Task):
    def __init__(
        self,
        head: nn.Module,
        loss: nn.Module,
        label_key: str = "labels",
        metrics: Any = None,
    ) -> None:
        super().__init__(head=head, loss=loss, label_key=label_key, metrics=metrics)

    @staticmethod
    def _maybe_squeeze(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if logits.ndim == labels.ndim + 1 and logits.shape[-1] == 1:
            return logits.squeeze(-1)
        return logits

    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self.loss(self._maybe_squeeze(logits, labels), labels.to(logits.dtype))

    def postprocess_for_metrics(self, logits: torch.Tensor) -> torch.Tensor:
        if logits.ndim > 1 and logits.shape[-1] == 1:
            return logits.squeeze(-1)
        return logits
