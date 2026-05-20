"""Regression task adapter.

``logits`` are the raw model outputs. A trailing singleton dimension (common
when the head's ``num_labels=1``) is squeezed in :meth:`format_output` so that
both the loss (e.g. ``MSELoss``) and regression metrics (``MAE``, ``R2``) see
a 1-D tensor of shape ``(batch,)`` without any extra bookkeeping.

Only ``preds`` and ``logits`` metric kinds are defined; ``probs`` has no
meaning for regression and will raise rather than silently return something
weird.
"""

from __future__ import annotations

from typing import Any

import torch

from model_trainer.tasks.base import Task


class RegressionTask(Task):
    def format_output(self, output: Any) -> torch.Tensor:
        logits = super().format_output(output)
        if logits.ndim > 1 and logits.shape[-1] == 1:
            return logits.squeeze(-1)
        return logits

    def prepare_metric_input(self, logits: torch.Tensor, kind: str) -> torch.Tensor:
        if kind in ("preds", "logits"):
            return logits
        raise ValueError(
            f"RegressionTask does not define metric input {kind!r}. Use 'preds' or 'logits'."
        )
