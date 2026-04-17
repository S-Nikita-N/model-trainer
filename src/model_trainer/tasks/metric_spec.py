"""MetricSpec: wraps a torchmetrics metric with its input kind.

Metrics that consume probabilities (AUROC, AveragePrecision, ...) and metrics
that consume discrete predictions (Accuracy, F1, ...) look identical from the
outside. ``MetricSpec`` adds a single explicit ``input`` field so the
``LitModule`` can dispatch correctly without any hard-coded name lists.

Example config::

    roc_auc:
      _target_: model_trainer.tasks.MetricSpec
      input: probs
      metric:
        _target_: torchmetrics.AUROC
        task: binary
"""

from __future__ import annotations

import torch.nn as nn
from torchmetrics import Metric

VALID_INPUTS = ("preds", "probs", "logits")


class MetricSpec(nn.Module):
    """A metric + the kind of input it expects.

    Subclasses ``nn.Module`` so it plays nicely inside ``nn.ModuleDict`` and
    gets moved to the right device together with the rest of the model.
    """

    def __init__(self, metric: Metric, input: str = "preds") -> None:
        super().__init__()
        if input not in VALID_INPUTS:
            raise ValueError(f"MetricSpec.input must be one of {VALID_INPUTS}, got {input!r}")
        self.metric = metric
        self.input = input

    def update(self, value, target) -> None:
        self.metric.update(value, target)

    def compute(self):
        return self.metric.compute()

    def reset(self) -> None:
        self.metric.reset()
