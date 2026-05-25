"""Task = head + loss + metrics + label_key.

A Task is an ``nn.Module`` that owns:
  - a ``head`` (any ``nn.Module``, typically from ``model_trainer.heads``)
  - a ``loss`` (any callable / ``nn.Module``)
  - a metrics config (a ``DictConfig`` of metric configs, instantiated lazily
    once per split by ``make_metric_group``)
  - a ``label_key`` — the top-level batch key from which to read labels

Subclasses override ``postprocess_for_metrics`` to apply the natural
activation (``sigmoid`` / ``softmax`` / identity). The loss is fed raw
``logits`` directly — every loss in ``torch.nn`` we care about
(``CrossEntropyLoss``, ``BCEWithLogitsLoss``, ``MSELoss``, ...) applies its
own activation internally for numerical stability.

The LitModule calls ``task(backbone_output, batch) -> (loss, info)`` and
``task.update_metrics(info, batch, group)``. ``info`` is opaque to the
LitModule — single-task variants put ``logits`` in there; ``MultiTask``
nests per-child dicts.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from model_trainer.hydra_builder import build_items_dict


class Task(nn.Module):
    def __init__(
        self,
        head: Any,
        loss: Any,
        label_key: str = "labels",
        metrics: Any = None,
    ) -> None:
        super().__init__()
        # ``_recursive_: false`` on the task config keeps ``metrics:`` un-built
        # (we build them lazily per-split in ``make_metric_group``), so head
        # and loss arrive as raw DictConfigs too — instantiate them here.
        from model_trainer.hydra_builder import build_item
        self.head = head if isinstance(head, nn.Module) else build_item(head)
        if self.head is None:
            raise ValueError("Task: `head` is required (got None / no _target_).")
        self.loss = loss if isinstance(loss, nn.Module) else build_item(loss)
        if self.loss is None:
            raise ValueError("Task: `loss` is required (got None / no _target_).")
        self.label_key = label_key
        self._metrics_cfg = metrics

    def _gather_head_inputs(self, batch: dict[str, Any]) -> dict[str, Any]:
        try:
            return {k: batch[k] for k in self.head.input_keys}
        except KeyError as e:
            raise KeyError(
                f"Task head {type(self.head).__name__} expects batch key {e.args[0]!r} "
                f"(declared in head.input_keys={list(self.head.input_keys)}); "
                f"batch has keys: {sorted(batch.keys())}."
            ) from None

    def forward(
        self,
        backbone_output: dict[str, Any],
        batch: dict[str, Any],
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        head_inputs = self._gather_head_inputs(batch)
        logits = self.head(backbone_output, **head_inputs)
        if self.label_key not in batch:
            raise KeyError(
                f"Task expects batch[{self.label_key!r}] (label_key); "
                f"batch has keys: {sorted(batch.keys())}."
            )
        labels = batch[self.label_key]
        loss = self.compute_loss(logits, labels)
        return loss, {"logits": logits}

    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self.loss(logits, labels)

    def postprocess_for_metrics(self, logits: torch.Tensor) -> torch.Tensor:
        return logits

    def update_metrics(
        self,
        info: dict[str, Any],
        batch: dict[str, Any],
        group: nn.ModuleDict,
    ) -> None:
        x = self.postprocess_for_metrics(info["logits"])
        labels = batch[self.label_key]
        for metric in group.values():
            metric.update(x, labels)

    def loss_components(self, info: dict[str, Any]) -> dict[str, torch.Tensor]:
        return {}

    def make_metric_group(self) -> nn.ModuleDict:
        return nn.ModuleDict(build_items_dict(self._metrics_cfg))

    def log_metrics(
        self,
        group: nn.ModuleDict,
        log_prefix: str,
        log_fn: Any,
    ) -> None:
        for name, metric in group.items():
            value = metric.compute()
            # Some torchmetrics return (value, threshold) — log the value only.
            if isinstance(value, (tuple, list)):
                value = value[0]
            log_fn(
                name=f"{log_prefix}_{name}",
                value=value,
                prog_bar=True,
                on_epoch=True,
                sync_dist=True,
            )
            metric.reset()
