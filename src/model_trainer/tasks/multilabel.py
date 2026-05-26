"""Multilabel classification: each output channel is an independent binary
decision with its own BCE loss.

Metric input is fed element-wise (after flattening + ignore-mask) so that a
``torchmetrics.<metric>(task="binary")`` metric computes a micro-style score
over all positions × channels. If you want per-label aggregation, configure
a ``task="multilabel"`` metric explicitly and override ``update_metrics``
downstream (TODO: per-label preserved shape).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from model_trainer.tasks.base import Task


class MultilabelClassificationTask(Task):
    def __init__(
        self,
        head: nn.Module,
        loss: nn.Module,
        label_key: str = "labels",
        metrics: Any = None,
        num_labels: int | None = None,
        ignore_index: int | None = -100,
        label_names: list[str] | None = None,
    ) -> None:
        super().__init__(head=head, loss=loss, label_key=label_key, metrics=metrics)
        self.num_labels = num_labels
        self.ignore_index = ignore_index
        self.label_names = label_names

    def _mask(self, logits: torch.Tensor, labels: torch.Tensor):
        if self.ignore_index is None:
            return logits, labels
        keep = labels != self.ignore_index
        if not keep.any():
            return None
        return logits[keep], labels[keep]

    def _mask_per_label(self, logits: torch.Tensor, labels: torch.Tensor):
        """Возвращает (K, num_labels) — для multilabel macro метрик.

        Оставляет строки, где хотя бы один канал не является паддингом.
        """
        if self.ignore_index is None:
            nl = logits.shape[-1]
            return logits.reshape(-1, nl), labels.reshape(-1, nl)
        nl = logits.shape[-1]
        flat_logits = logits.reshape(-1, nl)
        flat_labels = labels.reshape(-1, nl)
        keep = (flat_labels != self.ignore_index).any(dim=-1)
        if not keep.any():
            return None
        return flat_logits[keep], flat_labels[keep]

    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        filtered = self._mask_per_label(logits, labels)
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
        for m in group.values():
            if getattr(m, "num_labels", None) is not None:
                filtered = self._mask_per_label(logits, labels)
                if filtered is None:
                    continue
                flogits, flabels = filtered
                m.update(torch.sigmoid(flogits), flabels)
            else:
                filtered = self._mask(logits, labels)
                if filtered is None:
                    continue
                flogits, flabels = filtered
                m.update(torch.sigmoid(flogits), flabels)

    def postprocess_for_metrics(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(logits)

    def log_metrics(
        self,
        group: nn.ModuleDict,
        log_prefix: str,
        log_fn: Any,
    ) -> None:
        """Log multilabel metrics with both the macro mean and per-label values.

        If ``metric.compute()`` returns a per-label tensor (``average=None``),
        we log ``<prefix>_<name>`` as the mean across labels (macro) and one
        ``<prefix>_<name>_<label>`` per channel. Scalar / tuple returns are
        treated as in the base implementation.
        """
        for name, metric in group.items():
            value = metric.compute()
            if isinstance(value, (tuple, list)):
                value = value[0]
            if isinstance(value, torch.Tensor) and value.numel() > 1:
                log_fn(
                    name=f"{log_prefix}_{name}",
                    value=value.mean(),
                    prog_bar=True,
                    on_epoch=True,
                    sync_dist=True,
                )
                n = int(value.numel())
                if self.label_names is not None and len(self.label_names) == n:
                    channel_names = self.label_names
                else:
                    channel_names = [str(i) for i in range(n)]
                for channel, v in zip(channel_names, value):
                    log_fn(
                        name=f"{log_prefix}_{name}_{channel}",
                        value=v,
                        on_epoch=True,
                        sync_dist=True,
                    )
            else:
                log_fn(
                    name=f"{log_prefix}_{name}",
                    value=value,
                    prog_bar=True,
                    on_epoch=True,
                    sync_dist=True,
                )
            metric.reset()
