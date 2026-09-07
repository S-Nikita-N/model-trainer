"""Multilabel classification: each output channel is an independent binary
decision with its own BCE loss.

Metric input is fed element-wise (after flattening + ignore-mask) so that a
``torchmetrics.<metric>(task="binary")`` metric computes a micro-style score
over all positions × channels. If you want per-label aggregation, configure
a ``task="multilabel"`` metric explicitly and override ``update_metrics``
downstream (TODO: per-label preserved shape).

Partially annotated rows are supported via an optional per-row mask in
``batch["label_mask"]`` (same shape as labels, ``1`` = mask this channel out).
A masked channel is "unknown", not a negative: it is folded into
``ignore_index`` and then dropped from the loss, from the metrics, and from
logging — a label with no unmasked value in a split is not reported at all.
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
        mask_key: str = "label_mask",
    ) -> None:
        super().__init__(head=head, loss=loss, label_key=label_key, metrics=metrics)
        self.num_labels = num_labels
        self.ignore_index = ignore_index
        self.label_names = label_names
        self.mask_key = mask_key
        # Сколько незамаскированных значений видел каждый канал за сплит.
        # Ключ — id() группы метрик (у каждого сплита своя). Нужно, потому что
        # сама метрика пустой канал не отличает: с thresholds=None она падает с
        # IndexError, с биннингом молча возвращает 0.0.
        self._seen: dict[int, torch.Tensor] = {}

    def _labels(self, batch: dict[str, Any]) -> torch.Tensor:
        """Метки с наложенной маской: ``label_mask == 1`` -> ``ignore_index``."""
        labels = batch[self.label_key]
        mask = batch.get(self.mask_key)
        if mask is None or self.ignore_index is None:
            return labels
        return labels.masked_fill(mask.bool(), self.ignore_index)

    def forward(self, backbone_output, batch):
        head_inputs = self._gather_head_inputs(batch)
        logits = self.head(backbone_output, **head_inputs)
        return self.compute_loss(logits, self._labels(batch)), {"logits": logits}

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
        ``ignore_index`` в отдельных каналах при этом СОХРАНЯЕТСЯ: multilabel
        метрики принимают форму (N, L) целиком, и выбросить из неё отдельные
        позиции нельзя. Отсеивает их сама метрика — для этого её конфиг должен
        передавать тот же ``ignore_index`` (см. configs/metrics/*_multilabel.yaml).
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
        """BCE по каналам, где метка известна (``ignore_index`` — поэлементный).

        Форма (N, L) сохраняется, а не схлопывается отбором валидных элементов:
        ``pos_weight`` задаётся на канал и broadcast'ится по последней оси —
        после flatten он бы поехал. Поэтому считаем поэлементно и усредняем.
        """
        if self.ignore_index is None:
            return self.loss(logits, labels.to(logits.dtype))

        mask = labels != self.ignore_index
        if mask.all():
            return self.loss(logits, labels.to(logits.dtype))
        if not mask.any():
            # Ноль, но через граф: голая константа оставила бы часть параметров
            # без градиента, и DDP упал бы на unused parameters.
            return logits.sum() * 0.0

        target = torch.where(mask, labels, torch.zeros_like(labels)).to(logits.dtype)
        reduction = getattr(self.loss, "reduction", None)
        if reduction is None:
            # Лосс без переключателя reduction — отбираем валидные элементы.
            return self.loss(logits[mask], labels[mask].to(logits.dtype))
        try:
            self.loss.reduction = "none"
            per_element = self.loss(logits, target)
        finally:
            self.loss.reduction = reduction
        return (per_element * mask).sum() / mask.sum()

    def update_metrics(
        self,
        info: dict[str, Any],
        batch: dict[str, Any],
        group: nn.ModuleDict,
    ) -> None:
        logits = info["logits"]
        labels = self._labels(batch)
        if self.ignore_index is not None:
            nl = logits.shape[-1]
            valid = (labels.reshape(-1, nl) != self.ignore_index).sum(0)
            prev = self._seen.get(id(group))
            self._seen[id(group)] = valid if prev is None else prev + valid
        for m in group.values():
            if getattr(m, "num_labels", None) is not None:
                filtered = self._mask_per_label(logits, labels)
            else:
                filtered = self._mask(logits, labels)
            if filtered is None:
                continue
            flogits, flabels = filtered
            # BCEWithLogitsLoss keeps targets float (soft labels are allowed);
            # torchmetrics classification metrics want int ground truth.
            if flabels.dtype.is_floating_point:
                flabels = flabels.long()
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
        seen = self._seen.pop(id(group), None)
        for name, metric in group.items():
            value = metric.compute()
            if isinstance(value, (tuple, list)):
                value = value[0]
            if isinstance(value, torch.Tensor) and value.numel() > 1:
                # Канал, у которого за сплит не было ни одного незамаскированного
                # значения, не логируем вовсе и в macro-среднее не берём: метрика
                # вернула бы по нему 0.0 и тянула бы среднее вниз.
                valid = ~torch.isnan(value)
                if seen is not None and seen.numel() == value.numel():
                    valid &= seen.to(value.device) > 0
                if not valid.any():
                    metric.reset()
                    continue
                log_fn(
                    name=f"{log_prefix}_{name}",
                    value=value[valid].mean(),
                    prog_bar=True,
                    on_epoch=True,
                    sync_dist=True,
                )
                n = int(value.numel())
                if self.label_names is not None and len(self.label_names) == n:
                    channel_names = self.label_names
                else:
                    channel_names = [str(i) for i in range(n)]
                for channel, v, ok in zip(channel_names, value, valid, strict=True):
                    if not ok:
                        continue
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
