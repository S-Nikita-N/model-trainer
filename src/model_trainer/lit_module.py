"""Generic Lightning module.

Everything task-specific is configurable:
  - ``backbone`` comes from ``cfg.model`` via ``_target_``.
  - ``criterion`` comes from ``cfg.loss``.
  - ``task`` (how to turn logits into ``preds`` / ``probs`` / ... for metrics)
    comes from ``cfg.task``.
  - ``metrics`` are a dict of :class:`MetricSpec` objects, each tagged with
    the input kind it needs.

Per-dataloader validation / test set names are provided by the DataModule via
constructor arguments (``val_set_names`` / ``test_set_names``), not by snooping
into ``cfg.data``. This keeps the module decoupled from any particular
DataModule schema.

Batch contract: each batch is a ``dict`` with ``labels`` and whatever keys
the backbone's ``forward`` expects (typically ``input_ids`` and optionally
``attention_mask``).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn as nn
from omegaconf import DictConfig

from model_trainer.hydra_builder import build_item


class LitModule(pl.LightningModule):
    def __init__(
        self,
        cfg: DictConfig,
        val_set_names: Sequence[str] = ("val",),
        test_set_names: Sequence[str] = (),
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.backbone: nn.Module = build_item(cfg.model)
        self.criterion: nn.Module = build_item(cfg.loss)
        self.task: Task = build_item(cfg.get("task"), default=ClassificationTask())

        self.val_set_names = list(val_set_names)
        self.test_set_names = list(test_set_names)

        metrics_cfg = cfg.get("metrics") or {}
        self.train_metrics: nn.ModuleDict = nn.ModuleDict(build_items_dict(metrics_cfg))
        self.val_metrics: nn.ModuleDict = nn.ModuleDict(
            {name: nn.ModuleDict(build_items_dict(metrics_cfg)) for name in self.val_set_names}
        )
        self.test_metrics: nn.ModuleDict = nn.ModuleDict(
            {name: nn.ModuleDict(build_items_dict(metrics_cfg)) for name in self.test_set_names}
        )

        self.log_train_metrics_on_val_start: bool = bool(
            cfg.get("experiment", {}).get("log_train_metrics_on_val", True)
        )

    def forward(self, **inputs: Any) -> Any:
        return self.backbone(**inputs)

    @staticmethod
    def _log_prefix(prefix: str, set_name: str | None) -> str:
        """Skip redundant ``set_name`` in the log name when it equals ``prefix``.

        With a single validation set conventionally called ``"val"`` we get
        clean ``val_loss`` / ``val_accuracy`` names. With multiple named sets
        (e.g. ``in_domain`` + ``out_of_domain``) each metric is prefixed with
        its set name: ``val_in_domain_accuracy``.
        """
        if set_name is None or set_name == prefix:
            return prefix
        return f"{prefix}_{set_name}"

    def _step(
        self,
        batch: dict[str, Any],
        prefix: str,
        set_name: str | None = None,
    ) -> torch.Tensor:
        labels = batch["labels"]
        inputs = {k: v for k, v in batch.items() if k != "labels"}

        output = self.forward(**inputs)
        logits = self.task.format_output(output)
        loss = self.criterion(logits, labels)

        self.log(
            name=f"{self._log_prefix(prefix, set_name)}_loss",
            value=loss,
            prog_bar=True,
            on_epoch=True,
            on_step=(prefix == "train"),
            sync_dist=(prefix != "train"),
            add_dataloader_idx=False,
        )

        self._update_metrics(logits=logits, labels=labels, prefix=prefix, set_name=set_name)
        return loss

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self._step(batch, "train")

    def validation_step(
        self,
        batch: dict[str, Any],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> torch.Tensor:
        set_name = self.val_set_names[dataloader_idx]
        return self._step(batch, "val", set_name=set_name)

    def test_step(
        self,
        batch: dict[str, Any],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> torch.Tensor:
        set_name = self.test_set_names[dataloader_idx]
        return self._step(batch, "test", set_name=set_name)

    def _get_metric_group(self, prefix: str, set_name: str | None) -> nn.ModuleDict:
        if prefix == "train":
            return self.train_metrics
        if prefix == "val":
            return self.val_metrics[set_name]
        if prefix == "test":
            return self.test_metrics[set_name]
        raise ValueError(f"Invalid prefix: {prefix!r}")

    def _update_metrics(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        prefix: str,
        set_name: str | None = None,
    ) -> None:
        metrics = self._get_metric_group(prefix, set_name)
        for spec in metrics.values():
            value = self.task.prepare_metric_input(logits, spec.input)
            target = self.task.prepare_metric_target(labels, spec.input)
            spec.update(value, target)

    def _log_metrics(self, prefix: str, set_name: str | None = None) -> None:
        metrics = self._get_metric_group(prefix, set_name)
        log_prefix = self._log_prefix(prefix, set_name)

        for name, spec in metrics.items():
            value = spec.compute()
            # Some torchmetrics (e.g. RecallAtFixedPrecision) return (value, threshold).
            if isinstance(value, (tuple, list)):
                value = value[0]
            self.log(
                name=f"{log_prefix}_{name}",
                value=value,
                prog_bar=True,
                on_epoch=True,
                sync_dist=True,
            )
            spec.reset()

    def _reset_metric_group(self, group: nn.ModuleDict) -> None:
        for entry in group.values():
            if isinstance(entry, nn.ModuleDict):
                self._reset_metric_group(entry)
            else:
                entry.reset()

    def on_validation_epoch_start(self) -> None:
        if self.log_train_metrics_on_val_start and not self.trainer.sanity_checking:
            self._log_metrics("train")
        self._reset_metric_group(self.val_metrics)

    def on_test_epoch_start(self) -> None:
        self._reset_metric_group(self.test_metrics)

    def on_validation_epoch_end(self) -> None:
        if self.trainer.sanity_checking:
            self._reset_metric_group(self.val_metrics)
            return
        for set_name in self.val_set_names:
            self._log_metrics("val", set_name=set_name)

    def on_test_epoch_end(self) -> None:
        for set_name in self.test_set_names:
            self._log_metrics("test", set_name=set_name)

    def configure_optimizers(self) -> Any:
        optimizer = build_item(self.cfg.optim, params=self.parameters())

        scheduler_cfg = self.cfg.get("scheduler")
        if scheduler_cfg is None:
            return optimizer

        params = scheduler_cfg.get("params")
        if params is None or not params.get("_target_"):
            return optimizer

        settings = dict(scheduler_cfg.get("settings") or {})
        step_kwargs = scheduler_cfg.get("step_kwargs") or []
        extra: dict[str, Any] = {"optimizer": optimizer}

        if step_kwargs:
            total_steps = self._get_total_steps()
            warmup_ratio = scheduler_cfg.get("warmup_ratio")
            if warmup_ratio is not None and "num_warmup_steps" in step_kwargs:
                extra["num_warmup_steps"] = max(1, int(total_steps * warmup_ratio))
            if "num_training_steps" in step_kwargs:
                extra["num_training_steps"] = total_steps

        scheduler = build_item(params, default=None, **extra)
        if scheduler is None:
            return optimizer
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, **settings}}

    def _get_total_steps(self) -> int:
        if self.trainer is None:
            raise RuntimeError(
                "Trainer is not attached to LightningModule yet. "
                "Total steps are available only during trainer.fit()."
            )
        total_steps = int(getattr(self.trainer, "estimated_stepping_batches", 0))
        if total_steps <= 0:
            raise RuntimeError(
                "trainer.estimated_stepping_batches is unavailable or zero. "
                "Check your dataloaders and trainer configuration."
            )
        return total_steps
