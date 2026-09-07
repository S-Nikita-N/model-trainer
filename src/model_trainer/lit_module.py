"""Single generic LightningModule.

Three responsibilities:
  1. Build ``backbone`` (from ``cfg.model``) and ``task`` (from ``cfg.task``).
  2. Pipe each batch: ``backbone(**batch[backbone.input_keys]) → task(backbone_output, batch) → (loss, info)``.
  3. Per-split metric trees, built lazily by ``task.make_metric_group()`` —
     same call works for a single task (flat dict of metrics) and for
     ``MultiTask`` (nested dict per child).

Task is responsible for everything task-specific: head call, loss compute,
metric update + log + reset (recursively). LitModule just orchestrates.

DataModule contract: the batch is a flat dict. ``backbone.input_keys`` and
each ``head.input_keys`` are sliced out of it; the task reads its
``label_key`` from the same flat dict. Missing keys raise ``KeyError`` —
synchronization of dataset / backbone / head / loss / metric is the
developer's responsibility (see brainstorm notes).
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
        self.task: nn.Module = build_item(cfg.task)

        if not hasattr(self.backbone, "input_keys"):
            raise AttributeError(
                f"Backbone {type(self.backbone).__name__} must define `input_keys: list[str]`."
            )
        if not hasattr(self.task, "make_metric_group"):
            raise AttributeError(
                f"Task {type(self.task).__name__} must define `make_metric_group()`."
            )

        self.val_set_names = list(val_set_names)
        self.test_set_names = list(test_set_names)

        self.train_metrics: nn.ModuleDict = self.task.make_metric_group()
        self.val_metrics: nn.ModuleDict = nn.ModuleDict(
            {n: self.task.make_metric_group() for n in self.val_set_names}
        )
        self.test_metrics: nn.ModuleDict = nn.ModuleDict(
            {n: self.task.make_metric_group() for n in self.test_set_names}
        )

        self.log_train_metrics_on_val_start: bool = bool(
            cfg.get("experiment", {}).get("log_train_metrics_on_val", True)
        )

    @staticmethod
    def _log_prefix(prefix: str, set_name: str | None) -> str:
        if set_name is None or set_name == prefix:
            return prefix
        return f"{prefix}_{set_name}"

    def _step(
        self,
        batch: dict[str, Any],
        prefix: str,
        set_name: str | None = None,
    ) -> torch.Tensor:
        backbone_inputs = {k: batch[k] for k in self.backbone.input_keys}
        backbone_output = self.backbone(**backbone_inputs)
        loss, info = self.task(backbone_output, batch)

        log_prefix = self._log_prefix(prefix, set_name)
        self.log(
            name=f"{log_prefix}_loss",
            value=loss,
            prog_bar=True,
            on_epoch=True,
            on_step=(prefix == "train"),
            sync_dist=(prefix != "train"),
            add_dataloader_idx=False,
        )
        for comp_name, comp in self.task.loss_components(info).items():
            self.log(
                name=f"{log_prefix}_loss_{comp_name}",
                value=comp,
                on_epoch=True,
                on_step=(prefix == "train"),
                sync_dist=(prefix != "train"),
                add_dataloader_idx=False,
            )

        group = self._get_metric_group(prefix, set_name)
        self.task.update_metrics(info, batch, group)
        return loss

    def training_step(
        self,
        batch: dict[str, Any],
        batch_idx: int,
    ) -> torch.Tensor:
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

    def _log_metrics(self, prefix: str, set_name: str | None = None) -> None:
        group = self._get_metric_group(prefix, set_name)
        log_prefix = self._log_prefix(prefix, set_name)
        self.task.log_metrics(group, log_prefix, self.log)

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
            
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                **settings,
            },
        }

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
