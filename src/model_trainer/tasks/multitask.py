"""Multi-task composition.

``MultiTask`` holds a dict of child Tasks, runs each over the same
``backbone_output``, sums their losses with optional per-task ``weights``,
and namespaces metrics under the child's name in the metric group tree.

Hydra composition::

    task=multitask
    +task@task.tasks.fact=fact_head_preset
    +task@task.tasks.evidence=evidence_head_preset
    +task.weights.fact=1.0
    +task.weights.evidence=0.5

Metrics for child tasks live under that child's ``metrics:`` field — same
``+metrics@task.tasks.fact.metrics.f1=f1`` pattern.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from model_trainer.hydra_builder import build_items_dict


class MultiTask(nn.Module):
    def __init__(
        self,
        tasks: Any = None,
        weights: dict[str, float] | None = None,
    ) -> None:
        super().__init__()
        instantiated = build_items_dict(tasks) if tasks else {}
        if not instantiated:
            raise ValueError(
                "MultiTask requires at least one child task; compose them via "
                "`+task@task.tasks.<name>=<preset>` overrides."
            )
        # Use ModuleDict so children's parameters are registered with the LitModule.
        self.tasks: nn.ModuleDict = nn.ModuleDict(instantiated)
        self.weights: dict[str, float] = {n: float(v) for n, v in (weights or {}).items()}

    def forward(
        self,
        backbone_output: dict[str, Any],
        batch: dict[str, Any],
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        total: torch.Tensor | None = None
        info: dict[str, Any] = {}
        for name, task in self.tasks.items():
            child_loss, child_info = task(backbone_output, batch)
            w = self.weights.get(name, 1.0)
            term = w * child_loss
            total = term if total is None else total + term
            info[name] = {"loss": child_loss.detach(), "logits": child_info["logits"]}
        assert total is not None  # constructor guarantees at least one task
        return total, info

    def update_metrics(
        self,
        info: dict[str, Any],
        batch: dict[str, Any],
        group: nn.ModuleDict,
    ) -> None:
        for name, task in self.tasks.items():
            child_info = info[name]
            task.update_metrics(child_info, batch, group[name])

    def loss_components(self, info: dict[str, Any]) -> dict[str, torch.Tensor]:
        return {name: child_info["loss"] for name, child_info in info.items()}

    def make_metric_group(self) -> nn.ModuleDict:
        return nn.ModuleDict({name: task.make_metric_group() for name, task in self.tasks.items()})

    def log_metrics(
        self,
        group: nn.ModuleDict,
        log_prefix: str,
        log_fn: Any,
    ) -> None:
        for task_name, task in self.tasks.items():
            task.log_metrics(group[task_name], f"{log_prefix}_{task_name}", log_fn)
