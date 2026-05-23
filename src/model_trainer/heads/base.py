"""Base class for task heads.

A ``Head`` is the only place where backbone output meets task-specific
batch data (sentence offsets, masks, etc.). It declares:

- ``input_keys``: which batch keys it pulls as ``forward(**kwargs)`` args.
- ``backbone_input_key``: which key of the backbone's output dict to read.

The LitModule uses ``input_keys`` to slice the batch before calling the
head. Mismatches surface as ``KeyError`` at runtime (loud, by design — see
the multi-task brainstorm: dataset/backbone/head/loss synchronization is the
developer's responsibility, not the framework's).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class Head(nn.Module):
    def __init__(
        self,
        input_keys: list[str] | None = None,
        backbone_input_key: str = "hidden_states",
    ) -> None:
        super().__init__()
        self.input_keys: list[str] = list(input_keys or [])
        self.backbone_input_key = backbone_input_key

    def forward(self, backbone_output: dict[str, Any], **head_inputs: Any) -> torch.Tensor:
        raise NotImplementedError
