"""Task protocol. Implement your own by subclassing ``Task``.

A task encapsulates everything task-specific that the generic ``LitModule``
would otherwise have to guess: how to turn ``logits`` into whatever a metric
expects, and (optionally) how to format logged values. This is the single
extension point for new task types (regression, seq2seq, token-level
classification, contrastive, ...).
"""

from __future__ import annotations

from typing import Any

import torch


class Task:
    """Base class. Override the methods relevant to your task."""

    def prepare_metric_input(self, logits: torch.Tensor, kind: str) -> torch.Tensor:
        """Return the tensor a metric with ``input=<kind>`` expects.

        ``kind`` is a free-form string coming from a :class:`MetricSpec`
        (e.g. ``"preds"``, ``"probs"``, ``"logits"``). Raise ``ValueError``
        for unknown kinds so misconfigured metrics fail loudly.
        """
        raise NotImplementedError

    def format_output(self, output: Any) -> torch.Tensor:
        """Extract logits (or the value a criterion expects) from a backbone output.

        Default handles HuggingFace-style outputs with a ``.logits`` attribute
        and raw tensors. Override for custom heads.
        """
        return getattr(output, "logits", output)
