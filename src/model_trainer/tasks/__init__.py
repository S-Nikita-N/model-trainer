"""Task adapters and metric specs.

A ``Task`` is a small, config-instantiated object that knows how to go from raw
model outputs to whatever a metric expects (predictions, probabilities, etc.)
and how to format the per-step logged value. This keeps ``LitModule`` free of
task-specific branching (binary vs multiclass, classification vs regression).
"""

from model_trainer.tasks.base import Task
from model_trainer.tasks.classification import ClassificationTask
from model_trainer.tasks.metric_spec import MetricSpec

__all__ = ["Task", "ClassificationTask", "MetricSpec"]
