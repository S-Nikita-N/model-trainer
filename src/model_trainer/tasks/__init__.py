from model_trainer.tasks.base import Task
from model_trainer.tasks.binary import BinaryClassificationTask
from model_trainer.tasks.classification import ClassificationTask
from model_trainer.tasks.multilabel import MultilabelClassificationTask
from model_trainer.tasks.multitask import MultiTask
from model_trainer.tasks.regression import RegressionTask

__all__ = [
    "Task",
    "ClassificationTask",
    "BinaryClassificationTask",
    "MultilabelClassificationTask",
    "RegressionTask",
    "MultiTask",
]
