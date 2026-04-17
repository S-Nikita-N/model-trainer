import pytest
import torch
import torchmetrics

from model_trainer.tasks import ClassificationTask, MetricSpec


class TestClassificationTask:
    def test_binary_preds(self):
        task = ClassificationTask(num_classes=2)
        logits = torch.tensor([[1.0, 2.0], [3.0, 0.5]])
        preds = task.prepare_metric_input(logits, "preds")
        assert torch.equal(preds, torch.tensor([1, 0]))

    def test_binary_probs_returns_positive_column(self):
        task = ClassificationTask(num_classes=2, positive_class=1)
        logits = torch.tensor([[0.0, 0.0], [-10.0, 10.0]])
        probs = task.prepare_metric_input(logits, "probs")
        assert probs.shape == (2,)
        assert pytest.approx(probs[0].item(), abs=1e-5) == 0.5
        assert probs[1] > 0.999

    def test_multiclass_probs_returns_full_distribution(self):
        task = ClassificationTask(num_classes=3)
        logits = torch.randn(4, 3)
        probs = task.prepare_metric_input(logits, "probs")
        assert probs.shape == (4, 3)
        torch.testing.assert_close(probs.sum(dim=-1), torch.ones(4))

    def test_logits_passthrough(self):
        task = ClassificationTask(num_classes=2)
        logits = torch.randn(3, 2)
        assert torch.equal(task.prepare_metric_input(logits, "logits"), logits)

    def test_unknown_kind_raises(self):
        task = ClassificationTask(num_classes=2)
        with pytest.raises(ValueError, match="Unknown metric input kind"):
            task.prepare_metric_input(torch.randn(2, 2), "weird")

    def test_invalid_init(self):
        with pytest.raises(ValueError):
            ClassificationTask(num_classes=1)
        with pytest.raises(ValueError):
            ClassificationTask(num_classes=2, positive_class=5)


class TestMetricSpec:
    def test_wraps_torchmetric_and_computes(self):
        spec = MetricSpec(metric=torchmetrics.Accuracy(task="binary"), input="preds")
        spec.update(torch.tensor([1, 0, 1]), torch.tensor([1, 0, 0]))
        assert spec.compute().item() == pytest.approx(2 / 3)

    def test_rejects_unknown_input(self):
        with pytest.raises(ValueError):
            MetricSpec(metric=torchmetrics.Accuracy(task="binary"), input="bogus")
