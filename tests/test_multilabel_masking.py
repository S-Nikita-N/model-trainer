"""label_mask: 1 = маскируем. Маска действует и в лоссе, и в метриках."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
from torchmetrics import AveragePrecision  # noqa: E402

from model_trainer.tasks.multilabel import MultilabelClassificationTask  # noqa: E402

ASSIST_MASK = [0, 0, 1, 1]  # первые два канала размечены, остальные — нет


def _task(loss=None, num_labels=4, names=None):
    head = torch.nn.Identity()
    head.input_keys = ()
    return MultilabelClassificationTask(
        head=head,
        loss=loss or torch.nn.BCEWithLogitsLoss(),
        num_labels=num_labels,
        ignore_index=-100,
        label_names=names,
    )


def _batch(labels, mask=None):
    b = {"labels": torch.tensor(labels, dtype=torch.float32)}
    if mask is not None:
        b["label_mask"] = torch.tensor(mask)
    return b


def test_mask_folds_into_ignore_index():
    task = _task()
    b = _batch([[1.0, 0.0, 1.0, 0.0]], [ASSIST_MASK])
    assert task._labels(b).tolist() == [[1.0, 0.0, -100.0, -100.0]]


def test_loss_ignores_masked_channels():
    task = _task()
    b = _batch([[1.0, 0.0, 1.0, 0.0]], [ASSIST_MASK])
    labels = task._labels(b)
    base = task.compute_loss(torch.zeros(1, 4), labels)
    moved = torch.zeros(1, 4)
    moved[0, 2:] = 50.0  # замаскированные логиты уезжают куда угодно
    assert torch.allclose(base, task.compute_loss(moved, labels))


def test_loss_equals_mean_over_unmasked():
    task = _task()
    logits = torch.tensor([[2.0, -2.0, 0.5, 0.1]])
    b = _batch([[1.0, 0.0, 1.0, 0.0]], [ASSIST_MASK])
    labels = task._labels(b)
    keep = labels != -100
    want = torch.nn.functional.binary_cross_entropy_with_logits(
        logits[keep], labels[keep], reduction="mean"
    )
    assert torch.allclose(task.compute_loss(logits, labels), want)


def test_no_mask_key_changes_nothing():
    task = _task()
    logits = torch.randn(4, 4)
    labels = (torch.rand(4, 4) > 0.5).float()
    b = _batch(labels.tolist())
    assert torch.allclose(
        task.compute_loss(logits, task._labels(b)),
        torch.nn.BCEWithLogitsLoss()(logits, labels),
    )


def test_pos_weight_stays_per_channel():
    pw = torch.tensor([1.0, 5.0, 1.0, 1.0])
    task = _task(loss=torch.nn.BCEWithLogitsLoss(pos_weight=pw))
    b = _batch([[1.0, 1.0, 1.0, 1.0]], [ASSIST_MASK])
    logits = torch.zeros(1, 4)
    got = task.compute_loss(logits, task._labels(b))
    want = torch.nn.functional.binary_cross_entropy_with_logits(
        logits[:, :2], torch.ones(1, 2), pos_weight=pw[:2], reduction="mean"
    )
    assert torch.allclose(got, want)
    assert task.loss.reduction == "mean"  # reduction восстановлен


def test_all_masked_keeps_graph():
    task = _task()
    logits = torch.randn(2, 4, requires_grad=True)
    b = _batch([[1.0] * 4] * 2, [[1] * 4] * 2)
    loss = task.compute_loss(logits, task._labels(b))
    assert loss.item() == 0.0
    loss.backward()  # иначе DDP упадёт на unused parameters
    assert logits.grad is not None


def test_fully_masked_label_is_not_reported():
    """Канал без единого незамаскированного значения не логируется и не входит в macro."""
    names = ["a", "b", "c", "d"]
    task = _task(names=names)
    group = torch.nn.ModuleDict(
        {
            "pr_auc": AveragePrecision(
                task="multilabel", num_labels=4, average=None,
                ignore_index=-100, thresholds=50,
            )
        }
    )
    b = _batch([[1.0, 0.0, 1.0, 1.0], [0.0, 1.0, 0.0, 0.0]], [ASSIST_MASK] * 2)
    logits = torch.randn(2, 4)
    task.update_metrics({"logits": logits}, b, group)

    logged: dict[str, float] = {}
    task.log_metrics(group, "val", lambda name, value, **kw: logged.__setitem__(name, float(value)))
    assert "val_pr_auc_a" in logged and "val_pr_auc_b" in logged
    assert "val_pr_auc_c" not in logged, "замаскированный канал не должен логироваться"
    assert "val_pr_auc_d" not in logged
    macro = (logged["val_pr_auc_a"] + logged["val_pr_auc_b"]) / 2
    assert logged["val_pr_auc"] == pytest.approx(macro), "macro считается только по живым каналам"
