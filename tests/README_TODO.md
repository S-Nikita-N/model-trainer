# Test backlog

The big task-centric refactor removed `MetricSpec`, `HallucinationLitModule`,
`HallucinationLoss`, and the per-task `tasks/hallucination.py`. The
following tests were deleted and need rewriting against the new API:

## Deleted (need rewrite)

- `test_tasks.py` — old assertions used `ClassificationTask.prepare_metric_input(logits, kind)` /
  `MetricSpec(metric=..., input=...)`. New API: tasks own head + loss + metrics,
  the `kind` dispatch is gone, `MetricSpec` is gone.
  - Cover: shape handling in `BinaryClassificationTask._squeeze_singleton`,
    ignore-mask in `compute_loss` / `update_metrics`, multilabel flatten,
    regression squeeze, `ClassificationTask` softmax binary-special-case.
- `test_metrics_compose.py` — old assertions used top-level
  `+metrics@metrics.f1=f1` composition + `_require_non_empty_metrics`. New API:
  metrics live under `task.metrics`, no global non-empty check.
  - Cover: `+metrics@task.metrics.f1=f1` composes correctly; for `MultiTask`,
    `+metrics@task.tasks.<child>.metrics.f1=f1` composes per-child.
- `test_train_smoke.py` — 4-task parametrized end-to-end run on dummy data.
  Old configs (`task=classification`, `task=multiclass_classification`, etc.)
  partially match new layout but need the new `head=identity` default and a
  metric on a head/task that matches the new dispatch (single output kind
  per task — no `input:` field anymore).
  - Cover: binary / multiclass / multilabel / regression end-to-end on
    `data=dummy model=lstm`, exit code 0.
- `test_hallucination_smoke.py` — needs total rewrite as a `task=multitask`
  run with two child tasks (binary fact + multilabel evidence), each with
  its own `head` baked in. The DataModule contract changed from a nested
  `labels` dict to flat top-level label keys.

## Kept (still passing)

- `test_hydra_builder.py` — exercises `build_item` / `build_items_dict` /
  `build_items_list`, unaffected by the refactor.

## Open follow-ups not covered by tests

- Per-label preserved-shape metrics for `MultilabelClassificationTask` (right
  now `update_metrics` flattens; binary-style torchmetric works, multilabel-
  shaped torchmetric does not).
- Backbone-and-heads split checkpointing (callback that writes
  `backbone.pt` + `head_<name>.pt` alongside the full Lightning checkpoint).
- `AutoModelForSequenceClassification` wrapper backbone (needs `input_keys`
  attribute + dict-style output) — was provided before the refactor, dropped
  because the bare HF class doesn't expose `input_keys`.
