# model-trainer

[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org)
[![PyTorch Lightning](https://img.shields.io/badge/lightning-2.6%2B-792ee5)](https://lightning.ai)
[![Hydra](https://img.shields.io/badge/hydra-1.3%2B-89b8cd)](https://hydra.cc)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache_2.0-green)](LICENSE)

A small, opinionated training harness built on **Hydra + PyTorch Lightning**.
Clone, point a few `_target_`s at your own classes, override on the CLI, and
train. **Configs first, code second.**

## TL;DR

```bash
git clone <this repo> && cd model-trainer
uv sync

# Smoke-run (dummy data + LSTM). Metrics are mandatory — at least one ``+metrics@...``:
uv run python -m model_trainer.train \
  '+metrics@metrics.f1=f1' '+metrics@metrics.roc_auc=roc_auc'
```

Everything else — data, model, loss, optimizer, scheduler, callbacks, logger,
metrics, task adapter — is swappable from the CLI without touching Python.

```bash
# Train a HuggingFace seq-cls model on your data, with wandb logging:
uv run python -m model_trainer.train \
  data=hf \
  data.tokenizer=bert-base-uncased \
  '+data.train_sets.train=/path/train.parquet' \
  '+data.valid_sets.val=/path/val.parquet' \
  model=transformers_auto_seq_cls \
  model.pretrained_model_name_or_path=bert-base-uncased \
  '+model.classifier_dropout=0.1' \
  '+metrics@metrics.accuracy=accuracy' '+metrics@metrics.f1=f1' \
  '+metrics@metrics.roc_auc=roc_auc' '+metrics@metrics.pr_auc=pr_auc' \
  scheduler=linear_warmup \
  logger=wandb

# Quick dev loop: 1 epoch, 2 train batches, 1 val batch
uv run python -m model_trainer.train \
  '+metrics@metrics.f1=f1' '+metrics@metrics.roc_auc=roc_auc' \
  trainer.max_epochs=1 trainer.limit_train_batches=2 trainer.limit_val_batches=1

# Multi-GPU DDP, bf16
uv run python -m model_trainer.train \
  trainer.accelerator=gpu trainer.devices=4 strategy=ddp trainer.precision=bf16-mixed
```

## Supported tasks

| Task         | `task=<name>`                 | Loss                | Typical `+metrics@...` stems | Batch `labels`              |
|--------------|-------------------------------|---------------------|------------------------------|-----------------------------|
| Binary       | `classification`              | `crossentropy`      | `accuracy`, `f1`, `roc_auc`, `pr_auc` | `long` `{0,1}`   |
| Multiclass   | `multiclass_classification`   | `crossentropy`      | `accuracy_multiclass`, `f1_macro_multiclass`, … | `long` `[0,C)` |
| Multilabel   | `multilabel_classification`   | `bce_with_logits`   | `f1_macro_multilabel`, …     | `float` `(L,)`              |
| Regression   | `regression`                  | `mse` / `l1`        | `mae`, `mse`, `r2`           | `float` scalar              |

You must pass at least one metric; empty `metrics` exits with `ValueError`. See `tests/test_train_smoke.py` for full per-task lists.

Adding a new task = subclass `Task`, one YAML under `configs/task/`. No
changes in `LitModule`. See `src/model_trainer/tasks/` for the three
reference implementations (≈30–40 lines each).

## Multi-task training

`task=multitask` composes several child tasks over one shared backbone. Each
child owns its own head, loss and metrics. Children are added via Hydra's
`+task@task.tasks.<name>=<preset>` syntax; their internal groups (`head`,
`loss`, `metrics`) are overridable from the CLI like any nested Hydra group.

```bash
# Sentence-level hallucination detection on RAGBench-style pickles:
# per-response-sentence binary "contains a fact" (head A) + per-pair
# multilabel "support / contradict" evidence (head B), joint loss.
uv run python -m model_trainer.train \
  data=ragbench model=hf_encoder task=multitask \
  data.train_path=/path/train.pkl \
  '+data.valid_sets.fact=/path/val_fact.pkl' \
  '+data.valid_sets.evidence=/path/val_evidence.pkl' \
  '+task@task.tasks.fact=binary' \
  '+task@task.tasks.evidence=multilabel' \
  task.tasks.fact.label_key=fact_labels \
  task.tasks.evidence.label_key=evidence_labels \
  task.tasks.evidence.num_labels=2 \
  'head@task.tasks.fact.head=sentence_mlp' \
  'head@task.tasks.evidence.head=pairwise_evidence' \
  '+metrics@task.tasks.fact.metrics.f1=f1' \
  '+metrics@task.tasks.fact.metrics.auroc=roc_auc' \
  '+metrics@task.tasks.evidence.metrics.f1=f1' \
  '+task.weights.fact=1.0' '+task.weights.evidence=1.0' \
  experiment.monitor=val_fact_f1 experiment.monitor_mode=max
```

Anatomy of the override paths:

- `+task@task.tasks.<name>=<preset>` — adds a new child task under
  `task.tasks.<name>`, loaded from `configs/task/<preset>.yaml`.
- `head@task.tasks.<name>.head=<stem>` — swaps that child's head with
  `configs/head/<stem>.yaml` (package modifier; the `head` group is absolute).
- `+metrics@task.tasks.<name>.metrics.<slot>=<stem>` — adds one metric from
  `configs/metrics/<stem>.yaml` into that child's metric dict.
- `+task.weights.<name>=<float>` — per-child loss weight (default 1.0).

Per-task val sets (`+data.valid_sets.fact=...` / `+data.valid_sets.evidence=...`)
let each head be evaluated on the split that actually carries its labels.

## Why

Most training projects drift into one of two states: a mess of CLI flags and
argparse forks, or a bespoke framework that's impossible to extend. Hydra +
Lightning already solve this; this repo is a thin, battle-sized assembly of
both with clear extension points and no surprises.

Guarantees:

- **No hard-coded hyperparameters in Python.** LR, warmup, patience, batch
  size, precision, strategy, etc. all live in YAML.
- **No hard-coded task logic in `LitModule`.** Binary / multiclass / multilabel /
  regression live in a pluggable `Task` object. Add a new one by implementing
  one class and one YAML.
- **Metrics are explicit.** Each is a `MetricSpec` in `configs/metrics/<stem>.yaml`,
  composed with `+metrics@metrics.<slot>=<stem>`. Missing metrics → hard error at startup.
- **No hard-coded validation / test set names.** The DataModule declares its
  splits via `val_set_names` / `test_set_names`; `LitModule` builds per-split
  metric trees generically.

## Repo layout

```
configs/
  train.yaml            # root config for training
  score.yaml            # root config for standalone inference
  data/                 # DataModules (dummy, hf)
  model/                # backbones (lstm, transformers_auto_seq_cls, ...)
  task/                 # task adapters (classification, multiclass, multilabel, regression)
  loss/                 # loss functions (crossentropy, bce_with_logits, mse, l1)
  optim/ scheduler/     # optimizers & LR schedulers
  metrics/              # one YAML per metric (MetricSpec); no bundle presets
  callbacks/            # ModelCheckpoint, EarlyStopping, ...
  trainer/ strategy/    # Lightning Trainer + strategy (ddp, fsdp, deepspeed)
  logger/ profiler/     # wandb, csv, simple, advanced
  experiment/           # name, seed, monitor, ckpt_path
  hydra/ paths/         # run dir, output paths
src/model_trainer/
  train.py              # Hydra entry point for training
  score.py              # Hydra entry point for scoring
  lit_module.py         # the single LightningModule
  hydra_builder.py      # instantiate helpers with sane defaults
  tasks/                # Task base + ClassificationTask + RegressionTask + MultilabelTask + MetricSpec
  datamodules/          # DummyDataModule, HFDataModule
  backbones/            # LSTMBackbone (add your own, point _target_)
  inference.py          # checkpoint-loading helpers used by score.py
tests/                  # unit tests + parametrized end-to-end smoke tests (one per task)
```

## Contracts

Minimum you have to provide to train on your own data & model:

**Batches** (from the DataModule) are `dict`s that:
- contain `"labels"` (for loss and metrics);
- contain whatever kwargs your backbone's `forward` expects (typically
  `input_ids`, `attention_mask`).

**DataModule** exposes two attributes set in `__init__`:
- `val_set_names: list[str]` — one per validation DataLoader.
- `test_set_names: list[str]` — one per test DataLoader (can be empty).

`val_dataloader()` / `test_dataloader()` return a **list** of DataLoaders,
one per name. See `datamodules/dummy.py` for the minimal example.

**Backbone** is any `nn.Module` with `forward(**batch_without_labels) -> logits`
or an object with a `.logits` attribute (HuggingFace convention).

**Task** turns `logits` into whatever each metric needs:
```python
class ClassificationTask(Task):
    def prepare_metric_input(self, logits, kind):
        if kind == "preds":  return logits.argmax(-1)
        if kind == "probs":  return torch.softmax(logits, -1)[..., self.positive_class]
        if kind == "logits": return logits
```

## Hydra 30-second tour

```bash
# Replace a whole group (file under configs/<group>/):
model=transformers_auto_seq_cls

# Override an existing field:
optim.lr=5e-5

# Add a NEW field that isn't in the default config (note the "+"):
'+model.classifier_dropout=0.1'

# Add-or-override (doesn't care whether the field exists):
'++model.num_labels=3'

# Remove a field:
'~callbacks.early_stopping'

# Multirun (sweep over LRs, one run per value):
python -m model_trainer.train --multirun optim.lr=1e-5,5e-5,1e-4
```

## Metrics

Each metric is a `MetricSpec` — a torchmetrics metric paired with the kind of
input it needs (`preds` / `probs` / `logits`). **One YAML per metric**, flat
under `configs/metrics/<stem>.yaml`.

`train.yaml` sets `metrics: {}`. **You must add at least one** via
`+metrics@metrics.<slot>=<stem>`; otherwise `train.py` raises immediately.

```bash
# Minimal binary set:
python -m model_trainer.train \
  '+metrics@metrics.accuracy=accuracy' \
  '+metrics@metrics.f1=f1' \
  '+metrics@metrics.roc_auc=roc_auc' \
  '+metrics@metrics.pr_auc=pr_auc'

# Or a smaller custom set:
python -m model_trainer.train \
  '+metrics@metrics.f1=f1' \
  '+metrics@metrics.roc_auc=roc_auc' \
  '+metrics@metrics.precision=precision'

# Replace a slot (only works for entries you added with ``+`` in the defaults list):
python -m model_trainer.train \
  '+metrics@metrics.pr_auc=pr_auc' \
  'metrics@metrics.pr_auc=recall_at_precision'
```

Add a new definition once under `configs/metrics/my_metric.yaml`, then
`+metrics@metrics.foo=my_metric`.

## Checkpointing, early stopping, monitor

`experiment.monitor` is the metric name (e.g. `val_loss`, `val_accuracy`, or
`val_<set>_<metric>` when you have multiple named val sets).
`experiment.monitor_mode` is `min` or `max`. Both are interpolated into the
ModelCheckpoint and EarlyStopping callbacks — change the monitor in one place.

```bash
python -m model_trainer.train \
  experiment.monitor=val_roc_auc experiment.monitor_mode=max
```

## Outputs & logging

- Runs go to `${MODEL_TRAINER_OUTPUT:-outputs}/runs/${experiment.name}/<timestamp>/`.
- Hydra writes `train.log` there and dumps the full resolved config.
- `logger=wandb` enables Weights & Biases; set `WANDB_MODE=offline` for
  offline mode and `wandb sync <run-dir>` later.

## Resuming / warm-starting from a checkpoint

```bash
# Load weights only, fresh optimizer / epoch / scheduler:
python -m model_trainer.train \
  experiment.ckpt_path=/path/to/best.ckpt experiment.ckpt_weights_only=true

# Full resume (including optimizer state):
python -m model_trainer.train \
  experiment.ckpt_path=/path/to/last.ckpt experiment.ckpt_weights_only=false
```

## Standalone inference

Like training, scoring is a Hydra entry point — model, tokenizer and task
are all instantiated via `_target_`, so swapping to a custom backbone is a
one-line YAML change:

```bash
uv run python -m model_trainer.score \
  checkpoint=outputs/runs/.../checkpoints/best.ckpt \
  model.pretrained_model_name_or_path=bert-base-uncased \
  input_path=/path/to/data.parquet \
  output_path=/path/to/scored.parquet

# Multiclass: also dump the full probability matrix
uv run python -m model_trainer.score \
  checkpoint=... model.pretrained_model_name_or_path=bert-base-uncased \
  model.num_labels=5 \
  input_path=... output_path=... probs_column=all_probs
```

## Development

```bash
uv sync --all-extras
uv run pytest          # unit + parametrized end-to-end smoke tests
uv run ruff check .    # lint
uv run ruff format .   # auto-format
uv run pre-commit install
```

CI (GitHub Actions) runs the lint + format check + test suite on every push
and PR across Python 3.11 and 3.12.

## Requirements

- Python ≥ 3.11
- PyTorch ≥ 2.0
- PyTorch Lightning ≥ 2.6
- Hydra ≥ 1.3, OmegaConf ≥ 2.3
- transformers ≥ 4.36, datasets ≥ 2.14, torchmetrics ≥ 1.0
- Optional extras: `wandb`, `pandas` (for scoring), `scikit-learn`.

## License

Apache 2.0 — see [LICENSE](LICENSE).
