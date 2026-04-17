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
poetry install

# Smoke-run the default pipeline (dummy data + LSTM, CPU-only is fine):
poetry run python -m model_trainer.train
```

Everything else — data, model, loss, optimizer, scheduler, callbacks, logger,
metrics, task adapter — is swappable from the CLI without touching Python.

```bash
# Train a HuggingFace seq-cls model on your data, with wandb logging:
poetry run python -m model_trainer.train \
  data=hf \
  data.tokenizer=bert-base-uncased \
  '+data.data_files.train_sets.train=/path/train.parquet' \
  '+data.data_files.valid_sets.val=/path/val.parquet' \
  model=transformers_auto_seq_cls \
  model.pretrained_model_name_or_path=bert-base-uncased \
  scheduler=linear_warmup \
  logger=wandb

# Quick dev loop: 1 epoch, 2 train batches, 1 val batch
poetry run python -m model_trainer.train \
  trainer.max_epochs=1 trainer.limit_train_batches=2 trainer.limit_val_batches=1

# Multi-GPU DDP, bf16
poetry run python -m model_trainer.train \
  trainer.accelerator=gpu trainer.devices=4 strategy=ddp trainer.precision=bf16-mixed
```

## Why

Most training projects drift into one of two states: a mess of CLI flags and
argparse forks, or a bespoke framework that's impossible to extend. Hydra +
Lightning already solve this; this repo is a thin, **battle-sized** assembly
of both with clear extension points and no surprises.

Guarantees:

- **No hard-coded hyperparameters in Python.** LR, warmup, patience, batch
  size, precision, strategy, etc. all live in YAML.
- **No hard-coded tasks in `LitModule`.** Binary vs multiclass logic lives in
  a pluggable `Task` object. Add regression / seq2seq by implementing one class.
- **No hard-coded metric lists.** Metrics are `MetricSpec(metric, input=...)`
  — the module dispatches on `input` ∈ {`preds`, `probs`, `logits`} instead of
  matching names against a static set.
- **No hard-coded validation / test set names.** The DataModule declares its
  splits via `val_set_names` / `test_set_names`; `LitModule` builds per-split
  metric trees generically.

## Repo layout

```
configs/
  train.yaml            # root config, pulls every group
  data/                 # DataModules (dummy, hf)
  model/                # backbones (lstm, transformers_auto_seq_cls, ...)
  task/                 # task adapters (classification, ...)
  loss/                 # loss functions
  optim/ scheduler/     # optimizers & LR schedulers
  metrics/              # torchmetrics wrapped in MetricSpec
  callbacks/            # ModelCheckpoint, EarlyStopping, ...
  trainer/ strategy/    # Lightning Trainer + strategy (ddp, fsdp, deepspeed)
  logger/ profiler/     # wandb, csv, simple, advanced
  experiment/           # name, seed, monitor, ckpt_path
  hydra/ paths/         # run dir, output paths
src/model_trainer/
  train.py              # Hydra entry point
  lit_module.py         # the single LightningModule
  hydra_builder.py      # instantiate helpers with sane defaults
  tasks/                # Task base + ClassificationTask + MetricSpec
  datamodules/          # DummyDataModule, HFDataModule
  backbones/            # LSTMBackbone (add your own and point _target_)
scripts/score.py        # standalone inference CLI
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
To add a new task type, subclass `Task`, put a YAML under `configs/task/`,
and override `task=<name>` on the CLI.

## How overrides work (the 30-second Hydra tour)

```bash
# Replace a whole group (file under configs/<group>/):
model=transformers_auto_seq_cls

# Override a single field inside a group:
optim.lr=5e-5

# Add a new key that isn't in the default config (note the +):
'+data.data_files.train_sets.train=/abs/path/train.parquet'

# Multi-run (grid over LRs):
python -m model_trainer.train --multirun optim.lr=1e-5,5e-5,1e-4
```

## Metrics

Each metric is a `MetricSpec` — a torchmetrics metric paired with the kind of
input it needs:

```yaml
# configs/metrics/roc_auc.yaml
roc_auc:
  _target_: model_trainer.tasks.MetricSpec
  input: probs      # "preds" | "probs" | "logits"
  metric:
    _target_: torchmetrics.AUROC
    task: binary
```

Pick a pre-made set (`metrics=default` gives `accuracy + f1 + roc_auc + pr_auc`),
build your own by listing files, or define a metric inline on the CLI.

## Checkpointing, early stopping, monitor

`experiment.monitor` is the metric name (e.g. `val_loss`, `val_accuracy`, or
`val_<set>_<metric>` when you have multiple named val sets). `experiment.monitor_mode`
is `min` or `max`. Both are interpolated into the ModelCheckpoint and EarlyStopping
callbacks — change the monitor in one place.

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

```bash
poetry run python scripts/score.py \
  --checkpoint outputs/runs/.../checkpoints/best-epoch=1-step=752-0.9384.ckpt \
  --model-path bert-base-uncased \
  --input /path/to/data.parquet \
  --output /path/to/scored.parquet \
  --text-column text \
  --batch-size 64
```

## Requirements

- Python ≥ 3.11
- PyTorch ≥ 2.0
- PyTorch Lightning ≥ 2.6
- Hydra ≥ 1.3, OmegaConf ≥ 2.3
- transformers ≥ 4.36, datasets ≥ 2.14, torchmetrics ≥ 1.0
- Optional extras: `wandb`, `rich`.

## License

Apache 2.0 — see [LICENSE](LICENSE).
