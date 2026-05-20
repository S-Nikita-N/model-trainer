"""Hydra entry point for standalone scoring.

Usage::

    python -m model_trainer.score \\
        checkpoint=/path/to/best.ckpt \\
        model.pretrained_model_name_or_path=bert-base-uncased \\
        input_path=/data/in.parquet \\
        output_path=/data/out.parquet

Model and tokenizer are instantiated via Hydra ``_target_``, so swapping to a
custom model is a one-line YAML change — no hard-coded ``AutoModelForSequenceClassification``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from model_trainer.hydra_builder import build_item
from model_trainer.inference import load_backbone_into
from model_trainer.tasks import ClassificationTask, Task

log = logging.getLogger(__name__)


_READERS = {
    ".csv": "read_csv",
    ".tsv": "read_csv",
    ".parquet": "read_parquet",
    ".pkl": "read_pickle",
    ".pickle": "read_pickle",
    ".json": "read_json",
    ".jsonl": "read_json",
}
_WRITERS = {
    ".csv": "to_csv",
    ".tsv": "to_csv",
    ".parquet": "to_parquet",
    ".pkl": "to_pickle",
    ".pickle": "to_pickle",
    ".json": "to_json",
    ".jsonl": "to_json",
}


def _infer_format(path: Path, table: dict[str, str]) -> str:
    suffix = path.suffix.lower()
    if suffix not in table:
        raise ValueError(
            f"Cannot infer format for {path}. Supported: {sorted(table)}. "
            "Pass input_format / output_format explicitly to override."
        )
    return table[suffix]


def _read_df(path: Path, fmt: str | None):
    import pandas as pd

    fn = fmt or _infer_format(path, _READERS)
    kwargs: dict[str, Any] = {}
    if fn == "read_json" and path.suffix.lower() == ".jsonl":
        kwargs["lines"] = True
    return getattr(pd, fn)(path, **kwargs)


def _write_df(df, path: Path, fmt: str | None) -> None:
    fn = fmt or _infer_format(path, _WRITERS)
    kwargs: dict[str, Any] = {}
    if fn == "to_csv":
        kwargs["index"] = False
    elif fn == "to_json":
        kwargs["orient"] = "records"
        if path.suffix.lower() == ".jsonl":
            kwargs["lines"] = True
    getattr(df, fn)(path, **kwargs)


def _get_config_path() -> str:
    return str(Path(__file__).resolve().parents[2] / "configs")


@hydra.main(version_base="1.3", config_path=_get_config_path(), config_name="score")
def main(cfg: DictConfig) -> None:
    log.info("Resolved config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))

    device = torch.device(
        cfg.device if cfg.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    model: torch.nn.Module = build_item(cfg.model)
    if model is None:
        raise ValueError("cfg.model must define `_target_`.")
    load_backbone_into(model, cfg.checkpoint)
    model = model.to(device).eval()

    tokenizer = build_item(cfg.tokenizer)
    if tokenizer is None:
        raise ValueError("cfg.tokenizer must define `_target_`.")

    task: Task = build_item(cfg.get("task"), default=ClassificationTask())

    input_path = Path(cfg.input_path)
    output_path = Path(cfg.output_path)
    df = _read_df(input_path, cfg.input_format)
    if cfg.text_column not in df.columns:
        raise ValueError(
            f"Column {cfg.text_column!r} missing in {input_path}. Found: {list(df.columns)}"
        )
    if cfg.limit is not None:
        df = df.head(cfg.limit)
        log.info("Limiting to first %d rows", cfg.limit)

    texts = df[cfg.text_column].astype(str).tolist()
    total = (len(texts) + cfg.batch_size - 1) // cfg.batch_size

    scores: list = []
    preds: list = []
    full_probs: list[list[float]] = []

    with torch.inference_mode():
        for start in tqdm(range(0, len(texts), cfg.batch_size), total=total, desc="Score"):
            batch = texts[start: start + cfg.batch_size]
            enc = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=cfg.max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            output = model(**enc)
            logits = task.format_output(output)

            # Predictions: task-native interpretation of the output.
            batch_preds = task.prepare_metric_input(logits, "preds").cpu()
            preds.extend(batch_preds.tolist())

            # Score: probability of positive class for classification tasks;
            # raw value for regression / anything where probs is undefined.
            try:
                probs = task.prepare_metric_input(logits, "probs").cpu()
                if probs.ndim == 1:
                    scores.extend(probs.tolist())
                elif probs.ndim == 2 and probs.shape[-1] > cfg.positive_class:
                    scores.extend(probs[..., cfg.positive_class].tolist())
                    if cfg.probs_column:
                        full_probs.extend(probs.tolist())
                else:
                    scores.extend(probs.tolist())
            except ValueError:
                scores.extend(batch_preds.tolist())

    df = df.copy()
    df[cfg.score_column] = scores
    df[cfg.pred_column] = preds
    if cfg.probs_column and full_probs:
        df[cfg.probs_column] = full_probs

    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_df(df, output_path, cfg.output_format)
    log.info("Saved %d rows to %s", len(df), output_path)


if __name__ == "__main__":
    main()
