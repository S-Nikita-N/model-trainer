#!/usr/bin/env python3
"""Standalone scoring CLI for model-trainer checkpoints.

Example::

    poetry run python scripts/score.py \\
        --checkpoint outputs/runs/.../checkpoints/best.ckpt \\
        --model-path bert-base-uncased \\
        --input /path/to/data.parquet \\
        --output /path/to/scored.parquet \\
        --text-column text \\
        --batch-size 32 \\
        --max-length 512 \\
        --num-labels 2 \\
        --positive-class 1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tqdm import tqdm

from model_trainer.inference import build_inference_pipeline, score_texts

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
        raise SystemExit(
            f"Cannot infer format for {path}. Supported: {sorted(table)}. Use --format to override."
        )
    return table[suffix]


def _read_df(path: Path, fmt: str | None):
    import pandas as pd  # local import keeps pandas optional

    fn = fmt or _infer_format(path, _READERS)
    kwargs = {"lines": True} if fn == "read_json" and path.suffix.lower() == ".jsonl" else {}
    return getattr(pd, fn)(path, **kwargs)


def _write_df(df, path: Path, fmt: str | None) -> None:
    fn = fmt or _infer_format(path, _WRITERS)
    kwargs: dict = {}
    if fn == "to_csv":
        kwargs["index"] = False
    if fn == "to_json":
        kwargs["orient"] = "records"
        if path.suffix.lower() == ".jsonl":
            kwargs["lines"] = True
    getattr(df, fn)(path, **kwargs)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--checkpoint", required=True, help="Path to a Lightning .ckpt")
    p.add_argument(
        "--model-path", required=True, help="HF model name or local path used at training"
    )
    p.add_argument("--input", required=True, help="Input table (csv/parquet/json/pickle)")
    p.add_argument("--output", required=True, help="Output table path (format auto-detected)")
    p.add_argument("--text-column", default="text")
    p.add_argument("--score-column", default="score")
    p.add_argument("--pred-column", default="pred")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--num-labels", type=int, default=2)
    p.add_argument("--positive-class", type=int, default=1, help="Class index used for `score`")
    p.add_argument("--device", default=None, help="cuda | cpu | cuda:0 (auto by default)")
    p.add_argument("--limit", type=int, default=None, help="Score only the first N rows (debug)")
    p.add_argument("--input-format", default=None, help="Override input format (e.g. read_csv)")
    p.add_argument("--output-format", default=None, help="Override output format (e.g. to_parquet)")
    p.add_argument("--local-files-only", action="store_true")
    args = p.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    df = _read_df(input_path, args.input_format)

    if args.text_column not in df.columns:
        raise SystemExit(
            f"Column {args.text_column!r} missing in {input_path}. "
            f"Found columns: {list(df.columns)}"
        )
    if args.limit is not None:
        df = df.head(args.limit)
        print(f"Limiting to first {args.limit} rows", file=sys.stderr)

    model, tokenizer, device = build_inference_pipeline(
        checkpoint=args.checkpoint,
        model_name_or_path=args.model_path,
        num_labels=args.num_labels,
        device=args.device,
        local_files_only=args.local_files_only,
    )
    print(f"Device: {device}", file=sys.stderr)

    texts = df[args.text_column].astype(str).tolist()
    scores: list[float] = []
    preds: list[int] = []
    total = (len(texts) + args.batch_size - 1) // args.batch_size
    for start in tqdm(range(0, len(texts), args.batch_size), total=total, desc="Score"):
        batch = texts[start : start + args.batch_size]
        batch_scores, batch_preds = score_texts(
            batch,
            model=model,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            max_length=args.max_length,
            positive_class=args.positive_class,
            device=device,
        )
        scores.extend(batch_scores)
        preds.extend(batch_preds)

    df = df.copy()
    df[args.score_column] = scores
    df[args.pred_column] = preds

    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_df(df, output_path, args.output_format)
    print(f"Saved {len(df)} rows to {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
