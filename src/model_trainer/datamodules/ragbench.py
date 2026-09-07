"""DataModule for sentence-level hallucination detection on RAGBench-style data.

**Pickle schema (assumed; single point of contact is ``_unpack_example``)**::

    one_example = {
        "context_sentences":     list[str],                 # N sentences
        "response_sentences":    list[str],                 # M sentences
        "context_fact_labels":   list[int | None],          # length N: 0/1, or None if no annotation
        "response_fact_labels":  list[int | None],          # length M: same
        "support_labels":        list[list[int]],           # M × N raw values
        "contradict_labels":     list[list[int]],           # M × N raw values
    }

A pickle file may be either a ``list[dict]`` of such examples or a
``pandas.DataFrame`` with the same columns. Extra columns / keys
(``id``, ``config``, ``split``, model name, ...) are simply ignored by the
DataModule but stay accessible for downstream analysis on the same file.

``None`` in a fact-label list means "this sentence has no annotation in the
source labeling dict" — such sentences are simply **excluded** from the
offsets we produce (no -100 sentinel needed in the model's view).

The raw ``support_labels`` / ``contradict_labels`` matrices stay full-size
(M × N) and ``_build_one`` extracts the factual M_fact × N_fact submatrix
from them at processing time.

**Batch contract (flat top-level dict)**::

    batch = {
        "input_ids":            (B, T),
        "attention_mask":       (B, T),       # token-level mask for the HF backbone

        # Sentence-level head input: pre-filtered to annotated sentences only
        # (context first, then response). Includes BOTH factual (label=1) and
        # non-factual (label=0) sentences — the head learns to discriminate.
        "sentence_offsets":     (B, S_ann, 2),

        # Pair-level head inputs: pre-filtered to factual + annotated only.
        "context_fact_offsets":  (B, N_fact, 2),
        "response_fact_offsets": (B, M_fact, 2),

        # Labels — clean 0/1, with -100 reserved for batch-padding ONLY.
        "fact_labels":          (B, S_ann),                # 0 or 1, -100 = batch pad
        "evidence_labels":      (B, M_fact, N_fact, 2),   # [..., 0]=support, [..., 1]=contradict; -100 = batch pad
    }

Note that ``input_ids`` still contains **all** sentences (so the encoder sees
the full document context); the offsets just index into a subset of them.

**Per-task val / test sets**: ``valid_sets`` / ``test_sets`` map a set name
to a pickle path. The set name becomes ``val_<name>_<metric>`` in the logs.
"""

from __future__ import annotations

import logging
import pickle
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

log = logging.getLogger(__name__)

IGNORE_INDEX = -100


def _unpack_example(raw: dict[str, Any]) -> dict[str, Any]:
    """Single point of contact with the pickle schema.

    Change here when the actual file format diverges from the assumption.
    """
    return {
        "context_sentences": list(raw["context_sentences"]),
        "response_sentences": list(raw["response_sentences"]),
        "context_fact_labels": raw.get("context_fact_labels"),
        "response_fact_labels": raw.get("response_fact_labels"),
        "support_labels": raw.get("support_labels"),
        "contradict_labels": raw.get("contradict_labels"),
    }


class RAGBenchDataset(Dataset):
    def __init__(self, examples: list[dict[str, Any]]) -> None:
        self._examples = examples

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self._examples[idx]


class RAGBenchDataModule(pl.LightningDataModule):
    def __init__(
        self,
        tokenizer: str,
        train_path: str | None = None,
        valid_sets: dict[str, str] | None = None,
        test_sets: dict[str, str] | None = None,
        max_length: int = 512,
        batch_size: int = 8,
        num_workers: int = 0,
        pin_memory: bool = False,
        drop_last: bool = True,
        local_files_only: bool = False,
        max_train_samples: int | None = None,
        max_val_samples: int | None = None,
        max_test_samples: int | None = None,
        **_: object,
    ) -> None:
        super().__init__()
        self.tokenizer_name = tokenizer
        self.train_path = train_path
        self.valid_paths = dict(valid_sets or {})
        self.test_paths = dict(test_sets or {})
        self.max_length = int(max_length)
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.pin_memory = bool(pin_memory)
        self.drop_last = bool(drop_last)
        self.max_train_samples = max_train_samples
        self.max_val_samples = max_val_samples
        self.max_test_samples = max_test_samples

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, local_files_only=local_files_only)
        if self.tokenizer.cls_token_id is None or self.tokenizer.sep_token_id is None:
            raise ValueError(
                f"Tokenizer for {tokenizer!r} must define cls_token and sep_token; "
                f"got cls={self.tokenizer.cls_token!r}, sep={self.tokenizer.sep_token!r}."
            )

        self.val_set_names: list[str] = list(self.valid_paths.keys())
        self.test_set_names: list[str] = list(self.test_paths.keys())

        self._train_ds: RAGBenchDataset | None = None
        self._valid_datasets: list[tuple[str, RAGBenchDataset]] = []
        self._test_datasets: list[tuple[str, RAGBenchDataset]] = []

    def _load_pickle(self, path: str) -> list[dict[str, Any]]:
        with Path(path).open("rb") as f:
            data = pickle.load(f)
        # Lazy import to avoid a hard pandas dependency for list-of-dict users.
        try:
            import pandas as pd
        except ImportError:
            pd = None
        if pd is not None and isinstance(data, pd.DataFrame):
            return data.to_dict("records")
        if isinstance(data, list):
            return data
        raise TypeError(
            f"{path}: expected list[dict] or pandas.DataFrame, got {type(data).__name__}"
        )

    def _build_one(self, raw: dict[str, Any]) -> dict[str, Any]:
        ex = _unpack_example(raw)
        cls_id = self.tokenizer.cls_token_id
        sep_id = self.tokenizer.sep_token_id

        # 1. Tokenize ALL sentences (the encoder needs the full document
        # context) and record per-sentence ``(start, end)`` token offsets.
        ids: list[int] = [cls_id]
        all_ctx_offsets: list[tuple[int, int]] = []
        all_resp_offsets: list[tuple[int, int]] = []

        for s in ex["context_sentences"]:
            toks = self.tokenizer(s, add_special_tokens=False, truncation=False)["input_ids"]
            start = len(ids)
            ids.extend(toks)
            all_ctx_offsets.append((start, len(ids)))
            ids.append(sep_id)

        for s in ex["response_sentences"]:
            toks = self.tokenizer(s, add_special_tokens=False, truncation=False)["input_ids"]
            start = len(ids)
            ids.extend(toks)
            all_resp_offsets.append((start, len(ids)))
            ids.append(sep_id)

        if len(ids) > self.max_length:
            raise ValueError(
                f"Example exceeds max_length={self.max_length} (got {len(ids)} tokens). "
                "Increase data.max_length or pre-filter / shorten the example."
            )

        # 2. Validate label-shape consistency with the sentence counts.
        n_ctx, n_resp = len(all_ctx_offsets), len(all_resp_offsets)
        ctx_fact_labels = list(ex["context_fact_labels"] or [None] * n_ctx)
        resp_fact_labels = list(ex["response_fact_labels"] or [None] * n_resp)
        if len(ctx_fact_labels) != n_ctx:
            raise ValueError(
                f"context_fact_labels length {len(ctx_fact_labels)} != "
                f"#context_sentences {n_ctx}"
            )
        if len(resp_fact_labels) != n_resp:
            raise ValueError(
                f"response_fact_labels length {len(resp_fact_labels)} != "
                f"#response_sentences {n_resp}"
            )

        # 3. ``sentence_offsets`` = annotated sentences only (label != None),
        # context-first then response-second. ``fact_labels`` are the 0/1
        # values for those same positions.
        ann_ctx_idx = [i for i, lab in enumerate(ctx_fact_labels) if lab is not None]
        ann_resp_idx = [j for j, lab in enumerate(resp_fact_labels) if lab is not None]

        sentence_offsets = (
            [all_ctx_offsets[i] for i in ann_ctx_idx]
            + [all_resp_offsets[j] for j in ann_resp_idx]
        )
        fact_labels = (
            [int(ctx_fact_labels[i]) for i in ann_ctx_idx]
            + [int(resp_fact_labels[j]) for j in ann_resp_idx]
        )

        # 4. Pair-head inputs: factual + annotated only on each side.
        fact_ctx_idx = [i for i in ann_ctx_idx if ctx_fact_labels[i] == 1]
        fact_resp_idx = [j for j in ann_resp_idx if resp_fact_labels[j] == 1]
        context_fact_offsets = [all_ctx_offsets[i] for i in fact_ctx_idx]
        response_fact_offsets = [all_resp_offsets[j] for j in fact_resp_idx]

        # 5. Extract the factual submatrix from the raw M × N labels. The raw
        # lists may have 0s on non-factual cells (lookup said "not in keys")
        # — those rows/cols simply don't appear in the filtered submatrix.
        full_support = ex["support_labels"] or [[0] * n_ctx for _ in range(n_resp)]
        full_contradict = ex["contradict_labels"] or [[0] * n_ctx for _ in range(n_resp)]
        if any(len(row) != n_ctx for row in full_support) or len(full_support) != n_resp:
            raise ValueError(
                f"support_labels must be {n_resp} × {n_ctx}, got "
                f"{len(full_support)} × {len(full_support[0]) if full_support else 0}"
            )
        if any(len(row) != n_ctx for row in full_contradict) or len(full_contradict) != n_resp:
            raise ValueError(
                f"contradict_labels must be {n_resp} × {n_ctx}, got "
                f"{len(full_contradict)} × {len(full_contradict[0]) if full_contradict else 0}"
            )
        support_labels = [
            [int(full_support[j][i]) for i in fact_ctx_idx] for j in fact_resp_idx
        ]
        contradict_labels = [
            [int(full_contradict[j][i]) for i in fact_ctx_idx] for j in fact_resp_idx
        ]

        return {
            "input_ids": ids,
            "sentence_offsets": sentence_offsets,
            "context_fact_offsets": context_fact_offsets,
            "response_fact_offsets": response_fact_offsets,
            "fact_labels": fact_labels,
            "support_labels": support_labels,
            "contradict_labels": contradict_labels,
        }

    def _process(
        self,
        raws: Sequence[dict[str, Any]],
        limit: int | None,
        name: str,
    ) -> RAGBenchDataset:
        if limit is not None:
            raws = list(raws)[: int(limit)]
        return RAGBenchDataset([self._build_one(r) for r in raws])

    def setup(self, stage: str | None = None) -> None:
        if self._train_ds is None and self.train_path is not None:
            self._train_ds = self._process(
                self._load_pickle(self.train_path), self.max_train_samples, "train"
            )
        if not self._valid_datasets and self.valid_paths:
            self._valid_datasets = [
                (n, self._process(self._load_pickle(p), self.max_val_samples, f"val/{n}"))
                for n, p in self.valid_paths.items()
            ]
        if not self._test_datasets and self.test_paths:
            self._test_datasets = [
                (n, self._process(self._load_pickle(p), self.max_test_samples, f"test/{n}"))
                for n, p in self.test_paths.items()
            ]

    @staticmethod
    def _pad_1d(x: list[int], length: int, fill: int) -> list[int]:
        return x + [fill] * (length - len(x))

    @staticmethod
    def _pad_2d(x: list[list[int]] | None, m: int, n: int, fill: int) -> list[list[int]]:
        if x is None:
            return [[fill] * n for _ in range(m)]
        rows: list[list[int]] = []
        for i in range(m):
            if i < len(x):
                rows.append(list(x[i]) + [fill] * (n - len(x[i])))
            else:
                rows.append([fill] * n)
        return rows

    def _collate(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        max_t = max(len(e["input_ids"]) for e in examples)
        max_s = max(1, max(len(e["sentence_offsets"]) for e in examples))
        max_m = max(1, max(len(e["response_fact_offsets"]) for e in examples))
        max_n = max(1, max(len(e["context_fact_offsets"]) for e in examples))
        pad_id = self.tokenizer.pad_token_id or 0

        input_ids: list[list[int]] = []
        attention_mask: list[list[int]] = []
        sent_offsets: list[list[list[int]]] = []
        ctx_fact_offsets: list[list[list[int]]] = []
        resp_fact_offsets: list[list[list[int]]] = []
        fact_labels: list[list[int]] = []
        evidence_labels: list[list[list[list[int]]]] = []

        for e in examples:
            t = len(e["input_ids"])
            input_ids.append(self._pad_1d(e["input_ids"], max_t, pad_id))
            attention_mask.append([1] * t + [0] * (max_t - t))

            # All offset tensors pad with ``[0, 0]`` — empty token range, so
            # mean_pool_by_offsets returns a zero vector for them. No separate
            # ``*_mask`` tensors needed; -100 in label tensors marks pad
            # positions for loss / metric filtering.
            s_off = [list(p) for p in e["sentence_offsets"]]
            sent_offsets.append(s_off + [[0, 0]] * (max_s - len(s_off)))

            cf_off = [list(p) for p in e["context_fact_offsets"]]
            ctx_fact_offsets.append(cf_off + [[0, 0]] * (max_n - len(cf_off)))

            rf_off = [list(p) for p in e["response_fact_offsets"]]
            resp_fact_offsets.append(rf_off + [[0, 0]] * (max_m - len(rf_off)))

            fl = list(e["fact_labels"])
            fact_labels.append(fl + [IGNORE_INDEX] * (max_s - len(fl)))

            support_pad = self._pad_2d(e["support_labels"], max_m, max_n, IGNORE_INDEX)
            contradict_pad = self._pad_2d(e["contradict_labels"], max_m, max_n, IGNORE_INDEX)
            ev = [
                [[support_pad[i][j], contradict_pad[i][j]] for j in range(max_n)]
                for i in range(max_m)
            ]
            evidence_labels.append(ev)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "sentence_offsets": torch.tensor(sent_offsets, dtype=torch.long),
            "context_fact_offsets": torch.tensor(ctx_fact_offsets, dtype=torch.long),
            "response_fact_offsets": torch.tensor(resp_fact_offsets, dtype=torch.long),
            "fact_labels": torch.tensor(fact_labels, dtype=torch.long),
            "evidence_labels": torch.tensor(evidence_labels, dtype=torch.long),
        }

    def train_dataloader(self) -> DataLoader:
        if self._train_ds is None:
            raise RuntimeError("RAGBenchDataModule: no train_path provided.")
        return DataLoader(
            self._train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            collate_fn=self._collate,
        )

    def val_dataloader(self) -> list[DataLoader]:
        return [
            DataLoader(
                ds,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                collate_fn=self._collate,
            )
            for _, ds in self._valid_datasets
        ]

    def test_dataloader(self) -> list[DataLoader]:
        return [
            DataLoader(
                ds,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                collate_fn=self._collate,
            )
            for _, ds in self._test_datasets
        ]
