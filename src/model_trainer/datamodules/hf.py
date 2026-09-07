"""HuggingFace-datasets-backed DataModule.

Supports any number of named validation / test sets and an arbitrary number of
training files (concatenated into a single train split). The file loader is
inferred from the extension but can be overridden via the ``loader`` config
field.

Config schema (all fields optional unless noted)::

    _target_: model_trainer.datamodules.HFDataModule
    tokenizer: <hf-hub-name-or-path>           # required
    train_sets:                                # 1..N files; concatenated
      clean: /path/to/train_clean.parquet
      noisy: /path/to/train_noisy.parquet
    valid_sets:                                # 0..N named sets
      in_domain: /path/to/val_in.csv
      out_of_domain: /path/to/val_ood.csv
    test_sets:                                 # 0..N named sets
      holdout: /path/to/test.parquet
    text_column: text
    label_column: label
    label_map: {positive: 1, negative: 0}      # optional, str -> int
    label_dtype: int64                         # float32 for multilabel vectors; null to skip
    max_length: 512
    batch_size: 32
    loader: null                               # override auto-detection
"""

from __future__ import annotations

from typing import Any

import pytorch_lightning as pl
from datasets import (
    Dataset,
    DatasetDict,
    List,
    Value,
    load_dataset,
)
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, default_data_collator

from model_trainer.hydra_builder import build_item

_EXTENSION_TO_LOADER = {
    ".csv": "csv",
    ".tsv": "csv",
    ".json": "json",
    ".jsonl": "json",
    ".parquet": "parquet",
    ".txt": "text",
    ".pkl": "pandas",
    ".pickle": "pandas",
}


def _infer_loader(path: str) -> str:
    lower = path.lower()
    for ext, loader in _EXTENSION_TO_LOADER.items():
        if lower.endswith(ext):
            return loader
    raise ValueError(
        f"Cannot infer dataset loader for {path!r}. "
        f"Set `data.loader` explicitly or use one of: {sorted(_EXTENSION_TO_LOADER)}."
    )


class HFDataModule(pl.LightningDataModule):
    def __init__(
        self,
        tokenizer: str,
        train_sets: dict[str, Any] | None = None,
        valid_sets: dict[str, Any] | None = None,
        test_sets: dict[str, Any] | None = None,
        data_files: dict[str, Any] | None = None,
        name: str | None = None,
        loader: str | None = None,
        max_length: int = 512,
        padding: bool | str = False,
        truncation: bool = True,
        truncation_side: str = "right",
        collator: Any | None = None,
        text_column: str = "text",
        label_column: str = "label",
        batch_size: int = 16,
        num_workers: int = 0,
        pin_memory: bool = False,
        drop_last: bool = True,
        max_train_samples: int | None = None,
        max_val_samples: int | None = None,
        max_test_samples: int | None = None,
        load_from_cache_file: bool = True,
        label_map: dict[str, int] | None = None,
        label_dtype: str | None = "int64",
        **_: object,
    ) -> None:
        super().__init__()
        if not tokenizer:
            raise ValueError("HFDataModule: `tokenizer` is required.")
        # Splits live at the top level (``train_sets`` / ``valid_sets`` /
        # ``test_sets``), matching RAGBenchDataModule. The nested ``data_files``
        # form is the older spelling and still works; entries given both ways
        # are merged, with the flat one winning on a key clash.
        legacy = dict(data_files or {})
        train_sets = {**(legacy.get("train_sets") or {}), **(train_sets or {})}
        valid_sets = {**(legacy.get("valid_sets") or {}), **(valid_sets or {})}
        test_sets = {**(legacy.get("test_sets") or {}), **(test_sets or {})}

        if not train_sets and not name:
            raise ValueError(
                "HFDataModule: provide at least one train file via "
                "`train_sets` or a dataset `name`."
            )

        self.name = name
        self.loader = loader
        self.train_sets = train_sets
        self.valid_sets = valid_sets
        self.test_sets = test_sets

        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation
        self.truncation_side = truncation_side
        self.collator_cfg = collator
        self.text_column = text_column
        self.label_column = label_column
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.load_from_cache_file = load_from_cache_file
        self.max_train_samples = max_train_samples
        self.max_val_samples = max_val_samples
        self.max_test_samples = max_test_samples
        self.label_map = dict(label_map) if label_map else {}
        self.label_dtype = label_dtype

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.tokenizer.truncation_side = truncation_side

        self.val_set_names: list[str] = list(valid_sets.keys())
        self.test_set_names: list[str] = list(test_sets.keys())

        self.train_dataset: Dataset | None = None
        self.valid_datasets: list[tuple[str, Dataset]] = []
        self.test_datasets: list[tuple[str, Dataset]] = []
        self._collate_fn = None

    def _resolve_loader(self, path: str) -> str:
        return self.loader or _infer_loader(path)

    def _load_raw(self) -> DatasetDict:
        if not self.train_sets:
            loader = self._resolve_loader(self.name)
            return load_dataset(path=loader, data_files=self.name)

        data_files: dict[str, Any] = {"train": list(self.train_sets.values())}
        data_files.update({f"valid_{k}": v for k, v in self.valid_sets.items()})
        data_files.update({f"test_{k}": v for k, v in self.test_sets.items()})
        first_path = next(iter(self.train_sets.values()))
        loader = self._resolve_loader(first_path)
        return load_dataset(path=loader, data_files=data_files)

    def _tokenize(self, batch: dict[str, Any]) -> dict[str, Any]:
        out = self.tokenizer(
            batch[self.text_column],
            truncation=self.truncation,
            padding=self.padding,
            max_length=self.max_length,
            return_tensors=None,
        )
        if self.label_column in batch:
            values = batch[self.label_column]
            # ``label_map`` only makes sense for scalar labels; multilabel rows
            # arrive as lists, which aren't hashable.
            out["labels"] = [self.label_map.get(x, x) for x in values] if self.label_map else values
        return out

    def _process(self, ds: Dataset, limit: int | None) -> Dataset:
        if limit is not None:
            ds = ds.select(range(min(limit, len(ds))))
        ds = ds.map(
            self._tokenize,
            batched=True,
            remove_columns=ds.column_names,
            load_from_cache_file=self.load_from_cache_file,
        )
        if "labels" in ds.column_names and self.label_dtype is not None:
            # Multilabel targets arrive as a list per row; keep the list nesting
            # and only pin the scalar dtype inside it.
            feature = ds.features["labels"]
            target = (
                List(Value(self.label_dtype))
                if isinstance(feature, List)
                else Value(self.label_dtype)
            )
            ds = ds.cast_column("labels", target)
        return ds

    def setup(self, stage: str | None = None) -> None:
        if self.train_dataset is not None:
            return

        raw = self._load_raw()

        # Multiple train files are passed to load_dataset under the same split
        # name, so HF has already concatenated them under raw["train"].
        self.train_dataset = self._process(raw["train"], self.max_train_samples)

        self.valid_datasets = [
            (name, self._process(raw[f"valid_{name}"], self.max_val_samples))
            for name in self.val_set_names
            if f"valid_{name}" in raw
        ]
        self.test_datasets = [
            (name, self._process(raw[f"test_{name}"], self.max_test_samples))
            for name in self.test_set_names
            if f"test_{name}" in raw
        ]

        if self.collator_cfg is None and self.padding is False:
            raise ValueError(
                "HFDataModule: a dynamic padding collator is required when "
                "padding=False. Either set padding='max_length'/True or provide "
                "`collator` in the config."
            )

        self._collate_fn = build_item(
            self.collator_cfg,
            default=default_data_collator,
            tokenizer=self.tokenizer,
        )

    def train_dataloader(self) -> DataLoader:
        assert self.train_dataset is not None, "Call setup() first."
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            collate_fn=self._collate_fn,
        )

    def val_dataloader(self) -> list[DataLoader]:
        return [
            DataLoader(
                ds,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                collate_fn=self._collate_fn,
            )
            for _, ds in self.valid_datasets
        ]

    def test_dataloader(self) -> list[DataLoader]:
        return [
            DataLoader(
                ds,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                collate_fn=self._collate_fn,
            )
            for _, ds in self.test_datasets
        ]
