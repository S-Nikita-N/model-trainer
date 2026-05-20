"""Dummy datamodule for smoke-testing the pipeline without real data.

Supports every task shape the built-in tasks care about, selected via
``label_kind``:

- ``"binary"``       — scalar ``int`` label in ``{0, 1}``.
- ``"multiclass"``   — scalar ``int`` label in ``[0, num_classes)``.
- ``"multilabel"``   — float tensor of shape ``(num_classes,)`` with 0/1 values.
- ``"regression"``   — float scalar label sampled from a standard normal.
"""

from __future__ import annotations

from typing import Literal

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

LabelKind = Literal["binary", "multiclass", "multilabel", "regression"]


class DummyDataset(Dataset):
    def __init__(
        self,
        length: int = 10_000,
        seq_len: int = 50,
        vocab_size: int = 30_000,
        num_classes: int = 2,
        label_kind: LabelKind = "binary",
    ) -> None:
        self.length = length
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.num_classes = num_classes
        self.label_kind = label_kind

    def __len__(self) -> int:
        return self.length

    def _make_label(self) -> torch.Tensor:
        if self.label_kind in ("binary", "multiclass"):
            top = 2 if self.label_kind == "binary" else self.num_classes
            return torch.randint(0, top, (1,), dtype=torch.long).squeeze(0)
        if self.label_kind == "multilabel":
            return (torch.rand(self.num_classes) > 0.5).float()
        if self.label_kind == "regression":
            return torch.randn((), dtype=torch.float32)
        raise ValueError(f"Unknown label_kind: {self.label_kind!r}")

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        x = torch.randint(0, self.vocab_size, (self.seq_len,), dtype=torch.long)
        return {"input_ids": x, "labels": self._make_label()}


class DummyDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
        drop_last: bool = True,
        length: int = 10_000,
        seq_len: int = 50,
        vocab_size: int = 30_000,
        num_classes: int = 2,
        val_length: int = 2_000,
        label_kind: LabelKind = "binary",
        **_: object,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.length = length
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.num_classes = num_classes
        self.val_length = val_length
        self.label_kind: LabelKind = label_kind

        self.val_set_names: list[str] = ["val"]
        self.test_set_names: list[str] = []

        self._train_ds: Dataset | None = None
        self._val_ds: Dataset | None = None

    def _make_dataset(self, length: int) -> DummyDataset:
        return DummyDataset(
            length=length,
            seq_len=self.seq_len,
            vocab_size=self.vocab_size,
            num_classes=self.num_classes,
            label_kind=self.label_kind,
        )

    def setup(self, stage: str | None = None) -> None:
        if self._train_ds is None:
            self._train_ds = self._make_dataset(self.length)
        if self._val_ds is None:
            self._val_ds = self._make_dataset(self.val_length)

    def train_dataloader(self) -> DataLoader:
        assert self._train_ds is not None, "Call setup() first"
        return DataLoader(
            self._train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            shuffle=True,
        )

    def val_dataloader(self) -> list[DataLoader]:
        assert self._val_ds is not None, "Call setup() first"
        return [
            DataLoader(
                self._val_ds,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                drop_last=False,
                shuffle=False,
            )
        ]

    def test_dataloader(self) -> list[DataLoader]:
        return []
