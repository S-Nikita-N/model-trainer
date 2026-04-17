"""Dummy datamodule for smoke-testing the whole pipeline without real data."""

from __future__ import annotations

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset


class DummyDataset(Dataset):
    def __init__(
        self,
        length: int = 10_000,
        seq_len: int = 50,
        vocab_size: int = 30_000,
        num_classes: int = 2,
    ) -> None:
        self.length = length
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.num_classes = num_classes

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        x = torch.randint(0, self.vocab_size, (self.seq_len,), dtype=torch.long)
        y = torch.randint(0, self.num_classes, (1,), dtype=torch.long).squeeze(0)
        return {"input_ids": x, "labels": y}


class DummyDataModule(pl.LightningDataModule):
    """Random integer sequences + random labels.

    Exposes ``val_set_names`` / ``test_set_names`` so :class:`LitModule` can
    build per-set metric trees generically (no dependency on config schema).
    """

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

        self.val_set_names: list[str] = ["val"]
        self.test_set_names: list[str] = []

        self._train_ds: Dataset | None = None
        self._val_ds: Dataset | None = None

    def setup(self, stage: str | None = None) -> None:
        if self._train_ds is None:
            self._train_ds = DummyDataset(
                length=self.length,
                seq_len=self.seq_len,
                vocab_size=self.vocab_size,
                num_classes=self.num_classes,
            )
        if self._val_ds is None:
            self._val_ds = DummyDataset(
                length=self.val_length,
                seq_len=self.seq_len,
                vocab_size=self.vocab_size,
                num_classes=self.num_classes,
            )

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
