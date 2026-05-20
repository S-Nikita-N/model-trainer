"""Entry point: Hydra assembles the config, we build DataModule + Model + Trainer.

Run from the repo root:

    python -m model_trainer.train
    python -m model_trainer.train model=hf_encoder data=ragbench task=multitask
    python -m model_trainer.train trainer.max_epochs=1 trainer.limit_train_batches=2

To plug your own code in: implement a DataModule / backbone / head / Task,
point ``_target_`` at them in a YAML under ``configs/``, override on the CLI.
"""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf

from model_trainer.hydra_builder import build_item, build_items_list
from model_trainer.lit_module import LitModule
from model_trainer.utils import set_seed

log = logging.getLogger(__name__)


def _get_config_path() -> str:
    return str(Path(__file__).resolve().parents[2] / "configs")


def _load_weights_only(model: pl.LightningModule, ckpt_path: str) -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state, strict=True)


@hydra.main(version_base="1.3", config_path=_get_config_path(), config_name="train")
def main(cfg: DictConfig) -> None:
    log.info("Resolved config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))

    seed = cfg.experiment.get("seed")
    if seed is not None:
        set_seed(int(seed))

    datamodule: pl.LightningDataModule = build_item(cfg.data)
    val_set_names = tuple(getattr(datamodule, "val_set_names", ("val",)))
    test_set_names = tuple(getattr(datamodule, "test_set_names", ()))

    model = LitModule(
        cfg=cfg,
        val_set_names=val_set_names,
        test_set_names=test_set_names,
    )

    ckpt_path = cfg.experiment.get("ckpt_path")
    ckpt_weights_only = bool(cfg.experiment.get("ckpt_weights_only", True))
    if ckpt_path and ckpt_weights_only:
        _load_weights_only(model, ckpt_path)
        ckpt_path = None

    trainer = pl.Trainer(
        **cfg.trainer,
        strategy=build_item(cfg.get("strategy"), default="auto"),
        logger=build_item(cfg.get("logger"), default=False),
        callbacks=build_items_list(cfg.get("callbacks"), default=[]),
        profiler=build_item(cfg.get("profiler")),
    )

    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)

    if not cfg.experiment.get("test_after_fit", False):
        return
    if not test_set_names:
        log.info("test_after_fit=True but DataModule exposes no test sets, skipping.")
        return

    ckpt_cb = trainer.checkpoint_callback
    best_path = getattr(ckpt_cb, "best_model_path", None) if ckpt_cb else None
    if best_path:
        # PyTorch >= 2.6 defaults to weights_only=True for torch.load. Lightning's
        # checkpoint embeds non-tensor metadata (OmegaConf, callbacks), so we
        # explicitly opt out and load weights manually before testing.
        _load_weights_only(model, best_path)
        log.info("Running test on best checkpoint: %s", best_path)
    else:
        log.info("No best checkpoint recorded, running test on the final in-memory weights.")

    trainer.test(model, datamodule=datamodule, ckpt_path=None)


if __name__ == "__main__":
    main()
