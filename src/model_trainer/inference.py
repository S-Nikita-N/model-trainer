"""Low-level inference helpers, reusable from Python and from the Hydra CLI.

The checkpoint must have been trained with
:class:`model_trainer.lit_module.LitModule`, so its weights live under the
``backbone.`` prefix. The model to load weights *into* is built separately
(via Hydra ``_target_`` in ``model_trainer.score``, or by hand in a notebook)
— we never assume it's a HuggingFace seq-cls model.
"""

from __future__ import annotations

from pathlib import Path

import torch


def load_backbone_into(model: torch.nn.Module, checkpoint: str | Path) -> torch.nn.Module:
    """Load ``backbone.*`` weights from a LitModule checkpoint into ``model`` in place."""
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    backbone_state = {
        key.removeprefix("backbone."): value
        for key, value in state.items()
        if key.startswith("backbone.")
    }
    if not backbone_state:
        raise ValueError(
            f"No 'backbone.*' keys in {checkpoint}. Is this a model_trainer.LitModule checkpoint?"
        )
    model.load_state_dict(backbone_state, strict=True)
    return model
