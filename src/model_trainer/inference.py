"""Utilities for standalone inference with a Lightning checkpoint.

Usable both from Python and from the ``scripts/score.py`` CLI. The checkpoint
must have been trained with :class:`model_trainer.lit_module.LitModule` (so
its weights live under the ``backbone.`` prefix); any HuggingFace-style
seq-cls model is supported out of the box.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def load_backbone(
    checkpoint: str | Path,
    model_name_or_path: str,
    num_labels: int = 2,
    device: torch.device | str = "cpu",
    local_files_only: bool = False,
) -> torch.nn.Module:
    """Load the backbone weights from a LitModule checkpoint into a fresh HF model."""
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)

    backbone_state = {
        key.removeprefix("backbone."): value
        for key, value in state.items()
        if key.startswith("backbone.")
    }
    if not backbone_state:
        raise ValueError(
            f"No keys starting with 'backbone.' in {checkpoint}. "
            "Is this a model_trainer.LitModule checkpoint?"
        )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path,
        num_labels=num_labels,
        local_files_only=local_files_only,
    )
    model.load_state_dict(backbone_state, strict=True)
    return model.to(device).eval()


@torch.inference_mode()
def score_texts(
    texts: Iterable[str],
    model: torch.nn.Module,
    tokenizer,
    batch_size: int = 32,
    max_length: int = 512,
    positive_class: int = 1,
    device: torch.device | str = "cpu",
) -> tuple[list[float], list[int]]:
    """Return (scores, preds) for the positive class.

    ``scores`` are P(class=positive_class). For multiclass this is the
    per-sample probability of the chosen class; ``preds`` is the argmax over
    all classes.
    """
    texts = list(texts)
    scores: list[float] = []
    preds: list[int] = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(**enc)
        logits = getattr(out, "logits", out)
        probs = torch.softmax(logits, dim=-1)
        scores.extend(probs[..., positive_class].cpu().tolist())
        preds.extend(logits.argmax(dim=-1).cpu().tolist())

    return scores, preds


def build_inference_pipeline(
    checkpoint: str | Path,
    model_name_or_path: str,
    num_labels: int = 2,
    device: torch.device | str | None = None,
    local_files_only: bool = False,
) -> tuple[torch.nn.Module, AutoTokenizer, torch.device]:
    """Convenience: resolve device, load tokenizer + model, return all three."""
    device = torch.device(
        device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, local_files_only=local_files_only)
    model = load_backbone(
        checkpoint=checkpoint,
        model_name_or_path=model_name_or_path,
        num_labels=num_labels,
        device=device,
        local_files_only=local_files_only,
    )
    return model, tokenizer, device
