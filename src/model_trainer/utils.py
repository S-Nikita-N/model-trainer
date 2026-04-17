from __future__ import annotations

import random

import numpy as np
import pytorch_lightning as pl
import torch


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, PyTorch (CPU + CUDA) and Lightning workers."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed, workers=True)
