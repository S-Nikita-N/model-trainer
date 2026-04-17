"""Hydra + PyTorch Lightning training harness."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("model-trainer")
except PackageNotFoundError:  # editable install without metadata
    __version__ = "0.0.0+local"

__all__ = ["__version__"]
