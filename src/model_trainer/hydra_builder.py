"""Thin wrappers around ``hydra.utils.instantiate``.

The goal is to make config groups forgiving: a missing group, a group with no
``_target_``, or ``None`` all fall through to a caller-provided default instead
of crashing. This is what lets us write ``strategy: auto`` (an empty YAML) or
``logger: null`` without branching in the entry point.
"""

from __future__ import annotations

from typing import Any

import hydra


def _has_target(cfg: Any) -> bool:
    return hasattr(cfg, "get") and bool(cfg.get("_target_", None))


def build_item(cfg: Any, default: Any = None, **kwargs: Any) -> Any:
    """Instantiate a single object from a config node.

    Returns ``default`` if ``cfg`` is ``None`` or has no ``_target_``.
    Extra ``kwargs`` are forwarded to the constructor (useful for wiring in
    objects that are not themselves in the config, e.g. ``params=model.parameters()``).
    """
    if cfg is None:
        return default
    if not _has_target(cfg):
        return default
    return hydra.utils.instantiate(cfg, **kwargs)


def build_items_list(cfg: Any, default: list[Any] | None = None, **kwargs: Any) -> list[Any]:
    """Instantiate every node in ``cfg`` that has a ``_target_``.

    Iterates over ``cfg.values()``; useful for callbacks-style configs that are
    a dict of heterogeneous objects.
    """
    if cfg is None:
        return [] if default is None else default
    out: list[Any] = []
    for item in cfg.values():
        if _has_target(item):
            out.append(hydra.utils.instantiate(item, **kwargs))
    return out


def build_items_dict(
    cfg: Any,
    default: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Instantiate every named node in ``cfg`` that has a ``_target_``.

    Preserves keys — used for metrics / loggers where naming matters.
    """
    if cfg is None:
        return {} if default is None else default
    out: dict[str, Any] = {}
    for key, value in cfg.items():
        if _has_target(value):
            out[key] = hydra.utils.instantiate(value, **kwargs)
    return out
