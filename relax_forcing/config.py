from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import yaml


class Config(dict):
    """Dictionary with attribute access used by the existing pipelines."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value


def _merge(base: dict, override: Mapping) -> dict:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _merge(dict(result[key]), value)
        else:
            result[key] = value
    return result


def load_config(path: str | Path) -> Config:
    """Load YAML with optional relative `_base_` files and deep merging."""
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    bases = data.pop("_base_", [])
    if isinstance(bases, str):
        bases = [bases]

    merged = {}
    for base in bases:
        merged = _merge(merged, load_config(path.parent / base))
    return Config(_merge(merged, data))
