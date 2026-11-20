"""Config loading. Thin attribute-access wrapper over the YAML in configs/.

Usage:
    cfg = load_config("configs/default.yaml")
    cfg.ppo.clip_eps           # 0.2
    cfg.reward.alpha_spread    # 0.3
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class Namespace:
    """Recursive dot-access over a dict, with dict-like fallbacks."""

    def __init__(self, d: dict):
        for k, v in d.items():
            setattr(self, k, Namespace(v) if isinstance(v, dict) else v)

    def __getitem__(self, k):
        return getattr(self, k)

    def get(self, k, default=None):
        return getattr(self, k, default)

    def to_dict(self) -> dict:
        out = {}
        for k, v in self.__dict__.items():
            out[k] = v.to_dict() if isinstance(v, Namespace) else v
        return out

    def __repr__(self):
        return f"Namespace({self.to_dict()!r})"


def load_config(path: str | Path) -> Namespace:
    with open(path, "r", encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f)
    return Namespace(raw)
