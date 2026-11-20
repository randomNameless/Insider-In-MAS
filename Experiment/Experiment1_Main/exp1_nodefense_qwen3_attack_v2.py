#!/usr/bin/env python
"""Attack-priority Qwen3-4B PPO run with reduced KL/entropy constraints."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT))

from attacker.config import Namespace, load_config  # noqa: E402
from attacker.trainer import Trainer  # noqa: E402
from exp1_nodefense import _banner, _check_invariants, run_preflight  # noqa: E402


DEFAULT_VARIANT = HERE / "exp1_nodefense_qwen3_attack_v2.yaml"


def _merge_namespace(dst: Namespace, src: Namespace) -> None:
    """Recursively apply a small variant config to the full base config."""
    for key, value in vars(src).items():
        if key == "base_config":
            continue
        current = getattr(dst, key, None)
        if isinstance(value, Namespace) and isinstance(current, Namespace):
            _merge_namespace(current, value)
        else:
            setattr(dst, key, value)


def load_variant(path: str | Path) -> Namespace:
    variant_path = Path(path).resolve()
    variant = load_config(variant_path)
    base_name = getattr(variant, "base_config", "exp1_nodefense.yaml")
    base_path = Path(base_name)
    if not base_path.is_absolute():
        base_path = variant_path.parent / base_path
    cfg = load_config(base_path)
    _merge_namespace(cfg, variant)
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Attack-priority Experiment 1 with full-SFT Qwen3-4B")
    ap.add_argument("--config", default=str(DEFAULT_VARIANT),
                    help="delta config containing base_config and overrides")
    ap.add_argument("--preflight", action="store_true")
    ap.add_argument("--episodes", type=int, default=4)
    args = ap.parse_args()

    cfg = load_variant(args.config)
    _check_invariants(cfg)
    print("Variant: Qwen3-4B attack-priority v2 (reduced KL + entropy)")
    print(f"  regularization: beta_kl_ref={cfg.trust_region.beta_kl_ref} "
          f"ent_pos={cfg.trust_region.ent_coef_position} "
          f"ent_msg={cfg.trust_region.ent_coef_message}")
    _banner(cfg)
    if args.preflight:
        run_preflight(cfg, args.episodes)
        return
    Trainer(cfg).fit()


if __name__ == "__main__":
    main()
