#!/usr/bin/env python
"""Experiment 1 retrain with the full-SFT Qwen3-4B malicious actor.

This is intentionally a thin variant of ``exp1_nodefense.py``: all PPO,
validation, checkpointing and preflight logic stays shared, while the actor
checkpoint and output directory are isolated from the Qwen3.5 run.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT))

from attacker.config import load_config
from attacker.trainer import Trainer
from exp1_nodefense import DEFAULT_CONFIG, _banner, _check_invariants, run_preflight


DEFAULT_ACTOR = "/path/to/LLaMA-Factory/saves/qwen3-4b/malicious-full-sft"
DEFAULT_OUT = "runs/exp1_nodefense_qwen3_full_sft_v1"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Experiment 1 with malicious full-SFT Qwen3-4B actor")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG),
                    help="shared PPO/game config; actor/out are overridden below")
    ap.add_argument("--actor-model", default=DEFAULT_ACTOR,
                    help="full merged/SFT Hugging Face model directory")
    ap.add_argument("--out-dir", default=DEFAULT_OUT,
                    help="fresh output directory for this Qwen3 run")
    ap.add_argument("--preflight", action="store_true")
    ap.add_argument("--episodes", type=int, default=4)
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg.actor.model_name = args.actor_model
    cfg.actor.sft_adapter = None
    cfg.train.out_dir = args.out_dir

    _check_invariants(cfg)
    print("Variant: malicious full-SFT Qwen3-4B (fresh RL LoRA)")
    _banner(cfg)
    if args.preflight:
        run_preflight(cfg, args.episodes)
        return
    Trainer(cfg).fit()


if __name__ == "__main__":
    main()
