#!/usr/bin/env python
"""Turn-PPO training (§7).

Two ways to get the SFT'd (fluent, in-format, adversarial) initial actor + pi_ref:
  A. MERGED model (recommended): set actor.model_name to the merged SFT model dir
     and DO NOT pass --sft-adapter. A fresh RL LoRA is added on top; pi_ref
     (adapters off) = the merged SFT model. This is the correct anchor.
  B. LoRA adapter: --sft-adapter <dir>. NOTE this makes pi_ref = the RAW base
     (disable_adapter drops the adapter), NOT the SFT model — weaker anchor.

    python scripts/run_train.py --config configs/prototype.yaml            # merged (A)
    python scripts/run_train.py --config configs/prototype.yaml --sft-adapter <dir>  # (B)

Run scripts/preflight.py first — the §10 assertions must pass.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from attacker.config import load_config
from attacker.trainer import Trainer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--sft-adapter", default=None,
                    help="SFT LoRA adapter dir (path B). Omit if actor.model_name "
                         "is already your MERGED SFT model (path A, recommended).")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.sft_adapter:
        setattr(cfg.actor, "sft_adapter", args.sft_adapter)
    if getattr(cfg.actor, "sft_adapter", None) is None:
        print(f"[actor] no --sft-adapter; actor.model_name = {cfg.actor.model_name}\n"
              "        -> pi_ref = this model with the fresh RL LoRA disabled.\n"
              "        Make sure it's your MERGED SFT model, NOT the raw base "
              "(else there is no SFT anchor).")
    Trainer(cfg).fit()


if __name__ == "__main__":
    main()
