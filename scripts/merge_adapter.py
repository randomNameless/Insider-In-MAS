#!/usr/bin/env python
"""Merge an RL LoRA adapter into its base (the SFT model) -> one full model.

    python scripts/merge_adapter.py \
        --base /path/to/models/malicious-merged \
        --adapter runs/proto/best \
        --out /path/to/models/malicious-rl-merged

Runs on CPU (no .to(cuda)) so it's safe while training is using the GPU. The
adapter's own base_model_name_or_path field is IGNORED — the --base you pass is
what the adapter is applied to (this is the whole point: the adapter was trained
on the SFT model, not the raw base named in adapter_config.json).
"""
import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="base model dir (the SFT merged model)")
    ap.add_argument("--adapter", required=True, help="LoRA adapter dir (e.g. runs/proto/best)")
    ap.add_argument("--out", required=True, help="output dir for the merged full model")
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "float32"])
    args = ap.parse_args()

    dtype = getattr(torch, args.dtype)
    print(f"[merge] base    = {args.base}")
    print(f"[merge] adapter = {args.adapter}")
    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=dtype, trust_remote_code=True)   # CPU (no .to(cuda))
    model = PeftModel.from_pretrained(base, args.adapter)        # applied to THIS base
    print("[merge] merging LoRA into base weights ...")
    model = model.merge_and_unload()
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print(f"[merge] saved merged full model -> {args.out}")


if __name__ == "__main__":
    main()
