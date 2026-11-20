#!/usr/bin/env python
"""train_dpo.py + RPO regulariser (2026-10-30): identical to debate_ppo/ju_dpo/train_dpo.py except for
--rpo-alpha (trl DPOConfig.rpo_alpha: adds alpha * NLL(chosen) to the DPO loss, Pang et al. 2024) and --loss-type.
Motivation: the v6-format strat-DPO (strat_dpo_v6_q3_4b) showed likelihood displacement (logps/chosen fell from -62 to -78
while the margin grew) and degenerate 200-token generations; see ~/runs/ppo_diag_20261028/STRATDPO_V6_RESULTS.md."""

from __future__ import annotations

import argparse
import json
import os

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/path/to/models/qwen3-4b")
    p.add_argument("--train-data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--lora-r", type=int, default=128)
    p.add_argument("--lora-alpha", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--epochs", type=float, default=1)
    p.add_argument("--per-device-batch", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--rpo-alpha", type=float, default=None, help="RPO: alpha * NLL(chosen) added to the loss")
    p.add_argument("--loss-type", default="sigmoid")
    p.add_argument("--max-steps", type=int, default=-1, help="smoke only")
    p.add_argument("--save-steps", type=int, default=500)
    p.add_argument("--eval-data", default=None, help="held-out pairs (strat_dpo); unset for the Ju arm")
    p.add_argument("--eval-steps", type=int, default=100)
    a = p.parse_args()

    tok = AutoTokenizer.from_pretrained(a.model)
    ds = load_dataset("json", data_files=a.train_data, split="train")
    ds = ds.remove_columns([c for c in ds.column_names if c not in ("prompt", "chosen", "rejected")])
    ev = None
    if a.eval_data and os.path.getsize(a.eval_data) > 0:
        ev = load_dataset("json", data_files=a.eval_data, split="train")
        ev = ev.remove_columns([c for c in ev.column_names if c not in ("prompt", "chosen", "rejected")])
    allp = ds["prompt"] + (ev["prompt"] if ev is not None else [])
    allc = ds["chosen"] + ds["rejected"] + (ev["chosen"] + ev["rejected"] if ev is not None else [])
    plen = max(len(tok(x)["input_ids"]) for x in allp)
    clen = max(len(tok(x)["input_ids"]) for x in allc)
    max_prompt = max(512, plen)
    max_len = max_prompt + max(512, clen)
    print(json.dumps({"pairs": len(ds), "longest_prompt_tokens": plen, "longest_message_tokens": clen,
                      "max_prompt_length": max_prompt, "max_length": max_len, **vars(a)}, indent=1), flush=True)

    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32, device_map="auto")
    model = get_peft_model(model, LoraConfig(
        r=a.lora_r, lora_alpha=a.lora_alpha, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    model.enable_input_require_grads()
    model.print_trainable_parameters()
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    args = DPOConfig(output_dir=a.out, logging_steps=10, save_steps=a.save_steps, num_train_epochs=a.epochs,
                     max_steps=a.max_steps, per_device_train_batch_size=a.per_device_batch,
                     gradient_accumulation_steps=a.grad_accum, learning_rate=a.lr, beta=a.beta, rpo_alpha=a.rpo_alpha, loss_type=a.loss_type,
                     max_length=max_len, max_prompt_length=max_prompt, report_to="none",
                     **({"eval_strategy": "steps", "eval_steps": a.eval_steps,
                         "per_device_eval_batch_size": a.per_device_batch} if ev is not None else {}))
    trainer = DPOTrainer(model=model, args=args, ref_model=None, processing_class=tok, train_dataset=ds,
                         eval_dataset=ev)
    trainer.train()
    trainer.save_model(a.out)
    print(f"saved LoRA adapter -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
