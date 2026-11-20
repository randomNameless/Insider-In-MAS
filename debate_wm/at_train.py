#!/usr/bin/env python
"""LoRA SFT of the honest debate victim on adversarial-training records (2026-11-08).

Same loss and batching as debate_wm/sft_train.py (CE on the reply tokens + kl_coef * KL(teacher belief || student
letter distribution)); here the model is Qwen3.5-9B with LoRA adapters (peft), since a full 9B fine-tune does not fit
one H100 with AdamW. The "teacher belief" of a rejection-sampled record is the victim's own readout after the kept
argument, so kl_coef > 0 only sharpens toward the accepted letter; default 0 (pure CE).
Saves the adapter at --out/{best,last} and the merged model at --merge-out (vLLM loads the merged weights).

    python -m debate_wm.at_train --model ~/models/qwen3.5-9b --train ~/data/at/train_cons.jsonl --val ~/data/at/val_cons.jsonl \
        --out ~/runs/at_cons_lora --merge-out ~/models/at-cons-9b
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from debate_wm.sft_train import LETTERS, batches, encode, evaluate, log, step_loss  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--merge-out", default=None)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--max-tokens", type=int, default=12288, help="padded tokens per micro-batch")
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--kl-coef", type=float, default=0.0)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--val-max", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    device = "cuda"
    os.makedirs(a.out, exist_ok=True)
    json.dump(vars(a), open(os.path.join(a.out, "args.json"), "w"), indent=1)

    tok = AutoTokenizer.from_pretrained(a.model)
    eos_id = tok.convert_tokens_to_ids("<|im_end|>")
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else eos_id
    letter_ids = torch.tensor([tok.encode(" " + c, add_special_tokens=False)[0] for c in LETTERS], device=device)

    log("tokenising ...")
    train = [encode(tok, json.loads(l), eos_id) for l in open(a.train)]
    val_all = [encode(tok, json.loads(l), eos_id) for l in open(a.val)]
    val = random.Random(1).sample(val_all, min(a.val_max, len(val_all)))
    for x in train[:200] + val[:200]:
        assert x["ids"][x["letter_pos"]] in letter_ids.tolist(), "letter position mis-aligned"
    log(f"train {len(train)} records ({sum(len(x['ids']) for x in train) / 1e6:.1f}M tokens), val {len(val)} (of {len(val_all)})")

    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).to(device)
    model = get_peft_model(model, LoraConfig(
        r=a.lora_r, lora_alpha=a.lora_alpha, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    model.print_trainable_parameters()
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    model.train()
    # step_loss() needs (backbone, lm_head) of the wrapped model
    base = model.get_base_model()
    import debate_wm.sft_train as S
    S.get_lm = lambda m: (base.get_decoder() if hasattr(base, "get_decoder") else base.model, base.get_output_embeddings())

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0,
                            betas=(0.9, 0.95), eps=1e-6)
    per_epoch = len(batches(train, a.max_tokens, True, random.Random(0)))
    total = int(math.ceil(per_epoch * a.epochs / a.accum))
    warm = max(1, int(total * a.warmup))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))
    log(f"{per_epoch} micro-batches/epoch, {total} optimizer steps")

    metrics_f = open(os.path.join(a.out, "metrics.jsonl"), "a")
    best = float("inf")
    v = evaluate(model, val, letter_ids, pad_id, a.max_tokens, device)
    log(f"step 0 {v}")
    metrics_f.write(json.dumps({"step": 0, **v}) + "\n"); metrics_f.flush()
    step, micro, t0 = 0, 0, time.time()
    while step < total:
        for b in batches(train, a.max_tokens, True, rng):
            batch = [train[i] for i in b]
            ce, kl, n = step_loss(model, batch, letter_ids, pad_id, a.kl_coef, device)
            ((ce + a.kl_coef * kl) / a.accum).backward()
            micro += 1
            if micro % a.accum:
                continue
            gn = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            step += 1
            if step % 10 == 0:
                log(f"step {step}/{total} ce={float(ce):.4f} kl={float(kl):.4f} gn={float(gn):.2f} "
                    f"lr={sched.get_last_lr()[0]:.2e} {(time.time() - t0) / step:.2f}s/step")
                metrics_f.write(json.dumps({"step": step, "ce": float(ce), "kl": float(kl), "grad_norm": float(gn)}) + "\n")
                metrics_f.flush()
            if step % a.eval_every == 0 or step == total:
                v = evaluate(model, val, letter_ids, pad_id, a.max_tokens, device)
                log(f"step {step} {v}")
                metrics_f.write(json.dumps({"step": step, **v}) + "\n"); metrics_f.flush()
                score = v["val_ce"] + a.kl_coef * v["val_kl"]
                if score < best:
                    best = score
                    model.save_pretrained(os.path.join(a.out, "best"))
                    json.dump({"step": step, **v}, open(os.path.join(a.out, "best", "meta.json"), "w"))
                    log(f"  saved best adapter (score {score:.4f})")
            if step >= total:
                break
    model.save_pretrained(os.path.join(a.out, "last"))
    log("training done")
    if a.merge_out:
        del model, opt
        torch.cuda.empty_cache()
        m = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16)
        m = PeftModel.from_pretrained(m, os.path.join(a.out, "best")).merge_and_unload()
        m.save_pretrained(a.merge_out, safe_serialization=True)
        tok.save_pretrained(a.merge_out)
        for extra in ("generation_config.json", "preprocessor_config.json", "chat_template.json", "video_preprocessor_config.json"):
            src = os.path.join(a.model, extra)
            if os.path.exists(src) and not os.path.exists(os.path.join(a.merge_out, extra)):
                import shutil
                shutil.copy(src, a.merge_out)
        json.dump({"base": a.model, "adapter": os.path.join(a.out, "best"), **json.load(open(os.path.join(a.out, "best", "meta.json")))},
                  open(os.path.join(a.merge_out, "at_meta.json"), "w"), indent=1)
        log(f"merged best adapter -> {a.merge_out}")


if __name__ == "__main__":
    main()
