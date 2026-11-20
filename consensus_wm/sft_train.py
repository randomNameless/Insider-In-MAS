#!/usr/bin/env python
"""Plain-text SFT of a consensus-game world model (records from consensus_wm/build_sft.py).

loss = CE on target tokens "Message: ...\\nPosition: k<|im_end|>" only (no teacher logprobs, so
black-box w.r.t. the 9B). Same optimiser/scheduler/batching as debate_wm/sft_train.py.

    python -m consensus_wm.sft_train --model ~/models/qwen3-4b --train A_train.jsonl \
        --val A_val.jsonl --out ~/runs/cwm_sft_A --epochs 1
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
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from debate_wm.sft_train import batches, get_lm, log  # noqa: E402


def encode(tok, rec, eos_id):
    prompt = tok.apply_chat_template(rec["messages"], tokenize=False, add_generation_prompt=True,
                                     enable_thinking=False)
    p_ids = tok(prompt, add_special_tokens=False).input_ids
    t_ids = tok(rec["target"], add_special_tokens=False).input_ids + [eos_id]
    return {"ids": p_ids + t_ids, "n_prompt": len(p_ids)}


def step_loss(model, batch, pad_id, device):
    body, head = get_lm(model)
    B = len(batch)
    Lmax = max(len(x["ids"]) for x in batch)
    ids = torch.full((B, Lmax), pad_id, dtype=torch.long)
    attn = torch.zeros((B, Lmax), dtype=torch.long)
    for i, x in enumerate(batch):
        ids[i, :len(x["ids"])] = torch.tensor(x["ids"])
        attn[i, :len(x["ids"])] = 1
    ids, attn = ids.to(device), attn.to(device)
    hid = body(input_ids=ids, attention_mask=attn).last_hidden_state
    rows, cols, tgt = [], [], []
    for i, x in enumerate(batch):
        for p in range(x["n_prompt"] - 1, len(x["ids"]) - 1):
            rows.append(i); cols.append(p); tgt.append(x["ids"][p + 1])
    logits = head(hid[torch.tensor(rows, device=device), torch.tensor(cols, device=device)]).float()
    return F.cross_entropy(logits, torch.tensor(tgt, device=device)), len(tgt)


@torch.no_grad()
def evaluate(model, items, pad_id, max_tokens, device):
    model.eval()
    tot, n_tot = 0.0, 0
    for b in batches(items, max_tokens, False, random.Random(0)):
        ce, n = step_loss(model, [items[i] for i in b], pad_id, device)
        tot += float(ce) * n; n_tot += n
    model.train()
    return {"val_ce": tot / n_tot}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--max-tokens", type=int, default=16384, help="padded tokens per micro-batch")
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--val-max", type=int, default=4000)
    ap.add_argument("--max-steps", type=int, default=0, help="debug: stop after this many steps")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    device = "cuda"
    os.makedirs(a.out, exist_ok=True)
    json.dump(vars(a), open(os.path.join(a.out, "args.json"), "w"), indent=1)

    tok = AutoTokenizer.from_pretrained(a.model)
    eos_id = tok.convert_tokens_to_ids("<|im_end|>")
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else eos_id

    log("tokenising ...")
    train = [encode(tok, json.loads(l), eos_id) for l in open(a.train)]
    val_all = [encode(tok, json.loads(l), eos_id) for l in open(a.val)]
    val = random.Random(1).sample(val_all, min(a.val_max, len(val_all)))
    log(f"train {len(train)} records, val {len(val)} (of {len(val_all)})")

    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).to(device)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr,
                            weight_decay=0.0, betas=(0.9, 0.95), eps=1e-6)
    per_epoch = len(batches(train, a.max_tokens, True, random.Random(0)))
    total = int(math.ceil(per_epoch * a.epochs / a.accum))
    if a.max_steps:
        total = min(total, a.max_steps)
    warm = max(1, int(total * a.warmup))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))
    log(f"{per_epoch} micro-batches/epoch, {total} optimizer steps")

    metrics_f = open(os.path.join(a.out, "metrics.jsonl"), "a")
    best = float("inf")
    v = evaluate(model, val, pad_id, a.max_tokens, device)
    log(f"step 0 {v}")
    metrics_f.write(json.dumps({"step": 0, **v}) + "\n"); metrics_f.flush()
    step, micro = 0, 0
    t0 = time.time()
    while step < total:
        for b in batches(train, a.max_tokens, True, rng):
            ce, n = step_loss(model, [train[i] for i in b], pad_id, device)
            (ce / a.accum).backward()
            micro += 1
            if micro % a.accum:
                continue
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            step += 1
            if step % 20 == 0:
                log(f"step {step}/{total} ce={float(ce):.4f} gn={float(gn):.2f} "
                    f"lr={sched.get_last_lr()[0]:.2e} {(time.time() - t0) / step:.2f}s/step")
                metrics_f.write(json.dumps({"step": step, "ce": float(ce), "grad_norm": float(gn)}) + "\n")
                metrics_f.flush()
            if step % a.eval_every == 0 or step == total:
                v = evaluate(model, val, pad_id, a.max_tokens, device)
                log(f"step {step} {v}")
                metrics_f.write(json.dumps({"step": step, **v}) + "\n"); metrics_f.flush()
                if v["val_ce"] < best:
                    best = v["val_ce"]
                    model.save_pretrained(os.path.join(a.out, "best"), safe_serialization=True)
                    tok.save_pretrained(os.path.join(a.out, "best"))
                    json.dump({"step": step, **v}, open(os.path.join(a.out, "best", "meta.json"), "w"))
                    log(f"  saved best (val_ce {best:.4f})")
            if step >= total:
                break
    model.save_pretrained(os.path.join(a.out, "last"), safe_serialization=True)
    tok.save_pretrained(os.path.join(a.out, "last"))
    log("done")


if __name__ == "__main__":
    main()
