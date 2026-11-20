#!/usr/bin/env python
"""SFT / distillation of a debate world model (student) on teacher trajectories (debate_wm/build_sft.py).

loss = CE(target tokens: "<argument>\\nAnswer: X<|im_end|>")
     + kl_coef * KL(teacher option distribution || student distribution over the option letters
                    at the answer-letter position)

The second term distils the teacher's BELIEF (the quantity the PPO potential reads), not only the
one letter that was sampled from it. Logits are computed only at target positions (Qwen3.5 has a
~248k vocabulary; full-sequence logits would dominate memory).

    python -m debate_wm.sft_train --model /path/to/models/qwen3.5-4b --train data/fit.jsonl \
        --val data/val.jsonl --out runs/wm_sft_v1
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

LETTERS = "ABCDEFGHIJ"


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def encode(tok, rec, eos_id):
    prompt = tok.apply_chat_template(rec["messages"], tokenize=False, add_generation_prompt=True,
                                     enable_thinking=False)
    p_ids = tok(prompt, add_special_tokens=False).input_ids
    t_ids = tok(rec["target"], add_special_tokens=False).input_ids + [eos_id]
    # answer-letter token = last target token before EOS; must be " X"
    letter_pos = len(p_ids) + len(t_ids) - 2
    return {"ids": p_ids + t_ids, "n_prompt": len(p_ids), "letter_pos": letter_pos,
            "probs": rec["probs"], "n_options": rec["n_options"]}


def batches(items, max_tokens, shuffle, rng):
    idx = list(range(len(items)))
    if shuffle:
        rng.shuffle(idx)
    # bucket by length inside windows of 512 for padding efficiency
    out = []
    for w in range(0, len(idx), 512):
        win = sorted(idx[w:w + 512], key=lambda i: len(items[i]["ids"]))
        cur, cur_max = [], 0
        for i in win:
            L = len(items[i]["ids"])
            if cur and max(cur_max, L) * (len(cur) + 1) > max_tokens:
                out.append(cur)
                cur, cur_max = [], 0
            cur.append(i)
            cur_max = max(cur_max, L)
        if cur:
            out.append(cur)
    if shuffle:
        rng.shuffle(out)
    return out


def get_lm(model):
    """(backbone returning hidden states, lm_head) for CausalLM or ConditionalGeneration wrappers."""
    head = model.get_output_embeddings()
    body = model.get_decoder() if hasattr(model, "get_decoder") else model.model
    return body, head


def step_loss(model, batch, letter_ids, pad_id, kl_coef, device):
    body, head = get_lm(model)
    B = len(batch)
    Lmax = max(len(x["ids"]) for x in batch)
    ids = torch.full((B, Lmax), pad_id, dtype=torch.long)
    attn = torch.zeros((B, Lmax), dtype=torch.long)
    for i, x in enumerate(batch):
        ids[i, :len(x["ids"])] = torch.tensor(x["ids"])
        attn[i, :len(x["ids"])] = 1
    ids, attn = ids.to(device), attn.to(device)
    hid = body(input_ids=ids, attention_mask=attn).last_hidden_state          # [B, L, H]
    rows, cols, tgt = [], [], []
    for i, x in enumerate(batch):
        for p in range(x["n_prompt"] - 1, len(x["ids"]) - 1):               # predict ids[p+1]
            rows.append(i); cols.append(p); tgt.append(x["ids"][p + 1])
    rows_t = torch.tensor(rows, device=device); cols_t = torch.tensor(cols, device=device)
    logits = head(hid[rows_t, cols_t]).float()                                # [T, V]
    ce = F.cross_entropy(logits, torch.tensor(tgt, device=device))
    # KL at the answer-letter position (the position whose next token is the letter)
    kl = logits.new_zeros(())
    if kl_coef > 0:
        lp_rows = hid[torch.arange(B, device=device),
                      torch.tensor([x["letter_pos"] - 1 for x in batch], device=device)]
        lg = head(lp_rows).float()                                            # [B, V]
        kls = []
        for i, x in enumerate(batch):
            k = x["n_options"]
            s_log = torch.log_softmax(lg[i, letter_ids[:k]], -1)
            t = torch.tensor(x["probs"], device=device).clamp_min(1e-8)
            t = t / t.sum()
            kls.append((t * (t.log() - s_log)).sum())
        kl = torch.stack(kls).mean()
    return ce, kl, len(tgt)


@torch.no_grad()
def evaluate(model, items, letter_ids, pad_id, max_tokens, device):
    model.eval()
    tot_ce, tot_n, tot_kl, n_kl, agree = 0.0, 0, 0.0, 0, 0
    body, head = get_lm(model)
    for b in batches(items, max_tokens, False, random.Random(0)):
        batch = [items[i] for i in b]
        ce, kl, n = step_loss(model, batch, letter_ids, pad_id, 1.0, device)
        tot_ce += float(ce) * n; tot_n += n
        tot_kl += float(kl) * len(batch); n_kl += len(batch)
    model.train()
    return {"val_ce": tot_ce / tot_n, "val_kl": tot_kl / n_kl}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--max-tokens", type=int, default=16384, help="padded tokens per micro-batch")
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--kl-coef", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--val-max", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fp32-master", action="store_true", help="fp32 weights + bf16 autocast")
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
    letter_ids = []
    for c in LETTERS:
        t = tok.encode(" " + c, add_special_tokens=False)
        assert len(t) == 1
        letter_ids.append(t[0])
    letter_ids = torch.tensor(letter_ids, device=device)

    log("tokenising ...")
    train = [encode(tok, json.loads(l), eos_id) for l in open(a.train)]
    val_all = [encode(tok, json.loads(l), eos_id) for l in open(a.val)]
    val = random.Random(1).sample(val_all, min(a.val_max, len(val_all)))
    for x in train[:200] + val[:200]:
        assert x["ids"][x["letter_pos"]] in letter_ids.tolist(), "letter position mis-aligned"
    log(f"train {len(train)} records, val {len(val)} (of {len(val_all)})")

    dtype = torch.float32 if a.fp32_master else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=dtype).to(device)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr,
                            weight_decay=0.0, betas=(0.9, 0.95), eps=1e-6)
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
    step, micro = 0, 0
    ep = 0.0
    t0 = time.time()
    while step < total:
        for b in batches(train, a.max_tokens, True, rng):
            batch = [train[i] for i in b]
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=a.fp32_master):
                ce, kl, n = step_loss(model, batch, letter_ids, pad_id, a.kl_coef, device)
                loss = (ce + a.kl_coef * kl) / a.accum
            loss.backward()
            micro += 1
            if micro % a.accum:
                continue
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            step += 1
            if step % 20 == 0:
                log(f"step {step}/{total} ce={float(ce):.4f} kl={float(kl):.4f} gn={float(gn):.2f} "
                    f"lr={sched.get_last_lr()[0]:.2e} {(time.time() - t0) / step:.2f}s/step")
                metrics_f.write(json.dumps({"step": step, "ce": float(ce), "kl": float(kl),
                                            "grad_norm": float(gn)}) + "\n"); metrics_f.flush()
            if step % a.eval_every == 0 or step == total:
                v = evaluate(model, val, letter_ids, pad_id, a.max_tokens, device)
                log(f"step {step} {v}")
                metrics_f.write(json.dumps({"step": step, **v}) + "\n"); metrics_f.flush()
                score = v["val_ce"] + a.kl_coef * v["val_kl"]
                if score < best:
                    best = score
                    model.save_pretrained(os.path.join(a.out, "best"), safe_serialization=True)
                    tok.save_pretrained(os.path.join(a.out, "best"))
                    json.dump({"step": step, **v}, open(os.path.join(a.out, "best", "meta.json"), "w"))
                    log(f"  saved best (score {score:.4f})")
            if step >= total:
                break
    model.save_pretrained(os.path.join(a.out, "last"), safe_serialization=True)
    tok.save_pretrained(os.path.join(a.out, "last"))
    log("done")


if __name__ == "__main__":
    main()
