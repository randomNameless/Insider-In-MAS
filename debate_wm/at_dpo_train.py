#!/usr/bin/env python
"""LoRA-DPO of the honest debate victim on consistency preference pairs (2026-11-09; debate_wm/build_dpo_pairs.py).

loss = -log sigmoid(beta * [(log pi(c|x) - log ref(c|x)) - (log pi(r|x) - log ref(r|x))])   (Rafailov et al. 2023)
Sequence log-probs are summed over the reply tokens (+ <|im_end|>). The reference model is the same network with the
LoRA adapter disabled (peft `disable_adapter`), so no second copy of the 9B weights is needed; reference log-probs are
computed once per batch under no_grad. trl is not installed in this venv, hence the explicit loop (same batching and
model handling as debate_wm/at_train.py). Saves the adapter at --out/{best,last} (best = lowest val loss) and the merged
TEXT model at --merge-out (then debate_wm/export_vllm.py makes it servable).

    python -m debate_wm.at_dpo_train --model ~/models/qwen3.5-9b --train ~/data/at/train_dpo.jsonl --val ~/data/at/val_dpo.jsonl \
        --out ~/runs/at_dpo_lora --merge-out ~/runs/at_dpo_lora/merged_text
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from debate_wm.sft_train import log  # noqa: E402


def encode(tok, rec, eos_id):
    prompt = tok.apply_chat_template(rec["prompt"], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    p = tok(prompt, add_special_tokens=False).input_ids
    c = tok(rec["chosen"], add_special_tokens=False).input_ids + [eos_id]
    r = tok(rec["rejected"], add_special_tokens=False).input_ids + [eos_id]
    return {"p": p, "c": c, "r": r, "len": len(p) + max(len(c), len(r))}


def batches(items, max_tokens, shuffle, rng):
    idx = list(range(len(items)))
    if shuffle:
        rng.shuffle(idx)
    out = []
    for w in range(0, len(idx), 512):
        win = sorted(idx[w:w + 512], key=lambda i: items[i]["len"])
        cur, cur_max = [], 0
        for i in win:
            L = items[i]["len"]
            if cur and max(cur_max, L) * 2 * (len(cur) + 1) > max_tokens:     # x2: chosen + rejected sequences
                out.append(cur); cur, cur_max = [], 0
            cur.append(i); cur_max = max(cur_max, L)
        if cur:
            out.append(cur)
    if shuffle:
        rng.shuffle(out)
    return out


def seq_logps(model, batch, pad_id, device):
    """Summed log-prob of the reply tokens for [chosen..., rejected...] -> tensor [2B]."""
    seqs = [(x["p"], x["c"]) for x in batch] + [(x["p"], x["r"]) for x in batch]
    Lmax = max(len(p) + len(t) for p, t in seqs)
    ids = torch.full((len(seqs), Lmax), pad_id, dtype=torch.long)
    attn = torch.zeros((len(seqs), Lmax), dtype=torch.long)
    mask = torch.zeros((len(seqs), Lmax), dtype=torch.bool)       # positions whose NEXT token is a reply token
    for i, (p, t) in enumerate(seqs):
        s = p + t
        ids[i, :len(s)] = torch.tensor(s); attn[i, :len(s)] = 1
        mask[i, len(p) - 1:len(s) - 1] = True
    ids, attn, mask = ids.to(device), attn.to(device), mask.to(device)
    logits = model(input_ids=ids, attention_mask=attn).logits          # [2B, L, V] (bf16)
    lp = torch.log_softmax(logits[:, :-1].float(), -1)
    tok_lp = lp.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)        # [2B, L-1]
    return (tok_lp * mask[:, :-1]).sum(-1)


def dpo_loss(model, batch, pad_id, beta, device):
    B = len(batch)
    with torch.no_grad(), model.disable_adapter():
        ref = seq_logps(model, batch, pad_id, device)
    pol = seq_logps(model, batch, pad_id, device)
    margin = beta * ((pol[:B] - ref[:B]) - (pol[B:] - ref[B:]))
    loss = -F.logsigmoid(margin).mean()
    acc = (margin > 0).float().mean()
    return loss, acc, margin.mean()


@torch.no_grad()
def evaluate(model, items, pad_id, beta, max_tokens, device):
    model.eval()
    tot, n, acc = 0.0, 0, 0.0
    for b in batches(items, max_tokens, False, random.Random(0)):
        batch = [items[i] for i in b]
        l, a, _ = dpo_loss(model, batch, pad_id, beta, device)
        tot += float(l) * len(batch); acc += float(a) * len(batch); n += len(batch)
    model.train()
    return {"val_loss": tot / n, "val_pref_acc": acc / n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--merge-out", default=None)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--warmup", type=float, default=0.05)
    ap.add_argument("--max-tokens", type=int, default=16384, help="padded tokens per micro-batch (chosen + rejected)")
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--val-max", type=int, default=600)
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
    log("tokenising ...")
    train = [encode(tok, json.loads(l), eos_id) for l in open(a.train)]
    val_all = [encode(tok, json.loads(l), eos_id) for l in open(a.val)]
    val = random.Random(1).sample(val_all, min(a.val_max, len(val_all)))
    log(f"train {len(train)} pairs, val {len(val)} (of {len(val_all)})")

    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).to(device)
    model = get_peft_model(model, LoraConfig(
        r=a.lora_r, lora_alpha=a.lora_alpha, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    model.print_trainable_parameters()
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0,
                            betas=(0.9, 0.95), eps=1e-6)
    per_epoch = len(batches(train, a.max_tokens, True, random.Random(0)))
    total = int(math.ceil(per_epoch * a.epochs / a.accum))
    warm = max(1, int(total * a.warmup))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))
    log(f"{per_epoch} micro-batches/epoch, {total} optimizer steps")
    mf = open(os.path.join(a.out, "metrics.jsonl"), "a")
    v = evaluate(model, val, pad_id, a.beta, a.max_tokens, device)
    log(f"step 0 {v}"); mf.write(json.dumps({"step": 0, **v}) + "\n"); mf.flush()
    best, step, micro, t0 = float("inf"), 0, 0, time.time()
    while step < total:
        for b in batches(train, a.max_tokens, True, rng):
            batch = [train[i] for i in b]
            loss, acc, mg = dpo_loss(model, batch, pad_id, a.beta, device)
            (loss / a.accum).backward()
            micro += 1
            if micro % a.accum:
                continue
            gn = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            step += 1
            if step % 5 == 0:
                log(f"step {step}/{total} loss={float(loss):.4f} pref_acc={float(acc):.2f} margin={float(mg):.3f} "
                    f"gn={float(gn):.2f} lr={sched.get_last_lr()[0]:.2e} {(time.time() - t0) / step:.1f}s/step")
                mf.write(json.dumps({"step": step, "loss": float(loss), "pref_acc": float(acc), "margin": float(mg),
                                     "grad_norm": float(gn)}) + "\n"); mf.flush()
            if step % a.eval_every == 0 or step == total:
                v = evaluate(model, val, pad_id, a.beta, a.max_tokens, device)
                log(f"step {step} {v}"); mf.write(json.dumps({"step": step, **v}) + "\n"); mf.flush()
                if v["val_loss"] < best:
                    best = v["val_loss"]
                    model.save_pretrained(os.path.join(a.out, "best"))
                    json.dump({"step": step, **v}, open(os.path.join(a.out, "best", "meta.json"), "w"))
                    log(f"  saved best adapter (val_loss {best:.4f})")
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
        json.dump({"base": a.model, "adapter": os.path.join(a.out, "best"), "method": "dpo",
                   **json.load(open(os.path.join(a.out, "best", "meta.json")))},
                  open(os.path.join(a.merge_out, "at_meta.json"), "w"), indent=1)
        log(f"merged best adapter -> {a.merge_out}")


if __name__ == "__main__":
    main()
