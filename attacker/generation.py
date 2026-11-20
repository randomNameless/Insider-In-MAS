"""Generation for the plain-text "Message:/Position:" format.

- Benign (world model, frozen): free-form batched decode -> parse_pos_msg.
- Attacker (trained): TWO-PHASE templated batched decode so segment masks are
  exact — phase 1 forces "Message: " and samples the message (stop at newline),
  phase 2 forces "\\nPosition: " and samples the integer digits. The message and
  position value tokens are the two trainable segments (§6.1).

Training log-probs always come from logprobs_at_batched on the reconstructed
sequence, so generation only needs to produce valid tokens (§10.A).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List

import torch

from .action import parse_pos_msg
from .prompts import MSG_PREFIX, POS_MIDDLE


@dataclass
class GenTurn:
    input_ids: torch.Tensor          # [L] full teacher-forceable sequence (cpu)
    value_idx: torch.Tensor          # [V] absolute indices of value tokens
    seg_masks: Dict[str, torch.Tensor]  # name -> BoolTensor[V], aligned to value_idx
    old_logp_gen: torch.Tensor       # [V] log-probs captured at generation time
    next_position: int
    message: str
    valid: bool
    prompt_len: int                  # number of prompt (chat) tokens


def _encode_chat(tokenizer, messages, device):
    ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt",
        return_dict=False, enable_thinking=False)
    if not isinstance(ids, torch.Tensor):
        ids = ids["input_ids"]
    return ids[0].to(device)


def _batched_decode(model, tokenizer, prompt_ids_list, max_new_tokens, sample,
                    temperature, keep_predicate: Callable[[str], bool], device):
    """Left-padded batched incremental decode with a per-token keep predicate.
    Stops a sequence when eos is sampled or keep_predicate(token_text) is False
    (that token is NOT recorded). Returns (gen_ids_list, gen_logps_list)."""
    B = len(prompt_ids_list)
    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id
    lens = [len(p) for p in prompt_ids_list]
    Lmax = max(lens)
    ids = torch.full((B, Lmax), pad_id, dtype=torch.long, device=device)
    attn = torch.zeros((B, Lmax), dtype=torch.long, device=device)
    for i, p in enumerate(prompt_ids_list):
        L = len(p)
        ids[i, Lmax - L:] = torch.tensor(p, device=device)
        attn[i, Lmax - L:] = 1
    position_ids = attn.long().cumsum(-1) - 1
    position_ids.masked_fill_(attn == 0, 0)

    out = model(input_ids=ids, attention_mask=attn, position_ids=position_ids,
                use_cache=True, logits_to_keep=1)   # only last-token logits (saves VRAM)
    logits = torch.nan_to_num(out.logits[:, -1, :], nan=0.0, posinf=1e4, neginf=-1e4)
    past = out.past_key_values
    next_pos = attn.sum(1)

    gen_ids = [[] for _ in range(B)]
    gen_lps = [[] for _ in range(B)]
    done = [False] * B
    for _ in range(max_new_tokens):
        lp = torch.log_softmax(logits.float(), dim=-1)
        if sample:
            probs = torch.softmax(logits.float() / max(temperature, 1e-6), dim=-1)
            toks = torch.multinomial(probs, 1).squeeze(1)
        else:
            toks = logits.argmax(-1)
        for i in range(B):
            if done[i]:
                continue
            ti = int(toks[i])
            if ti == eos_id or not keep_predicate(tokenizer.decode([ti])):
                done[i] = True
                continue
            gen_ids[i].append(ti)
            gen_lps[i].append(float(lp[i, ti]))
        if all(done):
            break
        feed = toks.clone()
        for i in range(B):
            if done[i]:
                feed[i] = pad_id
        attn = torch.cat([attn, torch.ones((B, 1), dtype=torch.long, device=device)], dim=1)
        out = model(input_ids=feed.unsqueeze(1), attention_mask=attn,
                    position_ids=next_pos.unsqueeze(1), past_key_values=past,
                    use_cache=True, logits_to_keep=1)
        logits = torch.nan_to_num(out.logits[:, -1, :], nan=0.0, posinf=1e4, neginf=-1e4)
        past = out.past_key_values
        next_pos = next_pos + 1
    return gen_ids, gen_lps


def _digit_token_conf(tokenizer, gen_ids, gen_lps, default: float = 1.0) -> float:
    """Decoder-peakedness confidence = mean probability of the generated
    digit tokens (i.e. the Position value). exp(mean logprob). Cheap: reuses the
    logprobs already produced during decoding. Used by the CP-WBFT defense."""
    import math
    lps = [lp for tid, lp in zip(gen_ids, gen_lps)
           if tokenizer.decode([tid]).strip().isdigit()]
    if not lps:
        return default
    return math.exp(sum(lps) / len(lps))


@torch.no_grad()
def generate_benign_batched(model, tokenizer, chats, fallback_positions,
                            *, max_new_tokens=200, sample=True, temperature=0.7,
                            max_pos=20, device=None):
    """Free-form batched generation for the frozen benign world model. Returns
    (parsed, confidences): a list of ParsedAction (position, message, valid) and a
    parallel list of decoder-peakedness confidences (for the CP-WBFT defense; Exp1/
    Exp2 ignore them)."""
    device = device or next(model.parameters()).device
    prompt_ids = [_encode_chat(tokenizer, m, device).tolist() for m in chats]
    gen_ids, gen_lps = _batched_decode(model, tokenizer, prompt_ids, max_new_tokens,
                                       sample, temperature, lambda t: True, device)
    parsed = [parse_pos_msg(tokenizer.decode(g), int(fb), max_pos)
              for g, fb in zip(gen_ids, fallback_positions)]
    confs = [_digit_token_conf(tokenizer, g, l) for g, l in zip(gen_ids, gen_lps)]
    return parsed, confs


@torch.no_grad()
def generate_attacker_batched(model, tokenizer, chats, fallback_positions,
                              *, max_msg_tokens=64, sample=True, temperature=1.0,
                              max_pos=20, device=None) -> List[GenTurn]:
    """Two-phase templated batched generation with exact segment masks."""
    device = device or next(model.parameters()).device
    chat_ids = [_encode_chat(tokenizer, m, device).tolist() for m in chats]
    msgprefix = tokenizer.encode(MSG_PREFIX, add_special_tokens=False)
    posmiddle = tokenizer.encode(POS_MIDDLE, add_special_tokens=False)

    # phase 1: message (stop at newline)
    p1 = [c + msgprefix for c in chat_ids]
    msg_ids, msg_lps = _batched_decode(
        model, tokenizer, p1, max_msg_tokens, sample, temperature,
        lambda t: "\n" not in t, device)

    # phase 2: integer position (stop at first non-digit token)
    p2 = [chat_ids[i] + msgprefix + msg_ids[i] + posmiddle for i in range(len(chat_ids))]
    pos_ids, pos_lps = _batched_decode(
        model, tokenizer, p2, 4, sample, temperature,
        lambda t: t.strip().isdigit(), device)

    results = []
    for i in range(len(chat_ids)):
        base = len(chat_ids[i])
        full = chat_ids[i] + msgprefix + msg_ids[i] + posmiddle + pos_ids[i]
        input_ids = torch.tensor(full, dtype=torch.long)
        msg_start = base + len(msgprefix)
        msg_abs = list(range(msg_start, msg_start + len(msg_ids[i])))
        pos_start = base + len(msgprefix) + len(msg_ids[i]) + len(posmiddle)
        pos_abs = list(range(pos_start, pos_start + len(pos_ids[i])))
        value_idx = torch.tensor(msg_abs + pos_abs, dtype=torch.long)
        V = len(msg_abs) + len(pos_abs)
        msg_mask = torch.zeros(V, dtype=torch.bool); msg_mask[:len(msg_abs)] = True
        pos_mask = torch.zeros(V, dtype=torch.bool); pos_mask[len(msg_abs):] = True

        pos_text = tokenizer.decode(pos_ids[i]).strip()
        if pos_text.isdigit():
            npos, valid = max(0, min(max_pos, int(pos_text))), True
        else:
            npos, valid = int(fallback_positions[i]), False

        results.append(GenTurn(
            input_ids=input_ids,
            value_idx=value_idx,
            seg_masks={"next_position": pos_mask, "message": msg_mask},
            old_logp_gen=torch.tensor(msg_lps[i] + pos_lps[i]),
            next_position=npos,
            message=tokenizer.decode(msg_ids[i]).strip(),
            valid=valid,
            prompt_len=base,
        ))
    return results


# ===================== training-time log-prob recompute =====================

def logprobs_at(model, input_ids: torch.Tensor, value_idx: torch.Tensor):
    """Single-sequence recompute; reference for the batched version (§10.A)."""
    ids = input_ids.unsqueeze(0)
    attn = torch.ones_like(ids)
    out = model(input_ids=ids, attention_mask=attn, use_cache=False)
    clean = torch.nan_to_num(out.logits[0].float(), nan=0.0, posinf=1e4, neginf=-1e4)
    logp = torch.log_softmax(clean, dim=-1)
    return logp[value_idx - 1, input_ids[value_idx]]


def logprobs_at_batched(model, input_ids_list, value_idx_list, pad_id,
                        return_entropy=False):
    """Batched (right-padded) recompute. Training-critical: old/new/ref log-probs
    all flow through here. Preflight asserts it matches logprobs_at.

    return_entropy=True additionally returns per-value-token entropy vectors;
    the trainer averages them for the bonus and logs segments separately."""
    device = next(model.parameters()).device
    B = len(input_ids_list)
    lens = [int(x.shape[0]) for x in input_ids_list]
    Lmax = max(lens)
    ids = torch.full((B, Lmax), pad_id, dtype=torch.long, device=device)
    attn = torch.zeros((B, Lmax), dtype=torch.long, device=device)
    for i, x in enumerate(input_ids_list):
        ids[i, :lens[i]] = x.to(device)
        attn[i, :lens[i]] = 1
    out = model(input_ids=ids, attention_mask=attn, use_cache=False)
    logits = out.logits
    res, ents = [], []
    for i in range(B):
        vidx = value_idx_list[i].to(device)
        rows = torch.nan_to_num(logits[i, vidx - 1, :].float(),
                                nan=0.0, posinf=1e4, neginf=-1e4)
        lp = torch.log_softmax(rows, dim=-1)
        tok = input_ids_list[i].to(device)[vidx]
        res.append(lp[torch.arange(vidx.shape[0], device=device), tok])
        if return_entropy:
            ents.append(-(lp.exp() * lp).sum(-1))
    if return_entropy:
        return res, ents
    return res
