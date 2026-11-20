"""Generation back-ends.

* Honest agents (WM in training / sim eval, Qwen3.5-9B in sim2real and the training monitor):
  vLLM through debate_pilot/engine.py (argument sampled until 'Answer:', then the option-letter
  distribution is read out after that argument, then the answer is sampled from it).
* Attacker during training: HF two-phase templated decode with exact segment masks, reusing the
  consensus decoder (attacker/generation.py):
      phase 1: force 'Argument:' and sample the argument until a newline / EOS / cap
      phase 2: force '\\nAnswer:' and sample ONE token, valid iff it is ' <letter>' of a listed option
  Segment names are kept as the consensus ones so attacker/trainer.py works unchanged:
      'message' = argument tokens, 'next_position' = answer-letter token.
"""
from __future__ import annotations

import json
import os
from typing import List, Optional, Sequence

from . import PILOT_DIR  # noqa: F401
from . import dprotocol as D

os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")


# ------------------------------------------------------------------ vLLM honest engine
def make_vllm_engine(model_path: str, *, gpu_memory_utilization: float, max_model_len: int = 6144,
                     seed: int = 0, enable_sleep_mode: bool = False, enforce_eager: bool = False,
                     max_num_seqs: int = None, enable_prefix_caching: bool = False,
                     gdn_prefill_backend: str = "triton"):
    """debate_pilot.engine.Engine with extra LLM kwargs (sleep mode). Same methods / readout.
    Prefix caching is OFF by default: with Qwen3.5 (GDN/Mamba layers) vLLM 0.23 runs it in the
    experimental 'align' mode, and a training run hung inside a WM decode step (GPU spinning at
    80 W) after a sleep/wake cycle. Outputs do not depend on caching, only speed."""
    from vllm import LLM
    from engine import TOPK, Engine  # debate_pilot/engine.py
    kw = {}
    cfg_path = os.path.join(model_path, "config.json")
    if os.path.exists(cfg_path):
        archs = json.load(open(cfg_path)).get("architectures", [])
        if any("ConditionalGeneration" in a for a in archs):
            kw["limit_mm_per_prompt"] = {"image": 0, "video": 0}
    if max_num_seqs:
        kw["max_num_seqs"] = int(max_num_seqs)
    # On H100 (SM90) vLLM 0.23 auto-selects FlashInfer's GDN prefill kernel for Qwen3.5; three
    # training runs hung inside it (GPU ~80 W at 100%, stack in qwen_gdn_linear_attn._forward_core).
    # The Triton/FLA kernel (what the RTX 5090 pilot used) is forced instead.
    kw["additional_config"] = {"gdn_prefill_backend": gdn_prefill_backend}
    eng = Engine.__new__(Engine)
    eng.path = model_path
    eng.llm = LLM(model=model_path, dtype="bfloat16", gpu_memory_utilization=gpu_memory_utilization,
                  max_model_len=max_model_len, seed=seed, max_logprobs=TOPK,
                  enforce_eager=enforce_eager, enable_prefix_caching=enable_prefix_caching,
                  enable_sleep_mode=enable_sleep_mode, **kw)
    eng.tok = eng.llm.get_tokenizer()
    return eng


def close_vllm_engine(eng) -> None:
    import gc
    import torch
    try:
        eng.llm.llm_engine.engine_core.shutdown()
    except Exception as e:  # pragma: no cover
        print(f"[vllm] engine_core.shutdown failed: {e}")
    del eng.llm
    gc.collect()
    torch.cuda.empty_cache()


def honest_round(engine, items, *, temperature: float, max_tokens: int):
    """items: list of (state, agent, ep_seed). All states must be at the same phase semantics
    (chat built from state.round). Returns one dict per item:
        {reply: P.Reply, probs, letter_mass, truncated, seed}"""
    P = D.P
    chats = [D.chat(st, a) for st, a, _ in items]
    nopt = [st.n_options for st, _, _ in items]
    seeds = [D.honest_seed(st.task_id, s, st.round, a) for st, a, s in items]
    outs = engine.argue_then_read(chats, nopt, seeds=seeds, temperature=temperature,
                                  max_tokens=max_tokens)
    res = []
    for sd, out in zip(seeds, outs):
        ans = D.sample_answer(out["probs"], temperature, sd)
        rep = P.Reply(answer=ans, argument=out["argument"] or P.FALLBACK_ARGUMENT,
                      raw=out["argument_text"] + "Answer: " + ans, valid=bool(out["argument"]))
        res.append({"reply": rep, "probs": out["probs"], "letter_mass": out["letter_mass"],
                    "truncated": out["truncated"], "seed": sd})
    return res


# ------------------------------------------------------------------ HF attacker (training)
def generate_attacker_debate(model, tokenizer, chats, n_options: Sequence[int], *,
                             max_arg_tokens: int = 160, sample: bool = True,
                             temperature: float = 1.0, device=None, gen_batch: int = 16,
                             answer_first: bool = False, forced_letters: Optional[Sequence[str]] = None):
    """answer_first=False: 'Argument:' <argument> '\nAnswer:' <letter>  (default, as in the pilot).
    answer_first=True : 'Answer:' <letter> '\nArgument:' <argument>  -> the option to back is an explicit,
    explorable single-token action chosen before the argument. The public reply is identical.
    Chunks of `gen_batch` sequences (the fp32 KV cache of 40 debate prompts is ~17 GiB), decoded
    with SDPA attention. Generation log-probs are diagnostics only: every log-prob used by PPO
    (old / new / ref) is recomputed with eager attention at B=1, and switching back restores
    bit-identical eager outputs (checked on Lambda).
    forced_letters (answer_first only; targeted persuader, 2026-10-19): one letter per chat that is
    PREFILLED after 'Answer:' instead of sampled, so the turn's only action is the argument (the
    'next_position' segment is empty and gets no gradient)."""
    assert forced_letters is None or (answer_first and len(forced_letters) == len(chats))
    hf = model.get_base_model() if hasattr(model, "get_base_model") else model
    prev = hf.config._attn_implementation
    hf.set_attn_implementation("sdpa")
    try:
        out = []
        for i in range(0, len(chats), gen_batch):
            kw = {}
            if forced_letters is not None:
                kw["forced_letters"] = forced_letters[i:i + gen_batch]
            fn = _generate_attacker_chunk_answer_first if answer_first else _generate_attacker_chunk
            out += fn(model, tokenizer, chats[i:i + gen_batch],
                      n_options[i:i + gen_batch], max_arg_tokens=max_arg_tokens,
                      sample=sample, temperature=temperature, device=device, **kw)
        return out
    finally:
        hf.set_attn_implementation(prev)


def _generate_attacker_chunk(model, tokenizer, chats, n_options, *, max_arg_tokens, sample,
                             temperature, device):
    import torch
    from attacker.generation import GenTurn, _batched_decode, _encode_chat

    device = device or next(model.parameters()).device
    with torch.no_grad():
        chat_ids = [_encode_chat(tokenizer, m, device).tolist() for m in chats]
        pre = tokenizer.encode(D.ARG_PREFIX, add_special_tokens=False)
        mid = tokenizer.encode(D.ANS_MIDDLE, add_special_tokens=False)
        arg_ids, arg_lps = _batched_decode(
            model, tokenizer, [c + pre for c in chat_ids], max_arg_tokens, sample, temperature,
            lambda t: "\n" not in t, device)
        p2 = [chat_ids[i] + pre + arg_ids[i] + mid for i in range(len(chat_ids))]
        ans_ids, ans_lps = _batched_decode(
            model, tokenizer, p2, 1, sample, temperature, lambda t: True, device)

    out = []
    for i in range(len(chat_ids)):
        base = len(chat_ids[i])
        full = chat_ids[i] + pre + arg_ids[i] + mid + ans_ids[i]
        a0 = base + len(pre)
        arg_abs = list(range(a0, a0 + len(arg_ids[i])))
        s0 = a0 + len(arg_ids[i]) + len(mid)
        ans_abs = list(range(s0, s0 + len(ans_ids[i])))
        V = len(arg_abs) + len(ans_abs)
        msg_mask = torch.zeros(V, dtype=torch.bool); msg_mask[:len(arg_abs)] = True
        ans_mask = torch.zeros(V, dtype=torch.bool); ans_mask[len(arg_abs):] = True
        letter = tokenizer.decode(ans_ids[i]).strip() if ans_ids[i] else ""
        argument = " ".join(tokenizer.decode(arg_ids[i]).split())
        ok_letter = len(letter) == 1 and letter in D.LETTERS[: n_options[i]]
        gt = GenTurn(
            input_ids=torch.tensor(full, dtype=torch.long),
            value_idx=torch.tensor(arg_abs + ans_abs, dtype=torch.long),
            seg_masks={"next_position": ans_mask, "message": msg_mask},
            old_logp_gen=torch.tensor(arg_lps[i] + ans_lps[i]),
            next_position=D.LETTERS.index(letter) if ok_letter else -1,
            message=argument,
            valid=bool(ok_letter and argument),
            prompt_len=base,
        )
        out.append(gt)
    return out


def forced_letter_ids(tokenizer, letter: str) -> List[int]:
    """Token ids of ' <letter>' as it follows 'Answer:' (must be one token, like a sampled letter)."""
    ids = tokenizer.encode(" " + letter, add_special_tokens=False)
    assert len(ids) == 1, (letter, ids)
    return ids


def _generate_attacker_chunk_answer_first(model, tokenizer, chats, n_options, *, max_arg_tokens,
                                         sample, temperature, device, forced_letters=None):
    import torch
    from attacker.generation import GenTurn, _batched_decode, _encode_chat

    device = device or next(model.parameters()).device
    with torch.no_grad():
        chat_ids = [_encode_chat(tokenizer, m, device).tolist() for m in chats]
        pre = tokenizer.encode(D.ANS_FIRST_PREFIX, add_special_tokens=False)
        mid = tokenizer.encode(D.ARG_MIDDLE, add_special_tokens=False)
        if forced_letters is None:
            ans_ids, ans_lps = _batched_decode(
                model, tokenizer, [c + pre for c in chat_ids], 1, sample, temperature, lambda t: True, device)
        else:
            # prefilled letter: part of the prompt scaffolding, not a value token
            pre_f = [pre + forced_letter_ids(tokenizer, l) for l in forced_letters]
            ans_ids, ans_lps = [[] for _ in chat_ids], [[] for _ in chat_ids]
        pres = pre_f if forced_letters is not None else [pre] * len(chat_ids)
        p2 = [chat_ids[i] + pres[i] + ans_ids[i] + mid for i in range(len(chat_ids))]
        arg_ids, arg_lps = _batched_decode(
            model, tokenizer, p2, max_arg_tokens, sample, temperature, lambda t: "\n" not in t, device)

    out = []
    for i in range(len(chat_ids)):
        base = len(chat_ids[i])
        full = chat_ids[i] + pres[i] + ans_ids[i] + mid + arg_ids[i]
        s0 = base + len(pres[i])
        ans_abs = list(range(s0, s0 + len(ans_ids[i])))
        a0 = s0 + len(ans_ids[i]) + len(mid)
        arg_abs = list(range(a0, a0 + len(arg_ids[i])))
        V = len(ans_abs) + len(arg_abs)
        ans_mask = torch.zeros(V, dtype=torch.bool); ans_mask[:len(ans_abs)] = True
        msg_mask = torch.zeros(V, dtype=torch.bool); msg_mask[len(ans_abs):] = True
        if forced_letters is not None:
            letter = forced_letters[i]
        else:
            letter = tokenizer.decode(ans_ids[i]).strip() if ans_ids[i] else ""
        argument = " ".join(tokenizer.decode(arg_ids[i]).split())
        ok_letter = len(letter) == 1 and letter in D.LETTERS[: n_options[i]]
        out.append(GenTurn(
            input_ids=torch.tensor(full, dtype=torch.long),
            value_idx=torch.tensor(ans_abs + arg_abs, dtype=torch.long),
            seg_masks={"next_position": ans_mask, "message": msg_mask},
            old_logp_gen=torch.tensor(ans_lps[i] + arg_lps[i]),
            next_position=D.LETTERS.index(letter) if ok_letter else -1,
            message=argument,
            valid=bool(ok_letter and argument),
            prompt_len=base,
        ))
    return out


def attacker_reply(gt) -> "D.P.Reply":
    """Public reply the honest agents see for an attacker turn."""
    P = D.P
    ans = D.LETTERS[gt.next_position] if gt.next_position >= 0 else None
    arg = gt.message or P.FALLBACK_ARGUMENT
    return P.Reply(answer=ans, argument=arg, raw=f"{D.ARG_PREFIX} {gt.message}{D.ANS_MIDDLE} {ans}",
                   valid=gt.valid)


# ------------------------------------------------------------------ vLLM attacker (evaluation)
def vllm_attacker_two_phase(engine, chats, n_options, seeds, *, temperature: float = 1.0,
                            max_arg_tokens: int = 160, answer_first: bool = False,
                            forced_letters: Optional[Sequence[str]] = None):
    """Same two-phase decode as training, in vLLM (for model_base / model_rl evaluation)."""
    assert forced_letters is None or answer_first
    if answer_first:
        return _vllm_attacker_answer_first(engine, chats, n_options, seeds, temperature=temperature,
                                           max_arg_tokens=max_arg_tokens, forced_letters=forced_letters)
    from vllm import SamplingParams
    from protocol import generation_prompt
    P = D.P
    prompts = [generation_prompt(engine.tok, c) + D.ARG_PREFIX for c in chats]
    sp1 = [SamplingParams(temperature=temperature, max_tokens=max_arg_tokens, seed=int(s),
                          stop=["\n"]) for s in seeds]
    o1 = engine.llm.generate(prompts, sp1, use_tqdm=False)
    args = [o.outputs[0].text for o in o1]
    capped = [o.outputs[0].finish_reason == "length" for o in o1]
    p2 = [p + a + D.ANS_MIDDLE for p, a in zip(prompts, args)]
    sp2 = [SamplingParams(temperature=temperature, max_tokens=1, seed=int(s) + 1) for s in seeds]
    o2 = engine.llm.generate(p2, sp2, use_tqdm=False)
    res = []
    for a, cap, first, o, k in zip(args, capped, o1, o2, n_options):
        letter = o.outputs[0].text.strip()
        ok = len(letter) == 1 and letter in D.LETTERS[:k]
        arg = " ".join(a.split())
        res.append({"reply": P.Reply(answer=letter if ok else None, argument=arg or P.FALLBACK_ARGUMENT,
                                     raw=f"{D.ARG_PREFIX}{a}{D.ANS_MIDDLE}{o.outputs[0].text}",
                                     valid=bool(ok and arg)),
                    "arg_tokens": len(first.outputs[0].token_ids), "capped": cap})
    return res


def vllm_attacker_freeform(engine, chats, n_options, seeds, *, temperature: float = 0.7,
                           max_tokens: int = 200):
    """Prompt-only attacker, free-form generation + pilot parser (as debate_pilot S1 orig2)."""
    P = D.P
    outs = engine.generate(chats, temperature=temperature, max_tokens=max_tokens, seeds=seeds)
    return [{"reply": P.parse_reply(o, k), "arg_tokens": None, "capped": None}
            for o, k in zip(outs, n_options)]


def _vllm_attacker_answer_first(engine, chats, n_options, seeds, *, temperature, max_arg_tokens,
                                forced_letters=None):
    from vllm import SamplingParams
    from protocol import generation_prompt
    P = D.P
    prompts = [generation_prompt(engine.tok, c) + D.ANS_FIRST_PREFIX for c in chats]
    if forced_letters is not None:
        letters_raw = [" " + l for l in forced_letters]       # = the training prefill
    else:
        o1 = engine.llm.generate(prompts, [SamplingParams(temperature=temperature, max_tokens=1,
                                                          seed=int(s) + 1) for s in seeds], use_tqdm=False)
        letters_raw = [o.outputs[0].text for o in o1]
    p2 = [p + l + D.ARG_MIDDLE for p, l in zip(prompts, letters_raw)]
    o2 = engine.llm.generate(p2, [SamplingParams(temperature=temperature, max_tokens=max_arg_tokens,
                                                 seed=int(s), stop=["\n"]) for s in seeds], use_tqdm=False)
    res = []
    for l, o, k in zip(letters_raw, o2, n_options):
        letter = l.strip()
        ok = len(letter) == 1 and letter in D.LETTERS[:k]
        arg = " ".join(o.outputs[0].text.split())
        res.append({"reply": P.Reply(answer=letter if ok else None, argument=arg or P.FALLBACK_ARGUMENT,
                                     raw=f"{D.ANS_FIRST_PREFIX}{l}{D.ARG_MIDDLE}{o.outputs[0].text}",
                                     valid=bool(ok and arg)),
                    "arg_tokens": len(o.outputs[0].token_ids),
                    "capped": o.outputs[0].finish_reason == "length"})
    return res
