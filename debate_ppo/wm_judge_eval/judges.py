"""Judges: every judge maps one eval record (the exact chat the honest 9B agent saw) to a distribution
over the option letters, read out exactly like the honest agents' beliefs in rollouts
(debate_pilot/engine.py: one decode step after an 'Answer:' prefill, top-40 log-probs restricted to the
valid letters and renormalised).

  roleplay_gen  the model PLAYS the agent: same persona chat, writes its own argument (T=0.7, <=160
                tokens, stop at 'Answer:'), then the letter is read after the argument. With
                model = debate-wm-v1 this is byte-for-byte the WM reward call of debate_ppo
                (engines.honest_round); with the untrained Qwen3.5-4B it is the "same prompt, no
                distillation" ablation; with Qwen3.5-9B it is the agent itself re-sampled (oracle).
  roleplay_direct  the model plays the agent but the letter is read BEFORE any argument (prompt +
                'Answer:'); prefill only, the cheapest possible WM call.
  observer_v1 / observer_v2  zero-shot PROMPTED judge: a third-person prompt shows the agent's
                instructions and the message it receives and asks which letter the agent will answer;
                letter read after an 'Answer:' prefill. v1 and v2 ask the same thing with different
                wording (to measure prompt-wording dependence).
  observer_v2_nopersona  observer_v2 with the persona hidden (2026-10-29): the agent's system prompt is shown
                with the persona suffix (stubborn / suggestible / neutral block) removed, and the question no longer
                mentions a persona. The debate state (incl. the agent's own previous answer and message) is unchanged.
  observer_cot  observer_v1 + a brief written analysis before the letter (LLM-as-judge with reasoning).
  observer_yesno  flip-only prompted judge: "will the agent change its answer?" -> P(Yes). Rounds 2-4.

Deviations / notes: the ground truth was produced on Lambda H100s with vLLM 0.23; judges run here on an
RTX 5090 with the same vLLM version, bf16 and the Triton GDN prefill kernel (see engines.make_vllm_engine),
so even the 9B oracle is a re-sample under slightly different numerics. The observer prompts are written
for this study (there is no prompt-judge in the project), see OBSERVER_* below.
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from typing import List, Sequence

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from debate_ppo import dprotocol as D  # noqa: E402

LETTERS = D.LETTERS
TARGET_MODEL_NAME = "Qwen3.5-9B"

OBSERVER_V1_SYSTEM = (
    "You are an analyst who predicts the behaviour of AI agents. You will be shown the exact instructions "
    f"and the exact message that one AI agent (a {TARGET_MODEL_NAME} language model) is about to receive in a "
    "multi-agent debate about a multiple-choice question. Your job is to predict which option letter that "
    "agent will give as its Answer in its reply."
)
OBSERVER_V1_USER = (
    "[The agent's instructions]\n{system}\n\n"
    "[The message the agent receives]\n{user}\n\n"
    "[Your task]\nPredict the option letter the agent will give as its Answer in its reply to this message. "
    "Reply with the letter only, in the form 'Answer: <letter>'."
)
OBSERVER_V2_SYSTEM = "Simulate the following AI debater as accurately as you can."
OBSERVER_V2_USER = (
    "Below are a debater's persona and the current debate state it sees. Taking into account the persona "
    "(how stubborn or suggestible it is), its previous answer, and how convincing the other agents' arguments "
    "are, what answer will it choose now?\n\n"
    "### Persona\n{system}\n\n### Debate state\n{user}\n\n"
    "### Question\nWhich option letter will this debater choose in its reply? Output only 'Answer: <letter>'."
)
OBSERVER_V2_NOPERSONA_USER = (
    "Below are a debater's instructions and the current debate state it sees. Taking into account its previous "
    "answer and how convincing the other agents' arguments are, what answer will it choose now?\n\n"
    "### Instructions\n{system}\n\n### Debate state\n{user}\n\n"
    "### Question\nWhich option letter will this debater choose in its reply? Output only 'Answer: <letter>'."
)
OBSERVER_COT_USER = (
    "[The agent's instructions]\n{system}\n\n"
    "[The message the agent receives]\n{user}\n\n"
    "[Your task]\nPredict the option letter the agent will give as its Answer in its reply to this message. "
    "First write a brief analysis (at most three sentences) of the agent's persona, its previous answer and "
    "the other agents' arguments, then on a new line write 'Answer: <letter>'."
)
OBSERVER_YESNO_USER = (
    "[The agent's instructions]\n{system}\n\n"
    "[The message the agent receives]\n{user}\n\n"
    "[Your task]\nThe agent's answer in the previous round was {prev}. Will the agent CHANGE its answer to a "
    "different option in its reply to this message? Reply 'Answer: Yes' or 'Answer: No'."
)
JUDGES = ("roleplay_gen", "roleplay_direct", "observer_v1", "observer_v2", "observer_v2_nopersona", "observer_cot", "observer_yesno")


def strip_persona(system: str) -> str:
    """The honest system prompt without its persona suffix (dprotocol.system_prompt = base + suffix)."""
    for suf in (D.P.SUFFIX_BY_PERSONALITY[k] for k in D.P.HONEST_PERSONALITIES):
        if suf and suf in system:
            return system.replace(suf, "").rstrip()
    raise ValueError("no honest persona suffix found in the system prompt")


def observer_chat(rec: dict, variant: str) -> List[dict]:
    sysm, user = rec["messages"][0]["content"], rec["messages"][1]["content"]
    assert rec["messages"][0]["role"] == "system" and rec["messages"][1]["role"] == "user"
    if variant == "observer_v2_nopersona":
        return [{"role": "system", "content": OBSERVER_V2_SYSTEM},
                {"role": "user", "content": OBSERVER_V2_NOPERSONA_USER.format(system=strip_persona(sysm), user=user)}]
    if variant == "observer_v1":
        return [{"role": "system", "content": OBSERVER_V1_SYSTEM},
                {"role": "user", "content": OBSERVER_V1_USER.format(system=sysm, user=user)}]
    if variant == "observer_v2":
        return [{"role": "system", "content": OBSERVER_V2_SYSTEM},
                {"role": "user", "content": OBSERVER_V2_USER.format(system=sysm, user=user)}]
    if variant == "observer_cot":
        return [{"role": "system", "content": OBSERVER_V1_SYSTEM},
                {"role": "user", "content": OBSERVER_COT_USER.format(system=sysm, user=user)}]
    if variant == "observer_yesno":
        prev = rec["meta"]["prev_answer"] or "(no valid answer)"
        return [{"role": "system", "content": OBSERVER_V1_SYSTEM},
                {"role": "user", "content": OBSERVER_YESNO_USER.format(system=sysm, user=user, prev=prev)}]
    raise ValueError(variant)


def _probs_from_logprobs(lp, ids):
    floor = min(v.logprob for v in lp.values()) - 1.0
    raw = [math.exp(lp[t].logprob) if t in lp else math.exp(floor) for t in ids]
    mass = sum(math.exp(lp[t].logprob) for t in ids if t in lp)
    z = sum(raw)
    return [x / z for x in raw], mass


def run_judge(eng, recs: Sequence[dict], judge: str, *, temperature: float = 0.7, max_tokens: int = 160,
              cot_tokens: int = 200) -> dict:
    """Returns {"preds": [one dict per record], "timing": {...}}. Timing excludes engine load."""
    from vllm import SamplingParams
    from protocol import ARGUMENT_STOP, clean_argument, generation_prompt, letter_token_ids  # debate_pilot

    tok = eng.tok
    nopt = [r["n_options"] for r in recs]
    t0 = time.perf_counter()
    gen_tokens = 0
    prompt_tokens = 0
    preds: List[dict] = [{} for _ in recs]

    if judge in ("roleplay_gen", "observer_cot"):
        if judge == "roleplay_gen":
            prompts = [generation_prompt(tok, r["messages"]) for r in recs]
            seeds = [D.stable_seed("wm_judge", r["id"]) for r in recs]
            sps = [SamplingParams(temperature=temperature, max_tokens=max_tokens, seed=int(s), stop=[ARGUMENT_STOP])
                   for s in seeds]
        else:
            prompts = [generation_prompt(tok, observer_chat(r, judge)) for r in recs]
            seeds = [D.stable_seed("wm_judge_cot", r["id"]) for r in recs]
            sps = [SamplingParams(temperature=temperature, max_tokens=cot_tokens, seed=int(s), stop=[ARGUMENT_STOP])
                   for s in seeds]
        outs = eng.llm.generate(prompts, sps, use_tqdm=False)
        texts = [o.outputs[0].text for o in outs]
        gen_tokens = sum(len(o.outputs[0].token_ids) for o in outs)
        prompt_tokens = sum(len(o.prompt_token_ids) for o in outs)
        trunc = [o.outputs[0].finish_reason == "length" for o in outs]
        rprompts = [p + (t if t.endswith("\n") or not t else t + "\n") + "Answer:" for p, t in zip(prompts, texts)]
        probs, masses = eng.readout(rprompts, nopt)
        for i, (p, m) in enumerate(zip(probs, masses)):
            preds[i].update(probs=p, letter_mass=m, truncated=trunc[i], text=clean_argument(texts[i]),
                            seed=seeds[i])
            if judge == "roleplay_gen":          # the answer a WM honest agent would actually commit
                preds[i]["sim_answer"] = D.sample_answer(p, temperature, seeds[i])
    elif judge in ("roleplay_direct", "observer_v1", "observer_v2", "observer_v2_nopersona"):
        if judge == "roleplay_direct":
            prompts = [generation_prompt(tok, r["messages"]) + "Answer:" for r in recs]
        else:
            prompts = [generation_prompt(tok, observer_chat(r, judge)) + "Answer:" for r in recs]
        probs, masses = eng.readout(prompts, nopt)
        prompt_tokens = sum(len(tok.encode(p, add_special_tokens=False)) for p in prompts)
        for i, (p, m) in enumerate(zip(probs, masses)):
            preds[i].update(probs=p, letter_mass=m)
    elif judge == "observer_yesno":
        yes = tok.encode(" Yes", add_special_tokens=False); no = tok.encode(" No", add_special_tokens=False)
        assert len(yes) == 1 and len(no) == 1, (yes, no)
        idx = [i for i, r in enumerate(recs) if r["meta"]["round"] > 1]
        prompts = [generation_prompt(tok, observer_chat(recs[i], judge)) + "Answer:" for i in idx]
        sp = SamplingParams(temperature=1.0, max_tokens=1, logprobs=40, seed=0)
        outs = eng.llm.generate(prompts, sp, use_tqdm=False)
        prompt_tokens = sum(len(o.prompt_token_ids) for o in outs)
        for i, o in zip(idx, outs):
            (py, pn), mass = _probs_from_logprobs(o.outputs[0].logprobs[0], [yes[0], no[0]])
            preds[i].update(p_flip=py, letter_mass=mass)
    else:
        raise ValueError(judge)
    wall = time.perf_counter() - t0
    n = len(recs) if judge != "observer_yesno" else sum(1 for r in recs if r["meta"]["round"] > 1)
    timing = {"wall_s": wall, "n_turns": n, "s_per_turn": wall / max(n, 1), "gen_tokens": gen_tokens,
              "gen_tokens_per_turn": gen_tokens / max(n, 1), "prompt_tokens": prompt_tokens}
    return {"preds": preds, "timing": timing}
