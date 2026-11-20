"""WM-guided Best-of-N for the targeted PPO persuader at inference time (2026-10-25).

Condition name: <base_cond>__wmbon<N><mode>, e.g. targeted_model_rl_sd120__wmbon8wb.
  base_cond  : any targeted_model_* condition (same chat, target hint, forced letter, two-phase answer-first decoder)
  N          : candidates sampled per attacker turn (vLLM n=N, T = two_phase_temp, cap = two_phase_arg_tokens)
  mode  wb   : the belief WM tracks the honest agents' beliefs itself ("WM beliefs"): round-1 prior = fit-split mean mass on
               the stated answer per persona (rest uniform), then its own prediction for the chosen message each round.
               Observable inputs only (messages + stated answers) -> deployable.
        tb   : the honest agents' TRUE belief readouts (the eval harness has them) feed the WM -> diagnostic upper bound.
Selection: the candidate with the highest WM-predicted mean P(target) over the three honest listeners next round
(invalid candidates ranked last). The honest agents' SAME-round replies are inputs of the WM, so in a wmbon run the
victim phase of a round is played BEFORE the attacker phase (eval_debate.cmd_play); that is protocol-neutral, because the
honest replies of round r never see the attacker's round-r message.
The belief WM is dcm_analysis/mmlupro/ppo_bon.PPOBeliefWM (DCM + judge + residual D fit on the recorded 9B debates,
results/abcd_mmlupro_ppo/d_model.pt = "WM_post"); its judge (Qwen2.5-1.5B) and D live next to the vLLM engine (keep
--gpu-util <= 0.6). Every candidate, its WM score and the tracked beliefs are stored in the debate record (d["wmbon"]).
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

from . import dprotocol as D

P = D.P
PAT = re.compile(r"^(.*)__wmbon(\d+)(tb|wb)$")
LATENT = Path(os.environ.get("LATENT_DIR", "/path/to/latent_agent_transfer/dcm_analysis"))
K = 10
# fit-split mean belief mass on the agent's own stated answer in round 1 (dcm_analysis data/mmlupro_ppo, 11,528 debates)
PRIOR_MASS = {"stubborn": 0.925, "suggestible": 0.814, "neutral": 0.745}


def parse(cond: str):
    m = PAT.match(cond)
    return (m.group(1), int(m.group(2)), m.group(3)) if m else None


def is_wmbon(cond: str) -> bool:
    return parse(cond) is not None


def candidates(engine, chats, n_options, seeds, forced_letters, *, n: int, temperature: float, max_arg_tokens: int):
    """N answer-first candidates per chat with the target letter prefilled (as _vllm_attacker_answer_first, n=N)."""
    from vllm import SamplingParams
    from protocol import generation_prompt
    prompts = [generation_prompt(engine.tok, c) + D.ANS_FIRST_PREFIX + " " + l + D.ARG_MIDDLE
               for c, l in zip(chats, forced_letters)]
    sp = [SamplingParams(n=n, temperature=temperature, max_tokens=max_arg_tokens, seed=int(s), stop=["\n"])
          for s in seeds]
    outs = engine.llm.generate(prompts, sp, use_tqdm=False)
    res = []
    for l, o, k in zip(forced_letters, outs, n_options):
        cs = []
        for c in o.outputs:
            arg = " ".join(c.text.split())
            cs.append({"reply": P.Reply(answer=l, argument=arg or P.FALLBACK_ARGUMENT,
                                        raw=f"{D.ANS_FIRST_PREFIX} {l}{D.ARG_MIDDLE}{c.text}", valid=bool(arg)),
                       "arg_tokens": len(c.token_ids), "capped": c.finish_reason == "length"})
        res.append(cs)
    return res


def _pad(p):
    p = [float(x) for x in p]
    return p + [0.0] * (K - len(p))


def _prior(persona: str, answer, n_options: int):
    m = PRIOR_MASS.get(persona, 0.8)
    if answer not in D.LETTERS[:n_options]:
        return _pad([1.0 / n_options] * n_options)
    p = [(1.0 - m) / (n_options - 1)] * n_options
    p[D.LETTERS.index(answer)] = m
    return _pad(p)


class Selector:
    def __init__(self, mode: str):
        assert mode in ("tb", "wb")
        self.mode = mode
        os.environ.setdefault("DCM_DATA", "data/mmlupro_ppo")
        os.environ.setdefault("DCM_PREFIX", "mmluproppo_")
        os.environ.setdefault("DCM_FIT_DATASETS", "mmlupro")
        self._cwd = os.getcwd()
        os.chdir(LATENT)                                   # dcm_analysis modules resolve data paths from their own tree
        sys.path.insert(0, str(LATENT))
        from mmlupro.ppo_bon import PPOBeliefWM
        self.wm = PPOBeliefWM(data_dir=LATENT / "data/mmlupro_ppo")
        os.chdir(self._cwd)

    def state(self, d: dict, r: int):
        seat = d["seat"]
        hs = D.honest_seats(seat)
        honest = {}
        for k, i in enumerate(hs):
            pend = d["pending"][str(i)]
            rep = pend["reply"]
            probs = _pad(pend["probs"]) if self.mode == "tb" else d["wmbon_beliefs"][r][k]
            h = {"persona": d["personalities"][i], "message": rep["raw"], "answer": rep["answer"], "probs": probs}
            if r >= 1:
                prev = d["history"][r - 1][i]
                h.update({"prev_message": prev["raw"], "prev_answer": prev["answer"],
                          "prev_probs": _pad(d["beliefs"][r - 1][k]) if self.mode == "tb" else d["wmbon_beliefs"][r - 1][k]})
            honest[i] = h
        return {"question_id": d["task_id"], "round": r + 1, "fourth_seat": seat, "gold": d["gold"], "target": d["target"],
                "honest": honest, "fourth_prev_message": d["history"][r - 1][seat]["raw"] if r >= 1 else None}

    def choose(self, d: dict, r: int, cands: list, n_options: int) -> dict:
        hs = D.honest_seats(d["seat"])
        if self.mode == "wb" and r == 0:
            d["wmbon_beliefs"] = [[_prior(d["personalities"][i], d["pending"][str(i)]["reply"]["answer"], n_options) for i in hs]]
        out = self.wm.score(self.state(d, r), [c["reply"].raw for c in cands])
        sc = [o["mean_p_target"] for o in out]
        valid = [c["reply"].valid for c in cands]
        best = max(range(len(cands)), key=lambda j: (valid[j], sc[j] if sc[j] is not None else -1.0))
        if self.mode == "wb":
            d["wmbon_beliefs"].append([_pad(out[best]["beliefs"][i]) for i in hs])
        d.setdefault("wmbon", []).append({"round": r, "mode": self.mode, "n": len(cands), "best": best, "scores": sc,
                                          "valid": valid, "cands": [c["reply"].__dict__ for c in cands],
                                          "pred_beliefs_best": {str(i): out[best]["beliefs"][i] for i in hs}})
        return cands[best]
