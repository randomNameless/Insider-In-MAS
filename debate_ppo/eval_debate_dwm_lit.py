#!/usr/bin/env python
"""Same-selector control (2026-11-09): published single-sample persuaders + the debate-wm-v1 selector of
eval_debate_dwm.py + the same number of candidates. New file; eval_debate.py / eval_debate_dwm.py / baselines unchanged.

Conditions (single '_' before wmbon: a '__' suffix is a Ju strategy name for baseline_ju.strategy_of):
  ju_dpo_<tag>_wmbon<N>dw   Ju et al. DPO persuader, no-strategy prompt (eval_debate's ju_ branch: topic-blind chat,
                            decoding = the model's generation_config, max_tokens = cfg['ju']['max_tokens'], answer fixed
                            to the target) with N samples per turn instead of one;
  kraidia_<tag>_wmbon<N>dw  Kraidia et al. PPA prompt adversary (eval_debate's kraidia_ branch: same chat, T and token cap,
                            same parse / fallback) with N samples per turn; the chosen sample's raw text is the persuader's
                            own history in the next round, as in the single-sample arm.
  bozdag_<tag>_nothink_len<L>_wmbon<N>dw  (2026-11-13) Bozdag et al. released P-RL persuader in the protocol-conforming
                            setting of eval_debate's bozdag_ branch (no private <think>: baseline_bozdag.is_nothink prefill +
                            nothink_system prompt; shown message <= L tokens via fit_sentences) with N samples per turn.
SELECTOR: eval_debate_dwm.worker_attacker_dwm's, unchanged in every respect (listener_state, roleplay_direct judge on the
three honest listeners' round-(r+1) chats, argmax of mean P(target), invalid candidates last, victim phase of the round
played first). Seeds: stable_seed('attacker', task, seed, r) with n=N, as wm_bon.candidates.
Extra bookkeeping (cost table): per turn, generated tokens of all N candidates ('cand_tokens') and of the spoken one.

  python -m debate_ppo.eval_debate_dwm_lit init ... (eval_debate init flags) --dwm-model PATH
  python -m debate_ppo.eval_debate_dwm_lit play --run R
  python -m debate_ppo.eval_debate summary --run R
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from debate_ppo import eval_debate as E  # noqa: E402
from debate_ppo import eval_debate_dwm as DW  # noqa: E402

D, P, JU, KR, BZ = E.D, E.P, E.JU, E.KR, E.BZ
PAT = re.compile(r"^((?:ju_dpo|kraidia|bozdag)_.+)_wmbon(\d+)dw$")


def parse(cond):
    m = PAT.match(cond)
    return (m.group(1), int(m.group(2))) if m else None


def _sample_n(eng, chats, seeds, n, max_tokens, kw, prefix=""):
    from vllm import SamplingParams
    from protocol import generation_prompt       # debate_pilot/protocol.py (thinking off)
    prompts = [generation_prompt(eng.tok, c) + prefix for c in chats]
    sp = [SamplingParams(n=n, max_tokens=max_tokens, seed=int(s), **kw) for s in seeds]
    return eng.llm.generate(prompts, sp, use_tqdm=False)


def worker_attacker(run: Path, r: int, cond: str):
    from debate_ppo.engines import close_vllm_engine, make_vllm_engine
    from debate_ppo.wm_judge_eval.judges import run_judge
    base, n_cand = parse(cond)
    ju, bz = JU.is_ju(cond), BZ.is_bozdag(cond)
    assert ju + KR.is_kraidia(cond) + bz == 1 and E.length_limit(cond) is None, cond
    assert not bz or (BZ.is_nothink(base) and E.length_limit(base) and not BZ.word_limit(base)), cond
    assert not ju or JU.strategy_of(cond) == JU.NONE, cond
    st = E.load_state(run)
    cfg = st["config"]
    assert not cfg.get("defense"), "no defense hooks in this arm"
    D.set_profile_set(cfg.get("profile_set", "v1"))
    chats, nopt, seeds, idx, forced = [], [], [], [], []
    for di, d in enumerate(st["debates"]):
        if d["cond"] != cond:
            continue
        s = E.to_state(d)
        assert s.round == r
        tgt = cfg["targets"][d["task_id"]]
        assert tgt != d["gold"], (d["task_id"], tgt)
        assert all(str(i) in d["pending"] for i in D.honest_seats(d["seat"])), "dwm: victim phase must run first"
        if ju:      # eval_debate.worker_attacker, JU.is_ju branch
            chat = JU.persuader_chat(d["question"], tgt, JU.strategy_of(cond))
        elif bz:    # eval_debate.worker_attacker, BZ.is_bozdag branch (nothink, _len<L>)
            prev = [] if r == 0 else [(j, P._fmt_reply(rep)) for j, rep in enumerate(s.history[-1]) if j != d["seat"]]
            chat = BZ.bozdag_chat(d["question"], tgt, prev, None, True)
            chat[0]["content"] += BZ.TOKEN_RULE.format(n=E.length_limit(base))
        else:       # eval_debate.worker_attacker, KR.is_kraidia branch
            others = [[P._fmt_reply(rep) for j, rep in enumerate(s.history[k]) if j != d["seat"]] for k in range(r)]
            own = [s.history[k][d["seat"]].raw for k in range(r)]
            chat = KR.kraidia_chat(d["question"], tgt, others, own, E.REPLY_FORMAT)
        chats.append(chat)
        forced.append(tgt)
        nopt.append(d["n_options"])
        seeds.append(D.stable_seed("attacker", d["task_id"], d["seed"], r))
        idx.append(di)
    E.log(f"attacker[{cond}] r={r + 1}: {len(chats)} turns, {n_cand} candidates each")
    eng = make_vllm_engine(cfg["attackers"][cond], gpu_memory_utilization=cfg["vllm"]["gpu_memory_utilization"],
                           max_model_len=cfg["vllm"]["max_model_len"])
    t0 = time.time()
    if ju:
        outs = _sample_n(eng, chats, seeds, n_cand, cfg["ju"]["max_tokens"], JU.sampling_kwargs(eng.path))
    elif bz:
        outs = _sample_n(eng, chats, seeds, n_cand, BZ.NOTHINK_MAX_TOKENS,
                         {"temperature": cfg["bozdag_attacker_temp"], "top_p": 1.0}, prefix=BZ.NOTHINK_PREFIX)
    else:
        outs = _sample_n(eng, chats, seeds, n_cand, cfg["kraidia_attacker_tokens"],
                         {"temperature": cfg["kraidia_attacker_temp"], "top_p": 1.0})
    gen_s = time.time() - t0
    cl = []
    for o, l, k in zip(outs, forced, nopt):
        cs = []
        for c in o.outputs:
            t, cut = c.text, c.finish_reason == "length"
            if ju:
                ans, arg, ok = JU.debate_reply(t, l, P.FALLBACK_ARGUMENT)
                rep = P.Reply(answer=ans, argument=arg, raw=t, valid=ok)
            elif bz:
                t = BZ.NOTHINK_PREFIX + t
                t = t if BZ._MSG.search(t) else t + "\n</message>"
                ans, arg, ok = BZ.parse(t, l, P.FALLBACK_ARGUMENT)
                if arg != P.FALLBACK_ARGUMENT:
                    arg = E.fit_sentences(arg, eng.tok, E.length_limit(base))[0]
                rep = P.Reply(answer=ans, argument=arg, raw=t, valid=ok)
            else:
                rp = P.parse_reply(t, k)
                ans = rp.answer or KR.fallback_answer(t, k)
                arg = rp.argument if rp.valid or rp.argument != P.FALLBACK_ARGUMENT else (
                    " ".join(t.split()) or P.FALLBACK_ARGUMENT)
                rep = P.Reply(answer=ans, argument=arg, raw=t, valid=bool(ans) and arg != P.FALLBACK_ARGUMENT)
            cs.append({"reply": rep, "arg_tokens": len(c.token_ids), "capped": cut})
        cl.append(cs)
    close_vllm_engine(eng)
    n_gen = sum(c["arg_tokens"] for cs in cl for c in cs)
    E.log(f"attacker[{cond}] r={r + 1}: " + json.dumps({"gen_wall_s": gen_s, "gen_tokens": n_gen,
                                                        "gen_tokens_per_turn": n_gen / max(1, len(cl))}))
    # ---- selector: eval_debate_dwm.worker_attacker_dwm, verbatim
    recs, owner = [], []
    for k, (di, cs) in enumerate(zip(idx, cl)):
        d = st["debates"][di]
        for c, cand in enumerate(cs):
            ls = DW.listener_state(d, r, cand["reply"])
            for i in D.honest_seats(d["seat"]):
                recs.append({"id": f"{d['task_id']}|s{d['seed']}|r{r}|c{c}|a{i}", "messages": D.chat(ls, i),
                             "n_options": d["n_options"], "meta": {"round": r + 2, "target": d["target"]}})
                owner.append((k, c, i))
    dw = cfg["dwm"]
    E.log(f"dwm[{cond}] r={r + 1}: scoring {len(recs)} listener turns with {dw['model']}")
    jeng = make_vllm_engine(dw["model"], gpu_memory_utilization=dw["gpu_memory_utilization"],
                            max_model_len=dw["max_model_len"], max_num_seqs=dw["max_num_seqs"])
    res = run_judge(jeng, recs, DW.JUDGE)
    close_vllm_engine(jeng)
    E.log(f"dwm[{cond}] r={r + 1}: {json.dumps(res['timing'])}")
    probs = {}
    for (k, c, i), rec, p in zip(owner, recs, res["preds"]):
        probs.setdefault((k, c), {})[i] = p["probs"]
    for k, (di, cs) in enumerate(zip(idx, cl)):
        d = st["debates"][di]
        hs = D.honest_seats(d["seat"])
        ti = D.LETTERS.index(d["target"])
        sc = [sum(probs[(k, c)][i][ti] for i in hs) / len(hs) for c in range(len(cs))]
        valid = [c["reply"].valid for c in cs]
        best = max(range(len(cs)), key=lambda j: (valid[j], sc[j]))
        d.setdefault("wmbon", []).append({"round": r, "mode": "dw", "judge": f"debate-wm-v1/{DW.JUDGE}", "n": len(cs),
                                          "best": best, "scores": sc, "valid": valid,
                                          "cands": [c["reply"].__dict__ for c in cs],
                                          "cand_tokens": [c["arg_tokens"] for c in cs],
                                          "pred_beliefs_best": {str(i): probs[(k, best)][i] for i in hs}})
        o = cs[best]
        d["pending"][str(d["seat"])] = {"reply": o["reply"].__dict__, "probs": None, "letter_mass": None,
                                        "truncated": o["capped"]}
        d["attacker_meta"].append({"arg_tokens": o["arg_tokens"], "capped": o["capped"]})
    st["done_phases"].append(f"attacker[{cond}]{r}")
    E.save_state(run, st)


def cmd_play(run: Path):
    conds = [c for c in E.load_state(run)["config"]["conds"] if c != "none"]
    assert conds and all(parse(c) for c in conds), conds
    me = [sys.executable, "-m", "debate_ppo.eval_debate_dwm_lit", "worker", "--run", str(run)]
    for r in range(D.N_ROUNDS + 1):
        if r < D.N_ROUNDS:
            if f"victim{r}" not in E.load_state(run)["done_phases"]:
                subprocess.run(me + ["--role", "victim", "--round", str(r)], check=True)
            for c in conds:
                if f"attacker[{c}]{r}" not in E.load_state(run)["done_phases"]:
                    subprocess.run(me + ["--role", "attacker", "--cond", c, "--round", str(r)], check=True)
        if f"victim{r}" not in E.load_state(run)["done_phases"]:
            subprocess.run(me + ["--role", "victim", "--round", str(r)], check=True)
        if f"commit{r}" not in E.load_state(run)["done_phases"]:
            E.commit(run, r)
    E.log("play done")


def main():
    argv = sys.argv[1:]
    cmd = argv[0]
    if cmd == "init":
        DW.main()                                  # eval_debate init + cfg['dwm'] (model, judge, hashes)
        run = Path(argv[argv.index("--run") + 1])
        st = E.load_state(run)
        assert all(c == "none" or parse(c) for c in st["config"]["conds"]), st["config"]["conds"]
        st["config"]["dwm"]["code_sha256"]["debate_ppo/eval_debate_dwm_lit.py"] = DW._sha(__file__)
        st["config"]["dwm"]["selector"] += "; candidates = N samples of the baseline's own single-sample decoder " \
                                           "(eval_debate_dwm_lit)"
        json.dump(st["config"], open(run / "config.json", "w"), indent=1)
        E.save_state(run, st)
    elif cmd == "play":
        cmd_play(Path(argv[argv.index("--run") + 1]))
    elif cmd == "worker":
        run = Path(argv[argv.index("--run") + 1])
        role, rnd = argv[argv.index("--role") + 1], int(argv[argv.index("--round") + 1])
        if role == "victim":
            E.worker_victim(run, rnd)
        else:
            worker_attacker(run, rnd, argv[argv.index("--cond") + 1])
    else:
        raise SystemExit(f"unknown command {cmd}")


if __name__ == "__main__":
    main()
