#!/usr/bin/env python
"""Oracle-vs-belief-WM ranking of persuader candidates (2026-10-29; plan ~/runs/ppo_diag_20261028/WM_BON_C120_VS_SD120_PLAN.md).
Why does WM-BoN8 help sd120 (DPO+PPO) but hardly c120 (plain PPO)? H1 the WM never saw this attacker, H2 the 8 candidates
are about equally effective on the real 9B, H3 the WM cannot rank. Every persuader is sampled on the SAME states:

  states  100 test questions of exp10_nn `none` (Qwen3.5-9B, profile v1nn), seed 3, attacker rounds 0,1,2 = 300 states.
          history = rounds 0..r of the recorded debate; the attacker takes the 4th (neutral) seat (as make_data_v6).
  gen     per persuader 8 candidates per state with the v6 decoder (chat + TARGET_HINT, target letter prefilled, T 1.0,
          <= 200 tokens, stop at "\\n"; = wm_bon.candidates with n=8). Seeds shared across persuaders.
  oracle  real Qwen3.5-9B honest agents (v1nn, T 0.7, 160 tokens) play round r+1 with the candidate in the seat, K=4
          rollouts with seeds shared by all candidates (and persuaders) of a state; score = mean over the 3 listeners of
          P(target) (honest_round readout, as make_data_v6.cmd_score and the eval harness).
  wm      exact WM-BoN scorer (wm_bon.Selector -> PPOBeliefWM, DCM + residual D, Qwen2.5-1.5B stance judge). Two belief
          inputs: `tb` = the recorded 9B beliefs of rounds r (and r-1), `prior` = the wb-mode persona prior (what
          WM-BoN wb uses at round 1).
  analyze tables (CPU): see cmd_analyze.

python debate_wm/rank_oracle.py states|gen|oracle|wm|analyze --out R [--persuader NAME=PATH]"""
from __future__ import annotations

import argparse
import collections
import json
import math
import random
import re
import statistics as st
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from debate_ppo import dprotocol as D  # noqa: E402
from debate_ppo.strat_dpo import make_data as MD  # noqa: E402
from debate_ppo.strat_dpo import make_data_v6 as M6  # noqa: E402

P = D.P
log, _jsonl, _write_jsonl = MD.log, MD._jsonl, MD._write_jsonl
R_ = "/path/to/runs"
M_ = "/path/to/models"
PERSUADERS = {"dpo": f"{M_}/strat-dpo-q3-4b", "ppo_c120": f"{M_}/exp9-v6-tgt-c120",
              "dpo_ppo_sd120": f"{M_}/exp9-v6sd-tgt-c120", "base_q3_4b": f"{M_}/qwen3-4b"}
N_CAND, K = 8, 4
HEDGE = re.compile(r"(?i)\bwait\b|\bhowever\b.*\bhowever\b|\bbut the correct\b|\bactually\b")


def cmd_states(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    col = json.load(open(f"{R_}/exp10_nn_none/state.json"))
    assert col["config"]["profile_set"] == "v1nn"
    targets = json.load(open(f"{R_}/targets_none9b_exp10_nn.json"))["targets"]
    deb = [d for d in col["debates"] if d["cond"] == "none" and d["seed"] == a.seed and len(d["history"]) == 3]
    tasks = sorted({d["task_id"] for d in deb})
    pick = set(random.Random(0).sample(tasks, a.n))
    rows = M6.build_states([d for d in deb if d["task_id"] in pick], targets, [a.seed], [0, 1, 2], split="test")
    by = {(d["task_id"]): d for d in deb}
    for r in rows:                       # recorded 9B beliefs of the honest seats for rounds 0..r (WM `tb` input)
        r["beliefs"] = by[r["task_id"]]["beliefs"][: r["round"] + 1]
    _write_jsonl(out / "states.jsonl", rows)
    log(f"{len(rows)} states from {len(pick)} questions (seed {a.seed}) -> {out / 'states.jsonl'}")


def cmd_gen(a):
    from vllm import SamplingParams
    from debate_ppo.engines import make_vllm_engine
    D.set_profile_set("v1nn")
    out = Path(a.out)
    states = _jsonl(out / "states.jsonl")
    for spec in a.persuader:
        name, path = spec.split("=", 1)
        f = out / f"cands_{name}.jsonl"
        if f.exists():
            log(f"gen {name}: exists, skip")
            continue
        eng = make_vllm_engine(path, gpu_memory_utilization=a.gpu_util, max_model_len=8192)
        prompts = [M6.v6_prompt(eng.tok, M6.attacker_chat_v6(s), s["target"]) for s in states]
        sp = [SamplingParams(n=N_CAND, temperature=1.0, max_tokens=200, stop=["\n"],
                             seed=D.stable_seed("rank_oracle", s["state_id"])) for s in states]
        t0 = time.time()
        outs = eng.llm.generate(prompts, sp, use_tqdm=False)
        rows = []
        for s, o in zip(states, outs):
            for j, c in enumerate(o.outputs):
                arg = " ".join(c.text.split())
                rows.append({"state_id": s["state_id"], "persuader": name, "cand": j, "raw": c.text, "argument": arg,
                             "words": len(arg.split()), "arg_tokens": len(c.token_ids),
                             "capped": c.finish_reason == "length", "valid": bool(arg)})
        _write_jsonl(f, rows)
        log(f"gen {name}: {len(rows)} candidates in {time.time() - t0:.0f}s, words {st.mean(r['words'] for r in rows):.1f}, "
            f"capped {st.mean(r['capped'] for r in rows):.3f}")
        del eng
        import gc, torch
        gc.collect(); torch.cuda.empty_cache()


def _reply(st_, c):
    return P.Reply(answer=st_["target"], argument=c["argument"] or P.FALLBACK_ARGUMENT,
                   raw=f"{D.ANS_FIRST_PREFIX} {st_['target']}{D.ARG_MIDDLE}{c['raw']}", valid=c["valid"])


def _all_cands(out):
    return [c for f in sorted(Path(out).glob("cands_*.jsonl")) for c in _jsonl(f)]


def cmd_oracle(a):
    from debate_ppo.engines import honest_round, make_vllm_engine
    D.set_profile_set("v1nn")
    out = Path(a.out)
    states = {s["state_id"]: s for s in _jsonl(out / "states.jsonl")}
    cands = _all_cands(out)
    f = out / "oracle.jsonl"
    done = {(r["persuader"], r["state_id"], r["cand"]) for r in _jsonl(f)} if f.exists() else set()
    todo = [c for c in cands if (c["persuader"], c["state_id"], c["cand"]) not in done]
    log(f"oracle: {len(cands)} candidates ({len(done)} done), K={K}, 9B")
    eng = make_vllm_engine(f"{M_}/qwen3.5-9b", gpu_memory_utilization=a.gpu_util, max_model_len=8192, enforce_eager=True)
    for i in range(0, len(todo), a.chunk):
        chunk = todo[i: i + a.chunk]
        items, owners = [], []
        for ci, c in enumerate(chunk):
            s = states[c["state_id"]]
            sst = MD.scoring_state(s, _reply(s, c))
            for k in range(K):
                for ag in D.honest_seats(s["seat"]):
                    items.append((sst, ag, D.stable_seed("rank_oracle_9b", s["state_id"], k)))
                    owners.append((ci, k))
        res = honest_round(eng, items, temperature=MD.WM_TEMPERATURE, max_tokens=MD.WM_MAX_TOKENS)
        per = collections.defaultdict(lambda: collections.defaultdict(list))
        for (ci, k), h in zip(owners, res):
            per[ci][k].append(h)
        rows = []
        for ci, c in enumerate(chunk):
            s = states[c["state_id"]]
            ti, gi = D.LETTERS.index(s["target"]), D.LETTERS.index(s["gold"])
            ks = [per[ci][k] for k in range(K)]
            rows.append({"persuader": c["persuader"], "state_id": c["state_id"], "cand": c["cand"],
                         "ptarget_k": [sum(h["probs"][ti] for h in hs) / len(hs) for hs in ks],
                         "pgold_k": [sum(h["probs"][gi] for h in hs) / len(hs) for hs in ks],
                         "to_target_k": [sum(h["reply"].answer == s["target"] for h in hs) / len(hs) for hs in ks]})
        _write_jsonl(f, rows, mode="a")
        log(f"  oracle {min(i + a.chunk, len(todo))}/{len(todo)}")


def cmd_wm(a):
    from debate_ppo import wm_bon as WB
    out = Path(a.out)
    states = {s["state_id"]: s for s in _jsonl(out / "states.jsonl")}
    cands = _all_cands(out)
    by = collections.defaultdict(list)
    for c in cands:
        by[c["state_id"]].append(c)
    sel = WB.Selector("tb")
    rows = []
    t0 = time.time()
    for n, (sid, cs) in enumerate(sorted(by.items())):
        s = states[sid]
        r, seat = s["round"], s["seat"]
        hs = D.honest_seats(seat)
        for mode in ("tb", "prior"):
            honest = {}
            for k, i in enumerate(hs):
                rep = s["history"][r][i]
                pr = WB._prior(s["personalities"][i], rep["answer"], s["n_options"])
                h = {"persona": s["personalities"][i], "message": rep["raw"], "answer": rep["answer"],
                     "probs": WB._pad(s["beliefs"][r][k]) if mode == "tb" else pr}
                if r >= 1:
                    pv = s["history"][r - 1][i]
                    h.update({"prev_message": pv["raw"], "prev_answer": pv["answer"],
                              "prev_probs": WB._pad(s["beliefs"][r - 1][k]) if mode == "tb"
                              else WB._prior(s["personalities"][i], pv["answer"], s["n_options"])})
                honest[i] = h
            wst = {"question_id": s["task_id"], "round": r + 1, "fourth_seat": seat, "gold": s["gold"],
                   "target": s["target"], "honest": honest,
                   "fourth_prev_message": s["history"][r - 1][seat]["raw"] if r >= 1 else None}
            res = sel.wm.score(wst, [_reply(s, c).raw for c in cs])
            for c, o in zip(cs, res):
                rows.append({"persuader": c["persuader"], "state_id": sid, "cand": c["cand"], "mode": mode,
                             "wm_ptarget": o["mean_p_target"]})
        if n % 50 == 0:
            log(f"  wm {n}/{len(by)} states, {time.time() - t0:.0f}s")
    _write_jsonl(out / "wm.jsonl", rows)
    log(f"wm: {len(rows)} scores -> {out / 'wm.jsonl'}")


# ---------------------------------------------------------------- analysis
def _rank(x):
    o = sorted(range(len(x)), key=lambda i: x[i])
    r = [0.0] * len(x)
    i = 0
    while i < len(o):
        j = i
        while j + 1 < len(o) and x[o[j + 1]] == x[o[i]]:
            j += 1
        for t in range(i, j + 1):
            r[o[t]] = (i + j) / 2
        i = j + 1
    return r


def spearman(a, b):
    ra, rb = _rank(a), _rank(b)
    ma, mb = st.mean(ra), st.mean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = math.sqrt(sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb))
    return num / den if den else 0.0


def state_metrics(cs, wm_mode="tb"):
    """cs: the 8 candidates of one (persuader, state) with ptarget_k, wm, words. -> per-state metrics."""
    o = [st.mean(c["ptarget_k"]) for c in cs]
    hA = [st.mean(c["ptarget_k"][:2]) for c in cs]
    hB = [st.mean(c["ptarget_k"][2:]) for c in cs]
    w = [c["wm"][wm_mode] for c in cs]
    n = len(cs)
    mean_o = st.mean(o)
    within = st.mean(st.pvariance(c["ptarget_k"]) for c in cs)          # rollout noise per candidate
    between = max(0.0, st.pvariance(o) - within / K)                     # noise-corrected candidate variance
    # split-half: select on A, evaluate on B (and B on A), averaged
    def sel_gain(score, ev):
        j = max(range(n), key=lambda i: score[i])
        return ev[j] - st.mean(ev)
    oracle_gain = (sel_gain(hA, hB) + sel_gain(hB, hA)) / 2
    wm_gain = (sel_gain(w, hB) + sel_gain(w, hA)) / 2                    # WM pick evaluated on held-out halves
    long_gain = (sel_gain([c["words"] for c in cs], hB) + sel_gain([c["words"] for c in cs], hA)) / 2
    short_gain = (sel_gain([-c["words"] for c in cs], hB) + sel_gain([-c["words"] for c in cs], hA)) / 2
    best_o = max(range(n), key=lambda i: o[i])
    wm_pick = max(range(n), key=lambda i: w[i])
    corr_range = math.sqrt(between) * 2 if between > 0 else 0.0          # ~ +-1 SD band
    return {"oracle_mean": mean_o, "between_var": between, "between_sd": math.sqrt(between), "noise_var": within / K,
            "oracle_gain": oracle_gain, "wm_gain": wm_gain, "long_gain": long_gain, "short_gain": short_gain,
            "spread_ge05": float(corr_range >= 0.05),
            "sp_wm": spearman(w, o), "sp_ceiling": spearman(hA, hB),
            "top1": float(wm_pick == best_o), "wm_mean": st.mean(w),
            "wm_err": st.mean(abs(x - y) for x, y in zip(w, o)), "wm_bias": st.mean(x - y for x, y in zip(w, o))}


def boot(vals, n=2000, seed=0):
    rng = random.Random(seed)
    m = st.mean(vals)
    bs = sorted(st.mean(vals[rng.randrange(len(vals))] for _ in vals) for _ in range(n))
    return m, bs[int(0.025 * n)], bs[int(0.975 * n) - 1]


def cmd_analyze(a):
    out = Path(a.out)
    states = {s["state_id"]: s for s in _jsonl(out / "states.jsonl")}
    orc = {(r["persuader"], r["state_id"], r["cand"]): r for r in _jsonl(out / "oracle.jsonl")}
    wm = collections.defaultdict(dict)
    for r in _jsonl(out / "wm.jsonl"):
        wm[(r["persuader"], r["state_id"], r["cand"])][r["mode"]] = r["wm_ptarget"]
    groups = collections.defaultdict(list)
    for c in _all_cands(out):
        key = (c["persuader"], c["state_id"], c["cand"])
        if key in orc and key in wm:
            groups[(c["persuader"], c["state_id"])].append({**c, **orc[key], "wm": wm[key]})
    pers = [p for p in PERSUADERS if any(k[0] == p for k in groups)]
    lines = []
    pr = lambda s="": lines.append(s)
    for mode in ("tb", "prior"):
        met = {p: {sid: state_metrics(cs, mode) for (pp, sid), cs in groups.items() if pp == p and len(cs) == N_CAND}
               for p in pers}
        common = sorted(set.intersection(*[set(m) for m in met.values()]))
        pr(f"## WM belief input = {mode} ({len(common)} common states)")
        if mode == "tb":
            pr("| persuader | words | cap rate | 9B oracle mean P(target) |")
            pr("|---|---|---|---|")
            for p in pers:
                cs = [c for (pp, sid), v in groups.items() if pp == p and sid in common for c in v]
                pr(f"| {p} | {st.mean(c['words'] for c in cs):.1f} | {st.mean(c['capped'] for c in cs):.3f} | "
                   f"{st.mean(st.mean(c['ptarget_k']) for c in cs):.3f} |")
            pr()
            pr("**H2: is there anything to pick on the real 9B?** (mean over states [95% bootstrap CI])")
            pr()
            pr("| persuader | noise-corrected between-candidate SD | split-half oracle gain (best − mean) | states with noise-corrected 2·SD ≥ 0.05 | rollout-noise SD of a candidate mean |")
            pr("|---|---|---|---|---|")
            for p in pers:
                f = lambda k: boot([met[p][s][k] for s in common])
                pr(f"| {p} | {f('between_sd')[0]:.3f} [{f('between_sd')[1]:.3f}, {f('between_sd')[2]:.3f}] | "
                   f"{f('oracle_gain')[0]:.3f} [{f('oracle_gain')[1]:.3f}, {f('oracle_gain')[2]:.3f}] | "
                   f"{f('spread_ge05')[0]:.2f} | {math.sqrt(st.mean(met[p][s]['noise_var'] for s in common)):.3f} |")
            pr()
        pr(f"**H3/H1: does the WM rank? (WM input {mode})**")
        pr()
        pr("| persuader | Spearman(WM, 9B) | noise ceiling Spearman(9B half A, half B) | top-1 hit (chance 0.125) | WM-pick gain (held-out) | share of oracle gain | longest-pick gain | shortest-pick gain | WM − 9B bias | WM MAE |")
        pr("|---|---|---|---|---|---|---|---|---|---|")
        for p in pers:
            f = lambda k: boot([met[p][s][k] for s in common])
            og, wg = f("oracle_gain")[0], f("wm_gain")
            pr(f"| {p} | {f('sp_wm')[0]:.3f} [{f('sp_wm')[1]:.3f}, {f('sp_wm')[2]:.3f}] | {f('sp_ceiling')[0]:.3f} | "
               f"{f('top1')[0]:.3f} | {wg[0]:+.3f} [{wg[1]:+.3f}, {wg[2]:+.3f}] | {wg[0] / og if og > 0 else float('nan'):.2f} | "
               f"{f('long_gain')[0]:+.3f} | {f('short_gain')[0]:+.3f} | {f('wm_bias')[0]:+.3f} | {f('wm_err')[0]:.3f} |")
        pr()
        if mode == "tb":
            pr("Paired differences between persuaders on the same states (a − b, bootstrap over states):")
            pr()
            pr("| a | b | Δ between-cand SD | Δ split-half oracle gain | Δ Spearman(WM, 9B) | Δ WM-pick gain |")
            pr("|---|---|---|---|---|---|")
            for x, y in [("dpo_ppo_sd120", "ppo_c120"), ("dpo", "ppo_c120"), ("dpo_ppo_sd120", "dpo"), ("base_q3_4b", "ppo_c120"),
                         ("base_q3_4b", "dpo")]:
                if x not in met or y not in met:
                    continue
                cells = []
                for k in ("between_sd", "oracle_gain", "sp_wm", "wm_gain"):
                    m, lo, hi = boot([met[x][s][k] - met[y][s][k] for s in common])
                    cells.append(f"{m:+.3f} [{lo:+.3f}, {hi:+.3f}]")
                pr(f"| {x} | {y} | " + " | ".join(cells) + " |")
            pr()
            pr("Per round (attacker round 1/2/3): split-half oracle gain / Spearman(WM, 9B) / WM-pick gain")
            pr()
            pr("| persuader | round 1 | round 2 | round 3 |")
            pr("|---|---|---|---|")
            for p in pers:
                cells = []
                for rd in range(3):
                    ss = [s for s in common if states[s]["round"] == rd]
                    cells.append(" / ".join(f"{st.mean(met[p][s][k] for s in ss):+.3f}" for k in ("oracle_gain", "sp_wm", "wm_gain")))
                pr(f"| {p} | " + " | ".join(cells) + " |")
            pr()
    # Part B: WM error by message feature (tb), per candidate, error = WM - 9B oracle mean; also residual vs state mean
    pr("## Part B: WM error by message feature (WM input tb)")
    pr()
    pr("Within-state residual = (WM − WM state mean) − (9B − 9B state mean): >0 means the WM over-ranks this kind of message.")
    pr()
    pr("| persuader | feature | share of candidates | within-state residual (feature) | within-state residual (others) |")
    pr("|---|---|---|---|---|")
    for p in pers:
        rows = []
        for (pp, sid), cs in groups.items():
            if pp != p or len(cs) != N_CAND:
                continue
            s = states[sid]
            o = [st.mean(c["ptarget_k"]) for c in cs]
            w = [c["wm"]["tb"] for c in cs]
            mo, mw = st.mean(o), st.mean(w)
            for c, oi, wi in zip(cs, o, w):
                gold_named = bool(re.search(r"(?i)(correct answer is|answer is|correct option is)\s*\(?(option\s*)?%s\b" % s["gold"], c["argument"]))
                rows.append({"res": (wi - mw) - (oi - mo), "capped": c["capped"], "hedge": bool(HEDGE.search(c["argument"])),
                             "gold": gold_named, "long": c["words"] >= 120})
        for feat in ("capped", "hedge", "gold", "long"):
            yes = [r["res"] for r in rows if r[feat]]
            no = [r["res"] for r in rows if not r[feat]]
            pr(f"| {p} | {feat} | {len(yes) / max(1, len(rows)):.3f} | {st.mean(yes) if yes else float('nan'):+.4f} | "
               f"{st.mean(no) if no else float('nan'):+.4f} |")
    pr()
    pr("Stance judge truncation: none. stance.prompts() puts the whole argument into the prompt and Judge.score feeds the "
       "full token sequence (no max length), so long messages are scored from their full text.")
    text = "\n".join(lines)
    (out / "RESULTS.md").write_text(text + "\n")
    print(text)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("states", "gen", "oracle", "wm", "analyze"):
        p = sub.add_parser(name)
        p.add_argument("--out", required=True)
        if name == "states":
            p.add_argument("--n", type=int, default=100)
            p.add_argument("--seed", type=int, default=3)
        if name == "gen":
            p.add_argument("--persuader", action="append", default=None, help="NAME=PATH (repeatable); default all four")
        if name in ("gen", "oracle"):
            p.add_argument("--gpu-util", type=float, default=0.85)
        if name == "oracle":
            p.add_argument("--chunk", type=int, default=800)
    a = ap.parse_args()
    if a.cmd == "gen" and not a.persuader:
        a.persuader = [f"{k}={v}" for k, v in PERSUADERS.items()]
    {"states": cmd_states, "gen": cmd_gen, "oracle": cmd_oracle, "wm": cmd_wm, "analyze": cmd_analyze}[a.cmd](a)


if __name__ == "__main__":
    main()
