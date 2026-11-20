#!/usr/bin/env python
"""WM-BoN with debate-wm-v1 direct readout as the selector (2026-10-31). New file; eval_debate.py / wm_bon.py unchanged.

Condition <targeted_model_*>__wmbon<N>dw: identical to eval_debate's <cond>__wmbon<N>wb arm (same chat + TARGET_HINT, target
letter prefilled, N candidates from wm_bon.candidates with T = two_phase_temp, cap = two_phase_arg_tokens, seeds
stable_seed('attacker', task, seed, r); victim phase of a round played before the attacker phase) except the SELECTOR:
for every candidate, the three honest listeners' round-(r+1) chat with the candidate in the attacker seat (dprotocol.chat on
history 0..r, round r = the honest agents' recorded round-r replies + the candidate) is read out by debate-wm-v1 with the
wm_judge_eval `roleplay_direct` judge (persona chat + 'Answer:' prefill, top-40 logprobs restricted to valid letters) and the
candidate with the highest mean P(target) wins (invalid candidates last, as wm_bon.Selector).
Engines: the persuader generates all candidates of the phase, is closed, then debate-wm-v1 scores all of them in one batch
(one model on the GPU at a time). Everything else (init, victim worker, commit, metrics) is eval_debate's own code.

  python -m debate_ppo.eval_debate_dwm init ... (same flags as eval_debate init) --dwm-model PATH
  python -m debate_ppo.eval_debate_dwm play --run R
  python -m debate_ppo.eval_debate summary --run R
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from debate_ppo import eval_debate as E  # noqa: E402
from debate_ppo import wm_bon as WB  # noqa: E402

D, P = E.D, E.P
PAT = re.compile(r"^(targeted_model_.*)__wmbon(\d+)dw$")
JUDGE = "roleplay_direct"


def parse(cond):
    m = PAT.match(cond)
    return (m.group(1), int(m.group(2))) if m else None


def _sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]


def listener_state(d: dict, r: int, cand_reply) -> "P.DebateState":
    s = P.DebateState(d["task_id"], d["question"], d["n_options"], d["personalities"])
    hist = [[P.Reply(**rep) for rep in rnd] for rnd in d["history"]]
    cur = [cand_reply if j == d["seat"] else P.Reply(**d["pending"][str(j)]["reply"]) for j in range(D.N_AGENTS)]
    s.history = hist + [cur]
    s.round = len(s.history)
    return s


def worker_attacker_dwm(run: Path, r: int, cond: str):
    from debate_ppo.engines import close_vllm_engine, make_vllm_engine
    from debate_ppo.wm_judge_eval.judges import run_judge
    base, n_cand = parse(cond)
    # this arm reuses exactly the plain targeted branch of eval_debate.worker_attacker; no other branch may match
    assert E.is_targeted(cond) and E.is_forced_target(cond) and not any(
        f(cond) for f in (E.is_gold_informed, E.is_amayuelas, E.TM.is_tomap, E.BZ.is_bozdag, E.JU.is_ju, E.ST.is_strat,
                          E.KR.is_kraidia, E.BON.is_bon, E.BON.is_amayuelas_bon, E.BON.is_kraidia_bon)), cond
    st = E.load_state(run)
    cfg = st["config"]
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
        chats.append(E.add_target_hint(D.chat(s, d["seat"]), tgt))
        forced.append(tgt)
        nopt.append(d["n_options"])
        seeds.append(D.stable_seed("attacker", d["task_id"], d["seed"], r))
        idx.append(di)
    E.log(f"attacker[{cond}] r={r + 1}: {len(chats)} turns, {n_cand} candidates each")
    eng = make_vllm_engine(cfg["attackers"][cond], gpu_memory_utilization=cfg["vllm"]["gpu_memory_utilization"],
                           max_model_len=cfg["vllm"]["max_model_len"])
    cl = WB.candidates(eng, chats, nopt, seeds, forced, n=n_cand, temperature=cfg["two_phase_temp"],
                       max_arg_tokens=cfg["two_phase_arg_tokens"])
    close_vllm_engine(eng)
    recs, owner = [], []
    for k, (di, cs) in enumerate(zip(idx, cl)):
        d = st["debates"][di]
        for c, cand in enumerate(cs):
            ls = listener_state(d, r, cand["reply"])
            for i in D.honest_seats(d["seat"]):
                recs.append({"id": f"{d['task_id']}|s{d['seed']}|r{r}|c{c}|a{i}", "messages": D.chat(ls, i),
                             "n_options": d["n_options"], "meta": {"round": r + 2, "target": d["target"]}})
                owner.append((k, c, i))
    dw = cfg["dwm"]
    E.log(f"dwm[{cond}] r={r + 1}: scoring {len(recs)} listener turns with {dw['model']}")
    jeng = make_vllm_engine(dw["model"], gpu_memory_utilization=dw["gpu_memory_utilization"],
                            max_model_len=dw["max_model_len"], max_num_seqs=dw["max_num_seqs"])
    res = run_judge(jeng, recs, JUDGE)
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
        d.setdefault("wmbon", []).append({"round": r, "mode": "dw", "judge": f"debate-wm-v1/{JUDGE}", "n": len(cs),
                                          "best": best, "scores": sc, "valid": valid,
                                          "cands": [c["reply"].__dict__ for c in cs],
                                          "pred_beliefs_best": {str(i): probs[(k, best)][i] for i in hs}})
        o = cs[best]
        d["pending"][str(d["seat"])] = {"reply": o["reply"].__dict__, "probs": None, "letter_mass": None,
                                        "truncated": o["capped"]}
        d["attacker_meta"].append({"arg_tokens": o["arg_tokens"], "capped": o["capped"]})
    st["done_phases"].append(f"attacker[{cond}]{r}")
    E.save_state(run, st)


def cmd_play(run: Path):
    conds = [c for c in E.load_state(run)["config"]["conds"] if c != "none"]
    assert conds and all(parse(c) for c in conds), conds          # victim phase first, as eval_debate does for wmbon
    me = [sys.executable, "-m", "debate_ppo.eval_debate_dwm", "worker", "--run", str(run)]
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
        dwm_model = argv[argv.index("--dwm-model") + 1]
        rest = argv[:argv.index("--dwm-model")] + argv[argv.index("--dwm-model") + 2:]
        sys.argv = [sys.argv[0]] + rest
        E.main()
        run = Path(rest[rest.index("--run") + 1])
        st = E.load_state(run)
        root = Path(__file__).resolve().parents[1]
        st["config"]["dwm"] = {
            "model": dwm_model, "judge": JUDGE, "gpu_memory_utilization": st["config"]["vllm"]["gpu_memory_utilization"],
            "max_model_len": 8192, "max_num_seqs": 128,
            "weights_sha256": {f.name: hashlib.sha256(open(f, "rb").read()).hexdigest()
                               for f in sorted(Path(dwm_model).glob("*.safetensors"))},
            "code_sha256": {f: _sha(root / f) for f in ("debate_ppo/eval_debate_dwm.py",
                                                         "debate_ppo/wm_judge_eval/judges.py")},
            "selector": "argmax over candidates of mean_i debate-wm-v1 roleplay_direct P_i(target) at round r+1 "
                        "(invalid last); candidates = wm_bon.candidates"}
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
            worker_attacker_dwm(run, rnd, argv[argv.index("--cond") + 1])
    else:
        raise SystemExit(f"unknown command {cmd}")


if __name__ == "__main__":
    main()
