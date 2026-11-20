"""Fluency / well-formedness of the attacker messages PICKED by sd120 + dwm-BoN8 (2026-10-31). CPU only, read only.
Compares: picks of exp10_nn_dwmbon (debate-wm-v1 direct selector), all 8 candidates of the same turns, picks of
exp10_nn_wmbon (DCM selector, wb), and single-sample sd120 of exp10_nn_ours. Flags are regex heuristics (defined below).
python debate_wm/dwm_pick_fluency.py RUNS_DIR OUT.md"""
import json, random, re, statistics as S, sys
from collections import Counter
from pathlib import Path

RUNS, OUT = Path(sys.argv[1]), Path(sys.argv[2])
SD = "targeted_model_rl_sd120"
HEDGE = re.compile(r"(?i)\bwait\b|\bhowever\b.*\bhowever\b|\bbut the correct\b|\bactually\b")   # = rank_oracle.HEDGE
SENT = re.compile(r"[^.!?]+[.!?]")
NONLATIN = re.compile(r"[Ѐ-ӿ֐-ۿ฀-๿぀-ヿ㐀-鿿가-힯�]")


def flags(arg, gold, target, capped=None):
    t = arg.strip()
    gold_said = bool(re.search(r"(?i)(correct answer is|answer is|correct option is|correct choice is)\s*\(?(option\s*)?%s\b" % gold, t))
    other = [m.group(3) for m in re.finditer(r"(?i)(correct answer is|correct option is|correct choice is)\s*\(?(option\s*)?([A-J])\b", t)]
    contra = any(x.upper() != target for x in other)
    sents = [" ".join(s.lower().split()) for s in SENT.findall(t) if len(s.split()) >= 4]
    rep = any(v >= 2 for v in Counter(sents).values()) or len(re.findall(r"(?i)final answer", t)) >= 2
    f = {"hedge": bool(HEDGE.search(t)), "gold_stated": gold_said, "contradicts_answer": contra, "repetition": rep,
         "nonlatin": bool(NONLATIN.search(t)), "no_terminal_punct": not re.search(r"[.!?\"')\]]\s*$", t)}
    if capped is not None:
        f["capped_200"] = bool(capped)
    f["score"] = sum(v for k, v in f.items() if k in ("hedge", "gold_stated", "contradicts_answer", "repetition", "nonlatin", "no_terminal_punct"))
    return f


def turns_bon(path, cond):
    out = []
    for d in json.load(open(path))["debates"]:
        if d["cond"] != cond:
            continue
        hs = [i for i in range(4) if i != d["seat"]]
        for w, m in zip(d["wmbon"], d["attacker_meta"]):
            r = w["round"]
            after = [d["history"][r + 1][i]["answer"] for i in hs] if r + 1 < len(d["history"]) else [x["answer"] for x in d["final"]]
            out.append({"d": d, "r": r, "pick": w["cands"][w["best"]]["argument"], "capped": m["capped"],
                        "cands": [c["argument"] for c in w["cands"]], "valid": w["valid"], "scores": w["scores"], "after": after})
    return out


def turns_single(path, cond):
    out = []
    for d in json.load(open(path))["debates"]:
        if d["cond"] != cond:
            continue
        hs = [i for i in range(4) if i != d["seat"]]
        for r in range(3):
            after = [d["history"][r + 1][i]["answer"] for i in hs] if r + 1 < 3 else [x["answer"] for x in d["final"]]
            out.append({"d": d, "r": r, "pick": d["history"][r][d["seat"]]["argument"], "capped": d["attacker_meta"][r]["capped"], "after": after})
    return out


def agg(F, keys):
    return {k: S.mean(f[k] for f in F) for k in keys}


dw = turns_bon(RUNS / "exp10_nn_dwmbon/state.json", SD + "__wmbon8dw")
wb = turns_bon(RUNS / "exp10_nn_wmbon/state.json", SD + "__wmbon8wb")
sg = turns_single(RUNS / "exp10_nn_ours/state.json", SD)
KEYS = ["hedge", "gold_stated", "contradicts_answer", "repetition", "nonlatin", "no_terminal_punct", "score"]
rows = {}
for name, T in (("dwm-BoN8 picks", dw), ("dwm-BoN8 all 8 candidates (same turns)", None), ("DCM-BoN8 (wb) picks", wb),
                ("DCM-BoN8 all 8 candidates", "wbc"), ("sd120 single sample", sg)):
    if T is None:
        F = [flags(c, t["d"]["gold"], t["d"]["target"]) for t in dw for c in t["cands"]]
        W = [len(c.split()) for t in dw for c in t["cands"]]; cap = None
    elif T == "wbc":
        F = [flags(c, t["d"]["gold"], t["d"]["target"]) for t in wb for c in t["cands"]]
        W = [len(c.split()) for t in wb for c in t["cands"]]; cap = None
    else:
        F = [flags(t["pick"], t["d"]["gold"], t["d"]["target"]) for t in T]
        W = [len(t["pick"].split()) for t in T]; cap = S.mean(bool(t["capped"]) for t in T)
    rows[name] = (len(F), S.mean(W), cap, agg(F, KEYS))

# within-turn: pick flag - mean flag of the 8 candidates of that turn (bootstrap over turns)
def within(T):
    res = {}
    for k in KEYS:
        diffs = []
        for t in T:
            fc = [flags(c, t["d"]["gold"], t["d"]["target"])[k] for c in t["cands"]]
            diffs.append(flags(t["pick"], t["d"]["gold"], t["d"]["target"])[k] - S.mean(fc))
        rng = random.Random(0)
        bs = sorted(S.mean(diffs[rng.randrange(len(diffs))] for _ in diffs) for _ in range(1000))
        res[k] = (S.mean(diffs), bs[25], bs[974])
    return res


wi_dw, wi_wb = within(dw), within(wb)
L = []
pr = L.append
pr("# sd120 + dwm-BoN8 选中的攻击者消息是否通顺（2026-10-31，CPU，只读）\n")
pr("数据：`exp10_nn_dwmbon`（sd120，选择器 = debate-wm-v1 直接读）的 1,200 局 × 3 个攻击回合 = 3,600 条选中消息；对照：同一回合的全部 8 条候选、"
   "`exp10_nn_wmbon` 的 sd120 + DCM-BoN8（wb）选中消息与候选、`exp10_nn_ours` 的 sd120 单采样。全部是正则启发式标记，不是人工评审。\n")
pr("标记定义：hedge = `wait` / 两次 `however` / `but the correct` / `actually`（同 rank_oracle.HEDGE）；gold_stated = 出现 \"correct answer is / answer is … <gold>\"；"
   "contradicts_answer = 出现 \"correct answer/option/choice is X\" 且 X ≠ 自己的 Answer 行（= target）；repetition = 有 ≥4 词的句子原样出现两次，或 \"final answer\" ≥2 次；"
   "nonlatin = 含中日韩/西里尔/阿拉伯等文字或替换符（数学符号不算）；no_terminal_punct = 不以 .!?\"') 结尾（通常是被 200 token 截断）；capped_200 = 生成到 200 token 上限（只有选中消息有记录）。"
   "flaw score = 前 6 项之和。\n")
pr("| 组 | n | 平均词数 | 200-token 截断 | hedge/自我纠正 | 说出 gold | 与自己 Answer 矛盾 | 重复 | 非英文/乱码 | 无结尾标点 | 平均 flaw score |")
pr("|---|---|---|---|---|---|---|---|---|---|---|")
for name, (n, w, cap, a) in rows.items():
    pr(f"| {name} | {n} | {w:.1f} | {'–' if cap is None else f'{cap:.3f}'} | {a['hedge']:.3f} | {a['gold_stated']:.3f} | {a['contradicts_answer']:.3f} | "
       f"{a['repetition']:.3f} | {a['nonlatin']:.4f} | {a['no_terminal_punct']:.3f} | {a['score']:.3f} |")
pr("")
pr("**选择器偏不偏向有毛病的消息**（同一回合内：选中消息的标记 − 该回合 8 条候选的平均；负 = 选中的比候选平均更干净；95% CI 按回合 bootstrap 1,000 次）\n")
pr("| 标记 | dwm（debate-wm-v1 直接读） | DCM（wb） |")
pr("|---|---|---|")
for k in KEYS:
    a, b = wi_dw[k], wi_wb[k]
    pr(f"| {k} | {a[0]:+.3f} [{a[1]:+.3f}, {a[2]:+.3f}] | {b[0]:+.3f} [{b[1]:+.3f}, {b[2]:+.3f}] |")
pr("")


def show(t, why):
    d = t["d"]
    f = flags(t["pick"], d["gold"], d["target"], t["capped"])
    bad = [k for k, v in f.items() if k != "score" and v]
    pr(f"**{why}** — `{d['task_id']}` seed {d['seed']}，攻击回合 {t['r'] + 1}，gold = {d['gold']}，target = {d['target']}，"
       f"三个诚实方下一回合答案 = {', '.join(t['after'])}；标记：{', '.join(bad) or '无'}；词数 {len(t['pick'].split())}\n")
    pr("> " + t["pick"].replace("\n", " ") + "\n")


pr("## 选中消息原文（6 条）\n")
rng = random.Random(1)
for t in rng.sample(dw, 3):
    show(t, "随机")
score = lambda t: (flags(t["pick"], t["d"]["gold"], t["d"]["target"])["score"], len(t["pick"]))
for t in sorted(dw, key=score, reverse=True)[:2]:
    show(t, "flaw score 最高")
clean = [t for t in dw if score(t)[0] == 0 and not t["capped"] and sum(a == t["d"]["target"] for a in t["after"]) == 3]
show(max(clean, key=lambda t: max(t["scores"])), "最好（无任何标记、未截断、三人都转到 target，选择器分数最高的一条）")
OUT.write_text("\n".join(L) + "\n")
print("\n".join(L))
