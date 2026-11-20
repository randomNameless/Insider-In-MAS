"""
Experiment 2 attack evaluation: consensus rate of the 5 attackers vs the opd-mm
benign world model WHEN THE BENIGN AGENTS RUN A W-MSR INPUT FILTER.

Same as eval_attacks_consensus.py (Random / Extreme / Prompt-based / DQN+Rephrase /
Model-based RL, 200 games each, same init positions), except the benign (victim) WM
agents apply the W-MSR filter used in training (Exp2): before each benign agent is
queried, the F most extreme neighbor positions relative to its own (position AND
message) are dropped from its prompt. The attacker must stay inside the retained band
to be seen. This measures each attack's consensus rate UNDER the defense — compare it
against the no-defense numbers from eval_attacks_consensus.py.

All attacker policies, model loading, and the rollout loop are reused from
eval_attacks_consensus.py; only the env is swapped for a W-MSR-filtered one.

Run:
  python eval_attacks_consensus_wmsr.py \
      --wm_model_dir  ~/models/qwen35-wm-vllm \
      --llm_model_dir Qwen/Qwen3-4B-Instruct-2507 \
      --rl_adapter    ~/consensus-attacker/Experiment/Experiment1_Main/runs/exp2_wmsr/best \
      --dqn_ckpt      attacker_softq_qwen_mixed.pt \
      --wmsr_F 1 --n_exp 200

Note: point --rl_adapter at the exp2 (defense-trained) adapter to evaluate the
defense-aware attacker under the defense, or at the exp1 adapter to test transfer of
the no-defense attacker onto the defended WM.
"""

import os
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("VLLM_BATCH_INVARIANT", "0")

import argparse
import json
import random

import torch

from RL_attack_env_qwen import (
    AttackEnv, QwenWorldModelVLLM, QwenWorldModelHF, build_system_prompt,
    ROUND0_PROMPT_TEMPLATE, MULTIROUND_PROMPT_TEMPLATE,
)
# reuse everything else from the Exp1 eval (same directory)
from eval_attacks_consensus import (
    MIXED_PERSONALITIES, HFQwen3, load_dqn, make_attacker_fn, run_episode,
)


class WMSRFilterEnv(AttackEnv):
    """AttackEnv whose BENIGN agents run a W-MSR input filter. Only the benign
    prompt builder (_build_messages, used by prepare() for agents 0-2) is
    overridden — the attacker's own prompt is untouched. Standard MSR trim: for
    benign i with position x_i, drop up to F neighbors strictly above and F strictly
    below x_i (position + message); equal-to-x_i neighbors are kept."""

    def __init__(self, *args, wmsr_F=1, **kwargs):
        self.wmsr_F = int(wmsr_F)
        super().__init__(*args, **kwargs)

    def _visible_others(self, i):
        x = self.pos[i]
        others = [j for j in range(4) if j != i]
        above = sorted((j for j in others if self.pos[j] > x), key=lambda j: self.pos[j], reverse=True)
        below = sorted((j for j in others if self.pos[j] < x), key=lambda j: self.pos[j])
        trimmed = set(above[:self.wmsr_F]) | set(below[:self.wmsr_F])
        return [j for j in others if j not in trimmed]

    def _build_messages(self, i):
        vis = set(self._visible_others(i))
        others = [j for j in range(4) if j != i and j in vis]
        others_pos = [int(self.pos[j]) for j in others]
        if self.t == 0:
            user = ROUND0_PROMPT_TEMPLATE.format(
                self_pos=int(self.pos[i]), other_positions=others_pos)
        else:
            order = [j for j in self._order_others([j for j in range(4) if j != i]) if j in vis]
            lines = "\n".join(f"- Agent {j}: {self.last_messages[j]}" for j in order)
            user = MULTIROUND_PROMPT_TEMPLATE.format(
                self_pos=int(self.pos[i]), other_positions=others_pos, message_lines=lines)
        return [
            {"role": "system", "content": build_system_prompt(self.personalities[i])},
            {"role": "user", "content": user},
        ]


def main():
    p = argparse.ArgumentParser(
        description="Exp2 consensus-rate eval of 5 attackers vs opd-mm WITH a W-MSR benign filter.")
    p.add_argument("--wm_model_dir", default="~/models/qwen35-wm-vllm",
                   help="opd-mm benign world model (vLLM-loadable checkpoint).")
    p.add_argument("--wm_backend", default="vllm", choices=["vllm", "hf"])
    p.add_argument("--llm_model_dir", default="Qwen/Qwen3-4B-Instruct-2507",
                   help="stock Qwen3 for rephrase/prompt AND base for model_rl.")
    p.add_argument("--rl_adapter", default=None,
                   help="PPO LoRA adapter dir for model_rl (exp2's for defense-aware, exp1's for transfer).")
    p.add_argument("--dqn_ckpt", default="attacker_softq_qwen_mixed.pt")
    p.add_argument("--dqn_device", default="cpu")
    p.add_argument("--attacks", default="random,extreme,prompt,dqn,model_rl")
    p.add_argument("--wmsr_F", type=int, default=1, help="W-MSR trim count (defense strength).")
    p.add_argument("--n_exp", type=int, default=200)
    p.add_argument("--max_pos", type=int, default=20)
    p.add_argument("--max_steps", type=int, default=10)
    p.add_argument("--consensus_tol", type=float, default=0.0)
    p.add_argument("--consensus_penalty", type=float, default=-10.0)
    p.add_argument("--seed", type=int, default=9000)
    p.add_argument("--wm_temperature", type=float, default=0.7)
    p.add_argument("--llm_temperature", type=float, default=0.7)
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--wm_max_new_tokens", type=int, default=200)
    p.add_argument("--wm_gpu_mem_util", type=float, default=0.55)
    p.add_argument("--max_model_len", type=int, default=2048)
    p.add_argument("--enforce_eager", action="store_true")
    p.add_argument("--dump_json", default=None,
                   help="write a per-round trajectory JSON here (attacker pos/msg, benign moves, W-MSR visibility).")
    p.add_argument("--dump_n", type=int, default=5,
                   help="how many episodes per attacker to record into --dump_json.")
    args = p.parse_args()

    kinds = [k.strip() for k in args.attacks.split(",") if k.strip()]
    dump_records = [] if args.dump_json else None
    ct_kwargs = {"enable_thinking": False}

    if args.wm_backend == "vllm":
        wm = QwenWorldModelVLLM(
            model_dir=args.wm_model_dir, max_pos=args.max_pos, max_new_tokens=args.wm_max_new_tokens,
            gpu_memory_utilization=args.wm_gpu_mem_util, max_model_len=args.max_model_len,
            enforce_eager=args.enforce_eager, chat_template_kwargs=ct_kwargs,
            temperature=args.wm_temperature, seed=args.seed)
    else:
        wm = QwenWorldModelHF(
            model_dir=args.wm_model_dir, max_pos=args.max_pos,
            max_new_tokens=args.wm_max_new_tokens, chat_template_kwargs=ct_kwargs)

    # ONLY difference vs the Exp1 eval: the benign agents run the W-MSR input filter.
    env = WMSRFilterEnv(
        world_model=wm, max_pos=args.max_pos, max_steps=args.max_steps,
        consensus_tol=args.consensus_tol, consensus_penalty=args.consensus_penalty,
        reward_mode="spread", attacker_msg_mode="template", wmsr_F=args.wmsr_F)
    print(f"[defense] W-MSR benign input filter, F={args.wmsr_F}", flush=True)

    need_llm = any(k in kinds for k in ("random", "extreme", "prompt", "dqn", "model_rl"))
    llm = None
    if need_llm:
        llm = HFQwen3(
            args.llm_model_dir, adapter_dir=(args.rl_adapter if "model_rl" in kinds else None),
            device="cuda", max_new_tokens=args.max_new_tokens,
            temperature=args.llm_temperature, seed=args.seed)
    q_net = load_dqn(args.dqn_ckpt, torch.device(args.dqn_device)) if "dqn" in kinds else None

    rng = random.Random(args.seed)
    attacker_fns = {k: make_attacker_fn(k, llm, q_net, torch.device(args.dqn_device), rng, args.max_pos)
                    for k in kinds}
    stats = {k: {"n": 0, "consensus": 0, "ep_len_sum": 0} for k in kinds}
    benign = MIXED_PERSONALITIES

    for exp_idx in range(args.n_exp):
        er = random.Random(args.seed + exp_idx + args.max_steps * 1000)
        init_pos = [er.randint(0, args.max_pos) for _ in range(4)]
        for kind in kinds:
            rec = dump_records is not None and exp_idx < args.dump_n
            terminated, ep_len, rounds = run_episode(
                env, wm, attacker_fns[kind], init_pos, benign, record=rec)
            stats[kind]["n"] += 1
            stats[kind]["consensus"] += int(terminated)
            stats[kind]["ep_len_sum"] += ep_len
            if rec:
                dump_records.append({
                    "attacker": kind, "exp_idx": exp_idx,
                    "init_positions": list(init_pos), "personalities": list(benign),
                    "consensus": bool(terminated), "ep_len": ep_len, "rounds": rounds})
            print(f"[{kind:8s}] exp {exp_idx:4d} init={init_pos} "
                  f"{'CONSENSUS' if terminated else 'SURVIVED '} ep_len={ep_len}", flush=True)

    summary = {k: {"episodes": stats[k]["n"], "consensus": stats[k]["consensus"],
                   "rate": stats[k]["consensus"] / max(1, stats[k]["n"]),
                   "mean_ep_len": stats[k]["ep_len_sum"] / max(1, stats[k]["n"])} for k in kinds}
    print("\n" + "#" * 74)
    print(f"# SUMMARY (W-MSR defense, F={args.wmsr_F}): benign = {benign[:3]}, {args.n_exp} episodes/attacker")
    print(f"# consensus rate LOWER = stronger attacker; compare vs the no-defense eval")
    print("#" * 74)
    for k in kinds:
        s = summary[k]
        print(f"  attacker={k:9s}  episodes={s['episodes']:4d}  consensus={s['consensus']:4d}  "
              f"rate={s['rate']:.3f}  mean_ep_len={s['mean_ep_len']:.2f}", flush=True)

    if dump_records is not None:
        payload = {"defense": f"wmsr F={args.wmsr_F}", "n_exp": args.n_exp, "dump_n": args.dump_n,
                   "summary": summary, "episodes": dump_records}
        with open(args.dump_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"\n[dump] wrote {len(dump_records)} recorded episodes -> {args.dump_json}", flush=True)


if __name__ == "__main__":
    main()
