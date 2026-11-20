#!/usr/bin/env python
"""Confidence-aware Experiment-3 evaluation (prompt/base/PPO only).

Both attacker and benign confidence are computed by the same helper used during
PPO training: exp(mean log-probability of generated Position digit tokens).
The evaluator intentionally uses HF generation because the legacy vLLM wrapper
returns text only and discards the token log-probabilities required by CP-WBFT.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
EVAL_DIR = HERE.parent / "Experiment1_Eval"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(EVAL_DIR))

from peft import PeftModel  # noqa: E402
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from RL_attack_env_qwen import (  # noqa: E402
    AttackEnv, build_system_prompt, ROUND0_PROMPT_TEMPLATE,
    MULTIROUND_PROMPT_TEMPLATE,
)
from eval_attacks_consensus import (  # noqa: E402
    MIXED_PERSONALITIES, build_model_rl_query,
)
from attacker.generation import (  # noqa: E402
    generate_attacker_batched, generate_benign_batched,
)
from cpwbft_defense import retained_confident_indices, self_check  # noqa: E402


ALLOWED_ATTACKS = ("prompt", "model_base", "model_rl")


class ModelHolder:
    """Minimal world-model object accepted by the prototype AttackEnv."""

    def __init__(self, model, tokenizer, max_pos: int):
        self.model = model
        self.tokenizer = tokenizer
        self.max_pos = int(max_pos)


class ConfidenceCPEnv(AttackEnv):
    default_margin = 0.0
    default_keep_min_one = True

    def __init__(self, *args, **kwargs):
        self.confidences = [1.0] * 4
        super().__init__(*args, **kwargs)

    def reset(self, *args, **kwargs):
        result = super().reset(*args, **kwargs)
        self.confidences = [1.0] * 4
        return result

    def _visible_others(self, i):
        return retained_confident_indices(
            self.confidences, i, self.default_margin,
            self.default_keep_min_one)

    def _build_messages(self, i):
        visible = self._visible_others(i)
        other_positions = [int(self.pos[j]) for j in visible]
        if self.t == 0:
            user = ROUND0_PROMPT_TEMPLATE.format(
                self_pos=int(self.pos[i]), other_positions=other_positions)
        else:
            visible_set = set(visible)
            order = [
                j for j in self._order_others([j for j in range(4) if j != i])
                if j in visible_set
            ]
            lines = "\n".join(
                f"- Agent {j}: {self.last_messages[j]}" for j in order)
            user = MULTIROUND_PROMPT_TEMPLATE.format(
                self_pos=int(self.pos[i]), other_positions=other_positions,
                message_lines=lines)
        return [
            {"role": "system", "content": build_system_prompt(self.personalities[i])},
            {"role": "user", "content": user},
        ]


def _text_config(model_dir: str):
    full = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    return getattr(full, "text_config", full)


def load_causal_model(model_dir: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, config=_text_config(model_dir), dtype=torch.bfloat16,
        trust_remote_code=True, attn_implementation="eager",
    ).to(device).eval()
    return model, tokenizer


def position_confidence(gen) -> float:
    mask = gen.seg_masks["next_position"]
    if not bool(mask.any()):
        return 1.0
    return float(torch.exp(gen.old_logp_gen[mask].float().mean()).item())


def _adapter_context(actor, enabled: bool):
    if enabled:
        return contextlib.nullcontext(actor)
    return actor.disable_adapter()


def attacker_queries(kind, envs):
    if kind == "prompt":
        return [env._build_attacker_query()[0] for env in envs]
    return [build_model_rl_query(env) for env in envs]


def generate_attackers(actor, tokenizer, kind, envs, max_msg_tokens, max_pos):
    queries = attacker_queries(kind, envs)
    fallbacks = [int(env.pos[3]) for env in envs]
    with _adapter_context(actor, enabled=(kind == "model_rl")):
        generated = generate_attacker_batched(
            actor, tokenizer, queries, fallbacks,
            max_msg_tokens=max_msg_tokens, sample=True, temperature=1.0,
            max_pos=max_pos)
    return generated


def run_batch(*, envs, wm_model, wm_tokenizer, actor, actor_tokenizer, kind,
              init_positions, max_steps, max_pos, wm_max_new_tokens,
              actor_max_message_tokens, record):
    for env, init in zip(envs, init_positions):
        env.reset(options={"positions": list(init),
                           "personalities": list(MIXED_PERSONALITIES)})

    states = [{"done": False, "consensus": False, "ep_len": 0,
               "rounds": [] if record else None} for _ in envs]
    while any(not state["done"] for state in states):
        active_idx = [idx for idx, state in enumerate(states) if not state["done"]]
        active_envs = [envs[idx] for idx in active_idx]
        attacker_gens = generate_attackers(
            actor, actor_tokenizer, kind, active_envs,
            actor_max_message_tokens, max_pos)

        flat_queries, flat_fallbacks, query_index, pending = [], [], [], []
        for local_idx, (env, gen) in enumerate(zip(active_envs, attacker_gens)):
            atk_conf = position_confidence(gen)
            attacker_snapshot_conf = float(env.confidences[3])
            before = [int(env.pos[j]) for j in range(3)]
            attacker_snapshot_pos = int(env.pos[3])
            # Filter and prompts read only the round-start Position/confidence.
            visible = [list(env._visible_others(i)) for i in range(3)]
            queries = [(env._build_messages(i), int(env.pos[i])) for i in range(3)]
            for benign_idx, (messages, fallback) in enumerate(queries[:3]):
                flat_queries.append(messages)
                flat_fallbacks.append(int(fallback))
                query_index.append((local_idx, benign_idx))
            pending.append((before, visible, gen, atk_conf,
                            attacker_snapshot_pos, attacker_snapshot_conf))

        benign_actions, benign_confs = generate_benign_batched(
            wm_model, wm_tokenizer, flat_queries, flat_fallbacks,
            max_new_tokens=wm_max_new_tokens, sample=True, temperature=0.7,
            max_pos=max_pos)
        grouped_actions = [[None] * 3 for _ in active_envs]
        grouped_confs = [[None] * 3 for _ in active_envs]
        for (local_idx, benign_idx), action, confidence in zip(
                query_index, benign_actions, benign_confs):
            grouped_actions[local_idx][benign_idx] = (
                int(action.next_position), action.message)
            grouped_confs[local_idx][benign_idx] = float(confidence)

        for local_idx, global_idx in enumerate(active_idx):
            env = envs[global_idx]
            before, visible, gen, atk_conf, snapshot_pos, snapshot_conf = pending[local_idx]
            env.prepare(int(gen.next_position))
            env._pending_atk_msg = gen.message
            env.confidences[3] = atk_conf
            _, _, terminated, truncated, _ = env.apply(grouped_actions[local_idx])
            for benign_idx, confidence in enumerate(grouped_confs[local_idx]):
                env.confidences[benign_idx] = confidence
            state = states[global_idx]
            state["ep_len"] += 1
            state["consensus"] = bool(terminated)
            state["done"] = bool(terminated or truncated or state["ep_len"] >= max_steps)
            if state["rounds"] is not None:
                state["rounds"].append({
                    "t": state["ep_len"] - 1,
                    "attacker_position": int(gen.next_position),
                    "attacker_message": gen.message,
                    "attacker_confidence": atk_conf,
                    "attacker_snapshot_position": snapshot_pos,
                    "attacker_snapshot_confidence": snapshot_conf,
                    "benign_positions_before": before,
                    "benign_positions_after": [int(env.pos[j]) for j in range(3)],
                    "benign_confidences_next": grouped_confs[local_idx],
                    "visible_others_per_benign": visible,
                    "attacker_filtered_by_benign": [
                        i for i in range(3) if 3 not in visible[i]],
                    "consensus_after": bool(terminated),
                })
    return states


def main() -> None:
    self_check()
    parser = argparse.ArgumentParser(description="Experiment 3 CP-WBFT evaluation")
    parser.add_argument("--wm_model_dir", required=True)
    parser.add_argument("--llm_model_dir", required=True)
    parser.add_argument("--rl_adapter", required=True)
    parser.add_argument("--attacks", default=",".join(ALLOWED_ATTACKS))
    parser.add_argument("--n_exp", type=int, default=200)
    parser.add_argument("--seed", type=int, default=9000)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--max_pos", type=int, default=20)
    parser.add_argument("--max_steps", type=int, default=10)
    parser.add_argument("--consensus_tol", type=float, default=0.0)
    parser.add_argument("--actor_max_message_tokens", type=int, default=48)
    parser.add_argument("--wm_max_new_tokens", type=int, default=200)
    parser.add_argument("--margin", type=float, default=0.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dump_json", required=True)
    parser.add_argument("--dump_n", type=int, default=200)
    args = parser.parse_args()

    kinds = [value.strip() for value in args.attacks.split(",") if value.strip()]
    unknown = sorted(set(kinds) - set(ALLOWED_ATTACKS))
    if unknown:
        parser.error(f"unsupported attacks {unknown}; allowed={ALLOWED_ATTACKS}")
    if args.eval_batch_size < 1 or args.n_exp < 1:
        parser.error("batch size and n_exp must be positive")

    ConfidenceCPEnv.default_margin = float(args.margin)
    print(f"[wm] loading {args.wm_model_dir} (HF bf16, confidence enabled)")
    wm_model, wm_tokenizer = load_causal_model(args.wm_model_dir, args.device)
    holder = ModelHolder(wm_model, wm_tokenizer, args.max_pos)
    print(f"[actor] loading {args.llm_model_dir} + {args.rl_adapter}")
    actor_base, actor_tokenizer = load_causal_model(args.llm_model_dir, args.device)
    actor = PeftModel.from_pretrained(
        actor_base, os.path.expanduser(args.rl_adapter)).eval()

    def make_env():
        return ConfidenceCPEnv(
            world_model=holder, max_pos=args.max_pos, max_steps=args.max_steps,
            consensus_tol=args.consensus_tol, consensus_penalty=-10.0,
            reward_mode="spread", attacker_msg_mode="template")

    initial_positions = []
    for episode in range(args.n_exp):
        rng = random.Random(args.seed + episode + args.max_steps * 1000)
        initial_positions.append([rng.randint(0, args.max_pos) for _ in range(4)])

    summaries, records = {}, []
    for kind in kinds:
        policy_seed = args.seed + 100_000
        random.seed(policy_seed)
        np.random.seed(policy_seed % (2 ** 32))
        torch.manual_seed(policy_seed)
        torch.cuda.manual_seed_all(policy_seed)
        results = []
        for start in range(0, args.n_exp, args.eval_batch_size):
            stop = min(start + args.eval_batch_size, args.n_exp)
            batch_results = run_batch(
                envs=[make_env() for _ in range(start, stop)],
                wm_model=wm_model, wm_tokenizer=wm_tokenizer,
                actor=actor, actor_tokenizer=actor_tokenizer, kind=kind,
                init_positions=initial_positions[start:stop],
                max_steps=args.max_steps, max_pos=args.max_pos,
                wm_max_new_tokens=args.wm_max_new_tokens,
                actor_max_message_tokens=args.actor_max_message_tokens,
                record=start < args.dump_n)
            results.extend(batch_results)
            for offset, result in enumerate(batch_results):
                episode = start + offset
                if episode < args.dump_n:
                    records.append({
                        "attacker": kind, "episode": episode,
                        "initial_positions": initial_positions[episode],
                        "consensus": result["consensus"],
                        "ep_len": result["ep_len"], "rounds": result["rounds"],
                    })
        consensus = sum(int(result["consensus"]) for result in results)
        mean_len = sum(result["ep_len"] for result in results) / len(results)
        summaries[kind] = {
            "episodes": len(results), "consensus": consensus,
            "rate": consensus / len(results), "mean_ep_len": mean_len,
        }

    print("\n" + "#" * 74)
    print("# EXPERIMENT 3: CP-WBFT confidence-aware evaluation")
    print("# attacks = prompt, model_base, model_rl; lower consensus = stronger")
    print("#" * 74)
    for kind in kinds:
        row = summaries[kind]
        print(f"attacker={kind:10s} episodes={row['episodes']:4d} "
              f"consensus={row['consensus']:4d} rate={row['rate']:.3f} "
              f"mean_ep_len={row['mean_ep_len']:.2f}")

    output = Path(args.dump_json).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment": 3,
        "defense": "cpwbft_position_token_decoder_peakedness",
        "margin": args.margin,
        "confidence_timing": "round-synchronous snapshot for all four agents",
        "wm_model_dir": args.wm_model_dir,
        "llm_model_dir": args.llm_model_dir,
        "rl_adapter": args.rl_adapter,
        "seed": args.seed, "n_exp": args.n_exp,
        "attacks": kinds, "summary": summaries, "episodes": records,
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                      encoding="utf-8")
    print(f"[dump] wrote {len(records)} trajectories -> {output}")


if __name__ == "__main__":
    main()
