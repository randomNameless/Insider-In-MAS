#!/usr/bin/env python3
"""Frozen Exp1 PPO message/position replacement, no defense; no training."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time

from channel_game import CHANNELS, PERSONALITIES, run_batch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
EVAL = REPO / "Experiment/Experiment1_Eval"
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("VLLM_BATCH_INVARIANT", "0")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_new(path, data):
    path = Path(path)
    with path.open("x", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)


class Tee:
    def __init__(self, console, log):
        self.console, self.log = console, log
    def write(self, text):
        self.console.write(text)
        self.log.write(text)
        self.log.flush()
    def flush(self):
        self.console.flush()
        self.log.flush()
    def __getattr__(self, name):
        return getattr(self.console, name)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--victim", required=True, choices=["wm", "qwen35_9b"])
    p.add_argument("--wm_model_dir")
    p.add_argument("--llm_model_dir", default=str(Path.home() / "LLaMA-Factory/saves/qwen3-4b/malicious-full-sft"))
    p.add_argument("--rl_adapter", default=str(REPO / "Experiment/Experiment1_Main/runs/exp1_nodefense_qwen3_ppo_dominant_v4/best"))
    p.add_argument("--env-module", help="Exact path to the existing RL_attack_env_qwen.py")
    p.add_argument("--seeds", nargs="+", type=int, default=[9000, 9001, 9002, 9003, 9004])
    p.add_argument("--conditions", nargs="+", choices=list(CHANNELS), default=list(CHANNELS))
    p.add_argument("--n_exp", type=int, default=200)
    p.add_argument("--eval_batch_size", type=int, default=16)
    p.add_argument("--wm_gpu_mem_util", type=float, default=0.60)
    p.add_argument("--max_model_len", type=int, default=2048)
    p.add_argument("--enforce_eager", action="store_true")
    p.add_argument("--out", default=str(HERE / "results/channel_v1"))
    p.add_argument("--check", action="store_true", help="Resolve files/hashes without loading models")
    a = p.parse_args()
    if a.n_exp < 1 or a.eval_batch_size < 1 or len(a.seeds) != len(set(a.seeds)):
        p.error("Require positive episode/batch counts and unique seeds")
    if len(a.conditions) != len(set(a.conditions)):
        p.error("Duplicate conditions")
    if not a.wm_model_dir:
        a.wm_model_dir = str(Path.home() / "models" / ("qwen35-wm" if a.victim == "wm" else "qwen3.5-9b"))
    return a


def preflight(a):
    for name in ("wm_model_dir", "llm_model_dir", "rl_adapter"):
        setattr(a, name, str(Path(getattr(a, name)).expanduser().resolve()))
    for directory in (a.wm_model_dir, a.llm_model_dir):
        path = Path(directory)
        if not (path / "config.json").is_file() or not list(path.glob("*.safetensors")):
            raise FileNotFoundError(f"Missing model config/weights: {path}")
    base_cfg = json.loads((Path(a.llm_model_dir) / "config.json").read_text())
    if base_cfg.get("model_type") != "qwen3":
        raise ValueError("Exp1 V4 requires the Qwen3 malicious full-SFT base")
    adapter = Path(a.rl_adapter)
    config = json.loads((adapter / "adapter_config.json").read_text())
    if config.get("peft_type") != "LORA":
        raise ValueError("Expected a complete LoRA adapter")
    weights = adapter / "adapter_model.safetensors"
    if not weights.is_file() or not weights.stat().st_size:
        raise FileNotFoundError(weights)
    sys.path[:0] = [str(REPO), str(EVAL)]
    if a.env_module:
        external = Path(a.env_module).expanduser().resolve()
        if external.name != "RL_attack_env_qwen.py" or not external.is_file():
            raise ValueError("--env-module must name an existing RL_attack_env_qwen.py")
        sys.path.insert(0, str(external.parent))
    spec = importlib.util.find_spec("RL_attack_env_qwen")
    if spec is None or not spec.origin:
        raise FileNotFoundError("Find the existing RL_attack_env_qwen.py and pass --env-module; do not substitute a new game")
    external = Path(spec.origin).resolve()
    sources = list((REPO / "attacker").glob("*.py")) + list(HERE.glob("*.py"))
    sources += [EVAL / "eval_attacks_consensus.py", external]
    tracked = sources + [weights, adapter / "adapter_config.json"]
    fingerprints = {str(p.resolve()): sha(p) for p in tracked}
    print(f"[BASE] {a.llm_model_dir}", flush=True)
    print(f"[PPO ADAPTER] {a.rl_adapter}\n[SHA256] {fingerprints[str(weights)]}", flush=True)
    print(f"[adapter declared base] {config.get('base_model_name_or_path')}", flush=True)
    print(f"[VICTIM] {a.wm_model_dir}\n[GAME MODULE] {external}", flush=True)
    print(f"[PLAN] {len(a.seeds)} seeds x {len(a.conditions)} conditions x {a.n_exp} episodes; sync, horizon=10, scored=0,1,2", flush=True)
    inventories = {}
    for directory in (a.wm_model_dir, a.llm_model_dir):
        p = Path(directory)
        inventories[directory] = {"config_sha256": sha(p / "config.json"),
                                  "weights_sizes_not_hashes": {x.name: x.stat().st_size for x in sorted(p.glob("*.safetensors"))}}
    return {"fingerprints": fingerprints, "models": inventories,
            "external_env": str(external)}


def ensure_unchanged(provenance):
    for path, expected in provenance["fingerprints"].items():
        if sha(path) != expected:
            raise RuntimeError(f"Source/adapter changed during run: {path}")


def execute(a, provenance, output):
    import numpy as np
    import torch
    import RL_attack_env_qwen as game
    from attacker.prompts import build_chat
    from eval_attacks_consensus import HFQwen3, build_model_rl_query, benign_predict_many
    if str(Path(game.__file__).resolve()) != provenance["external_env"]:
        raise RuntimeError("Imported a different game module")
    wm = game.QwenWorldModelVLLM(
        model_dir=a.wm_model_dir, max_pos=20, max_new_tokens=200,
        gpu_memory_utilization=a.wm_gpu_mem_util, max_model_len=a.max_model_len,
        enforce_eager=a.enforce_eager, chat_template_kwargs={"enable_thinking": False},
        temperature=0.7, seed=a.seeds[0])
    if not hasattr(wm, "sampling"):
        raise RuntimeError("Victim wrapper has no SamplingParams; cannot reset seed")
    llm = HFQwen3(a.llm_model_dir, adapter_dir=a.rl_adapter, device="cuda",
                  max_new_tokens=48, temperature=1.0, seed=a.seeds[0])

    def make_env():
        return game.AttackEnv(world_model=wm, max_pos=20, max_steps=10,
                              consensus_tol=0.0, consensus_penalty=-10.0,
                              reward_mode="spread", attacker_msg_mode="template")

    def build_inputs(env):
        # Construct ONCE and keep the exact chats passed to both generators.
        return {"ppo_agent3": build_model_rl_query(env),
                "victim_agents_0_to_3": [env._build_messages(i) for i in range(3)] + [
                    build_chat(3, "neutral", [int(x) for x in env.pos],
                               list(env.last_messages), int(env.t), msg_order="index")],
                "fallback_positions": [int(x) for x in env.pos]}

    def predict_ppo(inputs):
        raws = llm.generate_two_phase_batch([x["ppo_agent3"] for x in inputs],
                                           use_adapter=True, temperature=1.0)
        return [(x["fallback_positions"][3] if pos is None else max(0, min(20, int(pos))), msg)
                for x, (msg, pos) in zip(inputs, raws)]

    def predict_victim(inputs):
        return benign_predict_many(wm, [list(zip(x["victim_agents_0_to_3"],
                                                 x["fallback_positions"])) for x in inputs])

    for seed in a.seeds:
        ensure_unchanged(provenance)
        initials = []
        for idx in range(a.n_exp):
            er = random.Random(seed + idx + 10 * 1000)
            initials.append([er.randint(0, 20) for _ in range(4)])
        summaries, records = {}, []
        for condition in a.conditions:
            started = time.monotonic()
            random.seed(seed + 100_000)
            np.random.seed((seed + 100_000) % 2**32)
            torch.manual_seed(seed + 100_000)
            torch.cuda.manual_seed_all(seed + 100_000)
            wm.sampling.seed = seed + 200_000
            current = []
            for start in range(0, a.n_exp, a.eval_batch_size):
                stop = min(a.n_exp, start + a.eval_batch_size)
                states = run_batch([make_env() for _ in range(start, stop)],
                                   initials[start:stop], condition, build_inputs,
                                   predict_ppo, predict_victim)
                for idx, state in zip(range(start, stop), states):
                    current.append({"condition": condition, "exp_idx": idx,
                                    "initial_positions": initials[idx], **state})
                print(f"[seed {seed} {condition}] completed {stop}/{a.n_exp}", flush=True)
            count = sum(ep["consensus"] for ep in current)
            summary = {"episodes": len(current), "consensus": count,
                       "rate": count / len(current),
                       "mean_ep_len": sum(ep["ep_len"] for ep in current) / len(current)}
            summaries[condition] = summary
            records.extend(current)
            ensure_unchanged(provenance)
            # Completed conditions survive a later crash; never overwrite them.
            write_new(output / f"seed{seed}_{condition}.partial.json",
                      {"seed": seed, "condition": condition, "summary": summary, "episodes": current})
            print(f"condition={condition:20s} episodes={len(current):4d} consensus={count:4d} "
                  f"rate={summary['rate']:.3f} mean_ep_len={summary['mean_ep_len']:.2f} "
                  f"seconds={time.monotonic()-started:.1f}", flush=True)
        write_new(output / f"eval_seed{seed}.json", {
            "experiment": 8, "protocol": "channel_replacement_v1", "defense": "none",
            "round_timing": "synchronous_snapshot_atomic_commit", "seed": seed,
            "victim": a.victim, "n_exp": a.n_exp, "max_steps": 10,
            "consensus_scope": [0, 1, 2], "personalities": PERSONALITIES,
            "channels": CHANNELS, "settings": vars(a), "provenance": provenance,
            "summary": summaries, "episodes": records})
        print(f"[dump] {output / f'eval_seed{seed}.json'}", flush=True)


def main():
    a = parse_args()
    provenance = preflight(a)
    if a.check:
        return
    output = Path(a.out).expanduser().resolve() / a.victim
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output is not empty; use a NEW --out to preserve results: {output}")
    output.mkdir(parents=True, exist_ok=True)
    provenance["python"] = platform.python_version()
    provenance["victim_engine_initial_seed"] = a.seeds[0]
    provenance["packages"] = subprocess.check_output([sys.executable, "-m", "pip", "freeze", "--all"], text=True)
    write_new(output / "launch.json", {"settings": vars(a), "provenance": provenance})
    # Python logs for the entire victim session; native engine stderr may remain console-only.
    with (output / "evaluation.log").open("x", encoding="utf-8") as log:
        stdout, stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = Tee(stdout, log), Tee(stderr, log)
        try:
            print(json.dumps({"settings": vars(a), "provenance": provenance}, indent=2), flush=True)
            execute(a, provenance, output)
        except BaseException:
            import traceback
            traceback.print_exc()
            raise
        finally:
            sys.stdout, sys.stderr = stdout, stderr


if __name__ == "__main__":
    main()
