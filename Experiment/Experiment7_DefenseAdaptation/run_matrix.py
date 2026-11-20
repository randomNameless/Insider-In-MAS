#!/usr/bin/env python3
"""Frozen 4x4 adapter/defense matrix; stdlib planning and isolated evaluators."""
from __future__ import annotations

import argparse
import hashlib
import importlib.machinery
import json
import math
import os
from pathlib import Path
import platform
import random
import shlex
import subprocess
import sys
import time
from collections import defaultdict

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SCRIPTS = {
    "no_defense": "Experiment1_Eval/eval_attacks_consensus.py",
    "wmsr": "Experiment2_WMSR/eval_exp2_wmsr.py",
    "cpwbft": "Experiment3_CPWBFT/eval_exp3_cpwbft.py",
    "sac": "Experiment4_SAC/eval_exp4_sac.py",
}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_config(path, seen=None):
    path = Path(path).expanduser().resolve()
    seen = set() if seen is None else set(seen)
    require(path not in seen, f"Config inheritance cycle: {path}")
    seen.add(path)
    cfg = read(path)
    parent = cfg.pop("base_config", None)
    if parent:
        inherited = load_config(path.parent / parent, seen)
        inherited.update(cfg)
        return inherited
    return cfg


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def resolve(value):
    return str(Path(value.format(repo=REPO, home=Path.home())).expanduser().resolve())


def model_info(path):
    path = Path(path)
    cfg = path / "config.json"
    require(cfg.is_file(), f"Missing model config: {cfg}")
    require(any(path.glob("*.safetensors")) or any(path.glob("pytorch_model*.bin")),
            f"No model weights found in {path}")
    return {"path": str(path), "config_sha256": sha(cfg),
            "model_type": read(cfg).get("model_type"),
            "weight_inventory": [{"name": p.name, "bytes": p.stat().st_size}
                                 for p in sorted(path.iterdir())
                                 if p.suffix in (".safetensors", ".bin")]}


def adapter_info(path):
    path = Path(path)
    cfg = read(path / "adapter_config.json")
    require(cfg.get("peft_type") == "LORA", f"Not a LoRA: {path}")
    weight = path / "adapter_model.safetensors"
    require(weight.is_file() and weight.stat().st_size > 0, f"Missing weights: {weight}")
    return {"path": str(path), "weight_sha256": sha(weight),
            "config_sha256": sha(path / "adapter_config.json"),
            "declared_base": cfg.get("base_model_name_or_path"),
            "rank": cfg.get("r"), "target_modules": cfg.get("target_modules")}


def command(cfg, defense, victim, adapter, seed, output):
    script = REPO / "Experiment" / SCRIPTS[defense]
    cmd = [sys.executable, "-u", str(script),
           "--wm_model_dir", cfg["victims"][victim],
           "--llm_model_dir", cfg["base_model"],
           "--rl_adapter", cfg["adapters"][adapter], "--attacks", "model_rl",
           "--seed", str(seed), "--n_exp", str(cfg["n_exp"]),
           "--eval_batch_size", str(cfg["batch_size"]),
           "--max_pos", str(cfg["max_pos"]), "--max_steps", str(cfg["max_steps"]),
           "--wm_max_new_tokens", str(cfg["wm_tokens"]),
           "--dump_n", str(cfg["n_exp"]), "--dump_json", str(output)]
    token_flag = "--max_new_tokens" if defense in ("no_defense", "wmsr") else "--actor_max_message_tokens"
    cmd += [token_flag, str(cfg["actor_message_tokens"])]
    if defense != "cpwbft":
        cmd += ["--wm_backend", "vllm", "--wm_gpu_mem_util", str(cfg["gpu_mem_util"]),
                "--max_model_len", str(cfg["max_model_len"]), "--log_every", "40"]
    if defense in ("no_defense", "wmsr"):
        cmd += ["--consensus_tol", "0", "--wm_temperature", "0.7",
                "--llm_temperature", "0.7"]  # model_rl itself samples at 1.0.
    if defense == "wmsr":
        cmd += ["--wmsr_F", "1"]
    if defense == "cpwbft":
        cmd += ["--margin", "0", "--consensus_tol", "0"]
    if defense == "sac":
        cmd += ["--scorer_max_new_tokens", "8"]
    return cmd


def validate_result(path, spec):
    d = read(path)
    expected_defense = {"no_defense": "none", "wmsr": "paper_global_msr f=1",
                        "cpwbft": "cpwbft_position_token_decoder_peakedness", "sac": "SAC_F1_complete4"}
    require(d.get("defense") == expected_defense[spec["defense"]], "Wrong defense result")
    require(set(d["summary"]) == {"model_rl"}, "Expected only model_rl")
    eps = d["episodes"]
    cfg, seed = spec["settings"], spec["seed"]
    n, limit = cfg["n_exp"], cfg["max_steps"]
    require(len(eps) == n, "Incomplete episode dump")
    ids = []
    for e in eps:
        require(e["attacker"] == "model_rl", "Unexpected attacker")
        idx = e.get("episode", e.get("exp_idx"))
        require(type(idx) is int and 0 <= idx < n, "Invalid episode ID")
        ids.append(idx)
        require(type(e["consensus"]) is bool, "Invalid consensus type")
        t = e["ep_len"]
        require(type(t) is int and 0 <= t <= limit, "Invalid episode length")
        require(e["consensus"] or t == limit, "Early nonconsensus termination")
        require(len(e["rounds"]) == t, "Incomplete trajectory")
        rng = random.Random(seed + idx + limit * 1000)
        expected = [rng.randint(0, cfg["max_pos"]) for _ in range(4)]
        require(e.get("initial_positions", e.get("init_positions")) == expected,
                "Initial-state pool mismatch")
    require(len(set(ids)) == n, "Duplicate episode IDs")
    s = d["summary"]["model_rl"]
    count = sum(e["consensus"] for e in eps)
    length = sum(e["ep_len"] for e in eps) / n
    require(s["episodes"] == n and s["consensus"] == count, "Count mismatch")
    require(math.isclose(s["rate"], count / n, abs_tol=1e-8), "Rate mismatch")
    require(math.isclose(s["mean_ep_len"], length, abs_tol=1e-8), "Length mismatch")
    if spec["defense"] == "cpwbft":
        require("synchronous" in d.get("confidence_timing", ""), "Missing sync confidence timing")
    else:
        require(d.get("round_timing") == "synchronous_snapshot_atomic_commit", "Missing sync timing")
    return s


def source_info(env_module):
    roots = [REPO / "attacker", HERE] + [REPO / "Experiment" / name for name in
             ("Experiment1_Eval", "Experiment2_WMSR", "Experiment3_CPWBFT", "Experiment4_SAC")]
    paths = sorted({p for root in roots for p in root.rglob("*.py")})
    info = {str(p.relative_to(REPO)): sha(p) for p in paths}
    info[str(env_module)] = sha(env_module)
    return info


def find_env(explicit):
    if explicit:
        p = Path(explicit).expanduser().resolve()
        require(p.name == "RL_attack_env_qwen.py" and p.is_file(), "Invalid --env-module")
        return p
    search = [str(REPO / "Experiment/Experiment1_Eval"), str(REPO), str(Path.cwd())] + sys.path
    spec = importlib.machinery.PathFinder.find_spec("RL_attack_env_qwen", search)
    require(spec is not None and spec.origin, "Cannot locate RL_attack_env_qwen.py; supply --env-module /absolute/path/RL_attack_env_qwen.py")
    return Path(spec.origin).resolve()


def verified_complete(output, spec):
    """Never infer successful completion from a merely nonempty JSON."""
    manifest = output.with_suffix(".manifest.json")
    if not manifest.exists() and not output.exists():
        return False
    require(manifest.exists() and output.exists(), f"Incomplete job at {output}; preserve it and use a NEW --out")
    old = read(manifest)
    require(old.get("status") == "complete" and old.get("job_id") == digest(spec),
            f"Config/code/environment changed or job incomplete: {output}; use a NEW --out")
    require(old["result_sha256"] == sha(output), f"Result checksum changed: {output}")
    validate_result(output, spec)
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=str(HERE / "matrix_config.json"))
    p.add_argument("--adapters", nargs="+")
    p.add_argument("--victims", nargs="+")
    p.add_argument("--defenses", nargs="+", choices=list(SCRIPTS))
    p.add_argument("--seeds", type=int, nargs="+")
    p.add_argument("--n-exp", type=int)
    p.add_argument("--batch-size", type=int)
    execution = p.add_mutually_exclusive_group()
    execution.add_argument("--reuse-seeds", action="store_true", help="Keep models for one victim/adapter/defense across seeds (separate protocol)")
    execution.add_argument("--fresh-processes", action="store_true", help="Disable model reuse, including in fast config (benchmark/fallback)")
    p.add_argument("--out")
    p.add_argument("--env-module", help="Actual external RL_attack_env_qwen.py")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--check", action="store_true", help="Default: validate paths, no GPU work or writes")
    p.add_argument("--print-commands", action="store_true")
    a = p.parse_args()
    cfg = load_config(a.config)
    if a.reuse_seeds:
        cfg["reuse_seed_models"] = True
    if a.fresh_processes:
        cfg["reuse_seed_models"] = False
    require(cfg["schema"] == 1, "Unsupported config schema")
    for key in ("base_model", "output"):
        cfg[key] = resolve(cfg[key])
    for key in ("adapters", "victims"):
        cfg[key] = {k: resolve(v) for k, v in cfg[key].items()}
    for key, value in (("n_exp", a.n_exp), ("batch_size", a.batch_size), ("seeds", a.seeds)):
        if value is not None:
            cfg[key] = value
    if a.out:
        cfg["output"] = resolve(a.out)
    adapters, victims, defenses = a.adapters or list(cfg["adapters"]), a.victims or list(cfg["victims"]), a.defenses or cfg["defenses"]
    for values, allowed in ((adapters, cfg["adapters"]), (victims, cfg["victims"]), (defenses, SCRIPTS)):
        require(bool(values) and len(values) == len(set(values)) and set(values) <= set(allowed), "Invalid/duplicate selection")
    require(cfg["n_exp"] > 0 and cfg["batch_size"] > 0 and cfg["max_steps"] > 0, "Invalid run size")
    require(cfg["seeds"] and len(set(cfg["seeds"])) == len(cfg["seeds"]), "Duplicate/empty seeds")
    base = model_info(cfg["base_model"])
    require(base["model_type"] == "qwen3", "These four adapters require the Qwen3 full-SFT base, not a Qwen3.5 victim")
    print(f"SFT BASE: {base['path']}\nPYTHON: {sys.executable}", flush=True)
    adapters_info = {}
    for name in adapters:
        info = adapters_info[name] = adapter_info(cfg["adapters"][name])
        print(f"\nADAPTER {name}\n  path: {info['path']}\n  SHA256: {info['weight_sha256']}\n"
              f"  declared base: {info['declared_base']}\n  rank: {info['rank']}\n"
              f"  actual load base: {base['path']}", flush=True)
        if info["declared_base"] != base["path"]:
            print("  WARNING: base label differs; verify this is the SAME full-SFT base, not raw Qwen3.")
    require(len({v["weight_sha256"] for v in adapters_info.values()}) == len(adapters_info),
            "Two adapter rows have identical weights; verify paths before running")
    victims_info = {v: model_info(cfg["victims"][v]) for v in victims}
    for name, info in victims_info.items():
        print(f"VICTIM {name}: {info['path']}")
    env_module = find_env(a.env_module)
    for directory in [REPO, REPO / "Experiment/Experiment1_Eval"] + [
            (REPO / "Experiment" / SCRIPTS[d]).parent for d in defenses]:
        shadow = directory / "RL_attack_env_qwen.py"
        require(not shadow.exists() or shadow.resolve() == env_module,
                f"Conflicting game module {shadow}; use the evaluator's actual module")
    sources = source_info(env_module)
    print(f"GAME MODULE: {env_module}\nCP-WBFT: HF victim + HF actor; other columns: vLLM victim + HF actor.")
    settings = {k: cfg[k] for k in ("n_exp", "batch_size", "max_steps", "max_pos", "actor_message_tokens", "wm_tokens", "gpu_mem_util", "max_model_len")}
    if cfg.get("reuse_seed_models", False):
        settings["execution_mode"] = "reuse_seeds_v2"
    jobs = []
    for victim in victims:
        for adapter in adapters:
            for defense in defenses:
                for seed in cfg["seeds"]:
                    output = Path(cfg["output"]) / victim / adapter / defense / f"eval_seed{seed}.json"
                    spec = {"experiment": 7, "protocol": "defense_adaptation_v1", "victim": victim,
                            "adapter": adapter, "defense": defense, "seed": seed, "settings": settings,
                            "base": base, "adapter_info": adapters_info[adapter], "victim_info": victims_info[victim],
                            "source_sha256": sources, "python": sys.executable,
                            "python_version": platform.python_version()}
                    cmd = command(cfg, defense, victim, adapter, seed, output)
                    jobs.append((output, spec, cmd))
                    if a.print_commands:
                        print(shlex.join(cmd))
    print(f"\nJOBS={len(jobs)} EPISODES={len(jobs)*cfg['n_exp']} SEEDS={cfg['seeds']}\nOUTPUT={cfg['output']}", flush=True)
    if cfg.get("reuse_seed_models", False):
        print(f"FAST: batch={cfg['batch_size']}; {len(victims)*len(adapters)*len(defenses)} model sessions instead of {len(jobs)}.\n"
              "Only model objects are reused; no generated answers or game states are cached.", flush=True)
    if not a.run:
        print("CHECK ONLY: no models loaded, no files written. Use --run after verifying the adapters.")
        return
    out = Path(cfg["output"])
    out.mkdir(parents=True, exist_ok=True)
    # A fresh process per cell/seed prevents monkey patches or adapters leaking.
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(env_module.parent), str(REPO), str(REPO / "Experiment/Experiment1_Eval"), env.get("PYTHONPATH", "")])
    packages = subprocess.check_output([sys.executable, "-m", "pip", "freeze", "--all"], text=True)
    for _, spec, _ in jobs:
        spec["environment_freeze"] = packages
    stamp = time.strftime("%Y%m%d_%H%M%S")
    write(out / f"launch_{stamp}.json", {"config": cfg, "selected_adapters": adapters,
          "selected_defenses": defenses, "selected_victims": victims, "game_module": str(env_module),
          "packages": packages, "source_sha256": sources})
    if cfg.get("reuse_seed_models", False):
        sessions = defaultdict(list)
        for output, spec, cmd in jobs:
            if verified_complete(output, spec):
                print(f"[verified skip] {output}", flush=True)
            else:
                sessions[spec["victim"], spec["adapter"], spec["defense"]].append(
                    {"output": str(output), "spec": spec, "command": cmd})
        session_dir = out / "sessions"
        session_dir.mkdir(exist_ok=True)
        for key, tasks in sessions.items():
            require(source_info(env_module) == sources, "Source changed during run")
            session = session_dir / ("_".join(key) + f"_{time.time_ns()}.json")
            write(session, {"jobs": tasks, "env_module": str(env_module)})
            print(f"[SESSION] {' / '.join(key)} seeds={[t['spec']['seed'] for t in tasks]}", flush=True)
            subprocess.run([sys.executable, "-u", str(HERE / "seed_worker.py"), str(session)],
                           cwd=str(HERE), env=env, check=True)
        print("Fast matrix selection complete. Run summarize_matrix.py to build the tables.")
        return
    for output, spec, cmd in jobs:
        output.parent.mkdir(parents=True, exist_ok=True)
        manifest = output.with_suffix(".manifest.json")
        job_id = digest(spec)
        if verified_complete(output, spec):
            print(f"[verified skip] {output}", flush=True)
            continue
        require(source_info(env_module) == sources, "Source changed during run; stop and use a new output")
        require(adapter_info(spec["adapter_info"]["path"]) == spec["adapter_info"], "Adapter changed during run")
        record = {"job_id": job_id, "status": "running", "spec": spec, "command": cmd,
                  "cwd": str(Path(cmd[2]).parent), "started": time.time()}
        write(manifest, record)
        print(f"\n[RUN] {spec['victim']} / {spec['adapter']} -> {spec['defense']} seed={spec['seed']}\n"
              f"ADAPTER={spec['adapter_info']['path']}\nSHA256={spec['adapter_info']['weight_sha256']}", flush=True)
        try:
            with output.with_suffix(".log").open("x", encoding="utf-8") as log:
                log.write(json.dumps(record, ensure_ascii=False) + "\n")
                log.flush()
                with subprocess.Popen(cmd, cwd=record["cwd"], env=env, stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace") as proc:
                    for line in proc.stdout:
                        print(line, end="", flush=True)
                        log.write(line)
                        log.flush()
                    require(proc.wait() == 0, f"Evaluator failed; see {output.with_suffix('.log')}")
            require(source_info(env_module) == sources, "Source changed during evaluation")
            require(adapter_info(spec["adapter_info"]["path"]) == spec["adapter_info"], "Adapter changed during evaluation")
            record.update(status="complete", summary=validate_result(output, spec), result_sha256=sha(output), finished=time.time())
        except BaseException as exc:
            record.update(status="failed", error=str(exc), finished=time.time())
            write(manifest, record)
            raise
        write(manifest, record)
        elapsed = record["finished"] - record["started"]
        print(f"[TIMING] {elapsed:.1f}s, {cfg['n_exp']/elapsed:.3f} episodes/s (including model load)", flush=True)
    print("Matrix selection complete. Run summarize_matrix.py to build the tables.")


if __name__ == "__main__":
    main()
