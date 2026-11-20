#!/usr/bin/env python3
"""One isolated model session per victim/adapter/defense; retain only weights.

Evaluator main() remains responsible for fresh games, summaries, counters and
per-seed Python/NumPy/Torch/vLLM sampling resets. No generated-text caching.
"""
from __future__ import annotations

import contextlib
import importlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

from run_matrix import (REPO, SCRIPTS, adapter_info, digest, read, require,
                        sha, source_info, validate_result, verified_complete, write)


def freeze(value):
    if isinstance(value, dict):
        return tuple(sorted((k, freeze(v)) for k, v in value.items()))
    if isinstance(value, (tuple, list)):
        return tuple(freeze(v) for v in value)
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    return (type(value).__name__, id(value))


class ModelCache:
    """Memoize constructors only. Unexpected construction parameters fail closed."""
    def __init__(self, factory, label, ignore_seed=False, multiple=False):
        self.factory, self.label = factory, label
        self.ignore_seed, self.multiple = ignore_seed, multiple
        self.values = {}

    def __call__(self, *args, **kwargs):
        key_kwargs = {k: v for k, v in kwargs.items() if not (self.ignore_seed and k == "seed")}
        key = freeze((args, key_kwargs))
        if key in self.values:
            print(f"[MODEL REUSE] {self.label}; evaluator will reset sampling for this seed", flush=True)
            return self.values[key]
        require(not self.values or self.multiple,
                f"Model construction changed inside session: {self.label}")
        result = self.factory(*args, **kwargs)
        self.values[key] = result
        return result


def install_caches(module, defense):
    """Patch the evaluator's constructor references, never its round loop."""
    if defense == "cpwbft":
        # Two distinct full models (victim + actor) and exactly one PEFT wrapper.
        # Returning the same PEFT wrapper prevents stacking the LoRA on each seed.
        module.load_causal_model = ModelCache(module.load_causal_model, "HF victim/actor", multiple=True)
        loader = module.PeftModel.from_pretrained
        module.PeftModel = SimpleNamespace(from_pretrained=ModelCache(loader, "CP-WBFT LoRA wrapper"))
    else:
        target = module.base_eval if defense == "wmsr" else module
        target.QwenWorldModelVLLM = ModelCache(target.QwenWorldModelVLLM, "vLLM victim", ignore_seed=True)
        target.HFQwen3 = ModelCache(target.HFQwen3, "HF actor + fixed LoRA", ignore_seed=True)
        # SACBenignEngineVLLM is NOT cached: scores/stats/filter state start fresh.


class Tee:
    def __init__(self, console, log):
        self.console, self.log = console, log

    def write(self, text):
        self.console.write(text)
        if self.log is not None and not self.log.closed:
            self.log.write(text)
        return len(text)

    def flush(self):
        self.console.flush()
        if self.log is not None and not self.log.closed:
            self.log.flush()

    def isatty(self):
        return False

    def __getattr__(self, name):
        return getattr(self.console, name)


def session_identity(spec):
    return {k: v for k, v in spec.items() if k != "seed"}


def run_session(session):
    jobs = session["jobs"]
    require(bool(jobs), "Empty session")
    first = jobs[0]["spec"]
    require(first["settings"].get("execution_mode") == "reuse_seeds_v2", "Not a reuse protocol")
    require(all(session_identity(j["spec"]) == session_identity(first) for j in jobs),
            "A model session cannot mix victims, adapters, defenses, settings or source versions")
    require(len({j["spec"]["seed"] for j in jobs}) == len(jobs), "Repeated seed in session")
    env_module = Path(session["env_module"])
    defense = first["defense"]
    script = REPO / "Experiment" / SCRIPTS[defense]
    module = None
    cwd, argv = Path.cwd(), list(sys.argv)
    # Logging handlers created at first import may retain these stream objects.
    # Retarget them, rather than leaving handlers attached to a closed seed log.
    stdout, stderr = Tee(sys.stdout, None), Tee(sys.stderr, None)
    try:
        for task in jobs:
            spec, cmd, output = task["spec"], task["command"], Path(task["output"])
            require(Path(cmd[2]).resolve() == script.resolve(), "Wrong evaluator in worker")
            require(cmd[cmd.index("--seed") + 1] == str(spec["seed"]), "Command seed mismatch")
            require(cmd[cmd.index("--rl_adapter") + 1] == spec["adapter_info"]["path"], "Command adapter mismatch")
            if verified_complete(output, spec):
                print(f"[verified skip] {output}", flush=True)
                continue
            require(source_info(env_module) == spec["source_sha256"], "Source changed")
            require(adapter_info(spec["adapter_info"]["path"]) == spec["adapter_info"], "Adapter changed")
            output.parent.mkdir(parents=True, exist_ok=True)
            manifest = output.with_suffix(".manifest.json")
            started = time.time()
            record = {"job_id": digest(spec), "status": "running", "spec": spec,
                      "command": cmd, "cwd": str(script.parent), "started": started,
                      "session_seed_order": [j["spec"]["seed"] for j in jobs],
                      "first_seed_in_session": jobs[0]["spec"]["seed"],
                      "execution": "seed_worker evaluator.main with constructor-only reuse"}
            write(manifest, record)
            try:
                with output.with_suffix(".log").open("x", encoding="utf-8") as log:
                    log.write(json.dumps(record, ensure_ascii=False) + "\n")
                    stdout.log = stderr.log = log
                    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                        print(f"\n[RUN] {spec['victim']} / {spec['adapter']} -> {defense} seed={spec['seed']}\n"
                              f"ADAPTER={spec['adapter_info']['path']}\nSHA256={spec['adapter_info']['weight_sha256']}", flush=True)
                        if module is None:
                            sys.path.insert(0, str(script.parent))
                            module = importlib.import_module(script.stem)
                            actual_env = sys.modules.get("RL_attack_env_qwen")
                            require(actual_env is not None and Path(actual_env.__file__).resolve() == env_module.resolve(),
                                    "Imported game module differs from the audited path")
                            install_caches(module, defense)
                        # Original evaluator starts a completely new set of games.
                        os.chdir(script.parent)
                        sys.argv = cmd[2:]
                        module.main()
                require(source_info(env_module) == spec["source_sha256"], "Source changed during evaluation")
                require(adapter_info(spec["adapter_info"]["path"]) == spec["adapter_info"], "Adapter changed during evaluation")
                record.update(status="complete", summary=validate_result(output, spec),
                              result_sha256=sha(output), finished=time.time())
            except BaseException as exc:
                record.update(status="failed", error=str(exc), finished=time.time())
                write(manifest, record)
                raise
            write(manifest, record)
            elapsed = record["finished"] - started
            print(f"[TIMING] seed={spec['seed']} {elapsed:.1f}s, "
                  f"{spec['settings']['n_exp']/elapsed:.3f} episodes/s; first seed includes model loading", flush=True)
    finally:
        stdout.log = stderr.log = None
        os.chdir(cwd)
        sys.argv = argv


if __name__ == "__main__":
    require(len(sys.argv) == 2, "Usage: seed_worker.py SESSION.json (normally launched by run_matrix.py)")
    run_session(read(sys.argv[1]))
