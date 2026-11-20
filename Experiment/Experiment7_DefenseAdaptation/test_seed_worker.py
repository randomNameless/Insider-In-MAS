"""Model-double integration tests, no GPU dependencies."""
import contextlib
import copy
import io
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from run_matrix import HERE, REPO, adapter_info, command, load_config, main, read, source_info, write
from seed_worker import ModelCache, Tee, install_caches, run_session


class CacheTests(unittest.TestCase):
    def test_tee_retargets_retained_logging_stream(self):
        console, first, second = io.StringIO(), io.StringIO(), io.StringIO()
        stream = Tee(console, first)
        stream.write("seed1")
        first.close()
        stream.write("between")
        stream.flush()
        stream.log = second
        stream.write("seed2")
        self.assertEqual(second.getvalue(), "seed2")
        self.assertEqual(console.getvalue(), "seed1betweenseed2")

    def test_only_seed_can_change(self):
        calls = []
        def factory(**kw):
            calls.append(kw)
            return object()
        cache = ModelCache(factory, "test", ignore_seed=True)
        a = cache(path="/model", seed=1, config={"a": 2})
        self.assertIs(a, cache(path="/model", seed=2, config={"a": 2}))
        self.assertEqual(len(calls), 1)
        with self.assertRaises(ValueError):
            cache(path="/other_model", seed=3, config={"a": 2})

    def test_cp_never_stacks_adapter(self):
        loaded, wrapped = [], []
        def load(*args):
            loaded.append(args)
            return object(), object()
        def wrap(*args):
            wrapped.append(args)
            return object()
        module = SimpleNamespace(load_causal_model=load, PeftModel=SimpleNamespace(from_pretrained=wrap))
        install_caches(module, "cpwbft")
        for _ in range(5):
            victim = module.load_causal_model("victim", "cuda")
            actor, tok = module.load_causal_model("base", "cuda")
            result = module.PeftModel.from_pretrained(actor, "adapter")
        self.assertEqual(len(loaded), 2)
        self.assertEqual(len(wrapped), 1)
        with self.assertRaises(ValueError):
            module.PeftModel.from_pretrained(actor, "DIFFERENT ADAPTER")

    def test_sac_engine_is_not_cached(self):
        module = SimpleNamespace(QwenWorldModelVLLM=lambda **k: object(), HFQwen3=lambda **k: object(),
                                 SACBenignEngineVLLM=lambda **k: object())
        engine = module.SACBenignEngineVLLM
        install_caches(module, "sac")
        self.assertIs(module.SACBenignEngineVLLM, engine)

    def test_wmsr_patch_only_its_base_module(self):
        base = SimpleNamespace(QwenWorldModelVLLM=lambda **k: object(), HFQwen3=lambda **k: object())
        module = SimpleNamespace(base_eval=base)
        install_caches(module, "wmsr")
        self.assertIsInstance(base.HFQwen3, ModelCache)


class WorkerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.model = self.root / "base"
        self.model.mkdir()
        write(self.model / "config.json", {"model_type": "qwen3"})
        (self.model / "model.safetensors").write_bytes(b"not a real model")
        self.adapter = self.root / "adapter"
        self.adapter.mkdir()
        write(self.adapter / "adapter_config.json", {"peft_type": "LORA", "r": 32})
        (self.adapter / "adapter_model.safetensors").write_bytes(b"not real weights")
        self.env = self.root / "RL_attack_env_qwen.py"
        self.env.write_text("# test only", encoding="utf-8")
        self.cfg = read(HERE / "matrix_config.json")
        self.cfg.update(base_model=str(self.model), victims={"wm": str(self.model)},
                        adapters={"exp1_nodefense": str(self.adapter)}, n_exp=2,
                        output=str(self.root / "results"), seeds=[9000])

    def tearDown(self):
        self.tmp.cleanup()

    def result(self, seed):
        eps = []
        for i in range(2):
            rng = random.Random(seed + i + 10000)
            eps.append({"attacker": "model_rl", "exp_idx": i, "consensus": False,
                        "ep_len": 10, "rounds": [{} for _ in range(10)],
                        "init_positions": [rng.randint(0, 20) for _ in range(4)]})
        return {"defense": "none", "round_timing": "synchronous_snapshot_atomic_commit", "episodes": eps,
                "summary": {"model_rl": {"episodes": 2, "consensus": 0, "rate": 0., "mean_ep_len": 10.}}}

    def test_fast_inherits_user_paths(self):
        parent = self.root / "matrix_config.json"
        write(parent, self.cfg)
        fast = self.root / "fast.json"
        write(fast, {"base_config": "matrix_config.json", "batch_size": 16, "reuse_seed_models": True})
        cfg = load_config(fast)
        self.assertEqual(cfg["adapters"], self.cfg["adapters"])
        self.assertEqual(cfg["batch_size"], 16)
        self.assertTrue(cfg["reuse_seed_models"])
        write(parent, {"base_config": "fast.json"})
        with self.assertRaises(ValueError):
            load_config(fast)

    def test_original_fresh_runner_still_completes(self):
        config = self.root / "config.json"
        write(config, self.cfg)
        owner = self
        class Process:
            def __init__(self, cmd, **kwargs):
                write(Path(cmd[cmd.index("--dump_json") + 1]), owner.result(9000))
                self.stdout = iter(["fake evaluator output\n"])
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return False
            def wait(self):
                return 0
        argv = ["run_matrix.py", "--run", "--config", str(config), "--env-module", str(self.env),
                "--defenses", "no_defense"]
        with patch("sys.argv", argv), patch("run_matrix.subprocess.check_output", return_value="test"), \
                patch("run_matrix.subprocess.Popen", Process), contextlib.redirect_stdout(io.StringIO()):
            main()
        result = self.root / "results/wm/exp1_nodefense/no_defense/eval_seed9000.manifest.json"
        self.assertEqual(read(result)["status"], "complete")

    def test_session_invokes_fresh_main_per_seed_and_reuses_models(self):
        settings = {k: self.cfg[k] for k in ("n_exp", "batch_size", "max_steps", "max_pos", "actor_message_tokens",
                                            "wm_tokens", "gpu_mem_util", "max_model_len")}
        settings["execution_mode"] = "reuse_seeds_v2"
        calls, mains = [], []
        def factory(**kwargs):
            calls.append(kwargs)
            return SimpleNamespace(sampling=SimpleNamespace(seed=None))
        module = SimpleNamespace(QwenWorldModelVLLM=factory, HFQwen3=factory)
        def eval_main():
            seed = int(sys.argv[sys.argv.index("--seed") + 1])
            wm = module.QwenWorldModelVLLM(path="victim", seed=seed)
            actor = module.HFQwen3(path="actor", seed=seed)
            # These resets are performed by actual evaluators, not the cache.
            random.seed(seed + 100000)
            wm.sampling.seed = seed + 200000
            mains.append((seed, wm.sampling.seed, random.random()))
            write(Path(sys.argv[sys.argv.index("--dump_json") + 1]), self.result(seed))
        module.main = eval_main
        jobs = []
        for seed in (9000, 9001):
            output = self.root / f"results/wm/exp1_nodefense/no_defense/eval_seed{seed}.json"
            spec = dict(victim="wm", adapter="exp1_nodefense", defense="no_defense", seed=seed,
                        settings=settings, source_sha256=source_info(self.env), adapter_info=adapter_info(self.adapter))
            jobs.append({"output": str(output), "spec": spec,
                         "command": command(self.cfg, "no_defense", "wm", "exp1_nodefense", seed, output)})
        original_path = list(sys.path)
        try:
            with patch("seed_worker.importlib.import_module", return_value=module), \
                    patch.dict(sys.modules, {"RL_attack_env_qwen": SimpleNamespace(__file__=str(self.env))}), \
                    contextlib.redirect_stdout(io.StringIO()):
                run_session({"jobs": jobs, "env_module": str(self.env)})
        finally:
            sys.path[:] = original_path
        self.assertEqual(len(calls), 2)
        self.assertEqual([x[:2] for x in mains], [(9000, 209000), (9001, 209001)])
        for job in jobs:
            m = read(Path(job["output"]).with_suffix(".manifest.json"))
            self.assertEqual(m["status"], "complete")
        bad = copy.deepcopy(jobs)
        bad[1]["spec"]["adapter"] = "wrong_row"
        with self.assertRaises(ValueError):
            run_session({"jobs": bad, "env_module": str(self.env)})


if __name__ == "__main__":
    unittest.main()
