"""CPU-only tests: no Torch/vLLM/model downloads required."""
import ast
import copy
import contextlib
import io
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

from run_matrix import HERE, REPO, SCRIPTS, adapter_info, command, digest, main, read, sha, validate_result, write
from summarize_matrix import summarize


class MatrixTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.cfg = dict(n_exp=2, batch_size=8, max_steps=10, max_pos=20,
                        actor_message_tokens=48, wm_tokens=200, gpu_mem_util=0.6,
                        max_model_len=2048, victims={"wm": "/victim"},
                        adapters={"exp1_nodefense": "/adapter"}, base_model="/sft")
        self.spec = dict(experiment=7, protocol="defense_adaptation_v1", victim="wm",
                         adapter="exp1_nodefense", defense="no_defense", seed=9000,
                         settings=self.cfg, base={}, source_sha256={}, python_version="test",
                         environment_freeze="test", victim_info={},
                         adapter_info={"path": "/adapter", "weight_sha256": "abc"})

    def tearDown(self):
        self.tmp.cleanup()

    def result(self):
        eps = []
        for i, t in enumerate([3, 10]):
            rng = random.Random(9000 + i + 10000)
            eps.append(dict(attacker="model_rl", exp_idx=i, consensus=i == 0,
                            ep_len=t, rounds=[{} for _ in range(t)],
                            init_positions=[rng.randint(0, 20) for _ in range(4)]))
        return dict(defense="none", round_timing="synchronous_snapshot_atomic_commit",
                    episodes=eps, summary={"model_rl": dict(episodes=2, consensus=1, rate=.5, mean_ep_len=6.5)})

    def test_commands_match_real_cli(self):
        # AST inspection tests actual parsers without importing GPU dependencies.
        for defense in SCRIPTS:
            cmd = command(self.cfg, defense, "wm", "exp1_nodefense", 9000, self.root / "out.json")
            name = "no_defense" if defense == "wmsr" else defense
            tree = ast.parse((REPO / "Experiment" / SCRIPTS[name]).read_text(encoding="utf-8"))
            allowed = {arg.value for node in ast.walk(tree) if isinstance(node, ast.Call)
                       and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument"
                       for arg in node.args if isinstance(arg, ast.Constant) and isinstance(arg.value, str)}
            if defense == "wmsr":
                allowed.add("--wmsr_F")
            flags = {v for v in cmd if v.startswith("--")}
            self.assertFalse(flags - allowed, (defense, flags - allowed))
            self.assertEqual(cmd[cmd.index("--attacks") + 1], "model_rl")
            self.assertEqual(cmd[cmd.index("--rl_adapter") + 1], "/adapter")
            self.assertEqual("--wm_backend" in cmd, defense != "cpwbft")

    def test_all_160_jobs_unique(self):
        keys = {(v, a, d, s) for v in ("wm", "qwen35_9b") for a in range(4)
                for d in SCRIPTS for s in range(9000, 9005)}
        self.assertEqual(len(keys), 160)

    def test_check_mode_no_gpu_no_output(self):
        cfg = read(HERE / "matrix_config.json")
        model = self.root / "model"
        model.mkdir()
        write(model / "config.json", {"model_type": "qwen3"})
        (model / "model.safetensors").write_bytes(b"fixture only")
        cfg["base_model"] = str(model)
        cfg["victims"] = {"wm": str(model)}
        for i, name in enumerate(cfg["adapters"]):
            adapter = self.root / name
            adapter.mkdir()
            write(adapter / "adapter_config.json", {"peft_type": "LORA", "r": 32,
                  "base_model_name_or_path": str(model)})
            (adapter / "adapter_model.safetensors").write_bytes(f"fixture {i}".encode())
            cfg["adapters"][name] = str(adapter)
        cfg["output"] = str(self.root / "must_not_exist")
        config = self.root / "config.json"
        write(config, cfg)
        module = self.root / "RL_attack_env_qwen.py"
        module.write_text("raise RuntimeError('check must not import game')", encoding="utf-8")
        capture = io.StringIO()
        with patch("sys.argv", ["run_matrix.py", "--check", "--config", str(config),
                                "--env-module", str(module)]), contextlib.redirect_stdout(capture):
            main()
        self.assertIn("JOBS=80 EPISODES=16000", capture.getvalue())
        self.assertIn("exp4_curriculum", capture.getvalue())
        self.assertFalse(Path(cfg["output"]).exists())

    def test_valid_and_bad_results(self):
        path = self.root / "eval.json"
        good = self.result()
        write(path, good)
        self.assertEqual(validate_result(path, self.spec)["rate"], .5)
        for mutation in (lambda d: d["episodes"].pop(),
                         lambda d: d["summary"]["model_rl"].update(rate=.9),
                         lambda d: d["episodes"][0].update(init_positions=[0]*4),
                         lambda d: d["episodes"][0].update(exp_idx=1),
                         lambda d: d.update(defense="SAC_F1_complete4"),
                         lambda d: d.update(round_timing="legacy"),
                         lambda d: d["episodes"][1].update(ep_len=4)):
            bad = copy.deepcopy(good)
            mutation(bad)
            write(path, bad)
            with self.assertRaises(ValueError):
                validate_result(path, self.spec)

    def test_adapter_identity_and_digest(self):
        write(self.root / "adapter_config.json", {"peft_type": "LORA", "r": 32})
        (self.root / "adapter_model.safetensors").write_bytes(b"test weight")
        info = adapter_info(self.root)
        self.assertEqual(info["weight_sha256"], sha(self.root / "adapter_model.safetensors"))
        self.assertNotEqual(digest(self.spec), digest({**self.spec, "seed": 1}))

    def test_summary_verifies_manifests(self):
        path = self.root / "wm/exp1_nodefense/no_defense/eval_seed9000.json"
        path.parent.mkdir(parents=True)
        write(path, self.result())
        write(path.with_suffix(".manifest.json"), dict(status="complete", spec=self.spec,
              job_id=digest(self.spec), result_sha256=sha(path)))
        report, missing = summarize(self.root, [9000])
        self.assertIn("0.500 ± N/A", report)
        self.assertTrue(missing)
        bad = self.result()
        bad["extra"] = True
        write(path, bad)
        with self.assertRaises(ValueError):
            summarize(self.root, [9000])

    def test_full_matrix_and_gain(self):
        from summarize_matrix import ROWS, COLS
        for adapter in ROWS:
            for defense in COLS:
                for seed in (9000, 9001):
                    spec = copy.deepcopy(self.spec)
                    spec.update(adapter=adapter, defense=defense, seed=seed)
                    spec["adapter_info"] = {"path": adapter, "weight_sha256": adapter}
                    d = self.result()
                    d["defense"] = {"no_defense": "none", "wmsr": "paper_global_msr f=1",
                                    "cpwbft": "cpwbft_position_token_decoder_peakedness",
                                    "sac": "SAC_F1_complete4"}[defense]
                    d["confidence_timing"] = "round-synchronous snapshot for all four agents"
                    for e in d["episodes"]:
                        rng = random.Random(seed + e["exp_idx"] + 10000)
                        e["init_positions"] = [rng.randint(0, 20) for _ in range(4)]
                    path = self.root / f"wm/{adapter}/{defense}/eval_seed{seed}.json"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    write(path, d)
                    write(path.with_suffix(".manifest.json"), dict(status="complete", spec=spec,
                          job_id=digest(spec), result_sha256=sha(path)))
        report, missing = summarize(self.root, [9000, 9001])
        self.assertFalse(missing)
        self.assertIn("0.500 ± 0.000", report)
        self.assertIn("0.500 (2/4)", report)
        self.assertIn("0.000 ± 0.000", report)


if __name__ == "__main__":
    unittest.main()
