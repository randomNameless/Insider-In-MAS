import copy
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from channel_game import CHANNELS, run_batch, select_action
from summarize_channels import validate
from eval_channels import execute, write_new


class FakeEnv:
    def __init__(self, max_steps=2):
        self.max_steps = max_steps
    def reset(self, options):
        self.pos = list(options["positions"])
        self.last_messages = [""] * 4
        self.t = 0
    def prepare(self, pos):
        self.pos[3] = pos
    def apply(self, preds):
        self.pos[:3] = [x[0] for x in preds]
        self.last_messages = [x[1] for x in preds] + [self._pending_atk_msg]
        self.t += 1
        return None, None, len(set(self.pos[:3])) == 1, self.t >= self.max_steps, {}


def inputs(env):
    return {"fallback_positions": list(env.pos), "old_messages": list(env.last_messages),
            "t": env.t}


def ppo(batch):
    return [(19, "PPO message") for _ in batch]


def victim(batch):
    return [[(1, "A"), (2, "B"), (3, "C"), (4, "neutral message")] for _ in batch]


class Tests(unittest.TestCase):
    def test_write_never_overwrites(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.json"
            write_new(path, {"original": True})
            with self.assertRaises(FileExistsError):
                write_new(path, {"original": False})
            self.assertTrue(json.loads(path.read_text())["original"])

    def test_evaluator_wiring_and_json_with_fake_models(self):
        calls = {"ppo_load": 0, "wm_load": 0, "victim_sizes": []}
        class WM:
            def __init__(self, **kwargs):
                calls["wm_load"] += 1
                self.sampling = SimpleNamespace(seed=kwargs["seed"])
        class LLM:
            def __init__(self, *args, **kwargs):
                calls["ppo_load"] += 1
            def generate_two_phase_batch(self, chats, **kwargs):
                return [("PPO", 19) for _ in chats]
        class Env(FakeEnv):
            def __init__(self, **kwargs):
                super().__init__(kwargs["max_steps"])
            def _build_messages(self, i):
                return [{"role": "user", "content": f"old={self.pos};i={i}"}]
        def benign_predict_many(wm, groups):
            calls["victim_sizes"].append([len(g) for g in groups])
            return [[(5, "A"), (5, "B"), (5, "C"), (7, "neutral")] for _ in groups]
        noop = lambda *args, **kwargs: None
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            external = output / "RL_attack_env_qwen.py"
            fake_modules = {
                "numpy": SimpleNamespace(random=SimpleNamespace(seed=noop)),
                "torch": SimpleNamespace(manual_seed=noop, cuda=SimpleNamespace(manual_seed_all=noop)),
                "RL_attack_env_qwen": SimpleNamespace(__file__=str(external), QwenWorldModelVLLM=WM, AttackEnv=Env),
                "attacker": SimpleNamespace(),
                "attacker.prompts": SimpleNamespace(build_chat=lambda *args, **kwargs: [{"role": "user", "content": "neutral"}]),
                "eval_attacks_consensus": SimpleNamespace(HFQwen3=LLM,
                    build_model_rl_query=lambda env: [{"role": "user", "content": str(env.pos)}],
                    benign_predict_many=benign_predict_many),
            }
            a = SimpleNamespace(wm_model_dir="victim", wm_gpu_mem_util=.6,
                max_model_len=2048, enforce_eager=False, seeds=[9000, 9001],
                llm_model_dir="sft", rl_adapter="adapter", n_exp=2,
                conditions=list(CHANNELS), eval_batch_size=2, victim="wm")
            provenance = {"external_env": str(external.resolve()), "fingerprints": {}}
            with patch.dict(sys.modules, fake_modules), contextlib.redirect_stdout(io.StringIO()):
                execute(a, provenance, output)
            self.assertEqual(calls["ppo_load"], 1)
            self.assertEqual(calls["wm_load"], 1)
            self.assertTrue(all(sizes == [4, 4] for sizes in calls["victim_sizes"]))
            for seed in a.seeds:
                data = json.loads((output / f"eval_seed{seed}.json").read_text())
                self.assertEqual(len(validate(data)), 4)
                self.assertTrue(all(ep["ep_len"] == 1 for ep in data["episodes"]))

    def test_mapping(self):
        p, n = {"position": 19, "message": "P"}, {"position": 4, "message": "N"}
        expected = [("P", 19), ("P", 4), ("N", 19), ("N", 4)]
        for c, (msg, pos) in zip(CHANNELS, expected):
            self.assertEqual(select_action(c, p, n), {"position": pos, "message": msg})

    def test_snapshot_and_actual_next_position_all_conditions(self):
        for c in CHANNELS:
            seen = []
            def generate(batch):
                seen.extend(copy.deepcopy(batch))
                return ppo(batch)
            result = run_batch([FakeEnv()], [[1, 2, 3, 7]], c, inputs, generate, victim)[0]
            first = result["rounds"][0]
            self.assertEqual(first["snapshot"]["positions"][3], 7)
            self.assertEqual(seen[1]["fallback_positions"][3], first["submitted_agent3"]["position"])
            self.assertEqual(seen[1]["old_messages"][3], first["submitted_agent3"]["message"])
            self.assertEqual(result["ep_len"], 2)

    def test_early_stop_and_active_batch(self):
        sizes = []
        def varied(batch):
            sizes.append(len(batch))
            return [[(5, "a"), (5, "b"), (5, "c"), (19, "n")]
                    if x["fallback_positions"][0] == 0 else victim([x])[0] for x in batch]
        results = run_batch([FakeEnv(), FakeEnv()], [[0, 2, 3, 7], [1, 2, 3, 7]],
                            "no_attack", inputs, ppo, varied)
        self.assertEqual(sizes, [2, 1])
        self.assertTrue(results[0]["consensus"])  # Agent 3 remains at 19, NOT scored.
        self.assertEqual([x["ep_len"] for x in results], [1, 2])

    def test_rejects_early_state_mutation(self):
        env = FakeEnv()
        def bad(batch):
            env.pos[3] = 19
            return ppo(batch)
        with self.assertRaisesRegex(RuntimeError, "before atomic commit"):
            run_batch([env], [[1, 2, 3, 7]], "full_attack", inputs, bad, victim)

    def test_rejects_prompt_mutation(self):
        def bad(batch):
            batch[0]["fallback_positions"][3] = 19
            return ppo(batch)
        with self.assertRaisesRegex(RuntimeError, "input prompts"):
            run_batch([FakeEnv()], [[1, 2, 3, 7]], "full_attack", inputs, bad, victim)

    def test_rejects_wrong_committed_message(self):
        class BadEnv(FakeEnv):
            def apply(self, preds):
                answer = super().apply(preds)
                self.last_messages[3] = "incorrect"
                return answer
        with self.assertRaisesRegex(RuntimeError, "chosen messages"):
            run_batch([BadEnv()], [[1, 2, 3, 7]], "no_attack", inputs, ppo, victim)

    def test_summary_validation_and_tampering(self):
        rows = []
        stats = {}
        for c in CHANNELS:
            state = run_batch([FakeEnv(10)], [[1, 2, 3, 7]], c, inputs, ppo, victim)[0]
            rows.append({"condition": c, "exp_idx": 0, "initial_positions": [1, 2, 3, 7], **state})
            stats[c] = {"episodes": 1, "consensus": 0, "rate": 0., "mean_ep_len": 10.}
        payload = {"protocol": "channel_replacement_v1", "round_timing": "synchronous_snapshot_atomic_commit",
                   "max_steps": 10, "consensus_scope": [0, 1, 2], "n_exp": 1,
                   "episodes": rows, "summary": stats}
        self.assertEqual(validate(payload), stats)
        broken = copy.deepcopy(payload)
        broken["episodes"][1]["rounds"][0]["submitted_agent3"]["position"] = 19
        with self.assertRaisesRegex(ValueError, "channel selection"):
            validate(broken)
        broken = copy.deepcopy(payload)
        broken["summary"]["no_attack"]["mean_ep_len"] = 5
        with self.assertRaisesRegex(ValueError, "Wrong summary"):
            validate(broken)


if __name__ == "__main__":
    unittest.main()
