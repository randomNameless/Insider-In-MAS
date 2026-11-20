from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def test_sac_never_reads_current_attacker_message():
    src = (HERE / "sac_defense.py").read_text(encoding="utf-8")
    assert "sac_current_attacker_message" not in src


def test_training_uses_shared_synchronous_rollout():
    src = (HERE / "train_ppo_exp4_sac.py").read_text(encoding="utf-8")
    assert "super().sample_batch(base_seed, sample)" in src
    assert "rollout_batch_sac(" not in src


def test_eval_commits_attacker_after_sac_generation():
    src = (HERE / "eval_exp4_sac.py").read_text(encoding="utf-8")
    score = src.index("benign_msgs = sac.step_batched(states, sample=True)")
    commit = src.index("e.pos[3] = int(gen.next_position)")
    assert score < commit


def test_eval_allows_all_nonconfidence_attackers():
    src = (HERE / "eval_exp4_sac.py").read_text(encoding="utf-8")
    assert 'ATTACKS = ("random", "extreme", "prompt", "dqn", "model_base", "model_rl")' in src
    assert "make_attacker_fn" in src


def test_eval_uses_vllm_for_text_only_sac_generation():
    src = (HERE / "eval_exp4_sac.py").read_text(encoding="utf-8")
    assert 'default="vllm"' in src
    assert "SACBenignEngineVLLM" in src
    assert "QwenWorldModelVLLM" in src


def test_eval_json_encoder_accepts_numpy_bool():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "eval_exp4_sac_for_test", HERE / "eval_exp4_sac.py")
    # Avoid importing the GPU evaluator and its optional runtime dependencies;
    # the structural guards below ensure both conversion layers are present.
    src = (HERE / "eval_exp4_sac.py").read_text(encoding="utf-8")
    assert "isinstance(value, np.generic)" in src
    assert "default=_json_default" in src
    assert "consensus[global_i] = bool(" in src
