"""Cross-entry structural regression tests for synchronous round semantics."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def text(path):
    return (ROOT / path).read_text(encoding="utf-8")


def test_shared_training_commits_attacker_after_benign():
    src = text("attacker/rollout.py")
    assert src.index("benign_msgs = benign_engine.step_batched") < src.index(
        "st.positions[ATTACKER_IDX] = atk_positions[idx]")


def test_exp1_eval_builds_snapshot_queries_before_prepare():
    src = text("Experiment/Experiment1_Eval/eval_attacks_consensus.py")
    query = 'queries = [(env._build_messages(i), int(env.pos[i])) for i in range(3)]'
    assert src.index(query) < src.index("env.prepare(atk_pos)")


def test_exp3_filter_runs_before_new_attacker_confidence_commit():
    src = text("Experiment/Experiment3_CPWBFT/eval_exp3_cpwbft.py")
    assert src.index("visible = [list(env._visible_others(i))") < src.index(
        "env.confidences[3] = atk_conf", src.index("def run_batch"))


def test_dqn_prepare_stages_without_parent_mutation():
    src = text("Experiment/Experiment1_Eval/train_attacker_maxentropy_qwen_mixed.py")
    prepare = src[src.index("    def prepare(self, action):"):src.index("    def apply(self, preds):")]
    assert "super().prepare" not in prepare
    assert "_sync_pending_action" in prepare
