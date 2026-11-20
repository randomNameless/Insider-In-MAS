"""Structural guards for the shared synchronous training rollout."""
from pathlib import Path


def test_attacker_commit_occurs_after_benign_generation():
    text = (Path(__file__).with_name("rollout.py")).read_text(encoding="utf-8")
    benign = text.index("benign_msgs = benign_engine.step_batched")
    commit = text.index("st.positions[ATTACKER_IDX] = atk_positions[idx]")
    assert benign < commit


def test_no_pre_benign_attacker_state_assignment():
    text = (Path(__file__).with_name("rollout.py")).read_text(encoding="utf-8")
    prefix = text[:text.index("benign_msgs = benign_engine.step_batched")]
    assert "st.positions[ATTACKER_IDX] =" not in prefix
    assert "st.confidences[ATTACKER_IDX] =" not in prefix
