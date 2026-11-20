import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from attacker.potential import phi
from attacker.reward import (
    RewardConfig,
    compute_turn_reward,
    diversity_penalty,
    length_penalty,
    message_similarity,
    r_spread,
    r_terminal,
    r_validity,
)


def test_spread_telescopes():
    cfg = RewardConfig()
    # a sequence of benign states; sum of r_spread == alpha*(phi_T - phi_0)
    states = [[10, 10, 10], [8, 10, 12], [5, 10, 15], [0, 10, 20]]
    total = 0.0
    for a, b in zip(states[:-1], states[1:]):
        total += r_spread(phi(a), phi(b), cfg.alpha_spread)
    assert abs(total - cfg.alpha_spread * (phi(states[-1]) - phi(states[0]))) < 1e-9


def test_terminal_early_fail_worse():
    early = r_terminal(is_terminal=True, consensus_reached=True, t=2, T=8, eta=0.1)
    late = r_terminal(is_terminal=True, consensus_reached=True, t=7, T=8, eta=0.1)
    assert early < late < 0
    survive = r_terminal(is_terminal=True, consensus_reached=False, t=8, T=8, eta=0.1)
    assert survive == 1.0
    nonterminal = r_terminal(is_terminal=False, consensus_reached=False, t=3, T=8, eta=0.1)
    assert nonterminal == 0.0


def test_validity_no_positive_reward():
    assert r_validity(True, -0.05) == 0.0
    assert r_validity(False, -0.05) == -0.05


def test_compute_turn_reward_components_separate():
    cfg = RewardConfig()
    tr = compute_turn_reward(
        benign_prev=[10, 10, 10], benign_next=[8, 10, 12],
        valid=True, is_terminal=False, consensus_reached=False,
        t=1, T=8, cfg=cfg,
    )
    assert tr.terminal == 0.0
    assert tr.validity == 0.0
    assert tr.spread > 0.0
    assert abs(tr.total - (tr.spread + tr.terminal + tr.validity)) < 1e-12


def test_message_regularization_penalties():
    cfg = RewardConfig(beta_length=0.05, target_message_tokens=32,
                       truncation_penalty=-0.1, beta_diversity=0.05,
                       diversity_threshold=0.5)
    assert length_penalty(20, False, cfg) == 0.0
    assert length_penalty(48, True, cfg) < -0.1
    sim = message_similarity("Please move toward position ten now",
                             "Please move toward position ten now")
    assert sim == 1.0
    assert diversity_penalty([sim], cfg) == -0.05


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("test_reward OK")
