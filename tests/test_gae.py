import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from attacker.gae import compute_gae, gae_and_returns


def test_gae_matches_manual():
    rewards = [0.0, 0.0, 1.0]
    values = [0.5, 0.6, 0.7]
    gamma, lam = 0.99, 0.90
    adv = compute_gae(rewards, values, gamma, lam, bootstrap_value=0.0)
    # manual backward recursion
    d2 = rewards[2] + gamma * 0.0 - values[2]
    a2 = d2
    d1 = rewards[1] + gamma * values[2] - values[1]
    a1 = d1 + gamma * lam * a2
    d0 = rewards[0] + gamma * values[1] - values[0]
    a0 = d0 + gamma * lam * a1
    assert abs(adv[0] - a0) < 1e-9
    assert abs(adv[1] - a1) < 1e-9
    assert abs(adv[2] - a2) < 1e-9


def test_returns_use_lambda_one():
    rewards = [0.1, -0.2, 1.0]
    values = [0.3, 0.1, 0.4]
    adv, returns = gae_and_returns(rewards, values, gamma=0.99,
                                   lam_actor=0.90, lam_critic=1.0)
    # with lam_critic=1.0, R_hat_t is the discounted reward-to-go (MC return)
    T = len(rewards)
    for t in range(T):
        mc = sum((0.99 ** (k - t)) * rewards[k] for k in range(t, T))
        assert abs(returns[t] - mc) < 1e-6


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("test_gae OK")
