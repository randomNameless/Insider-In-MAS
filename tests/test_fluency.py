import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from attacker.reward import RewardConfig, fluency_penalty

CFG = RewardConfig(beta_fluency=0.1, fluency_threshold=-4.0)


def test_clean_fluent_message_no_penalty():
    # in-distribution message (high base log-prob) + normal text -> 0 penalty
    p = fluency_penalty("Let's not rush, I think 12 is risky.", msg_ref_logp=-2.5, cfg=CFG)
    assert p == 0.0


def test_gibberish_low_logp_penalized():
    # word salad the base model finds unlikely (msg_ref_logp well below threshold)
    p = fluency_penalty("word salad here", msg_ref_logp=-9.0, cfg=CFG)
    # ppl hinge = min(0, -9 - (-4)) = -5 ; *0.1 = -0.5
    assert p < 0.0
    assert abs(p - 0.1 * (-5.0)) < 1e-9


def test_format_leak_penalized_even_if_high_logp():
    # repetition/leak can have HIGH logp -> ppl misses it, rule must catch it
    p = fluency_penalty("Position: 0 Position: 0 Position: 0", msg_ref_logp=-1.0, cfg=CFG)
    assert p < 0.0            # caught by the "Position:" leak rule + repeat rule


def test_char_run_penalized():
    p = fluency_penalty("go to 00000000 now", msg_ref_logp=-1.5, cfg=CFG)
    assert p < 0.0            # 6+ repeated chars


def test_symbol_soup_penalized():
    p = fluency_penalty("!!! @@@ ``` ;;; ###", msg_ref_logp=-1.5, cfg=CFG)
    assert p < 0.0            # <50% language chars


def test_penalty_only_never_positive():
    for lp in (-1.0, -3.0, -4.0, -6.0, -12.0):
        assert fluency_penalty("normal fluent sentence about moving", lp, CFG) <= 0.0


def test_beta_zero_disables():
    off = RewardConfig(beta_fluency=0.0, fluency_threshold=-4.0)
    assert fluency_penalty("Position: 0 Position: 0 garbage 00000", -20.0, off) == 0.0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("test_fluency OK")
