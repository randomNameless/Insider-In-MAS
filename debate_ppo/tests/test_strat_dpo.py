"""CPU-only tests for the strategy-catalog DPO pipeline (debate_ppo/strat_dpo)."""
import pytest

import debate_ppo  # noqa: F401  (sys.path for debate_pilot)
from debate_ppo import dprotocol as D
from debate_ppo.strat_dpo import make_data as MD
from debate_ppo.strat_dpo import strategies as S

P = D.P
Q = "What is 2+2?\n\nChoices (answer with the option letter):\nA. 3\nB. 4\nC. 5\nD. 22"


def _rep(arg, ans):
    return {"answer": ans, "argument": arg, "raw": f"Argument: {arg}\nAnswer: {ans}", "valid": True}


def _debate(seat=1, seed=0, n_rounds=3, gold="B"):
    hist = [[_rep(f"r{r} a{j} reasoning", "B" if j != 2 else "C") for j in range(4)] for r in range(n_rounds)]
    return {"task_id": "t1", "seed": seed, "cond": "none", "question": Q, "n_options": 4, "gold": gold,
            "seat": seat, "personalities": D.seat_personalities(seat, "neutral"), "history": hist}


def test_catalog():
    assert len(S.STRATEGY_IDS) == 27 and S.FABRICATION <= set(S.STRATEGY_IDS)
    assert [s[0] for s in S.STRATEGY_IDS].count("A") == 7 and [s[0] for s in S.STRATEGY_IDS].count("E") == 3
    for sid, (name, text) in S.STRATEGIES.items():
        assert "({TARGET})" in text or sid in ("A6", "B4", "D3", "D4", "D5", "D6") or "{TARGET}" in text, sid
        text.format(TARGET="C")                  # no stray braces


@pytest.mark.parametrize("strategy", [None] + S.STRATEGY_IDS)
def test_prompt_has_target_not_gold(strategy):
    st = MD.build_states([_debate()], {"t1": "C"}, [0], [1])[0]
    p = S.attacker_prompt(*MD.prompt_parts(st, "{}"), strategy=strategy)
    assert "Your assigned answer is (C)" in p and 'end with "(C)"' in p
    assert "{" not in p.replace("{}", "")        # every placeholder filled
    assert ("Strategy:" in p) == (strategy is not None)
    assert ("Apply the specified strategy" in p) == (strategy is not None)
    assert "Correct answer" not in p and "gold" not in p.lower()


def test_no_strategy_prompt_is_common_prompt_minus_strategy():
    st = MD.build_states([_debate()], {"t1": "C"}, [0], [1])[0]
    parts = MD.prompt_parts(st, "{}")
    a, b = S.attacker_prompt(*parts, strategy=None), S.attacker_prompt(*parts, strategy="A1")
    assert a.split("Requirements:")[0] == b.split("Strategy:")[0]
    assert a.split("- Refer to peers")[1] == b.split("- Refer to peers")[1]


def test_build_states_rounds_and_peers():
    rows = MD.build_states([_debate(seat=1), {**_debate(), "cond": "prompt"}, _debate(seed=1)], {"t1": "C"}, [0], [1, 2])
    assert [r["round"] for r in rows] == [1, 2]
    assert len(rows[0]["history"]) == 2 and len(rows[1]["history"]) == 3
    peers = MD.peers_of(rows[0])
    assert peers == [("r0 a0 reasoning", "B"), ("r0 a2 reasoning", "C"), ("r0 a3 reasoning", "B")]


def test_build_states_rejects_gold_target():
    with pytest.raises(AssertionError):
        MD.build_states([_debate()], {"t1": "B"}, [0], [1])


@pytest.mark.parametrize("r", [1, 2])
def test_scoring_state_replaces_seat_reply(r):
    D.set_profile_set("v1")
    st = MD.build_states([_debate(seat=1)], {"t1": "C"}, [0], [r])[0]
    s = MD.scoring_state(st, P.Reply(answer="C", argument="MSG", raw="MSG", valid=True))
    assert s.round == r + 1 and s.history[r][1].argument == "MSG"
    assert st["history"][r][1]["argument"] != "MSG"      # source state untouched
    for ag in D.honest_seats(1):
        u = D.chat(s, ag)[1]["content"]
        assert "MSG Answer: C." in u
        assert ("FINAL SUBMISSION" in u) == (r + 1 == D.FINAL_ROUND)


def test_check_message():
    ok = S.check_message("Option C fits the data best, so choose (C).", "C", False)
    assert ok["valid"] and ok["ends_with_target"]
    assert not S.check_message("Choose (B).", "C", False)["valid"]
    assert not S.check_message("Choose (C).", "C", True)["valid"]                 # cut
    assert not S.check_message("As Peer 2 said, (C).", "C", False)["valid"]      # private label
    assert not S.check_message(" ".join(["w"] * 200) + " (C)", "C", False)["valid"]
    assert S.check_message('It is (C)."', "C", False)["valid"]


def test_parse_opponent_model():
    t, ok = S.parse_opponent_model('```json\n{"peers": [], "swing_peers": []}\n```')
    assert ok and '"peers"' in t
    t, ok = S.parse_opponent_model('Here: {"a": 1} done')
    assert ok
    t, ok = S.parse_opponent_model("not json")
    assert not ok and t == "not json"


def test_clean_message():
    assert S.clean_message("<think>\n\n</think>\n\nHello\n world (C)") == "Hello world (C)"


def test_rank_pairs():
    cs = [(0.9, "A1"), (0.1, "B1"), (0.5, "C1"), (0.4, "D1")]
    ps = MD.rank_pairs(cs, 0.05, 2)
    assert [(c[1], r[1]) for c, r in ps] == [("A1", "B1"), ("C1", "D1")]
    assert len(MD.rank_pairs(cs, 0.3, 2)) == 1
    assert MD.rank_pairs([(0.5, "A1")], 0.0, 2) == []
    assert len(MD.rank_pairs([(0.5, "A1"), (0.2, "B1"), (0.1, "C1")], 0.0, 2)) == 1   # middle never paired with itself


def test_states_carry_split():
    assert {r["split"] for r in MD.build_states([_debate()], {"t1": "C"}, [0], [1, 2], split="val")} == {"val"}
    assert MD.build_states([_debate()], {"t1": "C"}, [0], [1])[0]["split"] == "fit"


# ---------------------------------------------------------------- evaluation conditions
def test_strat_conditions():
    from debate_ppo.eval_debate import is_two_phase, needs_target
    assert S.is_strat("strat_q3_4b") and S.is_strat("stratdpo_q3_4b") and not S.is_strat("targeted_4b")
    assert S.strategy_of("stratdpo_q3_4b") is None and S.strategy_of("strat_q3_4b__E2") == "E2"
    with pytest.raises(AssertionError):
        S.strategy_of("strat_q3_4b__Z9")
    assert needs_target("stratdpo_q3_4b") and not is_two_phase("stratdpo_q3_4b")


@pytest.mark.parametrize("r", [1, 2])
def test_eval_prompt_equals_training_prompt(r):
    from debate_ppo.eval_debate import strat_chat
    d = {**_debate(seat=1), "strat": [{"round": r, "opponent_model": "OM-JSON"}, {"round": r - 1, "check": {}}]}
    train = MD.build_states([_debate(seat=1)], {"t1": "C"}, [0], [r])[0]
    want = S.attacker_chat(*MD.prompt_parts(train, "OM-JSON"), strategy=None)
    assert strat_chat(d, r, "C", None, S.MAX_WORDS) == want


def test_eval_prompt_round1():
    from debate_ppo.eval_debate import strat_chat
    p = strat_chat({**_debate(), "history": []}, 0, "C", None, S.MAX_WORDS)[0]["content"]
    assert S.ROUND1_PEERS in p and S.ROUND1_OPPONENT_MODEL in p and "Strategy:" not in p
