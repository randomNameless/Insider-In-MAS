"""CPU-only logic tests for the debate PPO environment (no GPU, no model weights)."""
import random

import pytest

import debate_ppo  # noqa: F401  (sys.path for debate_pilot)
from attacker.gae import gae_and_returns
from debate_ppo import dprotocol as D
from debate_ppo.dreward import DebateRewardConfig, field_leak, r_belief, r_terminal

P = D.P


def _task(n=10, gold="C"):
    return {"task_id": "t1", "observation": "Q?\nA. a\nB. b", "answer": gold,
            "metadata": {"choices": list("x" * n)}}


# ---------------------------------------------------------------- parsing
def test_parse_reply_argument_first():
    r = P.parse_reply("Argument: Because X.\nAnswer: B", 10)
    assert r.answer == "B" and r.argument == "Because X." and r.valid


def test_parse_reply_rejects_out_of_range_letter():
    r = P.parse_reply("Argument: hmm\nAnswer: J", 4)
    assert r.answer is None and not r.valid


def test_parse_reply_takes_last_answer():
    r = P.parse_reply("Argument: Answer: A is wrong\nAnswer: (D)", 10)
    assert r.answer == "D"


def test_field_leak():
    assert field_leak("I think Answer: B is right")
    assert not field_leak("The answer is clearly B because of the law of demand.")


# ---------------------------------------------------------------- seats / prompts
@pytest.mark.parametrize("seat", range(4))
def test_seat_personalities(seat):
    ps = D.seat_personalities(seat)
    assert ps[seat] == D.ATTACKER_PERSONA
    assert [p for i, p in enumerate(ps) if i != seat] == D.HONEST_PERSONALITIES
    assert D.honest_seats(seat) == [i for i in range(4) if i != seat]


def test_gold_never_in_prompts():
    st = D.new_state(_task(gold="C"), seat=2)
    for r in range(D.N_ROUNDS + 1):
        for a in range(4):
            if r == D.FINAL_ROUND and a == 2:
                continue
            txt = str(D.chat(st, a))
            assert "gold" not in txt.lower() and "correct option is" not in txt.lower()
        if r < D.N_ROUNDS:
            D.commit(st, {i: P.Reply("A", f"arg{i}", "", True) for i in range(4)})


def test_round_sync_and_final_prompt():
    st = D.new_state(_task(), seat=1)
    assert "round 1 of 3" in D.chat(st, 0)[1]["content"]
    for r in range(3):
        D.commit(st, {i: P.Reply("ABCD"[i], f"r{r}a{i}", "", True) for i in range(4)})
    final = D.chat(st, 0)[1]["content"]
    assert "FINAL SUBMISSION" in final
    # reads round-3 (last) snapshot of everyone else, not itself
    assert "r2a1" in final and "r2a2" in final and "r2a3" in final and "r2a0" not in final
    assert "r1a1" not in final
    with pytest.raises(AssertionError):
        D.commit(st, {i: P.Reply("A", "x", "", True) for i in range(4)})


def test_commit_requires_all_agents():
    st = D.new_state(_task(), seat=0)
    with pytest.raises(AssertionError):
        D.commit(st, {0: P.Reply("A", "x", "", True)})


def test_round2_prompt_reads_previous_round_only():
    st = D.new_state(_task(), seat=3)
    D.commit(st, {i: P.Reply("A", f"first{i}", "", True) for i in range(4)})
    D.commit(st, {i: P.Reply("B", f"second{i}", "", True) for i in range(4)})
    txt = D.chat(st, 0)[1]["content"]
    assert "round 3 of 3" in txt and "second3" in txt and "first3" not in txt


# ---------------------------------------------------------------- potential / reward
def test_phi_honest_only_and_sign():
    probs = [[0.1, 0.9], [0.5, 0.5], [0.3, 0.7]]
    assert D.phi(probs, "B") == pytest.approx(-(0.9 + 0.5 + 0.7) / 3)
    with pytest.raises(AssertionError):
        D.phi(probs + [[1.0, 0.0]], "B")      # 4 agents (attacker included) is rejected


def test_belief_reward_telescopes():
    phis = [random.uniform(-1, 0) for _ in range(4)]
    tot = sum(r_belief(phis[t], phis[t + 1], 0.7) for t in range(3))
    assert tot == pytest.approx(0.7 * (phis[3] - phis[0]))


def test_majority_and_outcome():
    assert D.majority(["A", "A", "B"]) == "A"
    assert D.majority(["A", "B", "C"]) is None
    assert D.majority(["C", "C", "C"]) == "C"
    assert D.outcome(["A", "A", "C"], "C") == "wrong"
    assert D.outcome(["C", "C", "A"], "C") == "correct"
    assert D.outcome(["A", "B", "C"], "C") == "deadlock"
    assert D.outcome(["A", "B", "D"], "C") == "deadlock"


def test_terminal_values():
    c = DebateRewardConfig()
    assert r_terminal("wrong", c) == 1.0
    assert r_terminal("deadlock", c) == 0.5
    assert r_terminal("correct", c) == -1.0


def test_sample_answer_deterministic_and_in_range():
    probs = [0.1, 0.2, 0.7]
    a = [D.sample_answer(probs, 0.7, s) for s in range(200)]
    assert set(a) <= {"A", "B", "C"}
    assert a == [D.sample_answer(probs, 0.7, s) for s in range(200)]
    assert a.count("C") > a.count("A")


def test_gae_three_turns():
    adv, ret = gae_and_returns([0.1, -0.2, 1.0], [0.0, 0.0, 0.0], gamma=0.99, lam_actor=0.9, lam_critic=0.95)
    assert adv[2] == pytest.approx(1.0)
    assert ret[0] == pytest.approx(0.1 + 0.99 * 0.95 * (-0.2 + 0.99 * 0.95 * 1.0))


# ---------------------------------------------------------------- targeted persuader (2026-10-19)
from debate_ppo.dreward import r_terminal_targeted  # noqa: E402


def test_r_terminal_targeted():
    c = DebateRewardConfig()
    assert r_terminal_targeted(["B", "B", "C"], gold="C", target="B", cfg=c) == c.terminal_target
    assert r_terminal_targeted(["C", "C", "B"], gold="C", target="B", cfg=c) == c.terminal_correct
    assert r_terminal_targeted(["D", "D", "B"], gold="C", target="B", cfg=c) == c.terminal_other_wrong
    assert r_terminal_targeted(["A", "B", "C"], gold="C", target="B", cfg=c) == c.terminal_target_deadlock
    assert r_terminal_targeted([None, None, "B"], gold="C", target="B", cfg=c) == c.terminal_target_deadlock


def test_phi_target_and_telescoping():
    probs = [[0.1, 0.6, 0.3] + [0.0] * 7, [0.2, 0.2, 0.6] + [0.0] * 7, [0.0, 1.0, 0.0] + [0.0] * 7]
    assert abs(D.phi_target(probs, "B") - 0.6) < 1e-9
    phis = [0.1, 0.3, 0.2, 0.5]
    assert abs(sum(r_belief(phis[t], phis[t + 1], 1.0) for t in range(3)) - (0.5 - 0.1)) < 1e-12


def test_target_hint_only_in_attacker_system_prompt():
    from debate_ppo.eval_debate import TARGET_HINT
    assert TARGET_HINT == D.TARGET_HINT
    st = D.new_state(_task(gold="C"), seat=1)
    ch = D.add_target_hint(D.chat(st, 1), "E")
    assert "assigned option is E" in ch[0]["content"] and "assigned" not in D.chat(st, 1)[0]["content"]
    assert "assigned" not in ch[1]["content"]
    for a in (0, 2, 3):
        assert "assigned option" not in str(D.chat(st, a))


# ---------------------------------------------------------------- Best-of-N baselines (2026-10-20)
def test_bon_judge_score_rules():
    from debate_ppo import baseline_bon as BON
    assert BON.score_from_top({"1": -0.2, "2": -1.7}) == (-0.2, False)
    assert BON.score_from_top({"(1": -0.5}) == (-0.5, False)
    assert BON.score_from_top({"(": -0.1, "2": -3.0}, {"1": -0.4, "2": -1.1}) == (-0.5, True)
    assert BON.score_from_top({"(": -0.1}, {"2": -0.1}) == (BON.NO_SCORE, False)
    assert BON.score_from_top({"2": -0.1}) == (BON.NO_SCORE, False)


def test_bon_dummy_amayuelas_gold_kraidia_target():
    from debate_ppo import baseline_bon as BON
    assert BON.dummy_answer("amayuelasbon_4b", "C", "F") == "C"
    assert BON.dummy_answer("kraidiabon_4b", "C", "F") == "F"
    assert "Correct answer is: (C)" in BON.judge_chat("Q", "arg", "C")[0]["content"]


def test_bon_chats_carry_argument_and_target():
    from debate_ppo import baseline_bon as BON
    from debate_ppo.eval_debate import REPLY_FORMAT, amayuelas_chat
    g = BON.generator_chat("Q?", "F", [])
    assert "Your answer: F" in g[1]["content"] and "One agent solution" not in g[1]["content"]
    g = BON.generator_chat("Q?", "F", ["Argument: x Answer: A."])
    assert "Other agents responses were:" in g[1]["content"]
    k = BON.kraidia_bon_chat("Q?", "F", [["o1", "o2", "o3"]], ["own0"], ["ARG0", "ARG1"], REPLY_FORMAT)
    assert [m["role"] for m in k] == ["system", "user", "assistant", "user"]
    assert "Your Argument: ARG0" in k[1]["content"] and "Your Answer: (F)" in k[1]["content"]
    assert "Your Argument: ARG1" in k[3]["content"] and k[3]["content"].endswith("Your answer: (F)")
    base = amayuelas_chat([{"role": "system", "content": "s"}, {"role": "user", "content": "Q?"}], "F", 1)
    a = BON.amayuelas_bon_chat(base, "ARG")
    assert a[-1]["content"].endswith("Your Argument: ARG") and "Your answer: (F)" in a[-1]["content"]
    assert base[-1]["content"] != a[-1]["content"]


# ---------------------------------------------------------------- ToMAP port (2026-10-21)
def test_tomap_extract_answer_matches_their_rules():
    from debate_ppo import baseline_tomap as TM
    assert TM.extract_answer("<reason>a</reason><reason>b</reason><reason>c</reason><reason>d</reason>", "reason", 3) == ["b", "c", "d"]
    assert TM.extract_answer("<argument>open ended", "argument") == "open ended"     # unterminated tag -> to end
    assert TM.extract_answer("no tags", "argument") == ""                             # strict
    assert TM.extract_answer("no tags", "reason", 3) == [""]


def test_tomap_format_turns_and_extra_info():
    from debate_ppo import baseline_tomap as TM
    t = TM.format_turns([], "persuader")
    assert 'Alice (You) (turn 1): "Hi, I am Alice.' in t and 'Bob (turn 2): "Nice to meet you.' in t
    assert TM.format_turns([], "guesser").startswith("# There are no previous turns")
    g = TM.format_turns(["A1", "B1"], "guesser")
    assert 'Alice (turn 1): "A1"' in g and 'Bob (turn 2): "B1"' in g
    nodes = [["neg", "pos"], ["c1", "n1"], ["c2", "n2"], ["c3", "n3"]]
    assert "agreement" not in TM.format_extra_info(nodes, None)
    confs = [(1, 0), (1.0, 0.0), (0.5, 0.5), (0.0, 1.0)]
    e = TM.format_extra_info(nodes, confs)
    assert '"c1" (Bob\'s agreement on this claim is 8/8)' in e      # fully agrees with the counterclaim
    assert '"c2" (Bob\'s agreement on this claim is 4/8)' in e
    assert '"c3" (Bob\'s agreement on this claim is 0/8)' in e
    assert not e.endswith("\n")


def test_tomap_turn_order_and_chat():
    from debate_ppo import baseline_tomap as TM
    assert TM.turns_so_far(["A1", "A2"], ["B1", "B2"]) == ["A1", "B1", "A2", "B2"]
    assert TM.turns_so_far([], []) == []
    c = TM.persuader_chat('Question: "Q" Correct answer: "T"', [["n", "p"], ["c1", "x"]], None, ["A1", "B1"])
    assert 'persuade Bob to completely support the following claim:\n"Question: "Q" Correct answer: "T""' in c[0]["content"]
    assert c[1]["content"].startswith("# Here are some claims your OPPONENT")
    assert "<extra_info>" not in c[1]["content"] and "<turns>" not in c[1]["content"]
    cc = TM.counterclaim_chat("neg claim")
    assert '"<root_statement>"' in cc[0]["content"]            # their placeholder is left unfilled
    assert '"neg claim"' in cc[1]["content"] and "generate 3 supporting" in cc[1]["content"]


def test_tomap_negation_fallback():
    from debate_ppo import baseline_tomap as TM
    assert TM.extract_negation("<answer>\nA is not B\n</answer>") == "A is not B"
    assert TM.extract_negation("A is not B.") == "A is not B."


def test_length_limit_suffix_and_sentence_fit():
    from transformers import AutoTokenizer
    import glob
    from debate_ppo.eval_debate import length_limit, fit_sentences, LENGTH_RULE
    from debate_ppo import baseline_kraidia as KR
    assert length_limit("kraidia_9b_len200") == 200 and length_limit("kraidia_9b") is None
    assert KR.is_kraidia("kraidia_9b_len200") and "at most 200 tokens" in LENGTH_RULE.format(n=200)
    tok = AutoTokenizer.from_pretrained(glob.glob("/home/*/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/*")[0]
                                        if glob.glob("/home/*/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/*")
                                        else "/path/to/models/qwen3-4b")
    short = "One sentence. Two sentences."
    assert fit_sentences(short, tok, 200) == (short, len(tok(short, add_special_tokens=False)["input_ids"]), False)
    long = " ".join(f"This is sentence number {i} about option (B)." for i in range(60))
    shown, n, cut = fit_sentences(long, tok, 200)
    assert cut and n > 200 and shown.endswith(".") and len(tok(shown, add_special_tokens=False)["input_ids"]) <= 200
    assert long.startswith(shown)


# ---------------------------------------------------------------- Ju et al. 2025 (psychological persuasion + DPO)
_JU_UP = "external/psych_persuasion/strategy_agent.py"
_JU_BZ = "external/adversarial-persuasion/baselines/psychological_persuasion/strategy_prompts_mcq.py"
_JU_Q = "Which gas is most abundant?\n\nChoices (answer with the option letter):\nA. Oxygen\nB. Nitrogen\nC. Argon"


def _load(path, name):
    import importlib.util
    from pathlib import Path
    p = Path(__file__).resolve().parents[2] / path
    if not p.exists():
        pytest.skip(f"{p} missing")
    spec = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_ju_prompts_match_upstream_and_bozdag_port():
    from pathlib import Path
    from debate_ppo import baseline_ju as JU
    bz = _load(_JU_BZ, "bz_strategy_prompts_mcq")
    up = (Path(__file__).resolve().parents[2] / _JU_UP).read_text()
    assert JU.STRATEGY_MAP == bz.STRATEGY_MAP and JU.STRATEGIES == bz.STRATEGY_CHOICES
    for s, text in JU.STRATEGY_MAP.items():
        assert f'"{s}": lambda: "{text}"' in up, s                 # verbatim from their PersuaderAgent
        assert JU.system_prompt(s) == bz.build_system_prompt(s)
    assert JU.system_prompt(JU.NONE) == bz.build_system_prompt("NoneStrategy")
    chat = JU.persuader_chat(_JU_Q, "C", "conformity")
    assert chat[1]["content"] == bz.build_user_prompt("Which gas is most abundant?", "Argon")
    assert "Nitrogen" not in chat[1]["content"]                      # other options withheld
    assert JU.dpo_prompt(_JU_Q, "C") == bz.build_system_prompt("NoneStrategy") + bz.build_user_prompt("Which gas is most abundant?", "Argon")


def test_ju_conditions_and_listener_parse():
    from debate_ppo import baseline_ju as JU
    from debate_ppo.eval_debate import is_two_phase, needs_target
    assert JU.strategy_of("ju_q25_7b") == JU.NONE and JU.strategy_of("ju_q25_7b__authority_effect") == "authority_effect"
    assert JU.strategy_of("ju_dpo_q25_7b") == JU.NONE
    with pytest.raises(AssertionError):
        JU.strategy_of("ju_q25_7b__nonsense")
    assert JU.is_ju("ju_dpo_q25_7b") and needs_target("ju_q25_7b") and not is_two_phase("ju_q25_7b")
    assert JU.parse_listener("B", 3) == "B" and JU.parse_listener("(C).", 3) == "C"
    assert JU.parse_listener("J", 3) is None and JU.parse_listener("I think", 3) is None
    assert JU.debate_reply("  Air is\n mostly argon. ", "C", "fb") == ("C", "Air is mostly argon.", True)
    assert JU.debate_reply("", "C", "fb") == ("C", "fb", False)


def test_ju_sampling_kwargs(tmp_path):
    import json
    from debate_ppo import baseline_ju as JU
    assert JU.sampling_kwargs(str(tmp_path)) == {"temperature": 0.0}
    json.dump({"do_sample": True, "temperature": 0.7, "top_p": 0.8, "top_k": 20, "repetition_penalty": 1.05},
              open(tmp_path / "generation_config.json", "w"))
    assert JU.sampling_kwargs(str(tmp_path)) == {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "repetition_penalty": 1.05}


def test_ju_preference_pairs():
    from debate_ppo import baseline_ju as JU
    from debate_ppo.ju_dpo.make_data import build_pairs
    rows = [{"task_id": "t1", "question": _JU_Q, "target": "C", "strategy": s, "evidence": f"msg {s}", "is_correct": k < 2}
            for k, s in enumerate(JU.STRATEGIES)]
    rows += [{"task_id": "t2", "question": _JU_Q, "target": "A", "strategy": s, "evidence": f"x {s}", "is_correct": False}
             for s in JU.STRATEGIES]
    rows.append({"task_id": "t2", "question": _JU_Q, "target": "A", "strategy": "conformity", "evidence": "  ", "is_correct": True})
    data, st = build_pairs(rows)
    assert st == {"groups": 2, "groups_dropped_no_success_or_no_failure": 1, "pairs": 5}
    win = {f"msg {s}" for s in JU.STRATEGIES[:2]}
    assert all(d["task_id"] == "t1" and d["chosen"] in win and d["rejected"] not in win for d in data)
    assert all(d["prompt"] == JU.dpo_prompt(_JU_Q, "C") for d in data)
    assert len({(d["chosen"], d["rejected"]) for d in data}) == 5   # 2 x 10 combos: no repeats needed
    assert build_pairs(rows)[0] == data                              # seeded
