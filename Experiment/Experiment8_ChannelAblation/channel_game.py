"""Dependency-free synchronous channel intervention and auditable trajectories."""
from copy import deepcopy

CHANNELS = {
    "full_attack": ("ppo", "ppo"),
    "ppo_message_only": ("ppo", "neutral"),
    "ppo_position_only": ("neutral", "ppo"),
    "no_attack": ("neutral", "neutral"),
}
PERSONALITIES = ["stubborn", "suggestible", "neutral", "neutral"]


def snapshot(env):
    return {"t": int(env.t), "positions": [int(x) for x in env.pos],
            "messages": deepcopy(list(env.last_messages))}


def action(pair):
    pos, msg = pair
    if not isinstance(msg, str):
        raise ValueError("Expected a parsed string message")
    return {"position": int(pos), "message": msg}


def select_action(condition, ppo, neutral):
    message_source, position_source = CHANNELS[condition]
    candidates = {"ppo": ppo, "neutral": neutral}
    return {"message": candidates[message_source]["message"],
            "position": candidates[position_source]["position"]}


def run_batch(envs, initial_positions, condition, build_inputs, predict_ppo,
              predict_victim):
    """Callbacks never receive a staged action. Victim outputs are 0,1,2,neutral3.

    Both candidate policies are sampled in EVERY condition, including no_attack;
    only select_action determines what enters the game. No candidate replay.
    """
    if len(envs) != len(initial_positions) or not envs:
        raise ValueError("A nonempty, aligned batch is required")
    if condition not in CHANNELS:
        raise ValueError(condition)
    states = []
    for env, initial in zip(envs, initial_positions):
        env.reset(options={"positions": list(initial),
                           "personalities": list(PERSONALITIES)})
        states.append({"env": env, "done": False, "consensus": False,
                       "ep_len": 0, "rounds": []})
    while any(not s["done"] for s in states):
        active = [s for s in states if not s["done"]]
        before = [snapshot(s["env"]) for s in active]
        inputs = [build_inputs(s["env"]) for s in active]
        saved_inputs = deepcopy(inputs)

        def unchanged():
            if [snapshot(s["env"]) for s in active] != before:
                raise RuntimeError("Round-start state changed before atomic commit")
            if inputs != saved_inputs:
                raise RuntimeError("Generation callback changed its input prompts")

        unchanged()
        ppo_outputs = predict_ppo(inputs)
        unchanged()
        victim_outputs = predict_victim(inputs)
        unchanged()
        if len(ppo_outputs) != len(active) or len(victim_outputs) != len(active):
            raise ValueError("Generation returned the wrong batch size")
        for state, old, prompts, ppo_pair, victim_pairs in zip(
                active, before, saved_inputs, ppo_outputs, victim_outputs):
            if len(victim_pairs) != 4:
                raise ValueError("Need benign 0..2 plus neutral candidate 3")
            env = state["env"]
            ppo, neutral = action(ppo_pair), action(victim_pairs[3])
            submitted = select_action(condition, ppo, neutral)
            expected_actions = [action(x) for x in victim_pairs[:3]] + [submitted]
            if any(not 0 <= a["position"] <= 20 for a in expected_actions):
                raise ValueError("Out-of-range parsed position")
            # The original environment API stages agent 3 here, AFTER generation.
            env.prepare(submitted["position"])
            env._pending_atk_msg = submitted["message"]
            _, _, terminated, truncated, _ = env.apply(victim_pairs[:3])
            after = snapshot(env)
            if after["positions"] != [a["position"] for a in expected_actions]:
                raise RuntimeError("Environment did not commit the chosen positions")
            if after["messages"] != [a["message"] for a in expected_actions]:
                raise RuntimeError("Environment did not commit the chosen messages")
            if after["t"] != old["t"] + 1:
                raise RuntimeError("Unexpected environment round increment")
            if bool(terminated) != (len(set(after["positions"][:3])) == 1):
                raise RuntimeError("Consensus is not exact agreement of agents 0..2")
            state["consensus"] = bool(terminated)
            state["done"] = bool(terminated or truncated)
            state["ep_len"] += 1
            if bool(truncated) and not bool(terminated) and state["ep_len"] != env.max_steps:
                raise RuntimeError("Environment truncated before the configured horizon")
            if state["ep_len"] > env.max_steps:
                raise RuntimeError("Environment failed to enforce the horizon")
            state["rounds"].append({
                "t": old["t"], "snapshot": old, "inputs": prompts,
                "ppo_candidate": ppo, "neutral_candidate": neutral,
                "submitted_agent3": submitted,
                "message_source": CHANNELS[condition][0],
                "position_source": CHANNELS[condition][1],
                "benign_outputs": expected_actions[:3], "after": after,
                "consensus_after": bool(terminated), "truncated": bool(truncated),
            })
    return [{k: v for k, v in s.items() if k not in ("env", "done")} for s in states]
