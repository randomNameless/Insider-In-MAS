"""CPU-only differential test of the actual Exp5/Exp6 prompt and loop code.

Uses deterministic model doubles, not a GPU reproducibility claim.
"""
import ast
import hashlib
import json
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
EXP5 = HERE.parent / "Experiment5_AgentScaling"
sys.path.insert(0, str(EXP5))
from scaling_logic import (paired_initial_positions, consensus_reached,
                           paired_cooperative_added_personalities)
from topology_logic import adjacency, build_topology


def load_functions(path, names, namespace):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = [node for node in tree.body
             if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(nodes) == len(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def test_complete_matches_exp5_b5():
    prompts = runpy.run_path(str(HERE.parents[1] / "attacker" / "prompts.py"))
    common = dict(
        paired_initial_positions=paired_initial_positions,
        paired_cooperative_added_personalities=paired_cooperative_added_personalities,
        consensus_reached=consensus_reached, adjacency=adjacency,
        ROUND0_PROMPT_TEMPLATE=prompts["ROUND0_PROMPT_TEMPLATE"],
        MULTIROUND_PROMPT_TEMPLATE=prompts["MULTIROUND_PROMPT_TEMPLATE"],
        build_system_prompt=prompts["build_system_prompt"],
        build_attacker_system_prompt=lambda: prompts["build_system_prompt"]("malicious"),
        _render_wm_prompt=lambda wm, chat: chat,
        parse_pos_msg=lambda raw, fallback, max_pos: raw,
    )
    exp5 = load_functions(EXP5 / "eval_exp5_agent_scaling.py",
                          {"variable_chat", "received_others_snapshot", "run_batch"},
                          dict(common))
    exp6 = load_functions(HERE / "eval_exp6_topology.py",
                          {"local_chat", "run_batch"}, dict(common))

    # Confirm horizon-dependent initial states, multiple seeds and batch offsets.
    for seed, horizon, start in ((9000, 50, 0), (9001, 50, 7), (9000, 20, 0)):
        for mode in ("no_attack", "ppo"):
            results, events = [], []
            for version in (exp5, exp6):
                calls = []

                class WM:
                    step = 0

                    def _generate(self, chats):
                        calls.append(("wm", chats))
                        self.step += 1
                        width = 6 if mode == "no_attack" else 5
                        outputs = []
                        for index, chat in enumerate(chats):
                            episode, agent = divmod(index, width)
                            # First episode stops immediately, second at round2,
                            # third times out, testing active-batch compaction.
                            settled = episode == 0 and self.step <= 2
                            position = 7 if settled else agent
                            digest = hashlib.sha256(json.dumps(chat).encode()).hexdigest()
                            outputs.append((position, digest))
                        return outputs

                class Actor:
                    def generate_two_phase_batch(self, chats, **kwargs):
                        calls.append(("actor", chats, kwargs))
                        return [(hashlib.sha256(json.dumps(chat).encode()).hexdigest(),
                                 None if index == 0 else 30)
                                for index, chat in enumerate(chats)]

                args = SimpleNamespace(seed=seed, max_steps=horizon, max_pos=20,
                                       dump_n=200, actor_max_message_tokens=48,
                                       consensus_tol=0)
                kw = dict(wm=WM(), actor=Actor(), args=args, mode=mode,
                          episode_indices=range(start, start + 3))
                if version is exp5:
                    kw["n_benign"] = 5
                else:
                    kw.update(topology="complete", edges=build_topology("complete", seed))
                states = version["run_batch"](**kw)
                # Exp6's extra topology audit metadata has no Exp5 equivalent.
                for state in states:
                    for rnd in state["rounds"]:
                        rnd.pop("visible_neighbors", None)
                results.append(states)
                events.append(calls)
            assert events[0] == events[1], (seed, horizon, mode, "model inputs/order")
            assert results[0] == results[1], (seed, horizon, mode, "states/traces")
            assert [s["t"] for s in results[0]] == [1, 2, horizon]


if __name__ == "__main__":
    test_complete_matches_exp5_b5()
    print("PASS: Exp6 complete matches Exp5 B5 prompts, calls and trajectories (model doubles).")
