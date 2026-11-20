"""Exercise the actual round loop without importing CUDA/model dependencies."""
import ast
import sys
from pathlib import Path
from types import SimpleNamespace

from topology_logic import adjacency, build_topology
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Experiment5_AgentScaling"))
from scaling_logic import paired_cooperative_added_personalities


def test_paired_round_loop():
    source = Path(__file__).with_name("eval_exp6_topology.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    function = next(x for x in tree.body if isinstance(x, ast.FunctionDef)
                    and x.name == "run_batch")
    edges = build_topology("ring", 9000)
    adj = adjacency(6, edges)
    starts = ([1, 2, 3, 4, 5, 6], [5, 6, 7, 8, 9, 10])

    for mode in ("no_attack", "ppo"):
        seen = []

        def chat(i, personality, positions, messages, neighbors, t):
            seen.append((i, t, list(positions), list(messages), list(neighbors)))
            return (i, t, list(positions))

        class WM:
            def _generate(self, prompts):
                # Episode 0 succeeds immediately; episode 1 only at round 2.
                # First three agree even in the failing first round of ep1:
                # the loop must score all five. Control node never agrees.
                return [(19 if i == 5 else
                         (7 if positions[0] == 1 or t == 1 or i < 3 else i + 10),
                         f"message-{i}-t{t}") for i, t, positions in prompts]

        class Actor:
            def generate_two_phase_batch(self, chats, **kwargs):
                assert all(i == 5 for i, t, positions in chats)
                return [("new attacker output", 20) for _ in chats]

        namespace = dict(
            adjacency=adjacency,
            paired_initial_positions=lambda seed, ep, offset, max_pos, n: list(starts[ep]),
            paired_cooperative_added_personalities=paired_cooperative_added_personalities,
            local_chat=chat, _render_wm_prompt=lambda wm, x: x,
            parse_pos_msg=lambda raw, fallback, max_pos: raw,
            consensus_reached=lambda p, b, tol: len(set(p[:b])) == 1,
        )
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
        args = SimpleNamespace(seed=9000, max_pos=20, max_steps=50, dump_n=2,
                               actor_max_message_tokens=48, consensus_tol=0)
        states = namespace["run_batch"](
            wm=WM(), actor=Actor() if mode == "ppo" else None, args=args,
            topology="ring", edges=edges, episode_indices=range(2), mode=mode)
        assert [s["t"] for s in states] == [1, 2]
        assert all(s["consensus"] for s in states)
        assert len(seen) == 18  # Six agents for 2 initial games + 1 surviving game.
        assert all(s["personalities"].count("malicious") == int(mode == "ppo")
                   for s in states)
        assert all("malicious" not in s["personalities"][:5] for s in states)
        for i, t, positions, messages, neighbors in seen:
            assert neighbors == adj[i]
            if t == 0:
                assert positions in [list(x) for x in starts]
                assert messages == [None] * 6
            else:
                assert positions[:5] == [7, 7, 7, 13, 14]
                assert positions[5] == (20 if mode == "ppo" else 19)
        for state in states:
            for rnd in state["rounds"]:
                for obs in rnd["received_others_per_agent"]:
                    i = obs["agent"]
                    assert [x["agent"] for x in obs["received_others"]] == adj[i]
                    for other in obs["received_others"]:
                        j = other["agent"]
                        assert other["position"] == rnd["snapshot_positions"][j]
                        assert other["message"] == rnd["snapshot_messages"][j]
