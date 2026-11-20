"""Pure graph and paired-sampling helpers for Experiment 6."""
from __future__ import annotations

import random
from itertools import permutations
from collections import deque


TOPOLOGIES = (
    "complete",
    "ring",
    "path_attacker_endpoint",
    "path_attacker_middle",
    "star_attacker_center",
    "star_attacker_leaf",
)

ORIGINAL_BENIGN_PERSONALITIES = ("stubborn", "suggestible", "neutral")


def topology_benign_personalities(seed, episode, n_benign=5):
    """Historical V5 only; V6 uses Exp5's personality helper instead.

    Pair and balance a 1-stubborn/2-suggestible/2-neutral roster over slots.

    Every block of 30 episodes visits all unique assignments. Thus benign hub
    roles are not permanently assigned to the stubborn agent. Both arms and
    all graphs use the same assignment for each seed/episode.
    """
    if n_benign != 5:
        raise ValueError("six-node Exp6 requires exactly five scored benign agents")
    assignments = sorted(set(permutations(
        ("stubborn", "suggestible", "suggestible", "neutral", "neutral"))))
    block, offset = divmod(int(episode), len(assignments))
    random.Random(int(seed) + 60_000_000 + block * 1_000_003).shuffle(assignments)
    return list(assignments[offset])


def _canon(edges):
    return sorted({(min(int(a), int(b)), max(int(a), int(b)))
                   for a, b in edges if int(a) != int(b)})


def adjacency(n_nodes, edges):
    out = [set() for _ in range(n_nodes)]
    for a, b in _canon(edges):
        if not (0 <= a < n_nodes and 0 <= b < n_nodes):
            raise ValueError(f"edge {(a, b)} is outside 0..{n_nodes - 1}")
        out[a].add(b)
        out[b].add(a)
    return [sorted(row) for row in out]


def is_connected(n_nodes, edges, nodes=None):
    nodes = set(range(n_nodes) if nodes is None else nodes)
    if not nodes:
        return True
    adj = adjacency(n_nodes, edges)
    start = next(iter(nodes))
    seen, todo = {start}, [start]
    while todo:
        u = todo.pop()
        for v in adj[u]:
            if v in nodes and v not in seen:
                seen.add(v)
                todo.append(v)
    return seen == nodes


def _random_regular(n_nodes, degree, rng):
    if degree >= n_nodes or (n_nodes * degree) % 2:
        raise ValueError("invalid regular graph dimensions")
    for _ in range(20000):
        stubs = [node for node in range(n_nodes) for _ in range(degree)]
        rng.shuffle(stubs)
        edges, ok = set(), True
        for idx in range(0, len(stubs), 2):
            a, b = stubs[idx], stubs[idx + 1]
            edge = (min(a, b), max(a, b))
            if a == b or edge in edges:
                ok = False
                break
            edges.add(edge)
        if ok and is_connected(n_nodes, edges):
            return sorted(edges)
    raise RuntimeError("could not sample a simple connected regular graph")


def _small_world(n_nodes, k, beta, rng):
    # Watts-Strogatz: start from a k-nearest ring and rewire clockwise edges.
    if k % 2 or not 0 < k < n_nodes:
        raise ValueError("k must be positive, even, and smaller than n_nodes")
    edges = set()
    for u in range(n_nodes):
        for distance in range(1, k // 2 + 1):
            v = (u + distance) % n_nodes
            edges.add((min(u, v), max(u, v)))
    for u in range(n_nodes):
        for distance in range(1, k // 2 + 1):
            v = (u + distance) % n_nodes
            old = (min(u, v), max(u, v))
            if old not in edges or rng.random() >= beta:
                continue
            candidates = [w for w in range(n_nodes)
                          if w != u and (min(u, w), max(u, w)) not in edges]
            if candidates:
                edges.remove(old)
                w = rng.choice(candidates)
                edges.add((min(u, w), max(u, w)))
    return sorted(edges)


def build_topology(name, topology_seed, n_benign=5):
    """Build one fixed graph; malicious node is always index n_benign."""
    n_nodes, attacker = n_benign + 1, n_benign
    rng = random.Random((int(topology_seed) + 1) * 1_000_003 + 611)
    if name == "complete":
        edges = [(u, v) for u in range(n_nodes) for v in range(u + 1, n_nodes)]
    elif name == "ring":
        edges = [(u, (u + 1) % n_nodes) for u in range(n_nodes)]
    elif name == "path_attacker_endpoint":
        order = [attacker] + list(range(n_benign))
        edges = list(zip(order, order[1:]))
    elif name == "path_attacker_middle":
        order = list(range(n_benign))
        order.insert(n_benign // 2, attacker)
        edges = list(zip(order, order[1:]))
    elif name == "star_attacker_center":
        edges = [(attacker, node) for node in range(n_benign)]
    elif name == "star_attacker_leaf":
        center = 0
        edges = [(center, node) for node in range(1, n_nodes)]
    else:
        raise ValueError(f"unknown topology {name!r}; choose from {TOPOLOGIES}")
    return _canon(edges)


def _distances(n_nodes, edges, source, allowed=None):
    allowed = set(range(n_nodes) if allowed is None else allowed)
    adj = adjacency(n_nodes, edges)
    dist = {source: 0}
    todo = deque([source])
    while todo:
        u = todo.popleft()
        for v in adj[u]:
            if v in allowed and v not in dist:
                dist[v] = dist[u] + 1
                todo.append(v)
    return dist


def attacker_betweenness(n_nodes, edges, attacker):
    """Unnormalised fraction of unordered shortest paths crossing attacker."""
    others = [node for node in range(n_nodes) if node != attacker]
    hit = total = 0.0
    adj = adjacency(n_nodes, edges)
    for i, source in enumerate(others):
        for target in others[i + 1:]:
            # Count shortest paths with a small BFS dynamic programme.
            dist = {source: 0}
            paths = {source: 1}
            todo = deque([source])
            while todo:
                u = todo.popleft()
                for v in adj[u]:
                    if v not in dist:
                        dist[v] = dist[u] + 1
                        paths[v] = paths[u]
                        todo.append(v)
                    elif dist[v] == dist[u] + 1:
                        paths[v] += paths[u]
            if target not in dist:
                continue
            shortest = dist[target]
            total += 1.0
            da = _distances(n_nodes, edges, source)
            db = _distances(n_nodes, edges, attacker)
            if (attacker in da and target in db
                    and da[attacker] + db[target] == shortest):
                # Exact path fraction is unnecessary for audit metadata; record
                # whether the attacker lies on at least one shortest path.
                hit += 1.0
    return hit / total if total else 0.0


def graph_metrics(n_nodes, edges, attacker):
    adj = adjacency(n_nodes, edges)
    distances = []
    diameter = 0
    for source in range(n_nodes):
        dist = _distances(n_nodes, edges, source)
        for target in range(source + 1, n_nodes):
            if target in dist:
                distances.append(dist[target])
                diameter = max(diameter, dist[target])
    benign_nodes = range(attacker)
    return {
        "n_nodes": n_nodes,
        "n_edges": len(_canon(edges)),
        "average_degree": 2.0 * len(_canon(edges)) / n_nodes,
        "diameter": diameter if is_connected(n_nodes, edges) else None,
        "average_shortest_path": (sum(distances) / len(distances)
                                  if len(distances) == n_nodes * (n_nodes - 1) // 2
                                  else None),
        "attacker_degree": len(adj[attacker]),
        "attacker_betweenness_presence": attacker_betweenness(
            n_nodes, edges, attacker),
        "graph_connected": is_connected(n_nodes, edges),
        "benign_subgraph_connected": is_connected(
            n_nodes, edges, benign_nodes),
        "degrees": [len(row) for row in adj],
    }
