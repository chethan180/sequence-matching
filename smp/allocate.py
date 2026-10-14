"""Allocation: which feasible pairs to introduce today, each person at most once.

The pool is treated as a graph (members are nodes, feasible pairs are weighted edges) and the pairs are chosen jointly,
because taking the single best pair first can strand people who had no other partner.
"""
from .feasibility import feasible_edges
from .model import pair_values, soft_match_count


def greedy_matching(weighted):
    """Best pair first. `weighted` is a list of (weight, id_a, id_b)."""
    used, out = set(), []
    for w, a, b in sorted(weighted, key=lambda t: (-t[0], t[1], t[2])):
        if a not in used and b not in used:
            out.append([a, b])
            used.update((a, b))
    return out


def _component_matching(nodes, edges):
    """Maximum-weight matching of one connected component."""
    n = len(nodes)
    if n > 20:
        # large component: greedy start, then repeat improving single-edge swaps
        match = {}
        for w, a, b in sorted(edges, key=lambda t: (-t[0], t[1], t[2])):
            if a not in match and b not in match:
                match[a], match[b] = b, a
        weight = {}
        for w, a, b in edges:
            weight[(a, b)] = weight[(b, a)] = w
        improved = True
        while improved:
            improved = False
            for w, a, b in edges:
                if match.get(a) == b:
                    continue
                gain = w
                if a in match:
                    gain -= weight[(a, match[a])]
                if b in match:
                    gain -= weight[(b, match[b])]
                if gain > 1e-12:
                    for x in (a, b):
                        if x in match:
                            y = match.pop(x)
                            match.pop(y, None)
                    match[a], match[b] = b, a
                    improved = True
        return [[a, b] for a, b in match.items() if a < b]
    index = {x: i for i, x in enumerate(nodes)}
    adj = [[] for _ in range(n)]
    for w, a, b in edges:
        adj[index[a]].append((index[b], w))
        adj[index[b]].append((index[a], w))
    memo = {}

    def best(mask):
        if mask == 0:
            return 0.0, ()
        if mask in memo:
            return memo[mask]
        low = (mask & -mask).bit_length() - 1
        rest = mask & ~(1 << low)
        top = best(rest)
        for j, w in adj[low]:
            if rest >> j & 1:
                sub = best(rest & ~(1 << j))
                if sub[0] + w > top[0]:
                    top = (sub[0] + w, ((low, j),) + sub[1])
        memo[mask] = top
        return top

    return [[nodes[i], nodes[j]] for i, j in best((1 << n) - 1)[1]]


def global_matching(weighted):
    """Maximum total weight with each person in at most one pair, solved per connected component."""
    adjacency = {}
    for w, a, b in weighted:
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)
    seen, out = set(), []
    for start in sorted(adjacency):
        if start in seen:
            continue
        comp, stack = [], [start]
        seen.add(start)
        while stack:
            x = stack.pop()
            comp.append(x)
            for y in adjacency[x]:
                if y not in seen:
                    seen.add(y)
                    stack.append(y)
        comp.sort()
        members = set(comp)
        out.extend(_component_matching(comp, [e for e in weighted if e[1] in members]))
    return out


def edge_weights(state, edges, cfg):
    """(weight, id_a, id_b) for every feasible pair. `eps` is a constant per pair that favours more introductions."""
    eps = cfg.get('eps', 0.0)
    if cfg['score'] == 'logit':
        return [(eps + v, a['member_id'], b['member_id']) for v, (a, b) in zip(pair_values(state, edges), edges)]
    if cfg['score'] == 'soft':
        return [(eps + soft_match_count(a, b), a['member_id'], b['member_id']) for a, b in edges]
    from . import experimental
    return experimental.edge_weights(state, edges, cfg)


def choose_pairs(state, cfg):
    edges = feasible_edges(state)
    if not edges:
        return []
    weighted = edge_weights(state, edges, cfg)
    pairs = global_matching(weighted) if cfg['match'] == 'global' else greedy_matching(weighted)
    return [sorted(p) for p in pairs]
