"""Allocation: which feasible pairs to introduce today, each person at most once.

The pool is treated as a graph (members are nodes, feasible pairs are weighted edges) and the pairs are chosen jointly,
because taking the single best pair first can strand people who had no other partner.
"""
from .feasibility import feasible_edges, feasible_partners
from .matching import best_pairs
from .model import pair_values, soft_match_count
from .normalise import past_pairs

# Tiered integer weights: every pair is worth CARDINALITY, every never-introduced person in it COVERAGE, and the
# predicted value at most VALUE_SCALE. The tiers never overlap, so the matcher maximises the number of introductions
# first, then the number of people served for the first time, then predicted value.
CARDINALITY = 10 ** 12
COVERAGE = 10 ** 9
VALUE_SCALE = 10 ** 6
# Optional scarcity bonus for a first-time member with few feasible partners left (3 levels: 1, 2 or 3 partners).
# It sits below the coverage tier (at most 200 x 3 x 10^6 < 10^9) and mostly above the value tier.
SCARCITY = 10 ** 6


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


def tiered_weights(state, edges, cfg):
    """Integer weights for the exact matcher: cardinality, then first-time coverage, then predicted value."""
    served = {x for pair in past_pairs(state) for x in pair}
    values = pair_values(state, edges, adapt=cfg.get('adapt', False)) if cfg['score'] == 'logit' else [0.0] * len(edges)
    level = {}
    if cfg.get('scarcity'):
        # partners still possible for each complete member (busy ones included: they will be free again)
        _, partners = feasible_partners(state)
        level = {mid: max(0, 4 - len(ps)) for mid, ps in partners.items() if mid not in served}
    # optional waiting for a better match: a pair below the bar is not offered today, so both people wait
    waiting = state['day'] < cfg.get('release_day', 10 ** 6)
    min_value = cfg.get('min_value') if waiting else None
    min_soft = cfg.get('min_soft') if waiting else None
    out = []
    for v, (a, b) in zip(values, edges):
        if min_value is not None and v < min_value:
            continue
        if min_soft is not None and soft_match_count(a, b) < min_soft:
            continue
        w = CARDINALITY if cfg.get('cardinality', True) else 0
        if cfg.get('coverage', True):
            w += COVERAGE * ((a['member_id'] not in served) + (b['member_id'] not in served))
        w += SCARCITY * (level.get(a['member_id'], 0) + level.get(b['member_id'], 0))
        w += int(round(min(max(v, 0.0), 1.0) * VALUE_SCALE)) + 1
        out.append((w, a['member_id'], b['member_id']))
    return out


def choose_pairs(state, cfg):
    edges = feasible_edges(state)
    if not edges:
        return []
    if cfg['match'] == 'tiered':
        from .safety import seconds_left
        weighted = tiered_weights(state, edges, cfg)
        if not weighted:
            return []
        if seconds_left() > 2.0:
            return [sorted(p) for p in best_pairs(weighted)]
        return [sorted(p) for p in greedy_matching(weighted)]    # out of time: still valid, merely not optimal
    weighted = edge_weights(state, edges, cfg)
    pairs = global_matching(weighted) if cfg['match'] == 'global' else greedy_matching(weighted)
    return [sorted(p) for p in pairs]
