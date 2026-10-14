"""Reciprocal eligibility. The supplied `eligibility()` is the only gate: no score can override it.

A pair is feasible only when every hard constraint is known for both people and satisfied in both directions.
"""
import itertools
from kit import eligibility
from .normalise import is_complete, past_pairs


def feasible_edges(state):
    """Pairs that could be introduced today: both available, feasible, and never introduced before."""
    past = past_pairs(state)
    # a feasible pair needs every hard field of both people, so incomplete members are skipped up front
    avail = [m for m in state['members'] if m['available'] and is_complete(m)]
    edges = []
    for a, b in itertools.combinations(avail, 2):
        key = tuple(sorted((a['member_id'], b['member_id'])))
        if key in past or eligibility(a, b)['status'] != 'feasible':
            continue
        edges.append((a, b))
    return edges


def feasible_partners(state):
    """The complete members, and for each the partners they could still be introduced to (busy members included)."""
    past = past_pairs(state)
    complete = [m for m in state['members'] if is_complete(m)]
    partners = {}
    for a, b in itertools.combinations(complete, 2):
        key = tuple(sorted((a['member_id'], b['member_id'])))
        if key in past or eligibility(a, b)['status'] != 'feasible':
            continue
        partners.setdefault(a['member_id'], []).append(b)
        partners.setdefault(b['member_id'], []).append(a)
    return complete, partners
