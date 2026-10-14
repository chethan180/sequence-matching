"""Research-only variants from the Round 1 experiments. None of this is on the default path.

They are kept so that the earlier studies stay reproducible: an online-only response rate, Thompson sampling,
boosted trees, a staged second-meeting term, asks ranked by unlock potential, and a simple soft-ask rule.
"""
import random
from kit import eligibility
from .feasibility import feasible_edges
from .model import (PRIOR_ANSWER, PRIOR_STRENGTH, PRIOR_YES, directional_features, load_model, logistic_p, shrunk_rates,
                    sigmoid, soft_match_count)
from .normalise import EMPTY_STATS, SOFT_CORE, is_blocked, is_complete, member_stats, past_pairs

CONFIGS = {
    'ref_greedy':      dict(ask='baseline', match='greedy', score='soft'),
    'global_learned':  dict(ask='baseline', match='global', score='learned', eps=0.05, soft_w=0.15),
    'value_ask_only':  dict(ask='value', match='greedy', score='soft'),
    'value_global':    dict(ask='value', match='global', score='soft', eps=1.0),
    'smart':           dict(ask='value', match='global', score='learned', eps=0.05, soft_w=0.15),
    'soft_greedy':     dict(ask='baseline', match='greedy', score='soft', soft_asks=True),
    'value_soft':      dict(ask='value', match='global', score='soft', eps=1.0, soft_asks=True),
    'smart_soft':      dict(ask='value', match='global', score='learned', eps=0.05, soft_w=0.15, soft_asks=True),
    'logit_staged':    dict(ask='baseline', match='global', score='logit_staged', eps=0.01),
    'gbm_global':      dict(ask='baseline', match='global', score='gbm', eps=0.02),
    'thompson_global': dict(ask='baseline', match='global', score='thompson', eps=0.02, soft_w=0.15),
    'gbm_value':       dict(ask='value', match='global', score='gbm', eps=0.02),
    'logit_value':     dict(ask='value', match='global', score='logit', eps=0.02),
    'final_soft':      dict(ask='baseline', match='global', score='logit', eps=0.02, soft_asks=True),
    'final_soft_res':  dict(ask='baseline', match='global', score='logit', eps=0.02, soft_asks=True, soft_reserve=3),
    'final_voi_res':   dict(ask='baseline', match='global', score='logit', eps=0.02, soft_asks='voi', soft_reserve=3),
}


def propensities(state):
    """P(yes) from each person's own shrunk record alone (no compatibility)."""
    seen = {}
    for event in state['feedback']:
        if event['event'] != 'introduction_response' or event['member_id'] is None:
            continue
        s = seen.setdefault(event['member_id'], [0, 0, 0])  # total, answered, yes
        s[0] += 1
        if event['value'] is not None:
            s[1] += 1
            s[2] += event['value'] == 'yes'
    out = {}
    for m in state['members']:
        total, answered, yes = seen.get(m['member_id'], (0, 0, 0))
        p_answer = (answered + PRIOR_STRENGTH * PRIOR_ANSWER) / (total + PRIOR_STRENGTH)
        p_yes = (yes + PRIOR_STRENGTH * PRIOR_YES) / (answered + PRIOR_STRENGTH)
        out[m['member_id']] = p_answer * p_yes
    return out


def gbm_p(spec, x):
    raw = spec['init']
    for tree in spec['trees']:
        node = 0
        while tree['left'][node] != -1:
            node = tree['left'][node] if x[tree['feature'][node]] <= tree['threshold'][node] else tree['right'][node]
        raw += spec['lr'] * tree['value'][node]
    return sigmoid(raw)


def thompson_member_scores(state, stats):
    """One Beta draw per member for P(yes | answered), times the shrunk answer rate. Deterministic given the day."""
    rng = random.Random(7919 * (state['day'] + 1))
    out = {}
    for m in sorted(state['members'], key=lambda m: m['member_id']):
        st = stats.get(m['member_id'], EMPTY_STATS)
        p_answer, _ = shrunk_rates(st)
        a = PRIOR_STRENGTH * PRIOR_YES + st['yes']
        b = PRIOR_STRENGTH * (1 - PRIOR_YES) + (st['answered'] - st['yes'])
        out[m['member_id']] = p_answer * rng.betavariate(a, b)
    return out


def edge_weights(state, edges, cfg):
    eps = cfg.get('eps', 0.0)
    kind = cfg['score']
    if kind == 'learned':
        prop = propensities(state)
        weighted = []
        for a, b in edges:
            p = prop[a['member_id']] * prop[b['member_id']]
            if p < cfg.get('min_p', 0.0):
                continue
            p *= 1.0 + cfg.get('soft_w', 0.0) * soft_match_count(a, b)
            weighted.append((eps + p, a['member_id'], b['member_id']))
        return weighted
    if kind not in ('logit_staged', 'gbm', 'thompson'):
        raise ValueError('unknown score: ' + str(kind))
    model = load_model()
    stats = member_stats(state)
    day = state['day']
    fallback = propensities(state)
    draw = thompson_member_scores(state, stats) if kind == 'thompson' else None
    weighted = []
    for a, b in edges:
        if kind == 'thompson':
            w = draw[a['member_id']] * draw[b['member_id']] * (1.0 + cfg.get('soft_w', 0.0) * soft_match_count(a, b))
        else:
            xa = directional_features(a, b, stats, day)
            xb = directional_features(b, a, stats, day)
            if kind == 'gbm' and 'gbm' in model:
                w = gbm_p(model['gbm'], xa) * gbm_p(model['gbm'], xb)
            elif 'logit' in model:
                w = logistic_p(model['logit'], xa) * logistic_p(model['logit'], xb)
            else:
                w = fallback[a['member_id']] * fallback[b['member_id']]
            if kind == 'logit_staged' and 'second' in model:
                w *= logistic_p(model['second'], xa) * logistic_p(model['second'], xb)
        weighted.append((eps + w, a['member_id'], b['member_id']))
    return weighted


def value_asks(state, budget):
    """Hard bundles ranked by how many other members the person could still be feasible with."""
    members = state['members']
    past = past_pairs(state)
    askable = [m for m in members if m['available'] and not is_complete(m) and not is_blocked(m)]
    if not askable:
        return []
    scored = []
    for c in askable:
        value = 0.0
        for o in members:
            if o['member_id'] == c['member_id'] or is_blocked(o):
                continue
            if tuple(sorted((c['member_id'], o['member_id']))) in past:
                continue
            if eligibility(c, o)['status'] == 'infeasible':
                continue
            value += 1.0 if is_complete(o) else 0.3
        scored.append((-value, c['member_id']))
    scored.sort()
    return [{'member_id': mid, 'field': 'constraints'} for _, mid in scored[:budget // 3]]


def soft_ask_extras(state, spent_units):
    """Leftover budget: ask missing core soft fields of members who sit in a feasible pair today, field by field."""
    left = state['ask_budget_remaining'] - spent_units
    if left <= 0:
        return []
    in_edge = {}
    for a, b in feasible_edges(state):
        in_edge[a['member_id']] = a
        in_edge[b['member_id']] = b
    asks = []
    for k in SOFT_CORE:
        for mid in sorted(in_edge):
            m = in_edge[mid]
            if m['fields'][k] is None and m['field_status'][k] == 'not_asked':
                asks.append({'member_id': mid, 'field': k})
                if len(asks) == left:
                    return asks
    return asks
