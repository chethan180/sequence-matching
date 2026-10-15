"""Clarification: how the daily budget of 12 units is spent.

The hard-constraint bundle of one member costs 3 units and one soft field costs 1. An answer arrives the same day and
is the person's true self-report, unless they declined it. Only available members can be asked.

Hard bundles come first because they are what makes an introduction possible at all. Soft answers only change how
pairs are ranked, so they are bought with whatever is left, by value of information, and only when asked for.
"""
import random
from kit import HARD, baseline_asks
from .feasibility import feasible_partners
from .model import agreement_weights
from .normalise import SOFT_CORE, is_blocked, is_complete, past_pairs


def hard_bundle_asks(state):
    """Reference rule: the bundle for available members with a missing, not-declined hard field, in list order."""
    return baseline_asks(state)


# ------------------------------------------------------------------ Bayesian ordering of hard-constraint bundles
BAYES_DRAWS = 12


def compatible(a, fa, b, fb):
    """kit.eligibility for two complete answer sets (fa, fb), as a yes/no. Mirrors every hard check, both directions."""
    for y, fx, fy in ((b, fa, fb), (a, fb, fa)):
        if y['gender'] not in fx['who_to_meet'] or not fx['age_min'] <= y['age'] <= fx['age_max']:
            return False
        if y['zone'] not in fx['acceptable_zones']:
            return False
        if fx['partner_smoking'] == 'no_smoking' and fy['smoking'] in ('yes', 'occasionally'):
            return False
        if fx['partner_children'] == 'no_children' and fy['has_children'] is True:
            return False
    if fa['relationship_structure'] != fb['relationship_structure']:
        return False
    if {fa['wants_children'], fb['wants_children']} == {'yes', 'no'}:
        return False
    return bool(set(fa['schedule']) & set(fb['schedule']))


def _templates(state):
    """Complete hard answer sets observed so far, made relative to their owner's age and zone, so they can be lent to
    another person: an age range becomes 'years below / above my own age', extra zones are those beyond one's own."""
    out = []
    for m in state['members']:
        f = m['fields']
        if all(f[k] is not None for k in HARD):
            t = {k: f[k] for k in HARD}
            t['_below'], t['_above'] = m['age'] - f['age_min'], f['age_max'] - m['age']
            t['_extra'] = [z for z in f['acceptable_zones'] if z != m['zone']]
            out.append(t)
    return out


def _imagine(member, template):
    """Member's hard answers: the known ones as they are, each unknown one taken from the borrowed template."""
    f, out = member['fields'], {}
    for k in HARD:
        if f[k] is not None:
            out[k] = f[k]
        elif k == 'age_min':
            out[k] = max(18, member['age'] - template['_below'])
        elif k == 'age_max':
            out[k] = min(65, member['age'] + template['_above'])
        elif k == 'acceptable_zones':
            out[k] = [member['zone']] + [z for z in template['_extra'] if z != member['zone']]
        else:
            out[k] = template[k]
    return out


def match_chances(state, candidates):
    """For each candidate (a member with unknown hard answers), from what is observable now:
        P(at least one partner)  and  expected number of partners it would be feasible with.

    Bayesian by simulation: the prior for a person's unknown hard answers is the set of complete answer sets already
    observed in the pool (borrowed whole, so answers that go together stay together). Each draw imagines the unknown
    answers of every not-yet-complete member at once, then counts the candidate's feasible partners among everyone not
    blocked by a declined answer and not already introduced to them. Deterministic for a given day."""
    templates = _templates(state)
    if not templates:
        return {}
    rng = random.Random('bayes-asks|%d' % state['day'])
    past = past_pairs(state)
    pool = [m for m in state['members'] if not is_blocked(m)]
    hits = {c['member_id']: 0 for c in candidates}
    partners = {c['member_id']: 0 for c in candidates}
    for _ in range(BAYES_DRAWS):
        drawn = {m['member_id']: (m['fields'] if is_complete(m) else _imagine(m, rng.choice(templates))) for m in pool}
        for c in candidates:
            cid, fc, n = c['member_id'], drawn[c['member_id']], 0
            for o in pool:
                oid = o['member_id']
                if oid != cid and (min(cid, oid), max(cid, oid)) not in past and compatible(c, fc, o, drawn[oid]):
                    n += 1
            partners[cid] += n
            hits[cid] += n > 0
    return {cid: (hits[cid] / BAYES_DRAWS, partners[cid] / BAYES_DRAWS) for cid in hits}


def bayes_bundle_asks(state, skip_below=None):
    """The same members the reference rule would ask, ordered by expected number of partners (most first).
    With `skip_below`, members whose chance of any partner is below it are not asked today (they are re-scored daily)."""
    candidates = [m for m in state['members'] if m['available'] and not is_complete(m) and not is_blocked(m)]
    if not candidates:
        return []
    chances = match_chances(state, candidates)
    if not chances:
        return baseline_asks(state)
    order = sorted(candidates, key=lambda m: (-chances[m['member_id']][1], m['member_id']))
    if skip_below is not None:
        order = [m for m in order if chances[m['member_id']][0] >= skip_below]
    return [{'member_id': m['member_id'], 'field': 'constraints'} for m in order[:state['ask_budget_remaining'] // 3]]


def voi_soft_asks(state, spent_units):
    """Spend leftover units (1 each) on the soft answers with the highest value of information.

    A soft answer only matters through pairs we could still introduce, so the value of asking member m about field k is
        weight_k x sum over m's feasible, not-yet-introduced partners j of
            1.0 if j's answer on k is already known (the ask settles whether they agree)
            0.5 if j's answer is unknown but can still be asked (a second ask is needed)
            0.0 if j declined it (agreement can never be known)
    Picks are greedy and each pick updates the others. Only available members with a 'not_asked' field are asked,
    and nothing is asked when no ask has positive value."""
    left = state['ask_budget_remaining'] - spent_units
    if left <= 0:
        return []
    complete, partners = feasible_partners(state)
    if not partners:
        return []
    weights = agreement_weights()
    known = {(m['member_id'], k): m['fields'][k] is not None for m in complete for k in SOFT_CORE}
    askable = {(m['member_id'], k) for m in complete if m['available'] and m['member_id'] in partners
               for k in SOFT_CORE if m['fields'][k] is None and m['field_status'][k] == 'not_asked'}
    asks = []
    while len(asks) < left and askable:
        best = None
        for mid, k in askable:
            value = 0.0
            for j in partners[mid]:
                if known[(j['member_id'], k)]:
                    value += 1.0
                elif j['field_status'][k] == 'not_asked':
                    value += 0.5
            value *= weights[k]
            if value > 0 and (best is None or value > best[0] or (value == best[0] and (mid, k) < best[1])):
                best = (value, (mid, k))
        if best is None:
            break
        mid, k = best[1]
        asks.append({'member_id': mid, 'field': k})
        askable.discard((mid, k))
        known[(mid, k)] = True
    return asks


def choose_asks(state, cfg):
    if cfg['ask'] == 'none':
        asks = []
    elif cfg['ask'] == 'bayes':
        asks = bayes_bundle_asks(state, cfg.get('skip_below'))
    elif cfg['ask'] == 'value':
        from . import experimental
        asks = experimental.value_asks(state, state['ask_budget_remaining'])
    else:
        asks = hard_bundle_asks(state)
    if cfg.get('soft_reserve'):   # keep some units back from hard-constraint bundles for soft-field asks
        asks = asks[:max(0, (state['ask_budget_remaining'] - cfg['soft_reserve']) // 3)]
    if cfg.get('soft_asks') == 'voi':
        asks += voi_soft_asks(state, 3 * len(asks))
    elif cfg.get('soft_asks'):
        from . import experimental
        asks += experimental.soft_ask_extras(state, 3 * len(asks))
    return asks
