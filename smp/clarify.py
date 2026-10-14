"""Clarification: how the daily budget of 12 units is spent.

The hard-constraint bundle of one member costs 3 units and one soft field costs 1. An answer arrives the same day and
is the person's true self-report, unless they declined it. Only available members can be asked.

Hard bundles come first because they are what makes an introduction possible at all. Soft answers only change how
pairs are ranked, so they are bought with whatever is left, by value of information, and only when asked for.
"""
from kit import baseline_asks
from .feasibility import feasible_partners
from .model import agreement_weights
from .normalise import SOFT_CORE


def hard_bundle_asks(state):
    """Reference rule: the bundle for available members with a missing, not-declined hard field, in list order."""
    return baseline_asks(state)


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
