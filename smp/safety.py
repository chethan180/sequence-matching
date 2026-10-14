"""Safety: no response may ever be invalid.

One invalid ask or pair, a crash or malformed output makes an assessed episode invalid, and one invalid episode makes
the whole entry ineligible. So every response passes through these guards:

0. Input check. Records that do not fit the data contract are set aside (never repaired), so one damaged record
   cannot stop everybody else from being served.
1. Validation. Each ask and each pair is re-checked against the same rules the simulator enforces, using only the
   observation. Anything that would be rejected is dropped; the rest is kept in order.
2. Fallback. If the decision code raises, simpler rules are tried in turn; the last resort is an empty list, which is
   always a valid action (ask nothing, or wait one day).
3. A wall-clock guard that the allocation layer can consult before starting expensive work.

Nothing here raises. Every drop or fallback is recorded in EVENTS and reported on stderr, which is the diagnostics channel.
"""
import sys
import time
from kit import SOFT, eligibility
from .normalise import clean_state

BUNDLE = 'constraints'
BUNDLE_COST = 3
SOFT_COST = 1
EVENTS = []          # (kind, detail) for every drop or fallback in this process
_STARTED = time.monotonic()
TIME_BUDGET = 6.0    # seconds of compute we allow ourselves; the hard limit per call is 10 including start-up


def seconds_left():
    return TIME_BUDGET - (time.monotonic() - _STARTED)


def restart_clock():
    global _STARTED
    _STARTED = time.monotonic()


def _note(kind, detail):
    if len(EVENTS) < 1000:
        EVENTS.append((kind, detail))
    try:
        print('policy safety: %s: %s' % (kind, detail), file=sys.stderr)
    except Exception:
        pass


def validate_asks(state, asks):
    """The asks the simulator will accept, in the order given. Mirrors Simulator.resolve_asks."""
    out = []
    try:
        members = {m['member_id']: m for m in state.get('members', [])}
        budget = int(state.get('ask_budget_remaining', 0))
    except Exception as error:
        _note('asks dropped', 'unreadable state: %r' % (error,))
        return out
    seen, spent = set(), 0
    for ask in asks if isinstance(asks, (list, tuple)) else []:
        try:
            if not isinstance(ask, dict) or set(ask) != {'member_id', 'field'}:
                raise ValueError('malformed ask')
            mid, field = ask['member_id'], ask['field']
            if not isinstance(mid, str) or not isinstance(field, str):
                raise ValueError('ids and fields must be strings')
            member = members.get(mid)
            if member is None or not member.get('available'):
                raise ValueError('member missing or unavailable')
            if field != BUNDLE and field not in SOFT:
                raise ValueError('unknown field')
            if (mid, field) in seen:
                raise ValueError('duplicate ask')
            cost = BUNDLE_COST if field == BUNDLE else SOFT_COST
            if spent + cost > budget:
                raise ValueError('over budget')
        except Exception as error:
            _note('ask dropped', '%r: %s' % (ask, error))
            continue
        seen.add((mid, field))
        spent += cost
        out.append({'member_id': mid, 'field': field})
    return out


def validate_pairs(state, pairs):
    """The pairs the simulator will accept, in the order given. Mirrors Simulator.advance."""
    out = []
    try:
        members = {m['member_id']: m for m in state.get('members', [])}
        past = {tuple(sorted((i['user_a'], i['user_b']))) for i in state.get('introductions', [])}
    except Exception as error:
        _note('pairs dropped', 'unreadable state: %r' % (error,))
        return out
    used = set()
    for pair in pairs if isinstance(pairs, (list, tuple)) else []:
        try:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise ValueError('a pair needs exactly two ids')
            a, b = pair
            if not isinstance(a, str) or not isinstance(b, str) or a == b:
                raise ValueError('two different string ids needed')
            ma, mb = members.get(a), members.get(b)
            if ma is None or mb is None:
                raise ValueError('unknown member')
            if not ma.get('available') or not mb.get('available'):
                raise ValueError('member unavailable')
            if a in used or b in used:
                raise ValueError('person used twice in the batch')
            if tuple(sorted((a, b))) in past:
                raise ValueError('pair already introduced')
            if eligibility(ma, mb)['status'] != 'feasible':
                raise ValueError('not feasible')
        except Exception as error:
            _note('pair dropped', '%r: %s' % (pair, error))
            continue
        used.update((a, b))
        out.append([a, b])
    return out


def _clean_memory(memory):
    return memory if isinstance(memory, (dict, list)) else {}


def guarded(request, primary, fallbacks=()):
    """Run `primary(request)`; on any exception try each fallback in turn; validate whatever comes back.

    Always returns a response of the right shape with only valid actions. Never raises.
    """
    restart_clock()          # each request gets the full time budget (one request per process in the container)
    phase = request.get('phase') if isinstance(request, dict) else None
    key = 'pairs' if phase == 'match' else 'asks'
    state = request.get('state') if isinstance(request, dict) else None
    memory = _clean_memory(request.get('memory') if isinstance(request, dict) else None)
    if not isinstance(state, dict):
        _note('empty response', 'request has no state')
        return {key: [], 'memory': memory}
    check = validate_pairs if key == 'pairs' else validate_asks
    try:
        state, dropped = clean_state(state)
        if dropped:
            _note('input', '%d record(s) outside the data contract set aside' % dropped)
            request = dict(request, state=state)
    except Exception as error:
        _note('input', 'state could not be checked: %r' % (error,))
    for attempt, rule in enumerate((primary,) + tuple(fallbacks)):
        try:
            response = rule(request)
            items = check(state, response[key])
            return {key: items, 'memory': _clean_memory(response.get('memory', memory))}
        except Exception as error:
            _note('fallback', 'rule %d failed in phase %s: %r' % (attempt, phase, error))
    return {key: [], 'memory': memory}
