"""Input handling. Turns the observable JSON state into the views the other layers use; no modelling, no allocation.

A field is known only when its value is not null. Unknown is never replaced by a default: the status separates
`not_asked` (clarification can still reveal it) from `declined` (it stays unavailable for good).
"""
from kit import HARD, SOFT

SOFT_CORE = ['relationship_goal', 'relationship_pace', 'lifestyle', 'conversations']
SOFT_OTHER = ['emotional_availability', 'space_for_relationship', 'relocate']
EMPTY_STATS = {'n': 0, 'answered': 0, 'yes': 0, 'recv_n': 0, 'recv_yes': 0}


def past_pairs(state):
    """Pairs already introduced in this episode, as sorted id tuples. A pair may never be introduced twice."""
    return {tuple(sorted((i['user_a'], i['user_b']))) for i in state['introductions']}


def is_complete(member):
    """Every hard field is known, so the member can take part in a feasible pair."""
    return all(member['fields'][k] is not None for k in HARD)


def is_blocked(member):
    """A hard field was declined. It can never be clarified, so the member can never be introduced."""
    return any(member['field_status'][k] == 'declined' for k in HARD)


def member_stats(state):
    """Per member, from feedback already observed: responses seen, answered, yes, and how others answered them.

    `n` counts every response event including a timed-out one (value null); `answered` and `yes` count real answers.
    `recv_*` counts the answers other people gave about this member.
    """
    ends = {i['introduction_id']: (i['user_a'], i['user_b']) for i in state['introductions']}
    stats = {}

    def slot(mid):
        return stats.setdefault(mid, {'n': 0, 'answered': 0, 'yes': 0, 'recv_n': 0, 'recv_yes': 0})

    for e in state['feedback']:
        if e['event'] != 'introduction_response' or e['member_id'] is None:
            continue
        me = slot(e['member_id'])
        me['n'] += 1
        if e['value'] is not None:
            me['answered'] += 1
            me['yes'] += e['value'] == 'yes'
            a, b = ends.get(e['introduction_id'], (None, None))
            other = b if e['member_id'] == a else a
            if other is not None:
                o = slot(other)
                o['recv_n'] += 1
                o['recv_yes'] += e['value'] == 'yes'
    return stats


# ---------------------------------------------------------------- rejecting records that do not fit the contract
_LIST_FIELDS = ('who_to_meet', 'acceptable_zones', 'schedule')
_INT_FIELDS = ('age_min', 'age_max')


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def member_fits_contract(member):
    """True when a member record has the shape the data contract promises, so the other layers can rely on it."""
    if not isinstance(member, dict):
        return False
    if not all(isinstance(member.get(k), str) for k in ('member_id', 'pool_id', 'gender', 'zone')):
        return False
    if not _is_int(member.get('age')) or not isinstance(member.get('available'), bool):
        return False
    fields, status = member.get('fields'), member.get('field_status')
    if not isinstance(fields, dict) or not isinstance(status, dict):
        return False
    for key in HARD + SOFT:
        if key not in fields or key not in status:
            return False
        value = fields[key]
        if value is None:
            continue
        if key in _LIST_FIELDS:
            if not isinstance(value, list) or not value or not all(isinstance(v, str) for v in value):
                return False
        elif key in _INT_FIELDS:
            if not _is_int(value):
                return False
        elif not isinstance(value, (str, bool)):
            return False
    return True


def _intro_fits(intro):
    return isinstance(intro, dict) and all(isinstance(intro.get(k), str) for k in ('introduction_id', 'user_a', 'user_b'))


def _event_fits(event):
    if not isinstance(event, dict) or not isinstance(event.get('event'), str):
        return False
    if event['event'] != 'introduction_response':
        return True
    return 'member_id' in event and 'value' in event and 'introduction_id' in event


def clean_state(state):
    """The state with records that break the data contract set aside, and how many were set aside.

    Nothing is repaired or defaulted: a member that does not fit is simply not considered today, so one damaged record
    cannot stop everybody else from being served. When everything fits, the original object is returned untouched.
    """
    cleaned, dropped = {}, 0
    for key, fits in (('members', member_fits_contract), ('introductions', _intro_fits), ('feedback', _event_fits)):
        records = state.get(key)
        if not isinstance(records, list):
            cleaned[key] = []
            dropped += 1
            continue
        good = [r for r in records if fits(r)]
        dropped += len(records) - len(good)
        cleaned[key] = good
    for key in ('day', 'ask_budget_remaining'):
        if _is_int(state.get(key)):
            cleaned[key] = state[key]
        else:
            cleaned[key] = 0
            dropped += 1
    if not dropped:
        return state, 0
    merged = dict(state)
    merged.update(cleaned)
    return merged, dropped
