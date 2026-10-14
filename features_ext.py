"""Extended feature builder used for the feature study (research only; winners get ported to policy.py).

Every feature describes how likely `actor` is to say yes to `partner`, using only the observation at the
start of the day: profiles, previously answered feedback (observed_day <= today), and the introduction list.
Nothing is read from the simulator's hidden state.
"""
import kit
import policy

GENDERS = ['woman', 'man', 'non_binary']
SMOKE = {'no': 0, 'occasionally': 1, 'yes': 2}
WANTS = ['yes', 'no', 'unsure']
SOFT_CATS = {k: kit.OPTIONS[k] for k in kit.SOFT}


def ext_stats(state):
    """Chronological per-member record built only from feedback already observed."""
    ends = {i['introduction_id']: (i['user_a'], i['user_b']) for i in state['introductions']}
    answers = {}
    recv = {}
    n_intros = {}
    for i in state['introductions']:
        for m in (i['user_a'], i['user_b']):
            n_intros[m] = n_intros.get(m, 0) + 1
    events = sorted((e for e in state['feedback'] if e['event'] == 'introduction_response' and e['member_id']),
                    key=lambda e: (e['observed_day'], e['introduction_id']))
    for e in events:
        answers.setdefault(e['member_id'], []).append((e['observed_day'], e['value']))
        if e['value'] is not None:
            a, b = ends.get(e['introduction_id'], (None, None))
            other = b if e['member_id'] == a else a
            if other:
                recv.setdefault(other, []).append(e['value'] == 'yes')
    return {'answers': answers, 'recv': recv, 'n_intros': n_intros}


def _hist(stats, mid, day, prefix, out, groups, group):
    ans = stats['answers'].get(mid, [])
    answered = [v for _, v in ans if v is not None]
    yes = sum(v == 'yes' for v in answered)
    n = len(ans)
    p_answer = (len(answered) + 3 * policy.PRIOR_ANSWER) / (n + 3)
    p_yes = (yes + 3 * policy.PRIOR_YES) / (len(answered) + 3)
    last = answered[-1] if answered else None
    recent = answered[-2:]
    rv = stats['recv'].get(mid, [])
    rows = {
        'n': n, 'answered': len(answered), 'yes': yes, 'p_answer': p_answer, 'p_yes': p_yes,
        'last_yes': 0.5 if last is None else float(last == 'yes'),
        'recent_yes': (sum(v == 'yes' for v in recent) / len(recent)) if recent else 0.47,
        'days_since_last': (day - ans[-1][0]) if ans else 30,
        'n_intros': stats['n_intros'].get(mid, 0),
        'recv_rate': (sum(rv) + 3 * policy.PRIOR_YES) / (len(rv) + 3), 'recv_n': len(rv)}
    for k, v in rows.items():
        out[prefix + k] = v
        groups[prefix + k] = group


def extended_features(actor, partner, state, stats, degree, day):
    out, groups = {}, {}

    def put(group, name, value):
        out[name] = float(value)
        groups[name] = group

    fa, fb = actor['fields'], partner['fields']
    # ---- demographics
    put('demo', 'a_age', actor['age'])
    put('demo', 'p_age', partner['age'])
    put('demo', 'age_gap', abs(actor['age'] - partner['age']))
    put('demo', 'age_signed', partner['age'] - actor['age'])
    put('demo', 'same_zone', actor['zone'] == partner['zone'])
    put('demo', 'a_zone_a', actor['zone'] == 'zone_a')
    put('demo', 'p_zone_a', partner['zone'] == 'zone_a')
    for g in GENDERS:
        put('demo', 'a_g_' + g, actor['gender'] == g)
        put('demo', 'p_g_' + g, partner['gender'] == g)
    put('demo', 'same_gender', actor['gender'] == partner['gender'])
    # ---- hard-field derived (all known for a feasible pair)
    put('hard', 'a_children', bool(fa['has_children']))
    put('hard', 'p_children', bool(fb['has_children']))
    put('hard', 'a_smoke', SMOKE[fa['smoking']])
    put('hard', 'p_smoke', SMOKE[fb['smoking']])
    for w in WANTS:
        put('hard', 'a_wants_' + w, fa['wants_children'] == w)
        put('hard', 'p_wants_' + w, fb['wants_children'] == w)
    put('hard', 'sched_overlap', len(set(fa['schedule']) & set(fb['schedule'])))
    put('hard', 'a_slots', len(fa['schedule']))
    put('hard', 'p_slots', len(fb['schedule']))
    put('hard', 'a_zones', len(fa['acceptable_zones']))
    put('hard', 'p_zones', len(fb['acceptable_zones']))
    wa = fa['age_max'] - fa['age_min']
    wb = fb['age_max'] - fb['age_min']
    put('hard', 'a_age_width', wa)
    put('hard', 'p_age_width', wb)
    put('hard', 'p_pos_in_a_range', (partner['age'] - fa['age_min']) / max(1, wa))
    put('hard', 'a_pos_in_p_range', (actor['age'] - fb['age_min']) / max(1, wb))
    put('hard', 'a_who_size', len(fa['who_to_meet']))
    put('hard', 'p_who_size', len(fb['who_to_meet']))
    put('hard', 'a_strict_smoke', fa['partner_smoking'] == 'no_smoking')
    put('hard', 'a_strict_kids', fa['partner_children'] == 'no_children')
    put('hard', 'p_strict_smoke', fb['partner_smoking'] == 'no_smoking')
    put('hard', 'p_strict_kids', fb['partner_children'] == 'no_children')
    # ---- raw soft answers (unknown is its own category)
    for k in kit.SOFT:
        for who, f in (('a', fa), ('p', fb)):
            for c in SOFT_CATS[k]:
                put('soft_raw', f'{who}_{k}={c}', f[k] == c)
            put('soft_raw', f'{who}_{k}=?', f[k] is None)
    # ---- engineered similarity
    n_eq = n_ne = n_unk = 0
    for k in kit.SOFT:
        va, vb = fa[k], fb[k]
        if va is None or vb is None:
            put('soft_eq', 'eq_' + k, 0)
            put('soft_eq', 'ne_' + k, 0)
            n_unk += 1
        else:
            put('soft_eq', 'eq_' + k, va == vb)
            put('soft_eq', 'ne_' + k, va != vb)
            n_eq += va == vb
            n_ne += va != vb
    put('soft_eq', 'n_eq_all', n_eq)
    put('soft_eq', 'n_ne_all', n_ne)
    put('soft_eq', 'n_unknown_all', n_unk)
    # ---- history of the actor, of the partner, and how others treat the actor
    _hist(stats, actor['member_id'], day, 'a_', out, groups, 'hist_actor')
    _hist(stats, partner['member_id'], day, 'p_', out, groups, 'hist_partner')
    # ---- context
    put('context', 'day', day)
    put('context', 'late', day >= 35)
    put('context', 'a_since_arrival', day - actor['arrived_day'])
    put('context', 'p_since_arrival', day - partner['arrived_day'])
    put('context', 'a_degree', degree.get(actor['member_id'], 0))
    put('context', 'p_degree', degree.get(partner['member_id'], 0))
    put('context', 'n_available', sum(m['available'] for m in state['members']))
    return out, groups


def feature_layout():
    """Names, in a fixed order, and their groups (built once from a dummy pair)."""
    dummy = {'member_id': 'x', 'age': 30, 'gender': 'woman', 'zone': 'zone_a', 'arrived_day': 0, 'available': True,
             'fields': {'age_min': 25, 'age_max': 35, 'who_to_meet': ['man'], 'relationship_structure': 'monogamous',
                        'smoking': 'no', 'partner_smoking': 'any', 'has_children': False, 'partner_children': 'any',
                        'wants_children': 'yes', 'acceptable_zones': ['zone_a'], 'schedule': ['weekend_day'],
                        **{k: None for k in kit.SOFT}}}
    other = dict(dummy, member_id='y')
    x, groups = extended_features(dummy, other, {'members': [dummy]},
                                  {'answers': {}, 'recv': {}, 'n_intros': {}}, {}, 0)
    return list(x), groups
