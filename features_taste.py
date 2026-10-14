"""Features for the team's candidate ideas that need information the earlier collectors did not store.

  taste     : learn each person's OWN direction from their past yes/no answers. For a candidate, how did this person
              answer past partners who shared the candidate's value on a soft field? (shrunk toward the person's overall rate)
  drift     : pool-wide yes rate over the last 10 days vs overall (detects response-climate change)
  pool_shape: share of members with complete hard answers, number of distinct zones, pool size

Everything is built from the observation at the start of the decision day (feedback with observed_day <= today).
"""
import policy
from kit import HARD

CORE = [('goal', 'relationship_goal'), ('pace', 'relationship_pace'), ('life', 'lifestyle'), ('conv', 'conversations')]
TASTE_GROUPS = {
    'taste': ['taste_goal', 'taste_pace', 'taste_life', 'taste_conv', 'taste_sum', 'taste_n', 'gap_vs_yes', 'mean_gap_yes'],
    'drift': ['pool_recent_yes', 'pool_all_yes', 'drift_gap', 'pool_rec_n'],
    'pool_shape': ['pool_complete', 'pool_zones', 'pool_size'],
}
TASTE_NAMES = [n for v in TASTE_GROUPS.values() for n in v]
TASTE_GROUP_OF = {n: g for g, v in TASTE_GROUPS.items() for n in v}


def taste_context(state):
    ends = {i['introduction_id']: (i['user_a'], i['user_b']) for i in state['introductions']}
    past = {}
    rec = [0, 0]
    allr = [0, 0]
    day = state['day']
    for e in state['feedback']:
        if e['event'] != 'introduction_response' or e['member_id'] is None or e['value'] is None:
            continue
        allr[0] += 1
        allr[1] += e['value'] == 'yes'
        if e['observed_day'] >= day - 10:
            rec[0] += 1
            rec[1] += e['value'] == 'yes'
        a, b = ends.get(e['introduction_id'], (None, None))
        if a is None:
            continue
        partner = b if e['member_id'] == a else a
        past.setdefault(e['member_id'], []).append((partner, e['value'] == 'yes'))
    members = state['members']
    complete = sum(all(m['fields'][k] is not None for k in HARD) for m in members) / max(1, len(members))
    return {'past': past, 'byid': {m['member_id']: m for m in members}, 'rec': rec, 'all': allr,
            'complete': complete, 'zones': len({m['zone'] for m in members}), 'size': len(members)}


def taste_features(actor, partner, t, stats):
    st = stats.get(actor['member_id'], {'n': 0, 'answered': 0, 'yes': 0})
    base = (st['yes'] + policy.PRIOR_STRENGTH * policy.PRIOR_YES) / (st['answered'] + policy.PRIOR_STRENGTH)
    pp = t['past'].get(actor['member_id'], [])
    x = {}
    total, total_n = 0.0, 0
    for tag, key in CORE:
        v = partner['fields'][key]
        n_same = yes_same = 0
        if v is not None:
            for pid, said_yes in pp:
                pm = t['byid'].get(pid)
                if pm is not None and pm['fields'][key] == v:
                    n_same += 1
                    yes_same += said_yes
        rate = (yes_same + 3 * base) / (n_same + 3)
        x['taste_' + tag] = rate - base
        total += rate - base
        total_n += n_same
    x['taste_sum'] = total
    x['taste_n'] = total_n
    gaps = [abs(actor['age'] - t['byid'][pid]['age']) for pid, y in pp if y and pid in t['byid']]
    mean_gap = (sum(gaps) + 2 * 3.5) / (len(gaps) + 2)
    x['mean_gap_yes'] = mean_gap
    x['gap_vs_yes'] = abs(actor['age'] - partner['age']) - mean_gap
    rn, ry = t['rec']
    an, ay = t['all']
    x['pool_recent_yes'] = (ry + 20 * policy.PRIOR_YES) / (rn + 20)
    x['pool_all_yes'] = (ay + 20 * policy.PRIOR_YES) / (an + 20)
    x['drift_gap'] = x['pool_recent_yes'] - x['pool_all_yes']
    x['pool_rec_n'] = rn
    x['pool_complete'] = t['complete']
    x['pool_zones'] = t['zones']
    x['pool_size'] = t['size']
    return [float(x[n]) for n in TASTE_NAMES]
