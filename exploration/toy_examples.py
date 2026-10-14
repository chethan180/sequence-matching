"""Runs the real kit/policy code on the 8-person toy world from PROBLEM_EXPLAINED.md.

Run from the repo root:  python exploration/toy_examples.py
"""
import copy
import itertools
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import kit
import policy

SOFT = kit.SOFT  # goal, pace, lifestyle, conversations, emotional, space, relocate


def person(name, gender, age, zone, arrive, pref, soft, unknown=()):
    f = dict(pref)
    f.update(dict(zip(SOFT, soft)))
    status = {k: 'observed' for k in f}
    for k in unknown:
        f[k] = None
        status[k] = 'not_asked'
    return {'member_id': name, 'pool_id': 'toy', 'synthetic': True, 'age': age, 'gender': gender,
            'zone': zone, 'arrived_day': arrive, 'available': True, 'fields': f, 'field_status': status,
            'field_observed_day': {k: (0 if status[k] == 'observed' else None) for k in f}, 'source': 'toy'}


def pref(lo, hi, who, struct, smoke, psmoke, kids, pkids, wants, zones, sched):
    return dict(age_min=lo, age_max=hi, who_to_meet=who, relationship_structure=struct, smoking=smoke,
                partner_smoking=psmoke, has_children=kids, partner_children=pkids, wants_children=wants,
                acceptable_zones=zones, schedule=sched)


M = {}
M['Priya'] = person('Priya', 'woman', 28, 'zone_a', 0, pref(28, 36, ['man'], 'monogamous', 'no', 'no_smoking', False, 'any', 'yes', ['zone_a', 'zone_b'], ['weekday_evening', 'weekend_evening']),
                    ['long_term', 'steady', 'quiet', 'ideas', 'ready', 'ample', 'yes'])
M['Arun'] = person('Arun', 'man', 30, 'zone_a', 0, pref(25, 35, ['woman'], 'monogamous', 'no', 'any', False, 'any', 'yes', ['zone_a', 'zone_b'], ['weekend_evening', 'weekend_day']),
                   ['long_term', 'steady', 'quiet', 'ideas', 'ready', 'moderate', 'unsure'])
M['Sam'] = person('Sam', 'man', 40, 'zone_b', 0, pref(30, 45, ['woman'], 'monogamous', 'occasionally', 'any', True, 'any', 'no', ['zone_a', 'zone_b', 'zone_c'], ['weekday_evening']),
                  ['long_term', 'slow', 'mixed', 'practical', 'taking_time', 'limited', 'no'])
M['Kiran'] = person('Kiran', 'woman', 33, 'zone_a', 0, pref(28, 40, ['man'], 'monogamous', 'no', 'any', False, 'any', 'yes', ['zone_a', 'zone_b'], ['weekday_evening', 'weekend_evening']),
                    ['long_term', 'steady', 'quiet', 'ideas', 'ready', 'moderate', 'unsure'], unknown=('smoking', 'wants_children'))
M['Meera'] = person('Meera', 'woman', 35, 'zone_c', 5, pref(32, 45, ['man'], 'monogamous', 'no', 'any', True, 'any', 'no', ['zone_b', 'zone_c'], ['weekday_evening']),
                    ['long_term', 'slow', 'mixed', 'practical', 'taking_time', 'limited', 'no'])
M['Rohan'] = person('Rohan', 'man', 38, 'zone_b', 0, pref(30, 40, ['woman'], 'monogamous', 'no', 'any', False, 'any', 'unsure', ['zone_a', 'zone_b', 'zone_c'], ['weekday_evening', 'weekend_day']),
                    ['exploring', 'quick', 'quiet', 'ideas', 'ready', 'limited', 'no'])
M['Zoya'] = person('Zoya', 'non_binary', 27, 'zone_a', 0, pref(25, 32, ['woman', 'man', 'non_binary'], 'non_monogamous', 'no', 'any', False, 'any', 'unsure', ['zone_a'], ['weekend_evening']),
                   ['exploring', 'steady', 'mixed', 'playful', 'ready', 'moderate', 'unsure'])
M['Dev'] = person('Dev', 'man', 29, 'zone_a', 12, pref(24, 33, ['woman', 'non_binary'], 'non_monogamous', 'no', 'any', False, 'any', 'unsure', ['zone_a'], ['weekend_evening']),
                  ['exploring', 'steady', 'mixed', 'playful', 'ready', 'moderate', 'unsure'])


def make_state(names, day, reveal_kiran):
    ms = copy.deepcopy([M[n] for n in names])
    for m in ms:
        if m['member_id'] == 'Kiran' and reveal_kiran:
            m['fields']['smoking'] = 'no'
            m['fields']['wants_children'] = 'yes'
            m['field_status']['smoking'] = m['field_status']['wants_children'] = 'observed'
    return {'day': day, 'ask_budget_remaining': 12, 'members': ms, 'introductions': [], 'feedback': [], 'ask_log': []}


def soft_score(a, b):
    return sum(a['fields'][k] == b['fields'][k] for k in SOFT)


def random_average(edges, score, runs=2000):
    pairs, totals = [], []
    for seed in range(runs):
        order = edges[:]
        random.Random(seed).shuffle(order)
        used, got = set(), []
        for p in order:
            if not used.intersection(p):
                got.append(p)
                used.update(p)
        pairs.append(len(got))
        totals.append(sum(score(p) for p in got))
    return statistics.mean(pairs), statistics.mean(totals)


def run(title, names, day):
    print('=' * 86)
    print(title)
    print('=' * 86)
    s0 = make_state(names, day, False)
    s1 = make_state(names, day, True)
    rev = {m['member_id']: m for m in s1['members']}
    print('pairs that are not plainly infeasible (Kiran shown after her ask):')
    for a, b in itertools.combinations(s1['members'], 2):
        e = kit.eligibility(a, b)
        if e['status'] != 'infeasible':
            print(f"   {a['member_id']:6}-{b['member_id']:6} {e['status']:20} soft matches = {soft_score(a, b)}/7")
    print('baseline asks today:', [(x['member_id'], x['field']) for x in kit.baseline_asks(s0)])
    out = {'greedy': kit.baseline_match(s1),
           'no_asks': kit.baseline_match(s0),
           'global': policy.smart_match(s1, policy.CONFIGS['value_global'])}
    edges = [tuple(sorted((a['member_id'], b['member_id']))) for a, b in itertools.combinations(s1['members'], 2)
             if kit.eligibility(a, b)['status'] == 'feasible']
    print()
    print(f"{'policy':8} {'pairs introduced':46} {'pairs':>5} {'people':>6} {'soft score':>10}")
    for name in ('greedy', 'no_asks', 'random', 'global'):
        if name == 'random':
            n, t = random_average(edges, lambda p: soft_score(rev[p[0]], rev[p[1]]))
            print(f"{name:8} {'(random order, average of 2000 shuffles)':46} {n:5.2f} {2 * n:6.2f} {t:10.2f}")
        else:
            ps = [tuple(sorted(p)) for p in out[name]]
            t = sum(soft_score(rev[a], rev[b]) for a, b in ps)
            print(f"{name:8} {str(ps):46} {len(ps):5} {2 * len(ps):6} {t:10}")
    print()


run('EXAMPLE 1 - day 0: Priya, Arun, Sam, Kiran (needs an ask), Rohan, Zoya',
    ['Priya', 'Arun', 'Sam', 'Kiran', 'Rohan', 'Zoya'], 0)
run('EXAMPLE 2 - day 5: Meera has arrived, the whole chain exists',
    ['Priya', 'Arun', 'Sam', 'Kiran', 'Rohan', 'Meera', 'Zoya'], 5)
run('EXAMPLE 3 - day 12: Dev has arrived too (Zoya-Dev becomes possible)', list(M), 12)

print('=' * 86)
print('EXAMPLE 4 - the four-person case from the problem statement (values given directly)')
print('=' * 86)
weights = [(0.90, 'A', 'B'), (0.05, 'C', 'D'), (0.65, 'A', 'D'), (0.65, 'B', 'C')]
val = {(a, b): v for v, a, b in weights}
total = lambda ps: sum(val.get(tuple(sorted(p)), 0) for p in ps)
g = policy.greedy_matching(weights)
gl = policy.global_matching(weights)
n, t = random_average([(a, b) for _, a, b in weights], lambda p: val[p])
print('greedy :', g, 'total', round(total(g), 2))
print('random : average total', round(t, 3), 'with', round(n, 2), 'pairs on average')
print('global :', gl, 'total', round(total(gl), 2))
