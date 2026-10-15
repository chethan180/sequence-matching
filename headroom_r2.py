"""Head-room diagnostic: how much coverage and MSMI could any policy achieve on a world, and how much do we achieve?

DIAGNOSTIC ONLY. This reads the generator's hidden values (true answers, response rates, person and pair effects).
It is an analysis tool for the report and is never imported by the policy.

Per world (200 members, one scenario family):

coverage ceiling
    Share of the N members arrived by day 59 who have at least one *usable* partner: both can ever be introduced (no
    declined hard field), their true answers are compatible in both directions, and both are present on a common
    decision day (arrived, not yet left). No policy can serve anyone else, so this bounds coverage from above.

expected MSMI of an introduction
    The exact probability that an introduction made on a given day qualifies, from the simulator's own formula:
        P(both reply) x P(both yes | shared noise) x P(date happens = 0.78) x P(date within 30 days)
        x P(both answer the second-meeting question, both yes, both within 3 days | shared noise)
    integrated over the pair's shared noise N(0, 0.45^2) by Gauss-Hermite quadrature. Summed over a policy's actual
    introductions it gives the policy's MSMI with the outcome luck removed.

MSMI ceilings (upper bounds on expected MSMI per episode)
    all_pairs     every usable pair introduced once (ignores that people are busy 8 days after each introduction)
    slot_capped   a linear programme: choose pairs to maximise expected MSMI when each member can take at most one
                  introduction per 8 days of presence before day 60. Still an upper bound: it ignores that both
                  people must be free on the same day and that a mutual yes keeps them busy longer.

python headroom_r2.py --seeds 4001-4020 --output results/headroom_r2.json
"""
import argparse
import itertools
import json
import math
from multiprocessing import Pool
import numpy as np
from scipy.optimize import linprog
from kit import HARD, Simulator, generate
import policy
from smp.clarify import compatible

VARIANTS = ('development', 'sparse', 'cold_start', 'delayed', 'shift', 'drift')
NODES, WEIGHTS = np.polynomial.hermite_e.hermegauss(40)       # for E[f(Z)], Z ~ N(0, 1)
WEIGHTS = WEIGHTS / WEIGHTS.sum()
SHARED_SD = 0.45


def window_probability(variant):
    """P(date within 30 days of assignment): later reply delay (1-7) + date gap (1-14) [+ 5-12 if delayed] <= 30."""
    extra = range(5, 13) if variant == 'delayed' else [0]
    hits = total = 0
    for d1, d2, gap, x in itertools.product(range(1, 8), range(1, 8), range(1, 15), extra):
        total += 1
        hits += max(d1, d2) + gap + x <= 30
    return hits / total


def p_msmi(sim, a, b, day):
    """Exact probability that introducing a and b on `day` qualifies as MSMI (see module docstring)."""
    sim.day = day
    goal = 0.4 * (a['truth']['relationship_goal'] == b['truth']['relationship_goal'])
    total = 0.0
    for z, w in zip(NODES, WEIGHTS):
        s = SHARED_SD * z
        first = a['response_rate'] * sim._prob(a, b, s) * b['response_rate'] * sim._prob(b, a, s)
        second = 1.0
        for x in (a, b):
            p2 = 1 / (1 + math.exp(-(0.15 + x['second_bias'] + s + goal)))
            second *= x['response_rate'] * p2 * 0.6                  # answers, says yes, within 3 days (3 of 5 delays)
        total += w * first * second
    return total * 0.78 * window_probability(sim.world['variant'])


def present(m):
    """Decision days on which the member is in the pool (arrived, not yet left)."""
    return range(m['arrived_day'], min(m['exit_day'], 60))


def run(job):
    variant, seed = job
    world = generate(seed + 100000 * VARIANTS.index(variant), 200, 'evaluation', variant)
    members = world['members']
    by_id = {m['member_id']: m for m in members}
    sim = Simulator(world)
    arrived = [m for m in members if m['arrived_day'] <= 59]
    n = len(arrived)
    reachable = [m for m in members if m['field_status'] and not any(m['field_status'][k] == 'declined' for k in HARD)]
    usable = []
    for a, b in itertools.combinations(reachable, 2):
        common = range(max(a['arrived_day'], b['arrived_day']), min(a['exit_day'], b['exit_day'], 60))
        if len(common) and compatible(a, a['truth'], b, b['truth']):
            usable.append((a, b, common.start))
    has_partner = {x['member_id'] for a, b, _ in usable for x in (a, b)}
    value = [p_msmi(sim, a, b, day) for a, b, day in usable]

    out = {'variant': variant, 'seed': seed, 'arrived': n, 'usable_pairs': len(usable),
           'coverage_ceiling': len(has_partner & {m['member_id'] for m in arrived}) / n,
           'msmi_all_pairs': float(sum(value))}
    # slot-capped linear programme
    if usable:
        ids = sorted(has_partner)
        row = {mid: i for i, mid in enumerate(ids)}
        a_ub = np.zeros((len(ids), len(usable)))
        for k, (a, b, _) in enumerate(usable):
            a_ub[row[a['member_id']], k] = a_ub[row[b['member_id']], k] = 1
        slots = [math.ceil(len(present(by_id[mid])) / 8) for mid in ids]
        lp = linprog(-np.array(value), A_ub=a_ub, b_ub=slots, bounds=(0, 1), method='highs')
        out['msmi_slot_capped'] = float(-lp.fun)
    else:
        out['msmi_slot_capped'] = 0.0

    # what policies actually achieve on the same world
    for mode in ('greedy', 'submission'):
        s = Simulator(world)
        for _ in range(60):
            s.resolve_asks(policy.decide({'phase': 'ask', 'state': s.observe(), 'memory': None}, mode)['asks'])
            s.advance(policy.decide({'phase': 'match', 'state': s.observe(), 'memory': None}, mode)['pairs'])
        for _ in range(40):
            s.advance([])
        served = {x for i in s.introductions for x in (i['user_a'], i['user_b'])}
        expected = sum(p_msmi(sim, by_id[i['user_a']], by_id[i['user_b']], i['assigned_day']) for i in s.introductions)
        out[mode] = {'coverage': len(served & {m['member_id'] for m in arrived}) / n,
                     'expected_msmi': float(expected),
                     'realised_msmi': s.metrics()['mutual_second_meeting_intention'],
                     'introductions': len(s.introductions),
                     'usable_pairs_used': sum(1 for a, b, _ in usable
                                              if any({a['member_id'], b['member_id']} == {i['user_a'], i['user_b']}
                                                     for i in s.introductions))}
    return out


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--seeds', default='4001-4020')
    p.add_argument('--output', default='results/headroom_r2.json')
    p.add_argument('--procs', type=int, default=4)
    a = p.parse_args()
    lo, hi = map(int, a.seeds.split('-'))
    with Pool(a.procs) as pool:
        rows = pool.map(run, [(v, s) for v in VARIANTS for s in range(lo, hi + 1)], chunksize=1)

    def mean(xs):
        return sum(xs) / len(xs)
    summary = {}
    for v in VARIANTS + ('all',):
        rs = [r for r in rows if v == 'all' or r['variant'] == v]
        summary[v] = {
            'coverage_ceiling': mean([r['coverage_ceiling'] for r in rs]),
            'usable_pairs': mean([r['usable_pairs'] for r in rs]),
            'msmi_all_pairs': mean([r['msmi_all_pairs'] for r in rs]),
            'msmi_slot_capped': mean([r['msmi_slot_capped'] for r in rs]),
            **{f'{m}_{k}': mean([r[m][k] for r in rs]) for m in ('greedy', 'submission')
               for k in ('coverage', 'expected_msmi', 'realised_msmi', 'introductions', 'usable_pairs_used')}}
    with open(a.output, 'w') as f:
        json.dump({'rows': rows, 'summary': summary}, f, indent=1)
    cols = ('coverage_ceiling', 'greedy_coverage', 'submission_coverage', 'usable_pairs', 'submission_usable_pairs_used',
            'msmi_slot_capped', 'greedy_expected_msmi', 'submission_expected_msmi', 'submission_realised_msmi')
    print('family        ' + ' '.join(f'{c[:14]:>14}' for c in cols))
    for v, s in summary.items():
        print(f'{v:13} ' + ' '.join(f'{s[c]:14.3f}' for c in cols))
