"""How much is known about the four core soft fields when pairs are introduced, and what values do pairs have?

For every introduction made by the shipped policy, each core soft field of each person is classed as observed,
not_asked (could still be bought for 1 unit) or declined (can never be known); the pair is classed by how many core
fields are known for both people. Also records the distribution of the model's pair value over all feasible pairs
offered to the matcher, to set thresholds for the wait-for-a-better-match experiment.

python soft_status_study.py --seeds 7001-7005
"""
import argparse
import collections
import json
from kit import Simulator, generate
import policy
from smp.feasibility import feasible_edges
from smp.model import pair_values
from smp.normalise import SOFT_CORE

VARIANTS = ('development', 'sparse', 'cold_start', 'delayed', 'shift', 'drift')

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--seeds', default='7001-7005')
    a = p.parse_args()
    lo, hi = map(int, a.seeds.split('-'))
    status = collections.Counter()
    known_both = collections.Counter()
    soft_count = collections.Counter()
    values, chosen = [], []
    intros = 0
    for vi, v in enumerate(VARIANTS):
        for seed in range(lo, hi + 1):
            sim = Simulator(generate(seed + 100000 * vi, 200, 'evaluation', v))
            for _ in range(60):
                sim.resolve_asks(policy.decide({'phase': 'ask', 'state': sim.observe(), 'memory': None})['asks'])
                state = sim.observe()
                edges = feasible_edges(state)
                vals = dict(zip([tuple(sorted((x['member_id'], y['member_id']))) for x, y in edges],
                                pair_values(state, edges)))
                values.extend(vals.values())
                pairs = policy.decide({'phase': 'match', 'state': state, 'memory': None})['pairs']
                by = {m['member_id']: m for m in state['members']}
                for x, y in pairs:
                    intros += 1
                    chosen.append(vals[tuple(sorted((x, y)))])
                    k = 0
                    agree = 0
                    for f in SOFT_CORE:
                        for m in (by[x], by[y]):
                            status[(f, m['field_status'][f])] += 1
                        if by[x]['fields'][f] is not None and by[y]['fields'][f] is not None:
                            k += 1
                            agree += by[x]['fields'][f] == by[y]['fields'][f]
                    known_both[k] += 1
                    soft_count[agree] += 1
                sim.advance(pairs)
    print('introductions', intros)
    print('\nstatus of each core soft field, per person, at introduction:')
    for f in SOFT_CORE:
        n = sum(status[(f, s)] for s in ('observed', 'not_asked', 'declined'))
        print(f'  {f:20} observed {status[(f, "observed")] / n:.1%}  not_asked {status[(f, "not_asked")] / n:.1%}  declined {status[(f, "declined")] / n:.1%}')
    print('\ncore fields known for BOTH people:', {k: f'{known_both[k] / intros:.1%}' for k in range(5)})
    print('soft-match count (known AND equal):', {k: f'{soft_count[k] / intros:.1%}' for k in range(5)})
    qs = sorted(values)
    cq = sorted(chosen)
    def q(xs, f): return xs[int(f * (len(xs) - 1))]
    print('\npair value over feasible pairs offered: ' + '  '.join(f'p{int(f*100)} {q(qs, f):.3f}' for f in (0.1, 0.25, 0.5, 0.75, 0.9)))
    print('pair value of introduced pairs:        ' + '  '.join(f'p{int(f*100)} {q(cq, f):.3f}' for f in (0.1, 0.25, 0.5, 0.75, 0.9)))
    json.dump({'introductions': intros, 'status': {f'{k[0]}|{k[1]}': c for k, c in status.items()},
               'known_both': known_both, 'soft_match': soft_count,
               'value_quantiles_offered': {f: q(qs, f) for f in (0.1, 0.25, 0.5, 0.75, 0.9)},
               'value_quantiles_introduced': {f: q(cq, f) for f in (0.1, 0.25, 0.5, 0.75, 0.9)}},
              open('results/soft_status_study.json', 'w'), indent=1)
