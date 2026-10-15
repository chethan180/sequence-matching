"""Experiment: can we tell, before asking, whose hard-constraint bundle will lead to a match?

For every member the policy could ask (available, some hard answer unknown, none declined), on the first day they are
askable, three scores are computed from the observation only:
  bayes_p      Bayesian P(at least one feasible partner)       (smp.clarify.match_chances)
  bayes_n      Bayesian expected number of feasible partners
  potential    rule: number of members not already ruled out by a known failure (experimental.value_asks counts this)
The label is diagnostic and reads the generator's true answers: does the member have at least one truly feasible
partner among members who can be introduced at all. Labels are never available to a policy.

python ask_model_study.py --seeds 3001-3020 --output results/ask_model_study.json
"""
import argparse
import json
from multiprocessing import Pool
from kit import Simulator, eligibility, generate
import policy
from smp import clarify
from smp.normalise import is_blocked, is_complete, past_pairs

VARIANTS = ('development', 'sparse', 'cold_start', 'delayed', 'shift', 'drift')


def run(job):
    variant, seed = job
    world = generate(seed + 100000 * VARIANTS.index(variant), 200, 'evaluation', variant)
    truth = {m['member_id']: m for m in world['members']}
    sim = Simulator(world)
    first = {}
    for _ in range(60):
        state = sim.observe()
        cand = [m for m in state['members'] if m['available'] and not is_complete(m) and not is_blocked(m)
                and m['member_id'] not in first]
        if cand:
            chances = clarify.match_chances(state, cand)
            past = past_pairs(state)
            for c in cand:
                pot = sum(1 for o in state['members'] if o is not c and not is_blocked(o)
                          and tuple(sorted((c['member_id'], o['member_id']))) not in past
                          and eligibility(c, o)['status'] != 'infeasible')
                p, n = chances.get(c['member_id'], (None, None))
                first[c['member_id']] = {'day': state['day'], 'bayes_p': p, 'bayes_n': n, 'potential': pot}
        sim.resolve_asks(policy.decide({'phase': 'ask', 'state': state, 'memory': None}, 'tiered')['asks'])
        sim.advance(policy.decide({'phase': 'match', 'state': sim.observe(), 'memory': None}, 'tiered')['pairs'])
    final = sim.observe()['members']
    reachable = [m for m in final if not is_blocked(m)]          # members who can ever be introduced
    rows = []
    for mid, r in first.items():
        a = truth[mid]
        has = any(clarify.compatible(a, a['truth'], truth[o['member_id']], truth[o['member_id']]['truth'])
                  for o in reachable if o['member_id'] != mid)
        rows.append(dict(r, variant=variant, seed=seed, label=int(has)))
    return rows


def auc(scores, labels):
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return None
    wins = 0.0
    for s in pos:
        for t in neg:
            wins += 1.0 if s > t else 0.5 if s == t else 0.0
    return wins / (len(pos) * len(neg))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--seeds', default='3001-3020')
    p.add_argument('--output', default='results/ask_model_study.json')
    p.add_argument('--procs', type=int, default=6)
    a = p.parse_args()
    lo, hi = map(int, a.seeds.split('-'))
    with Pool(a.procs) as pool:
        rows = [r for part in pool.map(run, [(v, s) for v in VARIANTS for s in range(lo, hi + 1)]) for r in part]
    rows = [r for r in rows if r['bayes_p'] is not None]
    y = [r['label'] for r in rows]
    out = {'rows': len(rows), 'share_with_partner': sum(y) / len(y), 'auc': {}, 'calibration': [], 'by_variant': {}}
    for key in ('bayes_p', 'bayes_n', 'potential'):
        out['auc'][key] = auc([r[key] for r in rows], y)
    for lo_, hi_ in ((0, .05), (.05, .2), (.2, .4), (.4, .6), (.6, .8), (.8, 1.01)):
        b = [r for r in rows if lo_ <= r['bayes_p'] < hi_]
        if b:
            out['calibration'].append({'bin': [lo_, hi_], 'n': len(b), 'predicted': sum(r['bayes_p'] for r in b) / len(b),
                                       'observed': sum(r['label'] for r in b) / len(b)})
    for v in VARIANTS:
        rv = [r for r in rows if r['variant'] == v]
        yv = [r['label'] for r in rv]
        out['by_variant'][v] = {'n': len(rv), 'with_partner': sum(yv) / max(1, len(yv)),
                                'auc_bayes_n': auc([r['bayes_n'] for r in rv], yv),
                                'auc_potential': auc([r['potential'] for r in rv], yv)}
    out['skip'] = []
    for t in (0.05, 0.1, 0.2, 0.3):
        skipped = [r for r in rows if r['bayes_p'] < t]
        out['skip'].append({'threshold': t, 'skipped': len(skipped),
                            'skipped_without_partner': sum(1 - r['label'] for r in skipped),
                            'skipped_with_partner': sum(r['label'] for r in skipped)})
    with open(a.output, 'w') as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))
