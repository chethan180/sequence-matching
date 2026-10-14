"""Collect whole simulated worlds (members, every introduction, every answer with its observation day).

python collect_world.py --seeds 1001-1040 --behaviours greedy,random,global_learned --output results/worlds.pkl

The probabilistic model needs the interaction graph of a pool, not isolated rows: evidence about one person's hidden
traits comes from how they answered others and how others answered them. One record per episode:

  members   soft answers as category codes (0 = unknown), the day each soft answer became observable, arrival day
  intros    for each introduction: the two member indices, the day, each person's answer (1 yes, 2 no, 3 silent) and the
            day that answer became observable, plus the 19 baseline features of each direction at decision time

DIAGNOSTIC fields (prefix hid_) come from the simulator's hidden state and are stored only to measure ceilings.
They are never inputs to a model and never go near policy.py.
"""
import argparse
import pickle
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from kit import OPTIONS, SOFT, Simulator, generate
import policy

VARIANTS = ('development', 'sparse', 'cold_start', 'delayed', 'shift', 'drift')
CAT = {k: {v: i + 1 for i, v in enumerate(OPTIONS[k])} for k in SOFT}


def episode(job):
    mode, variant, seed = job
    cfg = policy.CONFIGS.get(mode, mode)
    sim = Simulator(generate(seed, 200, 'evaluation', variant))
    ids = list(sim.members)
    index = {m: i for i, m in enumerate(ids)}
    intros = []
    for _ in range(60):
        state = sim.observe()
        sim.resolve_asks(policy.decide({'phase': 'ask', 'state': state, 'memory': None}, cfg)['asks'])
        state = sim.observe()
        assert all(e['observed_day'] <= state['day'] for e in state['feedback']), 'future feedback leaked'
        pairs = policy.decide({'phase': 'match', 'state': state, 'memory': None}, cfg)['pairs']
        if pairs:
            day = state['day']
            byid = {m['member_id']: m for m in state['members']}
            stats = policy.member_stats(state)
            for a, b in pairs:
                a, b = sorted((a, b))
                ha, hb = sim.members[a], sim.members[b]
                intros.append({
                    'a': index[a], 'b': index[b], 'day': day,
                    'x19_a': policy.directional_features(byid[a], byid[b], stats, day),
                    'x19_b': policy.directional_features(byid[b], byid[a], stats, day),
                    'hid_p_a': sim._prob(ha, hb, 0.0) * ha['response_rate'],
                    'hid_p_b': sim._prob(hb, ha, 0.0) * hb['response_rate'],
                    'key': (a, b, day)})
        sim.advance(pairs)
    for _ in range(40):
        sim.advance([])
    events = sim.receive_feedback()
    iid = {(i['user_a'], i['user_b'], i['assigned_day']): i['introduction_id'] for i in sim.introductions}
    resp = {(e['introduction_id'], e['member_id']): e for e in events if e['event'] == 'introduction_response'}
    for r in intros:
        a, b, _ = r.pop('key')
        for side, mid in (('a', a), ('b', b)):
            e = resp[(iid[(a, b, r['day'])], mid)]
            r['val_' + side] = 1 if e['value'] == 'yes' else (2 if e['value'] == 'no' else 3)
            r['obs_' + side] = e['observed_day']
    final = {m['member_id']: m for m in sim.observe()['members']}
    soft = np.zeros((len(ids), len(SOFT)), dtype=np.int8)
    soft_day = np.full((len(ids), len(SOFT)), 999, dtype=np.int16)
    truth = np.zeros((len(ids), len(SOFT)), dtype=np.int8)
    arrived = np.zeros(len(ids), dtype=np.int16)
    hid = np.zeros((len(ids), 2), dtype=np.float32)
    for mid, i in index.items():
        hm = sim.members[mid]
        arrived[i] = hm['arrived_day']
        hid[i] = (hm['bias'], hm['response_rate'])
        fm = final.get(mid)
        for k, name in enumerate(SOFT):
            truth[i, k] = CAT[name][hm['truth'][name]]
            if fm is not None and fm['fields'][name] is not None:
                soft[i, k] = CAT[name][fm['fields'][name]]
                soft_day[i, k] = fm['field_observed_day'][name]
    return {'seed': seed, 'variant': VARIANTS.index(variant), 'behaviour': mode, 'soft': soft, 'soft_day': soft_day,
            'arrived': arrived, 'hid_truth': truth, 'hid_person': hid, 'intros': intros}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--seeds', default='1001-1040')
    p.add_argument('--behaviours', default='greedy,random,global_learned')
    p.add_argument('--output', default='results/worlds.pkl')
    p.add_argument('--procs', type=int, default=11)
    a = p.parse_args()
    lo, hi = a.seeds.split('-')
    jobs = [(m, v, s) for m in a.behaviours.split(',') for v in VARIANTS for s in range(int(lo), int(hi) + 1)]
    with Pool(a.procs) as pool:
        worlds = pool.map(episode, jobs, chunksize=1)
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    with open(a.output, 'wb') as f:
        pickle.dump(worlds, f)
    n = sum(len(w['intros']) for w in worlds)
    print('worlds', len(worlds), 'introductions', n, 'rows', 2 * n)
