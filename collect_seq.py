"""Collect person-introduction rows with each person's ANSWER HISTORY as a sequence, for building a better yes-model.

python collect_seq.py --seeds 1001-1040 --behaviours greedy,random,global_learned --output results/seq_rows.npz

Per row (person X introduced to candidate C on day d), built only from the observation at the start of day d:
  x19        the 19 features policy.py uses today (baseline)
  now_eq     for each of the 7 soft fields: 0 unknown, 1 same, 2 different (X vs C)
  a_raw/c_raw each person's own soft answers as category indices (0 = unknown)
  hist_*     X's last 8 past introductions, oldest first: outcome (1 yes, 2 no, 3 silent, 0 padding), the same/different
             codes against that past partner, that partner's soft answers, the day it was assigned
  scalars    day, ages, zone match

DIAGNOSTIC COLUMNS (prefix hid_) come from the simulator's hidden state. They are stored ONLY to measure the ceiling a
perfect model could reach. They are never model inputs and never go near policy.py.
"""
import argparse
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from kit import OPTIONS, SOFT, Simulator, generate
import policy

VARIANTS = ('development', 'sparse', 'cold_start', 'delayed', 'shift', 'drift')
H = 8
CAT = {k: {v: i + 1 for i, v in enumerate(OPTIONS[k])} for k in SOFT}


def raw_codes(m):
    return [CAT[k].get(m['fields'][k], 0) if m['fields'][k] is not None else 0 for k in SOFT]


def eq_codes(a, b):
    out = []
    for k in SOFT:
        va, vb = a['fields'][k], b['fields'][k]
        out.append(0 if va is None or vb is None else (1 if va == vb else 2))
    return out


def episode(job):
    mode, variant, seed = job
    cfg = policy.CONFIGS.get(mode, mode)
    sim = Simulator(generate(seed, 200, 'evaluation', variant))
    logged = []
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
            intro = {i['introduction_id']: i for i in state['introductions']}
            past = {}
            for e in sorted((e for e in state['feedback'] if e['event'] == 'introduction_response' and e['member_id']),
                            key=lambda e: (e['observed_day'], e['introduction_id'])):
                i = intro.get(e['introduction_id'])
                if i is None:
                    continue
                partner = i['user_b'] if e['member_id'] == i['user_a'] else i['user_a']
                past.setdefault(e['member_id'], []).append((partner, e['value'], i['assigned_day']))
            for a, b in pairs:
                a, b = sorted((a, b))
                for actor, cand in ((a, b), (b, a)):
                    ma, mc = byid[actor], byid[cand]
                    h_out, h_eq, h_raw, h_day = [], [], [], []
                    for pid, val, aday in past.get(actor, [])[-H:]:
                        pm = byid.get(pid)
                        if pm is None:
                            continue
                        h_out.append(1 if val == 'yes' else (2 if val == 'no' else 3))
                        h_eq.append(eq_codes(ma, pm))
                        h_raw.append(raw_codes(pm))
                        h_day.append(aday)
                    pad = H - len(h_out)
                    hid_a, hid_c = sim.members[actor], sim.members[cand]
                    p_accept = sim._prob(hid_a, hid_c, 0.0)
                    logged.append({
                        'key': (a, b, day), 'actor': actor,
                        'x19': policy.directional_features(ma, mc, stats, day),
                        'now_eq': eq_codes(ma, mc), 'a_raw': raw_codes(ma), 'c_raw': raw_codes(mc),
                        'h_out': [0] * pad + h_out, 'h_eq': [[0] * 7] * pad + h_eq, 'h_raw': [[0] * 7] * pad + h_raw,
                        'h_day': [0] * pad + h_day,
                        'scal': [day, ma['age'], mc['age'], float(ma['zone'] == mc['zone'])],
                        'hid': [p_accept * hid_a['response_rate'], p_accept, hid_a['bias'], hid_a['response_rate'],
                                float(sum(hid_a['truth'][k] == hid_c['truth'][k] for k in SOFT[:4]))]})
        sim.advance(pairs)
    for _ in range(40):
        sim.advance([])
    events = sim.receive_feedback()
    intro = {(i['user_a'], i['user_b'], i['assigned_day']): i['introduction_id'] for i in sim.introductions}
    resp = {}
    for e in events:
        if e['event'] == 'introduction_response':
            resp[(e['introduction_id'], e['member_id'])] = e['value']
    for r in logged:
        v = resp.get((intro[r['key']], r['actor']))
        r['y'] = int(v == 'yes')
        r['ans'] = int(v is not None)
        r['seed'], r['variant'] = seed, VARIANTS.index(variant)
    return logged


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--seeds', default='1001-1040')
    p.add_argument('--behaviours', default='greedy,random,global_learned')
    p.add_argument('--output', default='results/seq_rows.npz')
    p.add_argument('--procs', type=int, default=11)
    a = p.parse_args()
    lo, hi = a.seeds.split('-')
    jobs = [(m, v, s) for m in a.behaviours.split(',') for v in VARIANTS for s in range(int(lo), int(hi) + 1)]
    with Pool(a.procs) as pool:
        chunks = pool.map(episode, jobs, chunksize=1)
    rows = [r for c in chunks for r in c]
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        a.output,
        x19=np.array([r['x19'] for r in rows], dtype=np.float32), now_eq=np.array([r['now_eq'] for r in rows], dtype=np.int8),
        a_raw=np.array([r['a_raw'] for r in rows], dtype=np.int8), c_raw=np.array([r['c_raw'] for r in rows], dtype=np.int8),
        h_out=np.array([r['h_out'] for r in rows], dtype=np.int8), h_eq=np.array([r['h_eq'] for r in rows], dtype=np.int8),
        h_raw=np.array([r['h_raw'] for r in rows], dtype=np.int8), h_day=np.array([r['h_day'] for r in rows], dtype=np.int16),
        scal=np.array([r['scal'] for r in rows], dtype=np.float32), hid=np.array([r['hid'] for r in rows], dtype=np.float32),
        y=np.array([r['y'] for r in rows], dtype=np.int8), ans=np.array([r['ans'] for r in rows], dtype=np.int8),
        seed=np.array([r['seed'] for r in rows], dtype=np.int32), variant=np.array([r['variant'] for r in rows], dtype=np.int8),
        features19=np.array(policy.FEATURES))
    print('rows', len(rows), 'yes rate', round(float(np.mean([r['y'] for r in rows])), 4))
