"""Collect labelled introductions with the extended feature set (research data for the feature study).

python collect_ext.py --seeds 1001-1040 --behaviours greedy,random,global_learned --output results/ext3_rows.npz

For every introduction the policy made, we store each person's features as they were BEFORE the decision
(built only from the observation) and what that person (and the pair) did afterwards.
An assertion checks that no feedback event newer than today ever appears in the observation.
"""
import argparse
import json
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from kit import Simulator, generate
import features_ext
import features_domain
import features_taste
import policy

VARIANTS = ('development', 'sparse', 'cold_start', 'delayed', 'shift', 'drift')
NAMES0, GROUPS0 = features_ext.feature_layout()
NAMES = NAMES0 + features_domain.DOMAIN_NAMES + features_taste.TASTE_NAMES
GROUPS = dict(GROUPS0, **features_domain.DOMAIN_GROUP_OF, **features_taste.TASTE_GROUP_OF)


def episode(job):
    mode, variant, seed = job
    sim = Simulator(generate(seed, 200, 'evaluation', variant))
    logged = []
    for _ in range(60):
        state = sim.observe()
        sim.resolve_asks(policy.decide({'phase': 'ask', 'state': state, 'memory': None}, policy.CONFIGS.get(mode, mode))['asks'])
        state = sim.observe()
        assert all(e['observed_day'] <= state['day'] for e in state['feedback']), 'future feedback leaked'
        pairs = policy.decide({'phase': 'match', 'state': state, 'memory': None}, policy.CONFIGS.get(mode, mode))['pairs']
        if pairs:
            stats = features_ext.ext_stats(state)
            ctx = features_domain.domain_context(state)
            pstats = policy.member_stats(state)
            tctx = features_taste.taste_context(state)
            degree = {}
            for a, b in policy.feasible_edges(state):
                degree[a['member_id']] = degree.get(a['member_id'], 0) + 1
                degree[b['member_id']] = degree.get(b['member_id'], 0) + 1
            byid = {m['member_id']: m for m in state['members']}
            for a, b in pairs:
                a, b = sorted((a, b))
                xa, _ = features_ext.extended_features(byid[a], byid[b], state, stats, degree, state['day'])
                xb, _ = features_ext.extended_features(byid[b], byid[a], state, stats, degree, state['day'])
                xa = dict(zip(NAMES, [xa[n] for n in NAMES0] + features_domain.domain_features(byid[a], byid[b], ctx) + features_taste.taste_features(byid[a], byid[b], tctx, pstats)))
                xb = dict(zip(NAMES, [xb[n] for n in NAMES0] + features_domain.domain_features(byid[b], byid[a], ctx) + features_taste.taste_features(byid[b], byid[a], tctx, pstats)))
                logged.append((a, b, state['day'], xa, xb))
        sim.advance(pairs)
    for _ in range(40):
        sim.advance([])
    events = sim.receive_feedback()
    intro = {(i['user_a'], i['user_b'], i['assigned_day']): i for i in sim.introductions}
    by = {}
    for e in events:
        by.setdefault(e['introduction_id'], []).append(e)
    rows = []
    for a, b, day, xa, xb in logged:
        i = intro[(a, b, day)]
        es = by.get(i['introduction_id'], [])
        resp = {e['member_id']: e['value'] for e in es if e['event'] == 'introduction_response'}
        mutual = resp.get(a) == 'yes' and resp.get(b) == 'yes'
        dates = [e for e in es if e['event'] == 'date_happened' and e['value'] is True]
        seconds = {e['member_id']: e for e in es if e['event'] == 'second_meeting_intention'}
        msmi = bool(dates) and dates[0]['occurred_day'] - day <= 30 and len(seconds) == 2 and all(
            e['value'] == 'yes' and e['occurred_day'] - dates[0]['occurred_day'] <= 3 for e in seconds.values())
        for actor, x in ((a, xa), (b, xb)):
            sec = seconds.get(actor)
            rows.append(([x[n] for n in NAMES], int(resp.get(actor) == 'yes'), int(resp.get(actor) is not None),
                         -1 if not dates else int(bool(sec) and sec['value'] == 'yes'),
                         int(mutual), int(bool(dates)), int(msmi), seed, VARIANTS.index(variant), day))
    return rows


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--seeds', default='1001-1040')
    p.add_argument('--behaviours', default='greedy,random,global_learned')
    p.add_argument('--output', default='results/ext3_rows.npz')
    p.add_argument('--procs', type=int, default=11)
    a = p.parse_args()
    lo, hi = a.seeds.split('-')
    jobs = [(m, v, s) for m in a.behaviours.split(',') for v in VARIANTS for s in range(int(lo), int(hi) + 1)]
    with Pool(a.procs) as pool:
        chunks = pool.map(episode, jobs, chunksize=1)
    flat = [r for c in chunks for r in c]
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    cols = list(zip(*flat))
    np.savez_compressed(
        a.output, X=np.array(cols[0], dtype=np.float32), y_yes=np.array(cols[1], dtype=np.int8),
        y_ans=np.array(cols[2], dtype=np.int8), y_sec=np.array(cols[3], dtype=np.int8),
        y_mutual=np.array(cols[4], dtype=np.int8), y_date=np.array(cols[5], dtype=np.int8),
        y_msmi=np.array(cols[6], dtype=np.int8), seed=np.array(cols[7], dtype=np.int32),
        variant=np.array(cols[8], dtype=np.int8), day=np.array(cols[9], dtype=np.int16),
        names=np.array(NAMES), groups=np.array([GROUPS[n] for n in NAMES]))
    print('rows', len(flat), 'features', len(NAMES), 'pairs', len(flat) // 2)
    print('groups', {g: sum(1 for n in NAMES if GROUPS[n] == g) for g in dict.fromkeys(GROUPS.values())})
