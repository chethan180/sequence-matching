"""Collect one row per INTRODUCED PAIR with the raw attributes of both people (as observed on the decision day)
and what happened (each person's answer, mutual yes, date, MSMI).

python collect_pairs.py --seeds 1001-1040 --behaviours greedy,random,global_learned --output results/pair_rows.csv.gz

Side A is the member with the smaller id, side B the larger (arbitrary orientation); the analysis uses unordered
combinations such as zone_a|zone_b. Features come only from the observation at the start of the decision day.
"""
import argparse
from multiprocessing import Pool
from pathlib import Path
import pandas as pd
from kit import SOFT, Simulator, generate
import policy

VARIANTS = ('development', 'sparse', 'cold_start', 'delayed', 'shift', 'drift')


def side(m, stats, p):
    f = m['fields']
    st = stats.get(m['member_id'], {'n': 0, 'answered': 0, 'yes': 0})
    d = {p + 'age': m['age'], p + 'gender': m['gender'], p + 'zone': m['zone'], p + 'arrived': m['arrived_day'],
         p + 'children': 'has_kids' if f['has_children'] else 'no_kids', p + 'smoking': f['smoking'],
         p + 'wants': f['wants_children'], p + 'structure': f['relationship_structure'],
         p + 'n_slots': len(f['schedule']), p + 'n_zones': len(f['acceptable_zones']),
         p + 'age_width': f['age_max'] - f['age_min'], p + 'who_size': len(f['who_to_meet']),
         p + 'strict_smoke': f['partner_smoking'], p + 'strict_kids': f['partner_children'],
         p + 'hist_n': st['n'], p + 'hist_answered': st['answered'], p + 'hist_yes': st['yes']}
    for k in SOFT:
        d[p + k] = f[k] if f[k] is not None else 'unknown'
    return d


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
            stats = policy.member_stats(state)
            byid = {m['member_id']: m for m in state['members']}
            for a, b in pairs:
                a, b = sorted((a, b))
                row = {'seed': seed, 'variant': variant, 'behaviour': mode, 'day': state['day'], 'a_id': a, 'b_id': b}
                row.update(side(byid[a], stats, 'a_'))
                row.update(side(byid[b], stats, 'b_'))
                row['sched_overlap'] = len(set(byid[a]['fields']['schedule']) & set(byid[b]['fields']['schedule']))
                logged.append(row)
        sim.advance(pairs)
    for _ in range(40):
        sim.advance([])
    events = sim.receive_feedback()
    intro = {(i['user_a'], i['user_b'], i['assigned_day']): i for i in sim.introductions}
    by = {}
    for e in events:
        by.setdefault(e['introduction_id'], []).append(e)
    for row in logged:
        i = intro[(row['a_id'], row['b_id'], row['day'])]
        es = by.get(i['introduction_id'], [])
        resp = {e['member_id']: e['value'] for e in es if e['event'] == 'introduction_response'}
        row['yes_a'] = int(resp.get(row['a_id']) == 'yes')
        row['yes_b'] = int(resp.get(row['b_id']) == 'yes')
        row['answered_a'] = int(resp.get(row['a_id']) is not None)
        row['answered_b'] = int(resp.get(row['b_id']) is not None)
        row['mutual'] = row['yes_a'] * row['yes_b']
        dates = [e for e in es if e['event'] == 'date_happened' and e['value'] is True]
        seconds = {e['member_id']: e for e in es if e['event'] == 'second_meeting_intention'}
        row['date'] = int(bool(dates))
        row['msmi'] = int(bool(dates) and dates[0]['occurred_day'] - row['day'] <= 30 and len(seconds) == 2 and all(
            e['value'] == 'yes' and e['occurred_day'] - dates[0]['occurred_day'] <= 3 for e in seconds.values()))
    return logged


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--seeds', default='1001-1040')
    p.add_argument('--behaviours', default='greedy,random,global_learned')
    p.add_argument('--output', default='results/pair_rows.csv.gz')
    p.add_argument('--procs', type=int, default=11)
    a = p.parse_args()
    lo, hi = a.seeds.split('-')
    jobs = [(m, v, s) for m in a.behaviours.split(',') for v in VARIANTS for s in range(int(lo), int(hi) + 1)]
    with Pool(a.procs) as pool:
        chunks = pool.map(episode, jobs, chunksize=1)
    df = pd.DataFrame([r for c in chunks for r in c])
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.output, index=False)
    print('pairs', len(df), 'columns', len(df.columns), 'mutual yes rate', round(df['mutual'].mean(), 4))
