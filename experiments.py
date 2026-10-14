"""In-process experiment runner: same 60+40 day protocol as evaluate.episode, many episodes in parallel.

Usage
  python experiments.py --configs greedy,logit_global --seeds 401-460 --variants all --output results/exp.json
  python experiments.py --collect --configs greedy,random --seeds 1001-1030 --output results/train_rows.npz

Policies are called through policy.decide exactly like the protocol (memory is JSON round-tripped).
Modes starting with `oracle_` peek at the simulator's hidden outcome probabilities. They are research
diagnostics that show the head-room of better scoring. They are NOT valid policies and never go in policy.py.
"""
import argparse
import json
import statistics
import time
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from kit import Simulator, generate
import policy

VARIANTS = ('development', 'sparse', 'cold_start', 'delayed', 'shift', 'drift')
INDEPENDENT = '--independent' in __import__('sys').argv
KEYS = ('msmi100', 'mutual100', 'coverage', 'ask_cost', 'assignments', 'dates', 'yes_rate', 'mutual_rate', 'known_core')


def oracle_pairs(sim, state, mode):
    edges = policy.feasible_edges(state)
    weighted = []
    for a, b in edges:
        ma, mb = sim.members[a['member_id']], sim.members[b['member_id']]
        p = (sim._prob(ma, mb, 0.0) * ma['response_rate']) * (sim._prob(mb, ma, 0.0) * mb['response_rate'])
        weighted.append((0.02 + p, a['member_id'], b['member_id']))
    pairs = policy.global_matching(weighted) if mode == 'oracle_global' else policy.greedy_matching(weighted)
    return [sorted(p) for p in pairs]


def run(job):
    mode, variant, seed, collect = job
    world_seed = seed + 100000 * VARIANTS.index(variant) if INDEPENDENT else seed
    sim = Simulator(generate(world_seed, 200, 'evaluation', variant))
    memory = None
    t0 = time.perf_counter()
    slowest = 0.0
    logged = []
    known_sum, known_n = 0, 0
    for _ in range(60):
        for phase in ('ask', 'match'):
            state = sim.observe()
            req = {'schema_version': '1.0.0', 'phase': phase, 'state': state, 'memory': memory}
            t = time.perf_counter()
            if mode.startswith('oracle_'):
                resp = policy.decide(req, 'greedy')
                if phase == 'match':
                    resp = {'pairs': oracle_pairs(sim, state, mode), 'memory': memory}
            elif mode in policy.CONFIGS:
                resp = policy.decide(req, policy.CONFIGS[mode])
            else:
                resp = policy.decide(req, mode)
            slowest = max(slowest, time.perf_counter() - t)
            memory = json.loads(json.dumps(resp['memory']))
            if phase == 'ask':
                sim.resolve_asks(resp['asks'])
            else:
                if resp['pairs']:
                    by = {m['member_id']: m for m in state['members']}
                    for a, b in resp['pairs']:
                        known_sum += sum(by[a]['fields'][k] is not None and by[b]['fields'][k] is not None for k in policy.SOFT_SIM)
                        known_n += 4
                if collect and resp['pairs']:
                    stats = policy.member_stats(state)
                    byid = {m['member_id']: m for m in state['members']}
                    for a, b in resp['pairs']:
                        a, b = sorted((a, b))
                        logged.append((a, b, state['day'],
                                       policy.directional_features(byid[a], byid[b], stats, state['day']),
                                       policy.directional_features(byid[b], byid[a], stats, state['day'])))
                sim.advance(resp['pairs'])
    arrived = {m['member_id'] for m in sim.observe()['members'] if m['arrived_day'] <= 59}
    for _ in range(40):
        sim.advance([])
    r = sim.metrics()
    events = sim.receive_feedback()
    responses = [e for e in events if e['event'] == 'introduction_response']
    yes_n = sum(e['value'] == 'yes' for e in responses)
    served = set()
    for i in sim.introductions:
        served.update((i['user_a'], i['user_b']))
    n = max(1, len(arrived))
    out = dict(mode=mode, variant=variant, seed=seed,
               assignments=r['assignments'], mutual=r['mutual_acceptances'], dates=r['dates'],
               msmi=r['mutual_second_meeting_intention'], ask_cost=r['ask_cost'],
               arrived=n, coverage=len(served & arrived) / n,
               msmi100=100 * r['mutual_second_meeting_intention'] / n,
               mutual100=100 * r['mutual_acceptances'] / n,
               yes_rate=yes_n / max(1, len(responses)),
               mutual_rate=r['mutual_acceptances'] / max(1, r['assignments']),
               known_core=known_sum / max(1, known_n),
               seconds=time.perf_counter() - t0, slowest_call=slowest)
    if collect:
        intro_id = {(i['user_a'], i['user_b'], i['assigned_day']): i['introduction_id'] for i in sim.introductions}
        by_intro = {}
        for e in events:
            by_intro.setdefault(e['introduction_id'], []).append(e)
        rows = []
        for a, b, day, xa, xb in logged:
            es = by_intro.get(intro_id[(a, b, day)], [])
            date = any(e['event'] == 'date_happened' and e['value'] is True for e in es)
            for actor, x in ((a, xa), (b, xb)):
                resp_e = [e for e in es if e['event'] == 'introduction_response' and e['member_id'] == actor][0]
                sec = [e for e in es if e['event'] == 'second_meeting_intention' and e['member_id'] == actor]
                sec_label = -1 if not date else int(bool(sec) and sec[0]['value'] == 'yes')
                rows.append((x, int(resp_e['value'] == 'yes'), int(resp_e['value'] is not None), sec_label,
                             seed, VARIANTS.index(variant), day))
        out['rows'] = rows
    return out


def parse_seeds(text):
    out = []
    for part in text.split(','):
        if '-' in part:
            a, b = part.split('-')
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def summarise(rows):
    out = {}
    for mode in sorted({r['mode'] for r in rows}):
        fam = {}
        for v in VARIANTS:
            rs = [r for r in rows if r['mode'] == mode and r['variant'] == v]
            if rs:
                fam[v] = {k: statistics.mean(r[k] for r in rs) for k in KEYS}
        out[mode] = {'by_variant': fam,
                     'overall': {k: statistics.mean(f[k] for f in fam.values()) for k in KEYS}}
    return out


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--configs', required=True)
    p.add_argument('--seeds', default='201-206')
    p.add_argument('--variants', default='all')
    p.add_argument('--output', default='results/exp.json')
    p.add_argument('--procs', type=int, default=11)
    p.add_argument('--independent', action='store_true', help='a different world seed for every scenario, so episodes are independent')
    p.add_argument('--collect', action='store_true', help='log pre-decision features and outcomes of every introduction')
    a = p.parse_args()
    modes = a.configs.split(',')
    variants = list(VARIANTS) if a.variants == 'all' else a.variants.split(',')
    jobs = [(m, v, s, a.collect) for m in modes for v in variants for s in parse_seeds(a.seeds)]
    with Pool(a.procs) as pool:
        rows = pool.map(run, jobs, chunksize=1)
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    if a.collect:
        flat = [r for row in rows for r in row.pop('rows')]
        np.savez_compressed(a.output,
                            X=np.array([r[0] for r in flat], dtype=np.float32),
                            y_yes=np.array([r[1] for r in flat], dtype=np.int8),
                            y_ans=np.array([r[2] for r in flat], dtype=np.int8),
                            y_sec=np.array([r[3] for r in flat], dtype=np.int8),
                            seed=np.array([r[4] for r in flat], dtype=np.int32),
                            variant=np.array([r[5] for r in flat], dtype=np.int8),
                            day=np.array([r[6] for r in flat], dtype=np.int16),
                            features=np.array(policy.FEATURES))
        print('rows saved:', len(flat))
    summ = summarise(rows)
    if not a.collect:
        Path(a.output).write_text(json.dumps({'rows': rows, 'summary': summ}, indent=1))
    print(f"{'config':16} {'MSMI/100':>9} {'mutual/100':>11} {'yes rate':>9} {'mut/intro':>10} {'coverage':>9} {'ask_cost':>9} {'assign':>7} {'dates':>6} {'known core':>11}")
    for m, s in summ.items():
        o = s['overall']
        print(f"{m:16} {o['msmi100']:9.3f} {o['mutual100']:11.2f} {o['yes_rate']:9.3f} {o['mutual_rate']:10.3f} "
              f"{o['coverage']:9.3f} {o['ask_cost']:9.1f} {o['assignments']:7.1f} {o['dates']:6.1f} {o['known_core']:11.3f}")
