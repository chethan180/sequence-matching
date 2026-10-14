"""Fingerprint the actions a policy configuration takes, so a refactor can be proven behaviour-preserving.

python tools/snapshot.py --output results/snapshot_before.json
python tools/snapshot.py --compare results/snapshot_before.json

Each episode runs the 60 decision days in-process and hashes every ask list and pair list in order (the same digest
evaluate.episode records). Identical hashes mean identical behaviour on that world.
"""
import argparse
import hashlib
import json
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kit import Simulator, generate  # noqa: E402
import policy  # noqa: E402

VARIANTS = ('development', 'sparse', 'cold_start', 'delayed', 'shift', 'drift')
MODES = ('greedy', 'no_asks', 'random', 'final', 'final_noask', 'logit_greedy', 'global_only', 'final_voi', 'final_soft_res',
         'logit_value', 'smart', 'gbm_global', 'thompson_global', 'logit_staged', 'soft_greedy', 'value_ask_only')


def run(job):
    mode, variant, seed, raw = job
    decide = policy.decide_raw if raw else policy.decide
    sim = Simulator(generate(seed, 200, 'evaluation', variant))
    digest = hashlib.sha256()
    memory = None
    for _ in range(60):
        for phase in ('ask', 'match'):
            resp = decide({'schema_version': '1.0.0', 'phase': phase, 'state': sim.observe(), 'memory': memory}, mode)
            memory = json.loads(json.dumps(resp['memory']))
            action = resp['asks' if phase == 'ask' else 'pairs']
            digest.update(json.dumps([phase, sim.day, action], sort_keys=True).encode())
            if phase == 'ask':
                sim.resolve_asks(action)
            else:
                sim.advance(action)
    return f'{mode}|{variant}|{seed}', digest.hexdigest(), len(sim.introductions), sim.asks_total, len(policy.safety.EVENTS)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output')
    p.add_argument('--compare')
    p.add_argument('--modes', default=','.join(MODES))
    p.add_argument('--seeds', default='11,12')
    p.add_argument('--procs', type=int, default=10)
    p.add_argument('--raw', action='store_true', help='call decide_raw (no validation or fallback)')
    a = p.parse_args()
    jobs = [(m, v, int(s), a.raw) for m in a.modes.split(',') for v in VARIANTS for s in a.seeds.split(',')]
    with Pool(a.procs) as pool:
        rows = pool.map(run, jobs, chunksize=1)
    snap = {k: {'sha': h, 'introductions': n, 'ask_cost': c} for k, h, n, c, _ in rows}
    print('safety interventions (drops or fallbacks) across all episodes:', sum(r[4] for r in rows))
    if a.output:
        Path(a.output).parent.mkdir(parents=True, exist_ok=True)
        Path(a.output).write_text(json.dumps(snap, indent=1))
        print('saved', a.output, len(snap), 'episodes')
    if a.compare:
        ref = json.loads(Path(a.compare).read_text())
        same = [k for k in snap if k in ref and ref[k]['sha'] == snap[k]['sha']]
        diff = [k for k in snap if k in ref and ref[k]['sha'] != snap[k]['sha']]
        new = [k for k in snap if k not in ref]
        print(f'identical {len(same)} | different {len(diff)} | not in reference {len(new)}')
        by_mode = {}
        for k in diff:
            by_mode.setdefault(k.split('|')[0], []).append(k)
        for m, ks in by_mode.items():
            print(f'  DIFFERENT {m}: {len(ks)} episodes, e.g. {ks[0]} introductions {ref[ks[0]]["introductions"]} -> {snap[ks[0]]["introductions"]}')
        raise SystemExit(1 if diff else 0)
