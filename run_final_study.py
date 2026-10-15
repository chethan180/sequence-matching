"""Final study through the OFFICIAL harness (evaluate.episode, real subprocess per call, JSON protocol).

python run_final_study.py --seeds 101-120 --output-dir results/official
Writes one evaluate.py-format JSON per method plus all_episodes.json. Episodes run in threads: each one spends nearly
all its time waiting on the policy subprocess, and threads avoid a Windows handle error seen with process pools.
"""
import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import evaluate
from kit import generate

METHODS = {
    'greedy':       [sys.executable, 'policy.py', '--baseline', 'greedy'],
    'no_asks':      [sys.executable, 'policy.py', '--baseline', 'no_asks'],
    'random':       [sys.executable, 'policy.py', '--baseline', 'random'],
    'submission':   [sys.executable, 'policy.py'],                       # exactly the container entrypoint
    'round1_final': [sys.executable, 'policy.py', '--mode', 'final'],
}


def run(job):
    method, variant, seed = job
    try:
        row = evaluate.episode(generate(seed, 200, 'evaluation', variant), METHODS[method])
    except (ValueError, TypeError, KeyError, OSError) as error:
        row = {'valid': False, 'error': str(error)}
    row.update(method=method, variant=variant, seed=seed)
    return row


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--seeds', default='101-120')
    p.add_argument('--output-dir', default='results/official')
    p.add_argument('--procs', type=int, default=10)
    a = p.parse_args()
    lo, hi = a.seeds.split('-')
    jobs = [(m, v, s) for m in METHODS for v in evaluate.VARIANTS for s in range(int(lo), int(hi) + 1)]
    with ThreadPoolExecutor(a.procs) as pool:
        rows = list(pool.map(run, jobs))
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for m in METHODS:
        mine = [r for r in rows if r['method'] == m]
        summary = evaluate.summarise(mine)
        (out / f'{m}.json').write_text(json.dumps({'release': evaluate.VERSION, 'mode': 'trusted_local', 'decision_days': 60,
                                                   'follow_up_days': 40, 'members_per_episode': 200, 'command': METHODS[m][1:],
                                                   'episodes': mine, 'summary': summary}, indent=2, allow_nan=False) + '\n')
        print(m, 'episodes', len(mine), 'valid', sum(r.get('valid', False) for r in mine),
              'primary score', None if summary.get('primary_score') is None else round(summary['primary_score'], 4))
    (out / 'all_episodes.json').write_text(json.dumps(rows, allow_nan=False))
