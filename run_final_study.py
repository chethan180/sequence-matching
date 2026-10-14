"""Final study through the OFFICIAL harness (evaluate.episode, real subprocess per call, JSON protocol).

python run_final_study.py --seeds 801-830 --output-dir results/final
Writes one evaluate.py-format JSON per method plus results/final/all_episodes.json.
"""
import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path
import evaluate
from kit import generate

METHODS = {
    'greedy':       [sys.executable, 'policy.py', '--baseline', 'greedy'],
    'no_asks':      [sys.executable, 'policy.py', '--baseline', 'no_asks'],
    'random':       [sys.executable, 'policy.py', '--baseline', 'random'],
    'final':        [sys.executable, 'policy.py'],                       # exactly the container entrypoint
    'abl_greedy_matching': [sys.executable, 'policy.py', '--mode', 'logit_greedy'],
    'abl_soft_score':      [sys.executable, 'policy.py', '--mode', 'global_only'],
    'abl_no_asks':         [sys.executable, 'policy.py', '--mode', 'final_noask'],
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
    p.add_argument('--seeds', default='801-830')
    p.add_argument('--output-dir', default='results/final')
    p.add_argument('--procs', type=int, default=11)
    a = p.parse_args()
    lo, hi = a.seeds.split('-')
    jobs = [(m, v, s) for m in METHODS for v in evaluate.VARIANTS for s in range(int(lo), int(hi) + 1)]
    with Pool(a.procs) as pool:
        rows = pool.map(run, jobs, chunksize=1)
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
