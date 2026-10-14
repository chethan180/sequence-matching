"""Does buying soft answers raise the yes-model's accuracy? Same worlds, with and without the ask algorithm.

Input: results/rows_regime.npz (experiments.py --collect --configs final,final_voi): rows of the policy WITHOUT soft asks
followed by rows of the policy WITH value-of-information soft asks, on the same 40 seeds x 6 scenarios.
Accuracy is out-of-fold by seed; intervals bootstrap the 40 seeds (paired between the two regimes).
"""
import json
import warnings
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import policy

warnings.filterwarnings('ignore')
rng = np.random.default_rng(0)
d = np.load('results/rows_regime.npz')
X, y, seed, variant = d['X'].astype(float), d['y_yes'].astype(int), d['seed'], d['variant']
cut = int(np.where(np.diff(variant.astype(int)) < 0)[0][0]) + 1          # rows are stored policy by policy
regime = np.zeros(len(y), dtype=int)
regime[cut:] = 1
names = list(d['features'])
known = 1.0 - X[:, names.index('n_unknown_core')]
seeds = np.unique(seed)
picks = [rng.choice(seeds, len(seeds)) for _ in range(300)]
out = {}
p = np.zeros(len(y))
for r in (0, 1):
    m = regime == r
    Xr, yr, sr = X[m], y[m], seed[m]
    pr = np.zeros(m.sum())
    for tr, te in GroupKFold(5).split(Xr, yr, sr):
        pr[te] = make_pipeline(StandardScaler(), LogisticRegression(C=0.03, max_iter=3000)).fit(Xr[tr], yr[tr]).predict_proba(Xr[te])[:, 1]
    p[m] = pr
rows_by = {(r, s): np.where((regime == r) & (seed == s))[0] for r in (0, 1) for s in seeds}


def auc_of(r, pk):
    idx = np.concatenate([rows_by[(r, s)] for s in pk])
    return roc_auc_score(y[idx], p[idx])


for r, lab in ((0, 'without soft asks'), (1, 'with the ask algorithm')):
    m = regime == r
    a = roc_auc_score(y[m], p[m])
    v = [auc_of(r, pk) for pk in picks]
    out[lab] = {'rows': int(m.sum()), 'known_core': float(known[m].mean()), 'auc': float(a), 'ci': [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))],
                'yes_rate': float(y[m].mean())}
    print(f'{lab:24} rows {int(m.sum()):6d}  core answers known for both {known[m].mean()*100:5.1f}%  AUC {a:.4f} [{np.percentile(v,2.5):.4f}, {np.percentile(v,97.5):.4f}]  yes rate {y[m].mean()*100:.1f}%')
dv = [auc_of(1, pk) - auc_of(0, pk) for pk in picks]
dpoint = out['with the ask algorithm']['auc'] - out['without soft asks']['auc']
print(f'change in AUC from buying soft answers: {dpoint:+.4f}  95% CI over seeds [{np.percentile(dv,2.5):+.4f}, {np.percentile(dv,97.5):+.4f}]')
out['delta'] = {'auc': float(dpoint), 'ci': [float(np.percentile(dv, 2.5)), float(np.percentile(dv, 97.5))]}
json.dump(out, open('results/regime_auc.json', 'w'), indent=1)
