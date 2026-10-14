"""Out-of-fold accuracy of the yes-model with honest (seed-clustered) intervals.

Every row is predicted by a model that never saw its seed (5-fold cross-validation grouped by seed, 40 seeds).
Intervals: 95% bootstrap over the 40 seeds, the only independent unit (see proofs.py). Observable inputs only.
"""
import json
import warnings
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import policy

warnings.filterwarnings('ignore')
rng = np.random.default_rng(0)
d = np.load('results/seq_rows.npz')
x19, now_eq, y, ans, seed = d['x19'].astype(float), d['now_eq'], d['y'].astype(int), d['ans'].astype(int), d['seed']
day = d['scal'][:, 0].astype(float)
names = policy.FEATURES
seeds, sid = np.unique(seed, return_inverse=True)
picks = [np.concatenate([np.where(sid == s)[0] for s in rng.integers(0, len(seeds), len(seeds))]) for _ in range(300)]


def oof(M):
    p = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(M, y, seed):
        p[te] = make_pipeline(StandardScaler(), LogisticRegression(C=0.03, max_iter=3000)).fit(M[tr], y[tr]).predict_proba(M[te])[:, 1]
    return p


def ci(fn):
    v = [fn(b) for b in picks]
    return np.percentile(v, 2.5), np.percentile(v, 97.5)


compat = np.column_stack([(now_eq == 1).astype(float), (now_eq == 2).astype(float), (day >= 35).astype(float), day / 60.0])
hist_cols = [names.index(n) for n in ('actor_p_answer', 'actor_p_yes', 'actor_n')]
models = {
    'baseline soft-match count (number of equal known fields)': (now_eq == 1).sum(1).astype(float),
    'agreement features only (same / different / unknown per field)': oof(compat),
    "person's own history only": oof(x19[:, hist_cols]),
    'proposed yes-model (19 features)': oof(x19),
}
out = {}
ref = models['baseline soft-match count (number of equal known fields)']
print(f'rows {len(y)} | seeds {len(seeds)} | every prediction is out-of-fold by seed')
print(f'{"predictor of said-yes":64} {"AUC":>7} {"95% CI (seeds)":>18} {"gain over baseline score":>26}')
for name, p in models.items():
    a = roc_auc_score(y, p)
    lo, hi = ci(lambda b: roc_auc_score(y[b], p[b]))
    dlo, dhi = ci(lambda b: roc_auc_score(y[b], p[b]) - roc_auc_score(y[b], ref[b]))
    dm = a - roc_auc_score(y, ref)
    print(f'{name:64} {a:7.4f} [{lo:.4f}, {hi:.4f}]   {dm:+.4f} [{dlo:+.4f}, {dhi:+.4f}]')
    out[name] = {'auc': float(a), 'ci': [float(lo), float(hi)], 'gain': float(dm), 'gain_ci': [float(dlo), float(dhi)]}
p = models['proposed yes-model (19 features)']
const = np.full(len(y), y.mean())
print(f'\nlog-loss: proposed {log_loss(y, p):.4f}   constant rate {log_loss(y, const):.4f}')
print('calibration of the proposed model (out-of-fold, 10 equal-count bins):')
cal = []
for b in np.array_split(np.argsort(p), 10):
    cal.append((float(p[b].mean() * 100), float(y[b].mean() * 100)))
    print(f'   predicted {cal[-1][0]:5.1f}%   observed {cal[-1][1]:5.1f}%')
z = np.log(p / (1 - p)).reshape(-1, 1)
lr = LogisticRegression(C=1e6, max_iter=1000).fit(z, y)
slo, shi = ci(lambda b: LogisticRegression(C=1e6, max_iter=500).fit(z[b], y[b]).coef_[0][0])
print(f'calibration slope {lr.coef_[0][0]:.3f} [{slo:.3f}, {shi:.3f}] (1 = perfect), intercept {lr.intercept_[0]:+.3f}; '
      f'largest bin error {max(abs(a - b) for a, b in cal):.1f} points')
out['calibration'] = {'bins': cal, 'slope': float(lr.coef_[0][0]), 'slope_ci': [float(slo), float(shi)], 'intercept': float(lr.intercept_[0])}
out['logloss'] = {'proposed': float(log_loss(y, p)), 'constant': float(log_loss(y, const))}
json.dump(out, open('results/oof_auc.json', 'w'), indent=1)
