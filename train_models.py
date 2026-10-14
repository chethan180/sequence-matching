"""Train and evaluate the response models offline, then export them for the pure-Python policy.

Input : results/train_rows.npz  (made by experiments.py --collect; one row per person per introduction,
        features observed *before* the decision, label = what that person later answered)
Output: model_weights.json (loaded by policy.py), results/offline_metrics.json

Split is by simulator world (seed), never by row: seeds <= 1030 train, seeds >= 1031 held out.
"""
import json
import math
import sys
from pathlib import Path
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

import policy

SPLIT_SEED = 1030
data = np.load('results/train_rows.npz')
X, y_yes, y_ans, y_sec = data['X'].astype(np.float64), data['y_yes'], data['y_ans'], data['y_sec']
seed, variant = data['seed'], data['variant']
features = list(data['features'])
assert features == policy.FEATURES, 'feature order changed; recollect the rows'

train = seed <= SPLIT_SEED
test = ~train
print('rows', len(X), '| train', int(train.sum()), '| held-out', int(test.sum()),
      '| yes rate', round(float(y_yes.mean()), 4), '| answered rate', round(float(y_ans.mean()), 4))

mean = X[train].mean(axis=0)
scale = X[train].std(axis=0)
scale[scale < 1e-9] = 1.0
Z = (X - mean) / scale


def evaluate(name, p, y, mask, report):
    p = np.clip(p[mask], 1e-6, 1 - 1e-6)
    yy = y[mask]
    row = {'auc': float(roc_auc_score(yy, p)), 'log_loss': float(log_loss(yy, p)),
           'brier': float(brier_score_loss(yy, p))}
    report[name] = row
    print(f'  {name:28} AUC {row["auc"]:.4f}   log-loss {row["log_loss"]:.4f}   Brier {row["brier"]:.4f}')


report = {'yes': {}}
print('\nTarget: this person said YES to the introduction (no-response counts as not-yes), held-out worlds')
base = float(y_yes[train].mean())
evaluate('constant (population rate)', np.full(len(X), base), y_yes, test, report['yes'])
hist = X[:, features.index('actor_p_answer')] * X[:, features.index('actor_p_yes')]
evaluate('own history only (shrunk)', hist, y_yes, test, report['yes'])
core_eq = sum(X[:, features.index(k)] for k in ('eq_goal', 'eq_pace', 'eq_life', 'eq_conv'))
evaluate('soft-match count only', 0.3 + 0.05 * core_eq, y_yes, test, report['yes'])

logit = LogisticRegression(C=1.0, max_iter=2000)
logit.fit(Z[train], y_yes[train])
p_logit = logit.predict_proba(Z)[:, 1]
evaluate('logistic regression', p_logit, y_yes, test, report['yes'])

gbm = GradientBoostingClassifier(n_estimators=80, max_depth=3, learning_rate=0.1, min_samples_leaf=50,
                                 subsample=0.8, random_state=0)
gbm.fit(X[train], y_yes[train])
p_gbm = gbm.predict_proba(X)[:, 1]
evaluate('gradient boosting (80 trees)', p_gbm, y_yes, test, report['yes'])

print('\nLogistic coefficients (standardised features; positive = more likely to say yes):')
coefs = sorted(zip(features, logit.coef_[0]), key=lambda t: -abs(t[1]))
for name, c in coefs:
    print(f'  {name:16} {c:+.3f}')
report['logit_coefficients'] = {n: float(c) for n, c in coefs}
print('  intercept', round(float(logit.intercept_[0]), 3))

print('\nGBM feature importance:')
imp = sorted(zip(features, gbm.feature_importances_), key=lambda t: -t[1])
for name, c in imp[:8]:
    print(f'  {name:16} {c:.3f}')
report['gbm_importance'] = {n: float(c) for n, c in imp}

print('\nCalibration of the logistic model on held-out worlds (10 equal-count bins):')
order = np.argsort(p_logit[test])
bins = np.array_split(order, 10)
cal = []
for b in bins:
    cal.append((float(p_logit[test][b].mean()), float(y_yes[test][b].mean())))
    print(f'  predicted {cal[-1][0]:.3f}   observed {cal[-1][1]:.3f}')
report['calibration_logit'] = cal

print('\nAUC of the logistic model per scenario variant (held-out worlds):')
report['auc_by_variant'] = {}
for v, name in enumerate(('development', 'sparse', 'cold_start', 'delayed', 'shift', 'drift')):
    m = test & (variant == v)
    if m.sum() > 50 and len(set(y_yes[m])) == 2:
        auc = roc_auc_score(y_yes[m], p_logit[m])
        report['auc_by_variant'][name] = float(auc)
        print(f'  {name:12} {auc:.4f}  (n={int(m.sum())})')

# second-meeting model: given a date happened, did this person say yes to meeting again
has_date = y_sec >= 0
sec = LogisticRegression(C=1.0, max_iter=2000)
sec.fit(Z[train & has_date], y_sec[train & has_date])
print('\nSecond-meeting model: rows with a date  train', int((train & has_date).sum()), ' held-out', int((test & has_date).sum()))
if (test & has_date).sum() > 30 and len(set(y_sec[test & has_date])) == 2:
    p_sec = sec.predict_proba(Z)[:, 1]
    report['second'] = {}
    evaluate('second-meeting logistic', p_sec, y_sec, test & has_date, report['second'])
    evaluate('second-meeting constant', np.full(len(X), float(y_sec[train & has_date].mean())), y_sec,
             test & has_date, report['second'])

# ---------------------------------------------------------------- export
trees = []
for est in gbm.estimators_[:, 0]:
    t = est.tree_
    trees.append({'left': t.children_left.tolist(), 'right': t.children_right.tolist(),
                  'feature': t.feature.tolist(), 'threshold': [float(v) for v in t.threshold],
                  'value': [float(v) for v in t.value[:, 0, 0]]})
prior = float(gbm.init_.class_prior_[1])
weights = {
    'features': features,
    'logit': {'intercept': float(logit.intercept_[0]), 'coef': [float(c) for c in logit.coef_[0]],
              'mean': [float(m) for m in mean], 'scale': [float(s) for s in scale]},
    'second': {'intercept': float(sec.intercept_[0]), 'coef': [float(c) for c in sec.coef_[0]],
               'mean': [float(m) for m in mean], 'scale': [float(s) for s in scale]},
    'gbm': {'init': math.log(prior / (1 - prior)), 'lr': float(gbm.learning_rate), 'trees': trees},
    'trained_on': {'rows': int(train.sum()), 'seeds': '1001-1030', 'target': 'said yes to the introduction'},
}
Path('model_weights.json').write_text(json.dumps(weights), encoding='utf-8')

# the pure-Python scorers used inside the container must reproduce sklearn
policy._MODEL = None
model = policy.load_model()
idx = np.random.default_rng(0).choice(len(X), 400, replace=False)
err_l = max(abs(policy.logistic_p(model['logit'], X[i].tolist()) - p_logit[i]) for i in idx)
err_g = max(abs(policy.gbm_p(model['gbm'], X[i].tolist()) - p_gbm[i]) for i in idx)
print(f'\nexport check (max |pure-Python - sklearn| on 400 rows): logistic {err_l:.2e}, gbm {err_g:.2e}')
assert err_l < 1e-6 and err_g < 1e-5
report['export_check'] = {'logistic': float(err_l), 'gbm': float(err_g)}
size_kb = Path('model_weights.json').stat().st_size / 1024
print(f'model_weights.json: {size_kb:.0f} KiB')
Path('results/offline_metrics.json').write_text(json.dumps(report, indent=1))
