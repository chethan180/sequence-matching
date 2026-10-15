"""Train the shipped yes-model (Round 2) and the research-only models from simulator rollouts.

Input : results/train_rows_r2.npz, made by
        python experiments.py --configs greedy,random,global_learned --seeds 1001-1040 --variants all \
               --independent --collect --output results/train_rows_r2.npz
        One row per person per introduction: features observed before the decision, labels = what that person later
        answered. Labels are observed answers only, never the simulator's hidden values.
Output: model_weights.json             shipped: logistic yes-model and second-meeting model on SHIPPED_FEATURES
        results/research_weights.json  research only (boosted trees on all logged features); never in the image
        results/offline_metrics_r2.json

Validation splits by world seed, never by row: seeds <= 1030 train, seeds > 1030 held out, every scenario family in
both. Once validated, the shipped weights are refitted on every seed.
"""
import json
import math
from pathlib import Path
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from smp import model as M

SPLIT_SEED = 1030
VARIANTS = ('development', 'sparse', 'cold_start', 'delayed', 'shift', 'drift')
data = np.load('results/train_rows_r2.npz')
X, y_yes, y_sec = data['X'].astype(np.float64), data['y_yes'], data['y_sec']
seed, variant = data['seed'], data['variant']
features = list(data['features'])
assert features == M.FEATURES, 'feature order changed; recollect the rows'
ship = [features.index(f) for f in M.SHIPPED_FEATURES]
train, test = seed <= SPLIT_SEED, seed > SPLIT_SEED
rng = np.random.default_rng(0)
report = {'rows': int(len(X)), 'train_rows': int(train.sum()), 'test_rows': int(test.sum()),
          'yes_rate': float(y_yes.mean()), 'priors': {'answer': M.PRIOR_ANSWER, 'yes': M.PRIOR_YES}}
print(f"rows {len(X)} | train {int(train.sum())} | held-out {int(test.sum())} | yes rate {y_yes.mean():.4f}")


def fit_logit(cols, rows, target):
    A = X[rows][:, cols]
    mean, scale = A.mean(axis=0), A.std(axis=0)
    scale[scale < 1e-9] = 1.0
    clf = LogisticRegression(C=1.0, max_iter=2000).fit((A - mean) / scale, target[rows])

    def predict(mask):
        return clf.predict_proba((X[mask][:, cols] - mean) / scale)[:, 1]
    spec = {'intercept': float(clf.intercept_[0]), 'coef': [float(c) for c in clf.coef_[0]],
            'mean': [float(v) for v in mean], 'scale': [float(v) for v in scale]}
    return predict, spec


def metrics(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {'auc': float(roc_auc_score(y, p)), 'log_loss': float(log_loss(y, p)), 'brier': float(brier_score_loss(y, p))}


def seed_bootstrap_diff(y, p1, p0, seeds, n=400):
    """AUC(p1) - AUC(p0) with a 95% interval from resampling whole held-out worlds."""
    groups = [np.where(seeds == s)[0] for s in np.unique(seeds)]
    diffs = []
    for _ in range(n):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        diffs.append(roc_auc_score(y[idx], p1[idx]) - roc_auc_score(y[idx], p0[idx]))
    return [float(np.mean(diffs)), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]


# ------------------------------------------------------------------ validation (held-out worlds)
print('\nHeld-out worlds, target: said yes (no response counts as not yes)')
p_all, _ = fit_logit(list(range(len(features))), train, y_yes)
p_ship, _ = fit_logit(ship, train, y_yes)
yt, st, vt = y_yes[test], seed[test], variant[test]
pa, ps = p_all(test), p_ship(test)
report['validation'] = {'constant': metrics(yt, np.full(len(yt), y_yes[train].mean())),
                        'logit_all_19': metrics(yt, pa), 'logit_shipped_14': metrics(yt, ps)}
report['validation']['shipped_minus_all_auc'] = seed_bootstrap_diff(yt, ps, pa, st)
gbm = GradientBoostingClassifier(n_estimators=80, max_depth=3, learning_rate=0.1, min_samples_leaf=50,
                                 subsample=0.8, random_state=0).fit(X[train], y_yes[train])
pg = gbm.predict_proba(X[test])[:, 1]
report['validation']['gbm_all_19'] = metrics(yt, pg)
for name, row in report['validation'].items():
    if isinstance(row, dict):
        print(f"  {name:20} AUC {row['auc']:.4f}  log-loss {row['log_loss']:.4f}  Brier {row['brier']:.4f}")
d = report['validation']['shipped_minus_all_auc']
print(f"  shipped - all features: {d[0]:+.4f}  95% [{d[1]:+.4f}, {d[2]:+.4f}]")

order = np.argsort(ps)
report['calibration_shipped'] = [(float(ps[b].mean()), float(yt[b].mean())) for b in np.array_split(order, 10)]
print('\nCalibration of the shipped model (10 equal-count bins, predicted vs observed):')
for pred, obs in report['calibration_shipped']:
    print(f'  {pred:.3f}  {obs:.3f}')

report['auc_by_variant'] = {}
print('\nAUC by scenario family: trained with all families | trained without that family')
for v, name in enumerate(VARIANTS):
    m = vt == v
    p_wo, _ = fit_logit(ship, train & (variant != v), y_yes)
    with_ = float(roc_auc_score(yt[m], ps[m]))
    without = float(roc_auc_score(yt[m], p_wo(test)[m]))
    report['auc_by_variant'][name] = {'with': with_, 'without': without, 'rows': int(m.sum())}
    print(f'  {name:12} {with_:.4f} | {without:.4f}   (n={int(m.sum())})')

has_date = y_sec >= 0
p_sec, _ = fit_logit(ship, train & has_date, y_sec)
ms = test & has_date
report['second_validation'] = {'rows_train': int((train & has_date).sum()), 'rows_test': int(ms.sum()),
                               'logit': metrics(y_sec[ms], p_sec(ms)),
                               'constant': metrics(y_sec[ms], np.full(int(ms.sum()), y_sec[train & has_date].mean()))}
print(f"\nSecond-meeting model (rows with a date): held-out AUC {report['second_validation']['logit']['auc']:.4f} "
      f"on {int(ms.sum())} rows")

# ------------------------------------------------------------------ final fit on every seed, export
everything = np.ones(len(X), dtype=bool)
predict_final, logit_spec = fit_logit(ship, everything, y_yes)
_, second_spec = fit_logit(ship, everything & has_date, y_sec)
logit_spec['features'] = second_spec['features'] = list(M.SHIPPED_FEATURES)
weights = {'features': list(M.SHIPPED_FEATURES), 'logit': logit_spec, 'second': second_spec,
           'priors': {'answer': M.PRIOR_ANSWER, 'yes': M.PRIOR_YES, 'strength': M.PRIOR_STRENGTH,
                      'source': 'training pools public_01-06'},
           'trained_on': {'rows': int(len(X)), 'seeds': '1001-1040', 'families': list(VARIANTS),
                          'behaviour_policies': ['greedy', 'random', 'global_learned'],
                          'target': 'said yes to the introduction'}}
Path('model_weights.json').write_text(json.dumps(weights), encoding='utf-8')

gbm_all = GradientBoostingClassifier(n_estimators=80, max_depth=3, learning_rate=0.1, min_samples_leaf=50,
                                     subsample=0.8, random_state=0).fit(X, y_yes)
trees = []
for est in gbm_all.estimators_[:, 0]:
    t = est.tree_
    trees.append({'left': t.children_left.tolist(), 'right': t.children_right.tolist(), 'feature': t.feature.tolist(),
                  'threshold': [float(v) for v in t.threshold], 'value': [float(v) for v in t.value[:, 0, 0]]})
prior = float(gbm_all.init_.class_prior_[1])
Path('results/research_weights.json').write_text(json.dumps({'gbm': {
    'init': math.log(prior / (1 - prior)), 'lr': float(gbm_all.learning_rate), 'trees': trees,
    'features': features}}), encoding='utf-8')

# the pure-Python scorer used inside the container must reproduce scikit-learn
spec = M.reload_model()['logit']
idx = rng.choice(len(X), 400, replace=False)
p_ref = predict_final(everything)[idx]
err = max(abs(M.logistic_p(spec, X[i][ship].tolist()) - p_ref[k]) for k, i in enumerate(idx))
assert err < 1e-6, err
report['export_check_max_abs_error'] = float(err)
report['final_coefficients'] = dict(zip(M.SHIPPED_FEATURES, logit_spec['coef']))
Path('results/offline_metrics_r2.json').write_text(json.dumps(report, indent=1))
print(f"\nexport check {err:.1e}; model_weights.json {Path('model_weights.json').stat().st_size / 1024:.1f} KiB")
