"""Do the domain-knowledge features add anything? Logistic regression and XGBoost on held-out worlds.

Input: results/ext2_rows.npz (collect_ext2.py). Same split as feature_study.py: train seeds <= 1030, test >= 1031.
Output: results/feature_study2.json + printed tables.
"""
import json
import warnings
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')
d = np.load('results/ext2_rows.npz')
X, names, groups = d['X'].astype(np.float64), list(d['names']), list(d['groups'])
seed, variant, day, y = d['seed'], d['variant'], d['day'], d['y_yes']
fit_m, val_m = seed <= 1020, (seed > 1020) & (seed <= 1030)
test = seed >= 1031
gi = {}
for i, g in enumerate(groups):
    gi.setdefault(g, []).append(i)
DOMAIN = ['desirability', 'demand_supply', 'range_fit', 'latency', 'trait_logic', 'recency_segment']
CORE = gi['soft_eq'] + gi['hist_actor']
OLD = [i for g in ('demo', 'hard', 'soft_raw', 'soft_eq', 'hist_actor', 'hist_partner', 'context') for i in gi[g]]
print(f'rows {len(X)} features {len(names)} train {int((fit_m | val_m).sum())} test {int(test.sum())}')


def fit_logit(cols):
    Xs = X[:, cols]
    mu, sd = Xs[fit_m].mean(0), Xs[fit_m].std(0)
    sd[sd < 1e-9] = 1.0
    Z = (Xs - mu) / sd
    best, ll_best = None, 9
    for C in (0.003, 0.03, 0.3):
        m = LogisticRegression(C=C, max_iter=3000).fit(Z[fit_m], y[fit_m])
        ll = log_loss(y[val_m], m.predict_proba(Z[val_m])[:, 1])
        if ll < ll_best:
            best, ll_best = m, ll
    return best.predict_proba(Z)[:, 1], best, mu, sd


def fit_xgb(cols):
    m = XGBClassifier(n_estimators=600, max_depth=4, learning_rate=0.04, subsample=0.8, colsample_bytree=0.8,
                      min_child_weight=30, reg_lambda=5.0, early_stopping_rounds=40, eval_metric='logloss',
                      n_jobs=4, random_state=0, verbosity=0)
    m.fit(X[fit_m][:, cols], y[fit_m], eval_set=[(X[val_m][:, cols], y[val_m])], verbose=False)
    return m.predict_proba(X[:, cols])[:, 1], m


keyw = seed * 10 + variant
worlds = np.unique(keyw[test])
rng = np.random.default_rng(1)
boot_idx = [np.concatenate([np.where(test & (keyw == w))[0] for w in rng.choice(worlds, len(worlds))])
            for _ in range(200)]


def auc(p, idx=None):
    return roc_auc_score(y[test] if idx is None else y[idx], p[test] if idx is None else p[idx])


def compare(label, cols, ref_p_l, ref_p_x):
    pl, *_ = fit_logit(cols)
    px, _ = fit_xgb(cols)
    dl = [auc(pl, b) - auc(ref_p_l, b) for b in boot_idx]
    dx = [auc(px, b) - auc(ref_p_x, b) for b in boot_idx]
    print(f'  {label:36} logistic {auc(pl):.4f} ({np.mean(dl):+.4f} [{np.percentile(dl,2.5):+.4f},{np.percentile(dl,97.5):+.4f}])'
          f' | xgboost {auc(px):.4f} ({np.mean(dx):+.4f} [{np.percentile(dx,2.5):+.4f},{np.percentile(dx,97.5):+.4f}])')
    return {'logit_auc': auc(pl), 'xgb_auc': auc(px), 'logit_delta': float(np.mean(dl)),
            'logit_ci': [float(np.percentile(dl, 2.5)), float(np.percentile(dl, 97.5))],
            'xgb_delta': float(np.mean(dx)), 'xgb_ci': [float(np.percentile(dx, 2.5)), float(np.percentile(dx, 97.5))]}


out = {}
print('\n=== A. Domain groups on their own (AUC, 0.5 = chance) ===')
out['alone'] = {}
for g in DOMAIN:
    pl, *_ = fit_logit(gi[g])
    px, _ = fit_xgb(gi[g])
    out['alone'][g] = {'logit': auc(pl), 'xgb': auc(px)}
    print(f'  {g:18} ({len(gi[g])} features)  logistic {auc(pl):.4f}   xgboost {auc(px):.4f}')

print('\n=== B. Added to the core set (similarity + own history = 28 features); delta AUC with world-bootstrap 95% CI ===')
core_l, *_ = fit_logit(CORE)
core_x, _ = fit_xgb(CORE)
print(f'  core only                          logistic {auc(core_l):.4f} | xgboost {auc(core_x):.4f}')
out['core'] = {'logit': auc(core_l), 'xgb': auc(core_x)}
out['add_group'] = {}
for g in DOMAIN:
    out['add_group'][g] = compare('core + ' + g, CORE + gi[g], core_l, core_x)
all_dom = [i for g in DOMAIN for i in gi[g]]
out['add_group']['ALL domain'] = compare('core + ALL domain (27)', CORE + all_dom, core_l, core_x)

print('\n=== C. Added to the full earlier set (139 features) ===')
old_l, *_ = fit_logit(OLD)
old_x, _ = fit_xgb(OLD)
print(f'  earlier 139 features               logistic {auc(old_l):.4f} | xgboost {auc(old_x):.4f}')
out['old139'] = {'logit': auc(old_l), 'xgb': auc(old_x)}
out['old_plus_domain'] = compare('139 + ALL domain', OLD + all_dom, old_l, old_x)

print('\n=== D. Single domain features added to core (logistic delta AUC; top 12) ===')
singles = []
for i in all_dom:
    pl, *_ = fit_logit(CORE + [i])
    singles.append((names[i], groups[i], auc(pl) - auc(core_l)))
singles.sort(key=lambda t: -t[2])
for n, g, v in singles[:12]:
    print(f'  {n:24} {g:16} {v:+.4f}')
out['single_top'] = singles[:12]

print('\n=== E. Interpretation aid: logistic coefficients of the best compact model (core + helpful domain groups) ===')
helpful = [g for g in DOMAIN if out['add_group'][g]['logit_delta'] > 0.001]
cols = CORE + [i for g in helpful for i in gi[g]]
pl, model, mu, sd = fit_logit(cols)
coef = sorted(zip([names[i] for i in cols], model.coef_[0]), key=lambda t: -abs(t[1]))
print('  groups kept:', helpful, '| AUC', round(auc(pl), 4))
for n, c in coef[:14]:
    print(f'  {n:26} {c:+.3f}')
out['compact'] = {'groups': helpful, 'auc': auc(pl), 'coef': coef[:20]}
json.dump(out, open('results/feature_study2.json', 'w'), indent=1, default=float)
print('\nsaved results/feature_study2.json')
