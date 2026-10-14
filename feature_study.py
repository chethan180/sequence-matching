"""Extensive feature study with logistic regression and XGBoost on held-out simulator worlds.

Input : results/ext_rows.npz (collect_ext.py)   Output: results/feature_study.json + printed tables
Train worlds: seeds <= 1030 (validation = 1021..1030 for early stopping / C). Test worlds: seeds >= 1031.
All splits are by world (seed), never by row.
"""
import json
import warnings
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')
d = np.load('results/ext_rows.npz')
X, names, groups = d['X'].astype(np.float64), list(d['names']), list(d['groups'])
seed, variant, day = d['seed'], d['variant'], d['day']
y = {'yes': d['y_yes'], 'ans': d['y_ans'], 'sec': d['y_sec'], 'mutual': d['y_mutual'],
     'date': d['y_date'], 'msmi': d['y_msmi']}
ORDER = ['demo', 'hard', 'soft_raw', 'soft_eq', 'hist_actor', 'hist_partner', 'context']
gidx = {g: [i for i, gg in enumerate(groups) if gg == g] for g in ORDER}

fit_mask = seed <= 1020
val_mask = (seed > 1020) & (seed <= 1030)
train = seed <= 1030
test = seed >= 1031
print(f'rows {len(X)}  features {len(names)}  train {int(train.sum())}  test {int(test.sum())}  '
      f'yes rate {y["yes"].mean():.3f}  pairs {len(X)//2}  MSMI pairs {int(y["msmi"].sum()//2)}')
print({g: len(v) for g, v in gidx.items()})


def metrics(p, t, mask):
    p = np.clip(p[mask], 1e-6, 1 - 1e-6)
    tt = t[mask]
    return roc_auc_score(tt, p), log_loss(tt, p), brier_score_loss(tt, p)


def fit_logit(cols, target, rows_fit, rows_val):
    Xs = X[:, cols]
    mu, sd = Xs[rows_fit].mean(0), Xs[rows_fit].std(0)
    sd[sd < 1e-9] = 1.0
    Z = (Xs - mu) / sd
    best, best_ll = None, 9
    for C in (0.003, 0.03, 0.3):
        m = LogisticRegression(C=C, max_iter=3000).fit(Z[rows_fit], target[rows_fit])
        ll = log_loss(target[rows_val], m.predict_proba(Z[rows_val])[:, 1])
        if ll < best_ll:
            best, best_ll = m, ll
    return best.predict_proba(Z)[:, 1], (best, mu, sd)


def fit_xgb(cols, target, rows_fit, rows_val):
    m = XGBClassifier(n_estimators=600, max_depth=4, learning_rate=0.04, subsample=0.8, colsample_bytree=0.8,
                      min_child_weight=30, reg_lambda=5.0, early_stopping_rounds=40, eval_metric='logloss',
                      n_jobs=4, random_state=0, verbosity=0)
    m.fit(X[rows_fit][:, cols], target[rows_fit], eval_set=[(X[rows_val][:, cols], target[rows_val])], verbose=False)
    return m.predict_proba(X[:, cols])[:, 1], m


def both(cols, target_name='yes', rows_mask=None):
    t = y[target_name]
    rf = fit_mask if rows_mask is None else fit_mask & rows_mask
    rv = val_mask if rows_mask is None else val_mask & rows_mask
    te = test if rows_mask is None else test & rows_mask
    pl, _ = fit_logit(cols, t, rf, rv)
    px, mx = fit_xgb(cols, t, rf, rv)
    return metrics(pl, t, te), metrics(px, t, te), pl, px, mx


def row(label, res):
    (la, ll, lb), (xa, xl, xb) = res[0], res[1]
    print(f'  {label:34} logistic AUC {la:.4f} ll {ll:.4f} | xgboost AUC {xa:.4f} ll {xl:.4f}')
    return {'logit': {'auc': la, 'logloss': ll, 'brier': lb}, 'xgb': {'auc': xa, 'logloss': xl, 'brier': xb}}


out = {}
all_cols = list(range(len(names)))

print('\n=== E1. Build-up ladder (target: said YES), held-out worlds ===')
out['ladder'] = {}
base_rate = float(y['yes'][train].mean())
print(f'  {"constant":34} AUC 0.5000  logloss {log_loss(y["yes"][test], np.full(test.sum(), base_rate)):.4f}')
cols = []
for g in ORDER:
    cols = cols + gidx[g]
    out['ladder']['+' + g] = row('+ ' + g + f' ({len(cols)} features)', both(cols))

print('\n=== E2. Each group ALONE ===')
out['alone'] = {}
for g in ORDER:
    out['alone'][g] = row(g + f' only ({len(gidx[g])})', both(gidx[g]))

print('\n=== E3. Leave-one-group-out from the full set ===')
full = both(all_cols)
out['full'] = row('ALL groups', full)
out['drop'] = {}
for g in ORDER:
    cols = [i for i in all_cols if groups[i] != g]
    out['drop'][g] = row('drop ' + g, both(cols))

print('\n=== E4. Most useful single features (XGBoost, gain and permutation on held-out worlds) ===')
mx = full[4]
px_full = full[3]
gain = mx.get_booster().get_score(importance_type='gain')
gain_named = sorted(((names[int(k[1:])], v) for k, v in gain.items()), key=lambda t: -t[1])[:15]
rng = np.random.default_rng(0)
base_auc = roc_auc_score(y['yes'][test], px_full[test])
perm = []
Xt = X[test].copy()
for i in range(len(names)):
    drops = []
    for _ in range(2):
        Xp = Xt.copy()
        Xp[:, i] = rng.permutation(Xp[:, i])
        drops.append(base_auc - roc_auc_score(y['yes'][test], mx.predict_proba(Xp)[:, 1]))
    perm.append((names[i], groups[i], float(np.mean(drops))))
perm.sort(key=lambda t: -t[2])
print('  permutation importance (AUC lost when the feature is shuffled):')
for n, g, v in perm[:15]:
    print(f'    {n:28} {g:13} {v:+.4f}')
out['permutation_top'] = perm[:25]
out['gain_top'] = gain_named

print('\n=== E5. Target decomposition: answer, then yes-given-answer ===')
t_ans = y['ans']
res_ans = both(all_cols, 'ans')
out['answer_model'] = row('P(answers at all)', res_ans)
ans_rows = y['ans'] == 1
yes_given = y['yes'].copy()
res_yg = both(all_cols, 'yes', ans_rows)
out['yes_given_answer'] = row('P(yes | answered)', res_yg)
p_dec_l = fit_logit(all_cols, t_ans, fit_mask, val_mask)[0] * fit_logit(all_cols, y['yes'], fit_mask & ans_rows, val_mask & ans_rows)[0]
p_dec_x = fit_xgb(all_cols, t_ans, fit_mask, val_mask)[0] * fit_xgb(all_cols, y['yes'], fit_mask & ans_rows, val_mask & ans_rows)[0]
dec = (metrics(p_dec_l, y['yes'], test), metrics(p_dec_x, y['yes'], test))
out['decomposed_yes'] = row('answer x yes|answered  => YES', dec)
out['direct_yes'] = row('direct P(yes)', full)

print('\n=== E6. Pair level: can we predict MUTUAL yes / date / MSMI? ===')
assert np.all(y['mutual'][0::2] == y['mutual'][1::2]) and np.all(seed[0::2] == seed[1::2])
pair_seed = seed[0::2]
pair_fit, pair_val, pair_test = pair_seed <= 1020, (pair_seed > 1020) & (pair_seed <= 1030), pair_seed >= 1031
prod_l = full[2][0::2] * full[2][1::2]
prod_x = full[3][0::2] * full[3][1::2]
Xpair = np.hstack([X[0::2], X[1::2]])
out['pair'] = {}
for tname in ('mutual', 'date', 'msmi'):
    tt = y[tname][0::2]
    pos_tr, pos_te = int(tt[pair_fit | pair_val].sum()), int(tt[pair_test].sum())
    line = {'positives_train': pos_tr, 'positives_test': pos_te}
    res = {}
    for label, score in (('product of logistic yes', prod_l), ('product of xgboost yes', prod_x)):
        res[label] = float(roc_auc_score(tt[pair_test], score[pair_test])) if pos_te > 5 else None
    m = XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.04, subsample=0.8, colsample_bytree=0.5,
                      min_child_weight=40, reg_lambda=10.0, early_stopping_rounds=30, eval_metric='logloss',
                      n_jobs=4, random_state=0, verbosity=0)
    m.fit(Xpair[pair_fit], tt[pair_fit], eval_set=[(Xpair[pair_val], tt[pair_val])], verbose=False)
    res['direct pair xgboost'] = float(roc_auc_score(tt[pair_test], m.predict_proba(Xpair[pair_test])[:, 1])) if pos_te > 5 else None
    line['auc'] = res
    out['pair'][tname] = line
    print(f'  {tname:7} positives train {pos_tr:6d} test {pos_te:5d} | ' +
          ' | '.join(f'{k}: {v:.4f}' for k, v in res.items() if v is not None))

print('\n=== E7. Cold start: does history help early? (XGBoost AUC by day, held-out worlds) ===')
no_hist = [i for i in all_cols if not groups[i].startswith('hist')]
p_nohist = fit_xgb(no_hist, y['yes'], fit_mask, val_mask)[0]
out['by_day'] = {}
for lo, hi in ((0, 10), (10, 20), (20, 40), (40, 60)):
    m = test & (day >= lo) & (day < hi)
    a_full = roc_auc_score(y['yes'][m], px_full[m])
    a_nh = roc_auc_score(y['yes'][m], p_nohist[m])
    out['by_day'][f'{lo}-{hi}'] = {'n': int(m.sum()), 'full': a_full, 'no_history': a_nh}
    print(f'  days {lo:2d}-{hi:2d}  n={int(m.sum()):6d}   all features {a_full:.4f}   without history {a_nh:.4f}')

print('\n=== E8. World-level bootstrap: full xgboost vs the 19 engineered features used in policy.py ===')
current = [names.index(n) for n in ('age_gap', 'same_zone', 'eq_relationship_goal', 'ne_relationship_goal',
                                    'eq_relationship_pace', 'ne_relationship_pace', 'eq_lifestyle', 'ne_lifestyle',
                                    'eq_conversations', 'ne_conversations', 'a_p_answer', 'a_p_yes', 'a_n',
                                    'p_recv_rate', 'p_p_yes', 'late', 'day')]
p_cur = fit_xgb(current, y['yes'], fit_mask, val_mask)[0]
worlds = np.unique(seed[test] * 10 + variant[test])
key = seed * 10 + variant
boot = []
for _ in range(200):
    pick = rng.choice(worlds, len(worlds))
    idx = np.concatenate([np.where(test & (key == w))[0] for w in pick])
    boot.append(roc_auc_score(y['yes'][idx], px_full[idx]) - roc_auc_score(y['yes'][idx], p_cur[idx]))
lo_, hi_ = np.percentile(boot, [2.5, 97.5])
print(f'  AUC full - AUC(policy.py-style subset): {np.mean(boot):+.4f}  95% CI [{lo_:+.4f}, {hi_:+.4f}]  (worlds={len(worlds)})')
out['bootstrap_full_vs_current'] = {'mean': float(np.mean(boot)), 'ci': [float(lo_), float(hi_)]}

json.dump(out, open('results/feature_study.json', 'w'), indent=1, default=float)
print('\nsaved results/feature_study.json')
