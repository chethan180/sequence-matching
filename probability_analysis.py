"""Probability analysis of every feature, new derived features, and overfitting checks for the yes-model.

Input : results/ext2_rows.npz  (166 features per person-introduction row, collected from simulator rollouts)
Output: printed tables + results/probability_analysis.json

Split (always by WORLD = seed x scenario variant, never by row):
  train worlds  seeds 1001-1030   (inside it, GroupKFold by seed for tuning; seeds 1021-1030 act as validation)
  test  worlds  seeds 1031-1040   (touched only for the final numbers)
Screening of features (FDR) uses the train worlds for direction and the test worlds for confirmation.
"""
import json
import warnings
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')
rng = np.random.default_rng(0)
d = np.load('results/ext2_rows.npz')
X = d['X'].astype(np.float64)
names = list(d['names'])
idx = {n: i for i, n in enumerate(names)}
seed, variant, day = d['seed'], d['variant'], d['day']
y, ans = d['y_yes'].astype(int), d['y_ans'].astype(int)
world = seed * 10 + variant
train, test = seed <= 1030, seed >= 1031
out = {}


def c(n):
    return X[:, idx[n]]


print(f'rows {len(X)} | train rows {int(train.sum())} in {len(np.unique(world[train]))} worlds | '
      f'test rows {int(test.sum())} in {len(np.unique(world[test]))} worlds | overall yes rate {y.mean():.3f}')

# ---------------------------------------------------------------- world bootstrap indices for the test set
test_worlds = np.unique(world[test])
rows_by_world = {w: np.where(test & (world == w))[0] for w in test_worlds}
boot = [np.concatenate([rows_by_world[w] for w in rng.choice(test_worlds, len(test_worlds))]) for _ in range(200)]


def auc_ci(score, label=y, idx_sets=boot):
    vals = [roc_auc_score(label[b], score[b]) for b in idx_sets]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


# ================================================================ 1. probability tables for key features
def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return mid - half, mid + half


def table(title, labels, masks):
    print(f'\n{title}')
    print(f'  {"level":22} {"n_train":>8} {"yes%train":>10} {"n_test":>8} {"yes%test":>9} {"95% CI test":>14} {"answered%":>10} {"yes|answered%":>14}')
    rows = []
    for lab, m in zip(labels, masks):
        mt, ms = m & train, m & test
        if ms.sum() == 0:
            continue
        pt = y[mt].mean() * 100 if mt.sum() else float('nan')
        ps = y[ms].mean() * 100
        lo, hi = wilson(y[ms].sum(), ms.sum())
        pa = ans[ms].mean() * 100
        pya = y[ms & (ans == 1)].mean() * 100 if (ms & (ans == 1)).sum() else float('nan')
        print(f'  {lab:22} {int(mt.sum()):8d} {pt:10.1f} {int(ms.sum()):8d} {ps:9.1f} {lo*100:6.1f}-{hi*100:5.1f} {pa:10.1f} {pya:14.1f}')
        rows.append({'level': lab, 'n_train': int(mt.sum()), 'yes_train': float(pt), 'n_test': int(ms.sum()),
                     'yes_test': float(ps), 'ci_test': [float(lo * 100), float(hi * 100)], 'answered': float(pa),
                     'yes_given_answered': float(pya)})
    out[title] = rows


def status3(k):
    e, n_ = c('eq_' + k), c('ne_' + k)
    return ['same', 'different', 'unknown'], [e == 1, n_ == 1, (e == 0) & (n_ == 0)]


print('\n' + '=' * 100 + '\n1. PROBABILITY OF A YES BY FEATURE VALUE (train and held-out test worlds)\n' + '=' * 100)
for k in ('relationship_goal', 'relationship_pace', 'conversations', 'lifestyle', 'emotional_availability',
          'space_for_relationship', 'relocate'):
    labs, ms = status3(k)
    table(f'{k}: do the two people agree?', labs, ms)

core = ['relationship_goal', 'relationship_pace', 'lifestyle', 'conversations']
n_eq = sum(c('eq_' + k) for k in core)
n_ne = sum(c('ne_' + k) for k in core)
table('number of core soft fields that AGREE (of 4)', [str(i) for i in range(5)], [n_eq == i for i in range(5)])
table('number of core soft fields that DISAGREE (of 4)', [str(i) for i in range(5)], [n_ne == i for i in range(5)])
gk, pk = status3('relationship_goal'), status3('relationship_pace')
joint_labs, joint_ms = [], []
for gl, gm in zip(*gk):
    for pl, pm in zip(*pk):
        joint_labs.append(f'goal {gl[:4]} / pace {pl[:4]}')
        joint_ms.append(gm & pm)
table('goal x pace agreement (joint)', joint_labs, joint_ms)

p_yes = c('a_p_yes')
qs = np.quantile(p_yes[train], np.linspace(0, 1, 6))
qs[-1] += 1e-9
table("person's own shrunk yes-rate (quintiles, cut-points from train)",
      [f'{qs[i]:.3f}-{qs[i+1]:.3f}' for i in range(5)], [(p_yes >= qs[i]) & (p_yes < qs[i + 1]) for i in range(5)])
a_n = c('a_n')
table('how many responses we have seen from this person', ['0', '1', '2', '3', '4+'],
      [a_n == 0, a_n == 1, a_n == 2, a_n == 3, a_n >= 4])
table("person's last answer", ['no answer yet', 'last was no', 'last was yes'],
      [c('a_last_yes') == 0.5, c('a_last_yes') == 0, c('a_last_yes') == 1])
table('day of the episode', ['0-9', '10-19', '20-34', '35-44', '45-59'],
      [day < 10, (day >= 10) & (day < 20), (day >= 20) & (day < 35), (day >= 35) & (day < 45), day >= 45])
gap = c('age_gap')
table('age gap (years)', ['0-2', '3-5', '6-9', '10+'], [gap <= 2, (gap > 2) & (gap <= 5), (gap > 5) & (gap <= 9), gap >= 10])
table('same zone', ['yes', 'no'], [c('same_zone') == 1, c('same_zone') == 0])
table("person's gender", ['woman', 'man', 'non_binary'], [c('a_g_woman') == 1, c('a_g_man') == 1, c('a_g_non_binary') == 1])
table("partner's received-yes rate (quintiles)",
      [f'q{i+1}' for i in range(5)],
      [(c('p_recv_rate') >= np.quantile(c('p_recv_rate')[train], i / 5)) &
       (c('p_recv_rate') <= np.quantile(c('p_recv_rate')[train], (i + 1) / 5)) for i in range(5)])

# ================================================================ 2. screen every feature
print('\n' + '=' * 100 + '\n2. SCREENING ALL 166 FEATURES (single-feature AUC; direction from train, confirmation on test worlds)\n' + '=' * 100)
screen = []
for j, n in enumerate(names):
    x = X[:, j]
    if x[train].std() < 1e-9:
        continue
    a_tr = roc_auc_score(y[train], x[train])
    a_te = roc_auc_score(y[test], x[test])
    sign = 1.0 if a_tr >= 0.5 else -1.0
    vals = [roc_auc_score(y[b], sign * x[b]) for b in boot[:100]]
    oriented_te = a_te if sign > 0 else 1 - a_te
    p = 2 * min(np.mean(np.array(vals) <= 0.5), np.mean(np.array(vals) >= 0.5))
    p = max(p, 1 / 101)
    screen.append((n, float(a_tr), float(a_te), float(oriented_te), float(p), spearmanr(x[test], y[test])[0]))
ps = np.array([s[4] for s in screen])
order = np.argsort(ps)
m = len(ps)
bh = np.empty(m)
bh[order] = np.minimum.accumulate((ps[order] * m / (np.arange(m) + 1))[::-1])[::-1]
sig = [(s, bh[i]) for i, s in enumerate(screen) if bh[i] <= 0.05 and ((s[1] >= 0.5) == (s[2] >= 0.5))]
sig.sort(key=lambda t: -abs(t[0][3] - 0.5))
print(f'{len(screen)} features tested; {len(sig)} survive Benjamini-Hochberg FDR 5% with the same direction on train and test')
print(f'  {"feature":30} {"AUC train":>9} {"AUC test":>9} {"p (boot)":>9} {"FDR q":>7}')
for s, q in sig[:25]:
    print(f'  {s[0]:30} {s[1]:9.4f} {s[2]:9.4f} {s[4]:9.3f} {q:7.3f}')
out['screen_significant'] = [(s[0], s[1], s[2], q) for s, q in sig]
out['screen_n_tested'] = len(screen)

# ================================================================ 3. new features
print('\n' + '=' * 100 + '\n3. NEW DERIVED FEATURES (from existing columns only; nothing from the future)\n' + '=' * 100)
new = {}
eqf = {k: c('eq_' + k) for k in core}
nef = {k: c('ne_' + k) for k in core}
new['n_core_eq'] = n_eq
new['n_core_ne'] = n_ne
new['core_net'] = n_eq - n_ne
new['n_core_eq_sq'] = n_eq ** 2
new['core_all_agree'] = (n_eq == 4).astype(float)
new['core_all_known_none_agree'] = ((n_eq == 0) & (n_ne == 4)).astype(float)
new['goal_pace_both_eq'] = eqf['relationship_goal'] * eqf['relationship_pace']
new['goal_pace_both_ne'] = nef['relationship_goal'] * nef['relationship_pace']
new['goal_eq_pace_ne'] = eqf['relationship_goal'] * nef['relationship_pace']
new['goal_ne_pace_eq'] = nef['relationship_goal'] * eqf['relationship_pace']
n_hist, answered, yes_n = c('a_n'), c('a_answered'), c('a_yes')
new['a_no'] = answered - yes_n
new['a_silent'] = n_hist - answered
new['a_silent_rate'] = (new['a_silent'] + 3 * 0.22) / (n_hist + 3)
new['a_has_hist'] = (n_hist > 0).astype(float)
new['a_n_log'] = np.log1p(n_hist)
new['a_answered_log'] = np.log1p(answered)
pyes, pans = c('a_p_yes'), c('a_p_answer')
new['logit_pyes'] = np.log(pyes / (1 - pyes))
new['logit_pans'] = np.log(pans / (1 - pans))
combined = pyes * pans
new['p_combined'] = combined
new['logit_combined'] = np.log(combined / (1 - combined))
new['pyes_tau1'] = (yes_n + 0.47) / (answered + 1)
new['pyes_tau10'] = (yes_n + 4.7) / (answered + 10)
new['hinge_pyes_low'] = np.maximum(0, 0.40 - pyes)
new['hinge_pyes_high'] = np.maximum(0, pyes - 0.47)
new['recent_vs_overall'] = c('a_recent_yes') - pyes
new['streak_yes2'] = ((c('a_last_yes') == 1) & (c('a_recent_yes') == 1)).astype(float)
new['eq_goal_x_pyes'] = eqf['relationship_goal'] * pyes
new['ne_goal_x_pyes'] = nef['relationship_goal'] * pyes
new['core_net_x_pyes'] = new['core_net'] * pyes
new['core_net_x_hist'] = new['core_net'] * new['a_has_hist']
new['pyes_minus_partner'] = pyes - c('p_p_yes')
new['recv_minus_partner_recv'] = c('a_recv_rate') - c('p_recv_rate')
new['a_recv_n_log'] = np.log1p(c('a_recv_n'))
new['day_hinge35'] = np.maximum(0, day.astype(float) - 35)
new['day_sq'] = (day / 60.0) ** 2
new['age_gap_hinge8'] = np.maximum(0, gap - 8)
new['age_gap_sq'] = gap ** 2
new['days_since_last_log'] = np.log1p(c('a_days_since_last'))
new['a_new_member'] = (c('a_n_intros') == 0).astype(float)
print(f'{len(new)} new features created')

# screen the new ones the same way
new_screen = []
for n, x in new.items():
    a_tr = roc_auc_score(y[train], x[train])
    a_te = roc_auc_score(y[test], x[test])
    new_screen.append((n, a_tr, a_te))
new_screen.sort(key=lambda t: -abs(t[2] - 0.5))
print('  strongest new features by single-feature AUC (train / test):')
for n, a_tr, a_te in new_screen[:12]:
    print(f'    {n:28} {a_tr:.4f} / {a_te:.4f}')
out['new_feature_screen'] = new_screen

# ================================================================ 4. model sets
other_soft = ['emotional_availability', 'space_for_relationship', 'relocate']
current = {
    'eq_goal': eqf['relationship_goal'], 'ne_goal': nef['relationship_goal'], 'eq_pace': eqf['relationship_pace'],
    'ne_pace': nef['relationship_pace'], 'eq_life': eqf['lifestyle'], 'ne_life': nef['lifestyle'],
    'eq_conv': eqf['conversations'], 'ne_conv': nef['conversations'],
    'n_eq_other': sum(c('eq_' + k) for k in other_soft) / 3.0, 'n_unknown_core': (4 - n_eq - n_ne) / 4.0,
    'actor_p_answer': pans, 'actor_p_yes': pyes, 'actor_n': np.minimum(n_hist, 6) / 6.0,
    'partner_recv': c('p_recv_rate'), 'partner_p_yes': c('p_p_yes'), 'late': c('late'), 'day': day / 60.0,
    'age_gap': gap / 10.0, 'same_zone': c('same_zone')}
core28 = {n: c(n) for n in names if n.startswith(('eq_', 'ne_', 'n_eq_all', 'n_ne_all', 'n_unknown_all', 'a_'))
          and not n.startswith(('a_g_', 'a_age', 'a_zone', 'a_children', 'a_smoke', 'a_wants', 'a_slots', 'a_zones',
                                'a_who', 'a_strict', 'a_pos', 'a_degree', 'a_since', 'a_desir', 'a_demand',
                                'a_supply', 'a_latency', 'a_fast', 'a_ewma', 'a_gender', 'a_ageband', 'a_yes_vs',
                                'a_pos_in', 'a_relationship', 'a_lifestyle', 'a_conversations', 'a_emotional',
                                'a_space', 'a_relocate'))}


def mat(*dicts):
    cols, nm = [], []
    for dd in dicts:
        for k, v in dd.items():
            cols.append(v)
            nm.append(k)
    return np.column_stack(cols), nm


SETS = {}
SETS['S0 current policy features (19)'] = mat(current)
SETS['S1 core: similarity + own history (28)'] = mat(core28)
SETS['S2 current 19 + new'] = mat(current, new)
SETS['S3 core 28 + new'] = mat(core28, new)
for k, (M, nm) in SETS.items():
    print(f'  {k}: {M.shape[1]} columns')

groups_train = seed[train]
cv = GroupKFold(n_splits=5)
Cs = [0.0003, 0.003, 0.03, 0.3, 3.0]


def pipe(C, penalty='l2'):
    solver = 'lbfgs' if penalty == 'l2' else 'liblinear'
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, penalty=penalty, solver=solver, max_iter=3000))


def cv_choose(M, penalty='l2'):
    best = None
    res = {}
    for C in Cs:
        aucs, lls = [], []
        for tr, va in cv.split(M[train], y[train], groups_train):
            mdl = pipe(C, penalty).fit(M[train][tr], y[train][tr])
            p = mdl.predict_proba(M[train][va])[:, 1]
            aucs.append(roc_auc_score(y[train][va], p))
            lls.append(log_loss(y[train][va], p))
        res[C] = (float(np.mean(aucs)), float(np.mean(lls)))
        if best is None or res[C][1] < res[best][1]:
            best = C
    return best, res


print('\n' + '=' * 100 + '\n4. OVERFITTING CHECKS\n' + '=' * 100)
print('Tuning: regularisation strength C chosen by 5-fold CV grouped by SEED inside the train worlds (test never used).')
results = {}
preds = {}
for k, (M, nm) in SETS.items():
    C, res = cv_choose(M)
    mdl = pipe(C).fit(M[train], y[train])
    p_all = mdl.predict_proba(M)[:, 1]
    preds[k] = p_all
    tr_auc, te_auc = roc_auc_score(y[train], p_all[train]), roc_auc_score(y[test], p_all[test])
    tr_ll, te_ll = log_loss(y[train], p_all[train]), log_loss(y[test], p_all[test])
    cv_auc, cv_ll = res[C]
    lo, hi = auc_ci(p_all)
    results[k] = {'C': C, 'train_auc': tr_auc, 'cv_auc': cv_auc, 'test_auc': te_auc, 'test_auc_ci': [lo, hi],
                  'train_logloss': tr_ll, 'cv_logloss': cv_ll, 'test_logloss': te_ll,
                  'test_brier': float(brier_score_loss(y[test], p_all[test])), 'n_features': M.shape[1]}
    print(f'\n{k}: C={C}')
    print(f'   AUC   train {tr_auc:.4f} | grouped-CV {cv_auc:.4f} | TEST {te_auc:.4f}  (95% CI over worlds {lo:.4f}-{hi:.4f})   gap train-test {tr_auc-te_auc:+.4f}')
    print(f'   logloss train {tr_ll:.4f} | grouped-CV {cv_ll:.4f} | TEST {te_ll:.4f}   (constant-rate baseline {log_loss(y[test], np.full(test.sum(), y[train].mean())):.4f})')
out['models'] = results

print('\nPaired difference in TEST AUC against the current policy features (bootstrap over test worlds):')
base_p = preds['S0 current policy features (19)']
out['deltas'] = {}
for k in SETS:
    if k.startswith('S0'):
        continue
    dl = [roc_auc_score(y[b], preds[k][b]) - roc_auc_score(y[b], base_p[b]) for b in boot]
    lo, hi = np.percentile(dl, [2.5, 97.5])
    out['deltas'][k] = {'mean': float(np.mean(dl)), 'ci': [float(lo), float(hi)]}
    print(f'   {k:45} {np.mean(dl):+.4f}  [{lo:+.4f}, {hi:+.4f}]')

# regularisation path for S3
print('\nRegularisation path for S3 (grouped-CV AUC / logloss); too little regularisation would show CV getting worse:')
_, res3 = cv_choose(SETS['S3 core 28 + new'][0])
for C, (a, l) in res3.items():
    print(f'   C={C:<8} CV AUC {a:.4f}  logloss {l:.4f}')
out['reg_path_S3'] = {str(k): v for k, v in res3.items()}

# learning curve: does more data still help? (train on the first k seeds)
print('\nLearning curve for S3 (train on the first k world-seeds; test AUC on held-out worlds):')
M3 = SETS['S3 core 28 + new'][0]
C3 = results['S3 core 28 + new']['C']
out['learning_curve'] = []
for k in (3, 6, 12, 20, 30):
    m = train & (seed <= 1000 + k)
    mdl = pipe(C3).fit(M3[m], y[m])
    p = mdl.predict_proba(M3)[:, 1]
    row = (k, int(m.sum()), float(roc_auc_score(y[m], p[m])), float(roc_auc_score(y[test], p[test])))
    out['learning_curve'].append(row)
    print(f'   {k:2d} seeds ({row[1]:6d} rows): train AUC {row[2]:.4f}   test AUC {row[3]:.4f}')

# label-permutation sanity check: with shuffled labels the model must score about 0.5 on the test worlds
yp = y.copy()
for w in np.unique(world[train]):
    ii = np.where(train & (world == w))[0]
    yp[ii] = rng.permutation(y[ii])
mdl = pipe(C3).fit(M3[train], yp[train])
p = mdl.predict_proba(M3)[:, 1]
print(f'\nLabel-permutation control (labels shuffled inside every train world): train AUC {roc_auc_score(yp[train], p[train]):.4f}, test AUC {roc_auc_score(y[test], p[test]):.4f} (should be about 0.50)')
out['permutation_control'] = {'train_auc': float(roc_auc_score(yp[train], p[train])), 'test_auc': float(roc_auc_score(y[test], p[test]))}

# L1 selection inside the CV, then L2 refit on the selected columns
print('\nL1 (lasso) feature selection on S3, chosen by grouped CV, then refit and tested once:')
C1, res1 = cv_choose(M3, 'l1')
l1 = pipe(C1, 'l1').fit(M3[train], y[train])
coefs = l1[-1].coef_[0]
keep = [i for i, v in enumerate(coefs) if abs(v) > 1e-8]
nm3 = SETS['S3 core 28 + new'][1]
print(f'   C={C1}: keeps {len(keep)} of {M3.shape[1]} columns')
Ck, _ = cv_choose(M3[:, keep])
sel = pipe(Ck).fit(M3[train][:, keep], y[train])
p_sel = sel.predict_proba(M3[:, keep])[:, 1]
te_sel = roc_auc_score(y[test], p_sel[test])
lo, hi = auc_ci(p_sel)
print(f'   refit on selected: train AUC {roc_auc_score(y[train], p_sel[train]):.4f}  TEST AUC {te_sel:.4f} [{lo:.4f}-{hi:.4f}]')
sel_coefs = sorted(((nm3[i], float(sel[-1].coef_[0][j])) for j, i in enumerate(keep)), key=lambda t: -abs(t[1]))
print('   largest standardised coefficients in the selected model:')
for n, v in sel_coefs[:15]:
    print(f'     {n:28} {v:+.3f}')
out['l1_selected'] = {'n_kept': len(keep), 'test_auc': float(te_sel), 'coefs': sel_coefs[:25]}
preds['L1 selected'] = p_sel

# calibration of the final candidate on the test worlds
print('\nCalibration of the selected model on the test worlds (does a predicted 40% mean 40% say yes?):')
p = p_sel[test]
order = np.argsort(p)
cal = []
for b in np.array_split(order, 10):
    cal.append((float(p[b].mean() * 100), float(y[test][b].mean() * 100)))
    print(f'   predicted {cal[-1][0]:5.1f}%   observed {cal[-1][1]:5.1f}%')
z = np.log(p / (1 - p)).reshape(-1, 1)
lr = LogisticRegression(C=1e6, max_iter=1000).fit(z, y[test])
print(f'   calibration slope {lr.coef_[0][0]:.3f} (1.0 = perfect), intercept {lr.intercept_[0]:+.3f} (0 = perfect)')
out['calibration'] = {'bins': cal, 'slope': float(lr.coef_[0][0]), 'intercept': float(lr.intercept_[0])}
out['predicted_yes_range_test'] = [float(np.percentile(p, 1) * 100), float(np.percentile(p, 99) * 100)]
print(f'   spread of predictions (1st to 99th percentile): {out["predicted_yes_range_test"][0]:.1f}% to {out["predicted_yes_range_test"][1]:.1f}%')

json.dump(out, open('results/probability_analysis.json', 'w'), indent=1, default=float)
print('\nsaved results/probability_analysis.json')
