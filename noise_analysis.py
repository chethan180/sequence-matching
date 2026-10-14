"""Do indiscriminate responders (high yes-rate over several introductions) make compatibility features unreliable?

Hypothesis from the team: if someone has said yes a lot across multiple introductions, their stated features say little
about whom they will accept, so the model should trust agreement features less for them ("add noise to their features").

Input: results/ext2_rows.npz (person-level rows; history features are those observed before the decision)
Splits: by world; train seeds 1001-1030 (grouped CV by seed for tuning), test seeds 1031-1040.
"""
import json
import warnings
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
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
y = d['y_yes'].astype(int)
world = seed * 10 + variant
train, test = seed <= 1030, seed >= 1031


def c(n):
    return X[:, idx[n]]


core = ['relationship_goal', 'relationship_pace', 'lifestyle', 'conversations']
eq = {k: c('eq_' + k) for k in core}
ne = {k: c('ne_' + k) for k in core}
n_eq = sum(eq.values())
n_ne = sum(ne.values())
answered, yes_n, n_hist = c('a_answered'), c('a_yes'), c('a_n')
raw_rate = np.where(answered > 0, yes_n / np.maximum(answered, 1), np.nan)
out = {}

# ================================================================ 1. who is "indiscriminate"?
print('=' * 100 + '\n1. HOW COMMON ARE HIGH-YES-RATE, MULTI-INTRODUCTION PEOPLE?\n' + '=' * 100)
print('Prior answers = how many answers we had seen from this person before this decision.')
print(f'  {"prior answers":14} {"share of rows":>14} {"mean yes-rate so far":>22}')
for lo, hi in ((0, 0), (1, 1), (2, 2), (3, 3), (4, 99)):
    m = (answered >= lo) & (answered <= hi)
    print(f'  {str(lo) + ("+" if hi == 99 else ("" if lo == hi else "-" + str(hi))):14} {m.mean()*100:13.1f}% '
          f'{np.nanmean(raw_rate[m]) if m.any() and lo > 0 else float("nan"):22.2f}')

flag = (answered >= 2) & (raw_rate >= 0.6)
print(f'\nFlag "indiscriminate" = at least 2 prior answers AND yes-rate so far >= 60%: {flag.mean()*100:.1f}% of rows '
      f'({int(flag.sum())} rows; {int((flag & test).sum())} in the test worlds)')
out['flag_share'] = float(flag.mean())

# ================================================================ 2. is the agreement effect weaker for them?
print('\n' + '=' * 100 + '\n2. DOES AGREEMENT MATTER LESS FOR THEM? (yes rate by agreement and by how generous the person has been)\n' + '=' * 100)
groups = {
    'no history (0 prior answers)': answered == 0,
    'low yes-rate so far (<40%, 2+ answers)': (answered >= 2) & (raw_rate < 0.40),
    'middle (40-59%, 2+ answers)': (answered >= 2) & (raw_rate >= 0.40) & (raw_rate < 0.60),
    'HIGH yes-rate (>=60%, 2+ answers)': flag}
print('Yes rate when the core soft fields mostly agree (net agreement >= +1) vs mostly disagree (<= -1), all worlds:')
print(f'  {"person type":42} {"n":>7} {"yes% overall":>13} {"yes% agree":>11} {"yes% disagree":>14} {"lift (pts)":>11}')
net = n_eq - n_ne
rows = []
for lab, m in groups.items():
    ma, md = m & (net >= 1), m & (net <= -1)
    pa, pd_ = y[ma].mean() * 100, y[md].mean() * 100
    rows.append((lab, int(m.sum()), y[m].mean() * 100, pa, pd_, pa - pd_, int(ma.sum()), int(md.sum())))
    print(f'  {lab:42} {int(m.sum()):7d} {y[m].mean()*100:13.1f} {pa:11.1f} {pd_:14.1f} {pa-pd_:11.1f}   (n agree {int(ma.sum())}, n disagree {int(md.sum())})')
out['agreement_by_type'] = rows

# the right test is on the log-odds scale: probabilities compress near the extremes even if the effect is unchanged
print('\nTest on the log-odds scale (probabilities compress for generous people, so a smaller gap in points is not proof of noise):')
base_cols = np.column_stack([net, flag.astype(float), net * flag, c('a_p_yes'), c('day') / 60.0])
boots = []
tw = np.unique(world)
rows_by_world = {w: np.where(world == w)[0] for w in tw}
for _ in range(100):
    ii = np.concatenate([rows_by_world[w] for w in rng.choice(tw, len(tw))])
    m = LogisticRegression(C=1e6, max_iter=1000).fit(base_cols[ii], y[ii])
    boots.append(m.coef_[0][[0, 2]])
boots = np.array(boots)
full = LogisticRegression(C=1e6, max_iter=1000).fit(base_cols, y)
lo, hi = np.percentile(boots[:, 1], [2.5, 97.5])
print(f'   effect of one extra net agreement for typical people:      {full.coef_[0][0]:+.3f} log-odds')
print(f'   EXTRA effect for indiscriminate people (interaction):      {full.coef_[0][2]:+.3f}   95% CI over worlds [{lo:+.3f}, {hi:+.3f}]')
print('   (a negative interaction whose interval excludes zero would support "their features are less reliable")')
out['interaction'] = {'main': float(full.coef_[0][0]), 'interaction': float(full.coef_[0][2]), 'ci': [float(lo), float(hi)]}

# ================================================================ 3. build it into the probability
print('\n' + '=' * 100 + '\n3. BUILDING THE IDEA INTO THE YES-PROBABILITY (test AUC on held-out worlds)\n' + '=' * 100)
other_soft = ['emotional_availability', 'space_for_relationship', 'relocate']
agree_names = ['eq_goal', 'ne_goal', 'eq_pace', 'ne_pace', 'eq_life', 'ne_life', 'eq_conv', 'ne_conv']
agree_cols = [eq['relationship_goal'], ne['relationship_goal'], eq['relationship_pace'], ne['relationship_pace'],
              eq['lifestyle'], ne['lifestyle'], eq['conversations'], ne['conversations']]
rest = np.column_stack([sum(c('eq_' + k) for k in other_soft) / 3.0, (4 - n_eq - n_ne) / 4.0, c('a_p_answer'), c('a_p_yes'),
                        np.minimum(n_hist, 6) / 6.0, c('p_recv_rate'), c('p_p_yes'), c('late'), day / 60.0,
                        c('age_gap') / 10.0, c('same_zone')])
A = np.column_stack(agree_cols)
test_worlds = np.unique(world[test])
rows_test = {w: np.where(test & (world == w))[0] for w in test_worlds}
boot = [np.concatenate([rows_test[w] for w in rng.choice(test_worlds, len(test_worlds))]) for _ in range(200)]
groups_tr = seed[train]


def pipe(C):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000))


def tune(M, Cs=(0.0003, 0.003, 0.03, 0.3)):
    best, best_ll = None, 9
    for C in Cs:
        lls = []
        for tr, va in GroupKFold(5).split(M[train], y[train], groups_tr):
            lls.append(log_loss(y[train][va], pipe(C).fit(M[train][tr], y[train][tr]).predict_proba(M[train][va])[:, 1]))
        if np.mean(lls) < best_ll:
            best, best_ll = C, np.mean(lls)
    return best, best_ll


def run(label, M, M_fit=None):
    """Tune on clean validation folds; M_fit (optional) is a noisy version of M used only to fit."""
    C, ll = tune(M if M_fit is None else M)
    mdl = pipe(C).fit((M if M_fit is None else M_fit)[train], y[train])
    p = mdl.predict_proba(M)[:, 1]
    tr_a, te_a = roc_auc_score(y[train], p[train]), roc_auc_score(y[test], p[test])
    lo_, hi_ = np.percentile([roc_auc_score(y[b], p[b]) for b in boot], [2.5, 97.5])
    print(f'  {label:60} train {tr_a:.4f} | TEST {te_a:.4f} [{lo_:.4f}-{hi_:.4f}] | logloss {log_loss(y[test], p[test]):.4f}')
    return p, te_a, tr_a


M0 = np.hstack([A, rest])
print('Baseline and variants (test never used for tuning):')
p0, a0, _ = run('V0  baseline: the 19 policy features', M0)
res = {'V0': a0}
paired = {}


def delta(label, p):
    dl = [roc_auc_score(y[b], p[b]) - roc_auc_score(y[b], p0[b]) for b in boot]
    lo_, hi_ = np.percentile(dl, [2.5, 97.5])
    paired[label] = (float(np.mean(dl)), float(lo_), float(hi_))
    print(f'      paired change vs baseline: {np.mean(dl):+.4f}  [{lo_:+.4f}, {hi_:+.4f}]')


f = flag.astype(float)[:, None]
M1 = np.hstack([M0, f, A * f])
p1, a1, _ = run('V1  + indiscriminate flag and its interaction with every agreement feature', M1)
delta('V1', p1)

best = None
for lam in (0.25, 0.5, 0.75, 1.0):
    Mr = np.hstack([A * (1 - lam * f), rest, f])
    C, ll = tune(Mr)
    if best is None or ll < best[0]:
        best = (ll, lam, Mr)
print(f'  (reliability weight r = 1 - lambda for flagged people; lambda picked by grouped CV = {best[1]})')
p2, a2, _ = run(f'V2  agreement features scaled by reliability (lambda={best[1]}) + flag', best[2])
delta('V2', p2)

best3 = None
for q in (0.2, 0.4, 0.6):
    noisy = A.copy()
    pick = flag & train
    for j, (e_col, n_col) in enumerate(zip(range(0, 8, 2), range(1, 8, 2))):
        known = (A[:, e_col] + A[:, n_col]) > 0
        change = pick & known & (rng.random(len(A)) < q)
        new_eq = rng.random(len(A)) < 0.5
        noisy[change, e_col] = new_eq[change].astype(float)
        noisy[change, n_col] = (~new_eq[change]).astype(float)
    Mn = np.hstack([noisy, rest, f])
    Mclean = np.hstack([A, rest, f])
    C, ll = tune(Mclean)
    mdl = pipe(C).fit(Mn[train], y[train])
    p = mdl.predict_proba(Mclean)[:, 1]
    te = roc_auc_score(y[test], p[test])
    if best3 is None or te > best3[0] and False:
        pass
    # choose q by clean grouped CV on train rather than by test
    lls = []
    for tr, va in GroupKFold(5).split(Mn[train], y[train], groups_tr):
        lls.append(log_loss(y[train][va], pipe(C).fit(Mn[train][tr], y[train][tr]).predict_proba(Mclean[train][va])[:, 1]))
    cvll = float(np.mean(lls))
    if best3 is None or cvll < best3[0]:
        best3 = (cvll, q, p)
q_sel, p3 = best3[1], best3[2]
tr_a, te_a = roc_auc_score(y[train], p3[train]), roc_auc_score(y[test], p3[test])
lo_, hi_ = np.percentile([roc_auc_score(y[b], p3[b]) for b in boot], [2.5, 97.5])
print(f'  V3  noise injection: flagged train rows get scrambled agreement (q={q_sel} chosen by clean CV)'
      f'   train {tr_a:.4f} | TEST {te_a:.4f} [{lo_:.4f}-{hi_:.4f}]')
delta('V3', p3)
res.update({'V1': a1, 'V2': a2, 'V3': te_a})
out['models'] = res
out['paired_vs_baseline'] = paired

# ================================================================ 4. do features predict worse for them?
print('\n' + '=' * 100 + '\n4. DOES THE BASELINE MODEL PREDICT WORSE FOR INDISCRIMINATE PEOPLE? (test worlds)\n' + '=' * 100)
for lab, m in groups.items():
    mm = m & test
    if mm.sum() > 100 and len(set(y[mm])) == 2:
        print(f'  {lab:42} n={int(mm.sum()):6d}   AUC {roc_auc_score(y[mm], p0[mm]):.4f}   yes rate {y[mm].mean()*100:.1f}%')
        out.setdefault('auc_by_type', {})[lab] = float(roc_auc_score(y[mm], p0[mm]))
json.dump(out, open('results/noise_analysis.json', 'w'), indent=1, default=float)
print('\nsaved results/noise_analysis.json')
