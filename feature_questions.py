"""Test the candidate-feature questions (#1-#15) on held-out worlds.

Input : results/ext3_rows.npz (181 features per person-introduction row, incl. taste / drift / pool-shape)
        results/pair_rows.csv.gz (for #15, the date stage)
For every idea: yes-rate table on the held-out worlds and the change in test AUC when the idea's features are added to the
'core' model (similarity agreement + own history, logistic), with a bootstrap over the 60 test worlds.
Split by world: train seeds <= 1030 (grouped CV by seed for tuning), test seeds >= 1031.
"""
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')
rng = np.random.default_rng(0)
d = np.load('results/ext3_rows.npz')
X = d['X'].astype(np.float64)
names = list(d['names'])
groups = list(d['groups'])
idx = {n: i for i, n in enumerate(names)}
seed, variant, day = d['seed'], d['variant'], d['day']
y = d['y_yes'].astype(int)
world = seed * 10 + variant
train, test = seed <= 1030, seed >= 1031
VARIANTS = ['development', 'sparse', 'cold_start', 'delayed', 'shift', 'drift']


def c(n):
    return X[:, idx[n]]


core_idx = [i for i, g in enumerate(groups) if g in ('soft_eq', 'hist_actor')]
CORE = X[:, core_idx]
print(f'rows {len(X)} | train {int(train.sum())} | test {int(test.sum())} | yes rate {y.mean():.3f} | core columns {CORE.shape[1]}')

tw = np.unique(world[test])
rt = {w: np.where(test & (world == w))[0] for w in tw}
boot = [np.concatenate([rt[w] for w in rng.choice(tw, len(tw))]) for _ in range(200)]
gtr = seed[train]


def pipe(C):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000))


def tune(M):
    best, ll_best = None, 9
    for C in (0.0003, 0.003, 0.03, 0.3):
        ll = np.mean([log_loss(y[train][va], pipe(C).fit(M[train][tr], y[train][tr]).predict_proba(M[train][va])[:, 1])
                      for tr, va in GroupKFold(5).split(M[train], y[train], gtr)])
        if ll < ll_best:
            best, ll_best = C, ll
    return best


def fit_p(M):
    return pipe(tune(M)).fit(M[train], y[train]).predict_proba(M)[:, 1]


p_core = fit_p(CORE)
print(f'core model test AUC {roc_auc_score(y[test], p_core[test]):.4f}')
results = {}
n_tests = 0


def table(title, labels, masks):
    print(f'\n{title}')
    print(f'  {"level":34} {"n_test":>7} {"yes% test":>10} {"yes% train":>11}')
    for lab, m in zip(labels, masks):
        if (m & test).sum() == 0:
            continue
        print(f'  {lab:34} {int((m & test).sum()):7d} {y[m & test].mean()*100:10.1f} {y[m & train].mean()*100 if (m & train).sum() else float("nan"):11.1f}')


def add_test(key, label, extra, subset=None):
    """Change in test AUC when extra columns are added to the core model (optionally measured on a subset of test rows)."""
    global n_tests
    n_tests += 1
    E = extra if extra.ndim == 2 else extra[:, None]
    p = fit_p(np.hstack([CORE, E]))
    m = test if subset is None else (test & subset)
    if m.sum() < 200 or len(set(y[m])) < 2:
        print(f'  [{key}] {label}: too few test rows')
        return
    bm = [b[subset[b]] if subset is not None else b for b in boot]
    dl = [roc_auc_score(y[b], p[b]) - roc_auc_score(y[b], p_core[b]) for b in bm if len(b) > 100 and len(set(y[b])) == 2]
    lo, hi = np.percentile(dl, [2.5, 97.5])
    tag = 'SUPPORTED' if lo > 0 else ('HURTS' if hi < 0 else 'no effect')
    print(f'  [{key}] {label:62} AUC {roc_auc_score(y[m], p[m]):.4f} vs core {roc_auc_score(y[m], p_core[m]):.4f}  change {np.mean(dl):+.4f} [{lo:+.4f}, {hi:+.4f}]  {tag}')
    results[key] = {'label': label, 'delta': float(np.mean(dl)), 'ci': [float(lo), float(hi)], 'verdict': tag}


print('\n' + '=' * 110 + '\nA. IDEAS TESTABLE ON EXISTING FIELDS\n' + '=' * 110)

# ---- #1 zone flexibility, #2 zone thickness
table('#1 how many zones the PERSON accepts', ['1 zone', '2 zones'], [c('a_zones') == 1, c('a_zones') >= 2])
table('#1 how many zones the PARTNER accepts', ['1 zone', '2 zones'], [c('p_zones') == 1, c('p_zones') >= 2])
table('#2 zone thickness proxy: is the person in the biggest zone (zone a, about 59% of members)?', ['zone a', 'other zone'],
      [c('a_zone_a') == 1, c('a_zone_a') == 0])
table('#2 number of distinct zones in the pool (many zones = thin zones)', ['<=4 zones', '5-8 zones', '9+ zones'],
      [c('pool_zones') <= 4, (c('pool_zones') > 4) & (c('pool_zones') <= 8), c('pool_zones') >= 9])
add_test('1', 'zone flexibility (zones accepted by person and partner)', np.column_stack([c('a_zones'), c('p_zones')]))
add_test('2', 'zone thickness (zone a flag, distinct zones in pool)', np.column_stack([c('a_zone_a'), c('p_zone_a'), c('pool_zones')]))

# ---- #3 openness
op_cols = ['who_size', 'age_width', 'slots', 'zones']


def openness(side):
    z = []
    for n in op_cols:
        v = c(f'{side}_{n}')
        z.append((v - v[train].mean()) / v[train].std())
    z.append(-(c(f'{side}_strict_smoke')) - c(f'{side}_strict_kids'))
    return np.sum(z, axis=0)


oa, op_ = openness('a'), openness('p')
qa = np.quantile(oa[train], [1 / 3, 2 / 3])
table("#3 openness of the PERSON (breadth of who/age/schedule/zones, strictness) in terciles", ['least open', 'middle', 'most open'],
      [oa <= qa[0], (oa > qa[0]) & (oa <= qa[1]), oa > qa[1]])
qp = np.quantile(op_[train], [1 / 3, 2 / 3])
table("#3 openness of the PARTNER in terciles", ['least open', 'middle', 'most open'],
      [op_ <= qp[0], (op_ > qp[0]) & (op_ <= qp[1]), op_ > qp[1]])
add_test('3', 'openness score of person and partner', np.column_stack([oa, op_, c('a_who_size'), c('a_age_width'), c('a_slots')]))

# ---- #4 near-miss count, #5 one-sided fit
pos_p, pos_a = c('p_pos_in_a_range'), c('a_pos_in_p_range')
near = ((pos_p < 0.15) | (pos_p > 0.85)).astype(int) + ((pos_a < 0.15) | (pos_a > 0.85)).astype(int) + (c('sched_overlap') == 1).astype(int) \
    + ((c('a_wants_unsure') + c('p_wants_unsure')) == 1).astype(int)
table('#4 near-miss count (age at the edge of either range, only one shared schedule slot, one side unsure about children)',
      ['0', '1', '2', '3+'], [near == 0, near == 1, near == 2, near >= 3])
asym = np.abs(pos_p - pos_a)
table('#5 one-sided age fit: |partner position in my range - my position in theirs|', ['<0.15', '0.15-0.35', '>0.35'],
      [asym < 0.15, (asym >= 0.15) & (asym <= 0.35), asym > 0.35])
add_test('4', 'near-miss count and its parts', np.column_stack([near, pos_p, pos_a]))
add_test('5', 'one-sided fit (asymmetry of age position)', np.column_stack([asym, pos_p, pos_a]))

# ---- #10 silence vs no
a_n, a_ans, a_yes = c('a_n'), c('a_answered'), c('a_yes')
silent, noes = a_n - a_ans, a_ans - a_yes
table('#10 silence versus "no": what a past silence and a past no each imply',
      ['no history', 'one past answer: yes', 'one past answer: no', 'one past silence only', 'two past silences', 'past no + past silence'],
      [a_n == 0, (a_n == 1) & (a_yes == 1), (a_n == 1) & (noes == 1), (a_n == 1) & (silent == 1),
       (a_n == 2) & (silent == 2), (noes >= 1) & (silent >= 1)])
add_test('10', 'separate silent / no counts (silence as its own signal)', np.column_stack([silent, noes, silent / np.maximum(a_n, 1)]))

# ---- #11 wait time
since = c('a_since_arrival')
table('#11 days since the person arrived', ['0-4', '5-14', '15-29', '30+'],
      [since <= 4, (since > 4) & (since <= 14), (since > 14) & (since <= 29), since >= 30])
last = c('a_days_since_last')
table('#11 days since the person last answered (30 = never)', ['0-5', '6-12', '13-29', 'never'],
      [last <= 5, (last > 5) & (last <= 12), (last > 12) & (last < 30), last >= 30])
add_test('11', 'wait time (days since arrival, days since last answer)', np.column_stack([since, last, c('a_n_intros')]))

# ---- #13 pool shape, #14 weekday
table('#13 yes rate by scenario variant (what pool shape could proxy)', VARIANTS, [variant == i for i in range(6)])
add_test('13', 'pool shape (share complete, distinct zones, pool size)', np.column_stack([c('pool_complete'), c('pool_zones'), c('pool_size')]))
table('#14 day of the week (day mod 7)', [f'day%7={i}' for i in range(7)], [(day % 7) == i for i in range(7)])
add_test('14', 'weekday (day mod 7, one-hot)', np.column_stack([(day % 7) == i for i in range(7)]).astype(float))

print('\n' + '=' * 110 + '\nB. IDEAS THAT NEEDED NEW INFORMATION (now collected)\n' + '=' * 110)
has2 = a_ans >= 2
table('#8 personalised taste: the person\'s past yes-rate with partners sharing the candidate\'s traits, minus their overall rate (people with 2+ past answers)',
      ['strongly negative (<-0.05)', 'near zero', 'strongly positive (>+0.05)'],
      [has2 & (c('taste_sum') < -0.05), has2 & (np.abs(c('taste_sum')) <= 0.05), has2 & (c('taste_sum') > 0.05)])
cols8 = np.column_stack([c('taste_goal'), c('taste_pace'), c('taste_life'), c('taste_conv'), c('taste_sum'), c('taste_n')])
add_test('8', 'personalised taste (all rows)', cols8)
add_test('8b', 'personalised taste, measured only on people with 2+ past answers', cols8, subset=has2)
add_test('9', 'stated vs revealed: age gap of past yes-partners vs candidate', np.column_stack([c('gap_vs_yes'), c('mean_gap_yes')]))
cols12 = np.column_stack([c('pool_recent_yes'), c('pool_all_yes'), c('drift_gap'), c('pool_rec_n')])
table('#12 yes rate over time in the DRIFT scenario (response rates fall after day 35) vs the other scenarios',
      ['drift: days 0-34', 'drift: days 35-59', 'others: days 0-34', 'others: days 35-59'],
      [(variant == 5) & (day < 35), (variant == 5) & (day >= 35), (variant != 5) & (day < 35), (variant != 5) & (day >= 35)])
add_test('12', 'drift detector (recent pool yes-rate vs overall), all scenarios', cols12)
add_test('12b', 'drift detector, drift scenario after day 35 only', cols12, subset=(variant == 5) & (day >= 35))

# ================================================================ #15 date stage (pair file)
print('\n' + '=' * 110 + '\n#15 DATE AND SECOND-MEETING STAGES (pair file; unique introductions; test worlds)\n' + '=' * 110)
pf = pd.read_csv('results/pair_rows.csv.gz').drop_duplicates(subset=['seed', 'variant', 'a_id', 'b_id', 'day'])
pt = pf[pf.seed >= 1031]
mut = pt[pt.mutual == 1]
print(f'mutual-yes pairs in test worlds: {len(mut)}; of these a date happened in {mut.date.mean()*100:.1f}%; MSMI in {mut.msmi.mean()*100:.1f}%')
for col, lab in (('sched_overlap', 'shared schedule slots'), ):
    g = mut.groupby(col).agg(n=('date', 'size'), date_rate=('date', 'mean'))
    print(f'  P(date | mutual yes) by {lab}: ' + ', '.join(f'{k}: {v.date_rate*100:.1f}% (n={int(v.n)})' for k, v in g.iterrows()))
same_zone = (mut.a_zone == mut.b_zone)
print(f'  P(date | mutual yes): same zone {mut[same_zone].date.mean()*100:.1f}% (n={int(same_zone.sum())}), different zone {mut[~same_zone].date.mean()*100:.1f}% (n={int((~same_zone).sum())})')
dated = mut[mut.date == 1]
same_goal = (dated.a_relationship_goal == dated.b_relationship_goal) & (dated.a_relationship_goal != 'unknown')
diff_goal = (dated.a_relationship_goal != dated.b_relationship_goal) & (dated.a_relationship_goal != 'unknown') & (dated.b_relationship_goal != 'unknown')
print(f'  P(MSMI | date) overall {dated.msmi.mean()*100:.1f}% (n={len(dated)}); goal same {dated[same_goal].msmi.mean()*100:.1f}% (n={int(same_goal.sum())}), goal different {dated[diff_goal].msmi.mean()*100:.1f}% (n={int(diff_goal.sum())})')
results['date_stage'] = {'p_date_given_mutual': float(mut.date.mean()), 'p_msmi_given_date': float(dated.msmi.mean())}

print(f'\n{n_tests} model tests run; at a 95% interval about {n_tests * 0.05:.1f} of them would look "supported" by chance alone.')
json.dump(results, open('results/feature_questions.json', 'w'), indent=1, default=float)
print('saved results/feature_questions.json')
