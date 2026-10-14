"""Pair-level probability analysis: P(BOTH say yes) by single features, by merged feature combinations, and by lookup models.

Input : results/pair_rows.csv.gz  (collect_pairs.py: one row per introduced pair, raw attributes of both people)
Output: printed tables, results/pair_cells.csv (every cell tested), results/pair_analysis.json

Honest protocol
  * identical introductions made by different behaviour policies in the same world are de-duplicated
    (the simulator's outcome depends only on world, pair and day), so rows are unique introductions
  * split by WORLD: train seeds 1001-1030, test seeds 1031-1040
  * cells are DISCOVERED on train (z-test, Benjamini-Hochberg FDR 5%) and CONFIRMED on test
"""
import itertools
import json
import warnings
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')
df = pd.read_csv('results/pair_rows.csv.gz')
n_raw = len(df)
df = df.drop_duplicates(subset=['seed', 'variant', 'a_id', 'b_id', 'day']).reset_index(drop=True)
train = (df['seed'] <= 1030).values
test = ~train
y = df['mutual'].values
base_tr, base_te = y[train].mean(), y[test].mean()
print(f'pair rows {n_raw} -> {len(df)} unique introductions after removing replays of the same pair/day by another policy')
print(f'train pairs {int(train.sum())} | test pairs {int(test.sum())} | P(both yes): train {base_tr:.3%}, test {base_te:.3%}')
out = {'unique_pairs': int(len(df)), 'base_train': float(base_tr), 'base_test': float(base_te)}


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return mid - half, mid + half


# ---------------------------------------------------------------- unordered pair categories
SOFT = ['relationship_goal', 'relationship_pace', 'lifestyle', 'conversations', 'emotional_availability',
        'space_for_relationship', 'relocate']
cats = {}


def pair_cat(col):
    a, b = df['a_' + col].astype(str), df['b_' + col].astype(str)
    return pd.Series(np.where(a <= b, a + '|' + b, b + '|' + a))


def same_cat(col):
    a, b = df['a_' + col].astype(str), df['b_' + col].astype(str)
    return pd.Series(np.where((a == 'unknown') | (b == 'unknown'), 'unknown', np.where(a == b, 'same', 'diff')))


cats['zone_pair'] = pair_cat('zone')
cats['same_zone'] = same_cat('zone')
cats['gender_pair'] = pair_cat('gender')
cats['same_gender'] = same_cat('gender')
for col in ('children', 'smoking', 'wants', 'structure'):
    cats[col + '_pair'] = pair_cat(col)
    cats['same_' + col] = same_cat(col)
for col in SOFT:
    cats[col + '_pair'] = pair_cat(col)
    cats['same_' + col] = same_cat(col)
band = lambda s: pd.cut(s, [0, 25, 30, 35, 40, 100], labels=['21-25', '26-30', '31-35', '36-40', '41+']).astype(str)
ba, bb = band(df['a_age']), band(df['b_age'])
cats['age_band_pair'] = pd.Series(np.where(ba <= bb, ba + '|' + bb, bb + '|' + ba))
cats['same_age_band'] = pd.Series(np.where(ba == bb, 'same', 'diff'))
gap = (df['a_age'] - df['b_age']).abs()
cats['age_gap'] = pd.Series(pd.cut(gap, [-1, 2, 5, 9, 100], labels=['0-2', '3-5', '6-9', '10+']).astype(str))
cats['sched_overlap'] = df['sched_overlap'].astype(str)
cats['day_band'] = pd.Series(pd.cut(df['day'], [-1, 19, 39, 59], labels=['d0-19', 'd20-39', 'd40-59']).astype(str))


def hist_band(p):
    n, yes = df[p + 'hist_answered'], df[p + 'hist_yes']
    r = (yes + 3 * 0.47) / (n + 3)
    return pd.Series(np.where(n == 0, 'none', np.where(r < 0.40, 'low', np.where(r <= 0.52, 'mid', 'high'))))


ha, hb = hist_band('a_'), hist_band('b_')
cats['history_pair'] = pd.Series(np.where(ha <= hb, ha + '|' + hb, hb + '|' + ha))
cats['both_new'] = pd.Series(np.where((df['a_hist_n'] == 0) & (df['b_hist_n'] == 0), 'both_new',
                                      np.where((df['a_hist_n'] == 0) | (df['b_hist_n'] == 0), 'one_new', 'both_known')))
cats['arrival_pair'] = pd.Series(np.where((df['a_arrived'] == 0) & (df['b_arrived'] == 0), 'both_day0',
                                          np.where((df['a_arrived'] == 0) | (df['b_arrived'] == 0), 'one_late', 'both_late')))
cats = {k: v.values for k, v in cats.items()}
print(f'{len(cats)} pair-level attributes built')


def cell_table(label, keys, min_n_train=0):
    g = pd.DataFrame({'cell': keys, 'y': y, 'train': train})
    t = g[g.train].groupby('cell')['y'].agg(['size', 'sum'])
    s = g[~g.train].groupby('cell')['y'].agg(['size', 'sum'])
    tab = t.join(s, how='outer', lsuffix='_tr', rsuffix='_te').fillna(0)
    tab.columns = ['n_tr', 'k_tr', 'n_te', 'k_te']
    tab['rate_tr'] = tab.k_tr / tab.n_tr.replace(0, np.nan)
    tab['rate_te'] = tab.k_te / tab.n_te.replace(0, np.nan)
    ci = [wilson(k, n) for k, n in zip(tab.k_te, tab.n_te)]
    tab['lo_te'] = [c[0] for c in ci]
    tab['hi_te'] = [c[1] for c in ci]
    tab['lift_te'] = tab.rate_te / base_te
    return tab[tab.n_tr >= min_n_train]


def show(tab, title, rows=None):
    print(f'\n{title}')
    print(f'  {"cell":38} {"n_train":>7} {"both-yes% train":>15} {"n_test":>7} {"both-yes% test":>15} {"95% CI test":>13} {"lift":>5}')
    for cell, r in (tab if rows is None else tab.head(rows)).iterrows():
        if r.n_te == 0:
            continue
        print(f'  {cell:38} {int(r.n_tr):7d} {r.rate_tr*100:15.1f} {int(r.n_te):7d} {r.rate_te*100:15.1f} '
              f'{r.lo_te*100:6.1f}-{r.hi_te*100:5.1f} {r.lift_te:5.2f}')


print('\n' + '=' * 110 + '\n1. P(BOTH SAY YES) BY PAIR CATEGORY (unordered), single attributes\n' + '=' * 110)
for name in ('zone_pair', 'same_zone', 'gender_pair', 'same_gender', 'same_relationship_goal', 'relationship_goal_pair',
             'same_relationship_pace', 'relationship_pace_pair', 'same_lifestyle', 'same_conversations',
             'same_emotional_availability', 'same_space_for_relationship', 'same_relocate',
             'same_children', 'children_pair', 'same_smoking', 'smoking_pair', 'same_wants', 'wants_pair',
             'age_band_pair', 'age_gap', 'sched_overlap', 'history_pair', 'both_new', 'arrival_pair', 'day_band'):
    tab = cell_table(name, cats[name], min_n_train=1)
    tab = tab.sort_values('n_tr', ascending=False)
    show(tab, f'{name}', rows=14)

# ---------------------------------------------------------------- composition among mutual-yes pairs
print('\n' + '=' * 110 + '\n2. WHO ENDS UP IN A MUTUAL YES? (share of mutual-yes pairs vs share of all introduced pairs, test worlds)\n' + '=' * 110)
print(f'  {"attribute = value":42} {"% of all pairs":>15} {"% of mutual-yes pairs":>22}')
for name, val in (('same_zone', 'same'), ('same_zone', 'diff'), ('same_relationship_goal', 'same'), ('same_relationship_goal', 'diff'),
                  ('same_relationship_pace', 'same'), ('same_relationship_pace', 'diff'), ('same_gender', 'same'),
                  ('same_gender', 'diff'), ('same_lifestyle', 'same'), ('same_conversations', 'same'),
                  ('both_new', 'both_new'), ('both_new', 'both_known'), ('arrival_pair', 'both_day0')):
    m = cats[name] == val
    print(f'  {name + " = " + val:42} {m[test].mean()*100:15.1f} {m[test & (y == 1)].mean()*100:22.1f}')

# ================================================================ 3. merged features
print('\n' + '=' * 110 + '\n3. MERGED FEATURES: 2-way and 3-way combinations (discover on TRAIN with FDR, confirm on TEST)\n' + '=' * 110)
merge_attrs = ['zone_pair', 'same_zone', 'gender_pair', 'same_gender', 'same_relationship_goal', 'same_relationship_pace',
               'same_lifestyle', 'same_conversations', 'same_emotional_availability', 'same_space_for_relationship',
               'same_relocate', 'same_children', 'same_smoking', 'same_wants', 'age_band_pair', 'age_gap', 'sched_overlap',
               'history_pair', 'both_new', 'arrival_pair', 'day_band']
MIN_TR, MIN_TE = 100, 40


def combine(names):
    keys = np.char.add(f'{names[0]}=', cats[names[0]].astype(str))
    for n in names[1:]:
        keys = np.char.add(np.char.add(keys, f' & {n}='), cats[n].astype(str))
    return keys


def discover(combos, label):
    rows = []
    for combo in combos:
        keys = combine(combo)
        tab = cell_table(label, keys)
        tab = tab[(tab.n_tr >= MIN_TR) & (tab.n_te >= MIN_TE)]
        for cell, r in tab.iterrows():
            z = (r.k_tr - r.n_tr * base_tr) / np.sqrt(r.n_tr * base_tr * (1 - base_tr))
            rows.append((cell, r.n_tr, r.rate_tr, r.n_te, r.rate_te, r.lo_te, r.hi_te, r.lift_te, z, 2 * (1 - norm.cdf(abs(z)))))
    res = pd.DataFrame(rows, columns=['cell', 'n_tr', 'rate_tr', 'n_te', 'rate_te', 'lo_te', 'hi_te', 'lift_te', 'z_tr', 'p_tr'])
    if len(res) == 0:
        return res
    order = np.argsort(res.p_tr.values)
    m = len(res)
    q = np.empty(m)
    q[order] = np.minimum.accumulate((res.p_tr.values[order] * m / (np.arange(m) + 1))[::-1])[::-1]
    res['q_tr'] = q
    res['discovered'] = res.q_tr <= 0.05
    same_dir = np.sign(res.rate_tr - base_tr) == np.sign(res.rate_te - base_te)
    ci_excl = (res.lo_te > base_te) | (res.hi_te < base_te)
    res['replicated'] = res.discovered & same_dir & ci_excl
    return res


all_res = []
two = list(itertools.combinations(merge_attrs, 2))
res2 = discover(two, '2-way')
res2['order'] = 2
all_res.append(res2)
print(f'2-way: {len(two)} attribute pairs -> {len(res2)} cells with enough data; discovered on train (FDR 5%): {int(res2.discovered.sum())}; '
      f'confirmed on test: {int(res2.replicated.sum())}')
core3 = ['same_zone', 'same_relationship_goal', 'same_relationship_pace', 'same_lifestyle', 'same_conversations',
         'same_gender', 'history_pair', 'same_emotional_availability']
three = list(itertools.combinations(core3, 3))
res3 = discover(three, '3-way')
res3['order'] = 3
all_res.append(res3)
print(f'3-way: {len(three)} attribute triples -> {len(res3)} cells with enough data; discovered on train: {int(res3.discovered.sum())}; '
      f'confirmed on test: {int(res3.replicated.sum())}')
res_all = pd.concat(all_res)
res_all.to_csv('results/pair_cells.csv', index=False)
disc = res_all[res_all.discovered]
if len(disc):
    shrink = (disc.rate_tr - base_tr).abs().mean(), (disc.rate_te - base_te).abs().mean()
    print(f'winner\'s curse check: discovered cells deviate from the base rate by {shrink[0]*100:.1f} points on train but only {shrink[1]*100:.1f} points on test')
out['merged'] = {'two_way_tested': int(len(res2)), 'two_way_discovered': int(res2.discovered.sum()),
                 'two_way_confirmed': int(res2.replicated.sum()), 'three_way_tested': int(len(res3)),
                 'three_way_discovered': int(res3.discovered.sum()), 'three_way_confirmed': int(res3.replicated.sum())}


def show_cells(res, title, rows=14, ascending=False):
    r = res[res.replicated].sort_values('lift_te', ascending=ascending)
    print(f'\n{title}')
    print(f'  {"merged cell":88} {"n_tr":>5} {"tr%":>5} {"n_te":>5} {"te%":>5} {"lift":>5}')
    for _, x in r.head(rows).iterrows():
        print(f'  {x.cell[:88]:88} {int(x.n_tr):5d} {x.rate_tr*100:5.1f} {int(x.n_te):5d} {x.rate_te*100:5.1f} {x.lift_te:5.2f}')


show_cells(res_all, 'Highest-probability CONFIRMED merged cells (both say yes)', 14)
show_cells(res_all, 'Lowest-probability CONFIRMED merged cells', 12, ascending=True)

# ================================================================ 4. probability models from merged features
print('\n' + '=' * 110 + '\n4. PROBABILITY MODELS FROM MERGED FEATURES (target: both say yes; AUC on held-out worlds)\n' + '=' * 110)
soft4 = ['same_relationship_goal', 'same_relationship_pace', 'same_lifestyle', 'same_conversations']
joint = cats[soft4[0]].astype(str)
for n in soft4[1:]:
    joint = np.char.add(np.char.add(joint, '/'), cats[n].astype(str))
tab = cell_table('joint4', joint)
m_eb = 100
cell_p = {}
for cell, r in tab.iterrows():
    cell_p[cell] = (r.k_tr + m_eb * base_tr) / (r.n_tr + m_eb)
p_eb = np.array([cell_p.get(c, base_tr) for c in joint])
print('Probability table built from the MERGED four soft-field agreements (same/diff/unknown each), smoothed toward the base rate:')
shown = tab[tab.n_tr >= 150].copy()
shown['smoothed'] = [cell_p[c] for c in shown.index]
shown = shown.sort_values('smoothed', ascending=False)
print(f'  {"goal/pace/lifestyle/conversations":38} {"n_tr":>6} {"raw tr%":>8} {"smoothed%":>10} {"n_te":>6} {"test%":>7}')
for cell, r in pd.concat([shown.head(8), shown.tail(6)]).iterrows():
    print(f'  {cell:38} {int(r.n_tr):6d} {r.rate_tr*100:8.1f} {r.smoothed*100:10.1f} {int(r.n_te):6d} {r.rate_te*100:7.1f}')
auc_eb = (roc_auc_score(y[train], p_eb[train]), roc_auc_score(y[test], p_eb[test]))
print(f'  lookup AUC  train {auc_eb[0]:.4f}  test {auc_eb[1]:.4f}   ({len(cell_p)} cells)')
out['lookup_joint4'] = {'train_auc': float(auc_eb[0]), 'test_auc': float(auc_eb[1]), 'cells': len(cell_p)}

# logistic models on one-hot pair attributes
groups_tr = df['seed'].values[train]


def onehot(names, min_count=50):
    cols, labels = [], []
    for n in names:
        s = pd.Series(cats[n])
        for v, cnt in s[train].value_counts().items():
            if cnt >= min_count:
                cols.append((s == v).values.astype(float))
                labels.append(f'{n}={v}')
    return np.column_stack(cols), labels


def fit_eval(M, label):
    best, best_ll = None, 9
    for C in (0.0003, 0.003, 0.03, 0.3):
        lls = []
        for tr, va in GroupKFold(5).split(M[train], y[train], groups_tr):
            mdl = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000)).fit(M[train][tr], y[train][tr])
            lls.append(log_loss(y[train][va], mdl.predict_proba(M[train][va])[:, 1]))
        if np.mean(lls) < best_ll:
            best, best_ll = C, np.mean(lls)
    mdl = make_pipeline(StandardScaler(), LogisticRegression(C=best, max_iter=3000)).fit(M[train], y[train])
    p = mdl.predict_proba(M)[:, 1]
    tr_a, te_a = roc_auc_score(y[train], p[train]), roc_auc_score(y[test], p[test])
    print(f'  {label:58} cols {M.shape[1]:4d}  C={best:<7} train AUC {tr_a:.4f}  test AUC {te_a:.4f}  gap {tr_a-te_a:+.4f}')
    return float(tr_a), float(te_a), p, mdl


M1, l1 = onehot(['same_zone', 'zone_pair', 'same_gender', 'gender_pair', 'same_relationship_goal', 'same_relationship_pace',
                 'same_lifestyle', 'same_conversations', 'same_emotional_availability', 'same_space_for_relationship',
                 'same_relocate', 'same_children', 'same_smoking', 'same_wants', 'age_band_pair', 'age_gap', 'sched_overlap',
                 'history_pair', 'both_new', 'arrival_pair', 'day_band'])
r1 = fit_eval(M1, 'A. logistic on all single pair attributes (one-hot)')
conf = res_all[res_all.replicated].cell.tolist()


def cell_columns(cells):
    cols = []
    for c in cells:
        parts = c.split(' & ')
        m = np.ones(len(df), dtype=bool)
        for part in parts:
            name, val = part.split('=', 1)
            m &= cats[name] == val
        cols.append(m.astype(float))
    return np.column_stack(cols) if cols else np.zeros((len(df), 0))


Mb = np.hstack([M1, cell_columns(conf)])
rb = fit_eval(Mb, f'B. A + the {len(conf)} merged cells confirmed on test (selection leaks!)')
disc_cells = res_all[res_all.discovered].cell.tolist()
Mc = np.hstack([M1, cell_columns(disc_cells)])
rc = fit_eval(Mc, f'C. A + the {len(disc_cells)} merged cells discovered on TRAIN only (honest)')
every = res_all.cell.tolist()
Md = np.hstack([M1, cell_columns(every)])
rd = fit_eval(Md, f'D. A + ALL {len(every)} merged cells, no selection (overfit control)')
out['models'] = {'A_single': r1[:2], 'B_leaky_confirmed': rb[:2], 'C_train_discovered': rc[:2], 'D_all_cells': rd[:2]}

print('\nCalibration of model C on test pairs (predicted vs observed chance that BOTH say yes):')
p = rc[2][test]
order = np.argsort(p)
cal = []
for b in np.array_split(order, 8):
    cal.append((float(p[b].mean() * 100), float(y[test][b].mean() * 100)))
    print(f'   predicted {cal[-1][0]:5.1f}%   observed {cal[-1][1]:5.1f}%')
out['calibration_C'] = cal
print(f'   spread of predictions (1st to 99th percentile): {np.percentile(p,1)*100:.1f}% to {np.percentile(p,99)*100:.1f}%  (base rate {base_te*100:.1f}%)')
json.dump(out, open('results/pair_analysis.json', 'w'), indent=1, default=float)
print('\nsaved results/pair_analysis.json and results/pair_cells.csv')
