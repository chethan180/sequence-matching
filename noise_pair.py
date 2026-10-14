"""Noise idea at PAIR level: if someone has said yes a lot over several introductions, are their features less informative?

Target: both say yes. Unique introductions only (replays by other policies removed). Split by world: train seeds <= 1030, test >= 1031.
A person is 'generous' if they had >= K prior answers and a yes-rate so far >= T. A pair is flagged if EITHER side is generous.
We (1) test whether the effect of agreement on the log-odds shrinks for flagged pairs, with a world bootstrap, for several (K, T);
(2) test whether building it into the model (interaction / reliability weight / training noise) improves held-out AUC.
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
df = pd.read_csv('results/pair_rows.csv.gz').drop_duplicates(subset=['seed', 'variant', 'a_id', 'b_id', 'day']).reset_index(drop=True)
train = (df.seed <= 1030).values
test = ~train
y = df.mutual.values
world = (df.seed * 10 + df.variant.map({v: i for i, v in enumerate(['development', 'sparse', 'cold_start', 'delayed', 'shift', 'drift'])})).values
print(f'unique pairs {len(df)} | train {int(train.sum())} | test {int(test.sum())} | both-yes rate {y.mean():.3f}')

SOFT = ['relationship_goal', 'relationship_pace', 'lifestyle', 'conversations', 'emotional_availability',
        'space_for_relationship', 'relocate']
eq, ne = {}, {}
for k in SOFT:
    a, b = df['a_' + k].astype(str), df['b_' + k].astype(str)
    known = (a != 'unknown') & (b != 'unknown')
    eq[k] = (known & (a == b)).values.astype(float)
    ne[k] = (known & (a != b)).values.astype(float)
core = SOFT[:4]
net = sum(eq[k] - ne[k] for k in core)


def rate(side, k):
    n, yes = df[f'{side}_hist_answered'].values, df[f'{side}_hist_yes'].values
    return np.where(n >= k, yes / np.maximum(n, 1), np.nan)


def flag_for(k, t):
    ra, rb = rate('a', k), rate('b', k)
    return ((ra >= t) | (rb >= t))


out = {}
tw = np.unique(world)
rows_by_world = {w: np.where(world == w)[0] for w in tw}
print('\n' + '=' * 100 + '\n1. DOES THE EFFECT OF AGREEMENT SHRINK FOR PAIRS WITH A GENEROUS PERSON? (log-odds, all worlds, bootstrap over worlds)\n' + '=' * 100)
print(f'  {"definition (generous = ...)":38} {"flagged pairs":>14} {"effect typical":>15} {"interaction":>12} {"95% CI over worlds":>22}')
sens = []
for k, t in ((2, 0.5), (2, 0.6), (2, 0.7), (3, 0.5), (3, 0.6), (3, 0.7)):
    fl = flag_for(k, t).astype(float)
    cols = np.column_stack([net, fl, net * fl])
    full = LogisticRegression(C=1e6, max_iter=1000).fit(cols, y)
    boots = []
    for _ in range(100):
        ii = np.concatenate([rows_by_world[w] for w in rng.choice(tw, len(tw))])
        boots.append(LogisticRegression(C=1e6, max_iter=1000).fit(cols[ii], y[ii]).coef_[0][2])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    sens.append((k, t, int(fl.sum()), float(full.coef_[0][0]), float(full.coef_[0][2]), float(lo), float(hi)))
    print(f'  >= {k} answers and yes-rate >= {int(t*100)}%{"":14} {int(fl.sum()):14d} {full.coef_[0][0]:+15.3f} {full.coef_[0][2]:+12.3f} {"[" + format(lo, "+.3f") + ", " + format(hi, "+.3f") + "]":>22}')
out['sensitivity'] = sens

print('\nBoth-yes rate by agreement for pairs with / without a generous person (>= 2 answers, >= 60%):')
fl = flag_for(2, 0.6)
for lab, m in (('no generous person', ~fl), ('a generous person is involved', fl)):
    ma, md = m & (net >= 1), m & (net <= -1)
    print(f'  {lab:32} n={int(m.sum()):6d}  overall {y[m].mean()*100:5.1f}%   agree {y[ma].mean()*100:5.1f}% (n={int(ma.sum())})   disagree {y[md].mean()*100:5.1f}% (n={int(md.sum())})')

# ================================================================ 2. build it into the model
print('\n' + '=' * 100 + '\n2. BUILDING IT INTO THE PAIR MODEL (test AUC, held-out worlds)\n' + '=' * 100)
pa = (df.a_hist_yes + 3 * 0.47) / (df.a_hist_answered + 3)
pb = (df.b_hist_yes + 3 * 0.47) / (df.b_hist_answered + 3)
A = np.column_stack([eq[k] for k in SOFT] + [ne[k] for k in SOFT])
rest = np.column_stack([np.minimum(pa, pb), np.maximum(pa, pb), np.minimum(df.a_hist_n, 6) / 6 + np.minimum(df.b_hist_n, 6) / 6,
                        df.sched_overlap, df.day / 60.0])
Agr = np.column_stack([eq[k] for k in core] + [ne[k] for k in core])
base_M = np.hstack([A, rest])
test_worlds = np.unique(world[test])
rt = {w: np.where(test & (world == w))[0] for w in test_worlds}
boot = [np.concatenate([rt[w] for w in rng.choice(test_worlds, len(test_worlds))]) for _ in range(200)]
gtr = df.seed.values[train]


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


def evaluate(label, M, M_fit=None, p_ref=None):
    C = tune(M)
    mdl = pipe(C).fit((M if M_fit is None else M_fit)[train], y[train])
    p = mdl.predict_proba(M)[:, 1]
    tr_a, te_a = roc_auc_score(y[train], p[train]), roc_auc_score(y[test], p[test])
    msg = f'  {label:70} train {tr_a:.4f} | TEST {te_a:.4f}'
    if p_ref is not None:
        dl = [roc_auc_score(y[b], p[b]) - roc_auc_score(y[b], p_ref[b]) for b in boot]
        msg += f' | change {np.mean(dl):+.4f} [{np.percentile(dl,2.5):+.4f}, {np.percentile(dl,97.5):+.4f}]'
        out.setdefault('paired', {})[label] = (float(np.mean(dl)), float(np.percentile(dl, 2.5)), float(np.percentile(dl, 97.5)))
    print(msg)
    return p, te_a


p0, a0 = evaluate('V0 baseline (agreement on 7 fields, both people\'s history, overlap, day)', base_M)
f = flag_for(2, 0.6).astype(float)[:, None]
evaluate('V1 + generous flag and its interaction with every agreement feature', np.hstack([base_M, f, Agr * f]), p_ref=p0)
for lam in (0.25, 0.5):
    Ar = Agr * (1 - lam * f)
    M = np.hstack([A[:, [i for i in range(14) if i not in (0, 1, 2, 3, 7, 8, 9, 10)]], Ar, rest, f])
    evaluate(f'V2 agreement features scaled by reliability (lambda={lam})', M, p_ref=p0)
for q in (0.3, 0.6):
    noisy = Agr.copy()
    pick = (f[:, 0] == 1) & train
    for j in range(4):
        known = (Agr[:, j] + Agr[:, 4 + j]) > 0
        change = pick & known & (rng.random(len(df)) < q)
        new_eq = rng.random(len(df)) < 0.5
        noisy[change, j] = new_eq[change]
        noisy[change, 4 + j] = ~new_eq[change]
    clean = np.hstack([A[:, [i for i in range(14) if i not in (0, 1, 2, 3, 7, 8, 9, 10)]], Agr, rest, f])
    noisy_M = np.hstack([A[:, [i for i in range(14) if i not in (0, 1, 2, 3, 7, 8, 9, 10)]], noisy, rest, f])
    evaluate(f'V3 noise injection: scramble agreement of generous pairs in training (q={q})', clean, M_fit=noisy_M, p_ref=p0)
json.dump(out, open('results/noise_pair.json', 'w'), indent=1, default=float)
print('\nsaved results/noise_pair.json')
