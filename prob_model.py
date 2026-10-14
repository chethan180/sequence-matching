"""Probabilistic latent-trait model of "will this person say yes", with the probability analysis that motivates it.

Input : results/worlds.pkl (collect_world.py)      Output: printed tables + results/prob_model.json
Split by WORLD: fit seeds 1001-1020, validation 1021-1030 (tuning), test 1031-1040 (final numbers only).

Model (every latent is inferred ONLY from answers already observed on the decision day):
  answered_i        ~ Bernoulli(rho_i)                                   rho_i  = the person's response rate
  accept_i->j       ~ Bernoulli(sigmoid(c + g_i + sum_k a_k * same_k(i,j) + time))
                                                                        g_i    = the person's generosity, prior N(0, tau^2)
  same_k(i,j)       = 1 if the two people give the same answer on core soft field k
                      when an answer is unknown we carry a probability over its values and use P(same)
  yes               = answered and accept
Inference per pool and day: alternate (1) generosity given expected compatibility of each past partner (Newton MAP with
posterior variance), (2) each unknown soft answer given how that person answered others AND how others answered them.
Parameters (c, a_k, time effects, tau, prior strength of rho) are LEARNED from training worlds, never hard-coded.

Columns named hid_* are simulator internals used only as a diagnostic yardstick (ceilings, calibration of the inferred
traits). They are never model inputs.
"""
import json
import pickle
import sys
import time
import warnings
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')
path = sys.argv[1] if len(sys.argv) > 1 else 'results/worlds.pkl'
tag = sys.argv[2] if len(sys.argv) > 2 else 'main'
rng = np.random.default_rng(0)
CORE = [0, 1, 2, 3]                      # relationship_goal, relationship_pace, lifestyle, conversations
CORE_NAMES = ['relationship goal', 'pace', 'lifestyle', 'conversations']
NV = [2, 3, 3, 4]
V = 4
worlds = pickle.load(open(path, 'rb'))
print(f'{path}: {len(worlds)} episodes, {sum(len(w["intros"]) for w in worlds)} introductions')


def sig(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def logsig(x):
    return -np.logaddexp(0.0, -x)


# ---------------------------------------------------------------- pack each world into arrays
W = []
for w in worlds:
    it = w['intros']
    if not it:
        continue
    ia = np.array([r['a'] for r in it])
    ib = np.array([r['b'] for r in it])
    d = np.array([r['day'] for r in it])
    va = np.array([r['val_a'] for r in it])
    vb = np.array([r['val_b'] for r in it])
    W.append({
        'seed': w['seed'], 'variant': w['variant'], 'n': len(it),
        'soft': w['soft'][:, CORE].astype(int), 'sday': w['soft_day'][:, CORE].astype(int),
        'truth': w['hid_truth'][:, CORE].astype(int), 'hid_person': w['hid_person'],
        'ev_i': np.concatenate([ia, ib]), 'ev_j': np.concatenate([ib, ia]), 'ev_val': np.concatenate([va, vb]),
        'ev_obs': np.concatenate([[r['obs_a'] for r in it], [r['obs_b'] for r in it]]).astype(int),
        'ev_day': np.concatenate([d, d]),
        'x19': np.array([r['x19_a'] for r in it] + [r['x19_b'] for r in it], dtype=float),
        'hid_p': np.array([r['hid_p_a'] for r in it] + [r['hid_p_b'] for r in it], dtype=float)})
seed = np.concatenate([np.full(2 * w['n'], w['seed']) for w in W])
variant = np.concatenate([np.full(2 * w['n'], w['variant']) for w in W])
wid = np.concatenate([np.full(2 * w['n'], i) for i, w in enumerate(W)])
y = np.concatenate([(w['ev_val'] == 1) for w in W]).astype(int)
ans = np.concatenate([(w['ev_val'] != 3) for w in W]).astype(int)
day = np.concatenate([w['ev_day'] for w in W]).astype(float)
x19 = np.vstack([w['x19'] for w in W])
hid_p = np.concatenate([w['hid_p'] for w in W])
fit_m, val_m = seed <= 1020, (seed > 1020) & (seed <= 1030)
train, test = seed <= 1030, seed >= 1031
world_key = seed * 10 + variant
tw = np.unique(world_key[test])
rt = {k: np.where(test & (world_key == k))[0] for k in tw}
boot = [np.concatenate([rt[k] for k in rng.choice(tw, len(tw))]) for _ in range(200)]
print(f'rows {len(y)} | train {int(train.sum())} | test {int(test.sum())} | yes {y.mean():.3f} | answered {ans.mean():.3f}')
out = {}

# prior over each soft answer, from the answers that ARE observed in the training worlds
prior = np.zeros((4, V))
for k in range(4):
    vals = np.concatenate([w['soft'][:, k][w['soft'][:, k] > 0] for w in W if w['seed'] <= 1030])
    for v in range(NV[k]):
        prior[k, v] = (vals == v + 1).mean()
print('prior over soft answers (from observed answers in training worlds):')
for k in range(4):
    print(f'   {CORE_NAMES[k]:18} ' + '  '.join(f'{p:.3f}' for p in prior[k, :NV[k]]) + f'   -> P(two random people agree) = {(prior[k] ** 2).sum():.3f}')
logprior = np.where(prior > 0, np.log(np.maximum(prior, 1e-12)), -1e9)


# ---------------------------------------------------------------- inference for one pool on one day
def infer(soft, known, ei, ej, ey, eday, th, n_members, traits=True, generosity=True, T=3):
    q = np.broadcast_to(prior, (n_members, 4, V)).copy()
    kk = np.where(known)
    q[kk[0], kk[1], :] = 0.0
    q[kk[0], kk[1], soft[kk] - 1] = 1.0
    g = np.zeros(n_members)
    tau2 = th['tau'] ** 2
    a = th['a']
    tterm = th['c'] + th['late'] * (eday >= 35) + th['dday'] * (eday / 60.0)
    hess = np.full(n_members, 1.0 / tau2)
    if len(ei) == 0:
        return g, 1.0 / hess, q
    for _ in range(T):
        pe = (q[ei] * q[ej]).sum(-1)                               # (E,4)  P(same) per past introduction and field
        base = tterm + pe @ a
        if generosity:
            for _ in range(3):
                p = sig(base + g[ei])
                grad = np.bincount(ei, weights=ey - p, minlength=n_members) - g / tau2
                hess = np.bincount(ei, weights=p * (1 - p), minlength=n_members) + 1.0 / tau2
                g = g + grad / hess
        if not traits:
            break
        z0 = (base + g[ei])[:, None, None] - (a * pe)[:, :, None]   # everything except field k
        za = z0 + a[None, :, None] * q[ej]                          # if the ACTOR's answer on field k were v
        zp = z0 + a[None, :, None] * q[ei]                          # if the PARTNER's answer on field k were v
        yy = ey[:, None, None]
        la = yy * logsig(za) + (1 - yy) * logsig(-za)
        lp = yy * logsig(zp) + (1 - yy) * logsig(-zp)
        L = (np.bincount((ei[:, None] * 16 + CELL).ravel(), weights=la.reshape(len(ei), 16).ravel(), minlength=n_members * 16)
             + np.bincount((ej[:, None] * 16 + CELL).ravel(), weights=lp.reshape(len(ej), 16).ravel(), minlength=n_members * 16)
             ).reshape(n_members, 4, V)
        lq = logprior[None] + L
        lq -= lq.max(-1, keepdims=True)
        qn = np.exp(lq)
        qn /= qn.sum(-1, keepdims=True)
        q_new = 0.5 * q + 0.5 * qn
        q_new[kk[0], kk[1], :] = 0.0
        q_new[kk[0], kk[1], soft[kk] - 1] = 1.0
        q = q_new
    if generosity:
        pe = (q[ei] * q[ej]).sum(-1)
        p = sig(tterm + pe @ a + g[ei])
        hess = np.bincount(ei, weights=p * (1 - p), minlength=n_members) + 1.0 / tau2
    return g, 1.0 / hess, q


NF = 14
CELL = np.arange(16)[None, :]


def run_pass(th, idx=None, traits=True, generosity=True):
    """Online predictions for every row: features come from inference on the decision day using only past answers."""
    res = []
    for wi, w in enumerate(W):
        n = w['n']
        if idx is not None and not idx[wi]:
            res.append(np.zeros((2 * n, NF)))
            continue
        m_count = len(w['soft'])
        r = np.zeros((2 * n, NF))
        d = w['ev_day'][:n]
        for dd in np.unique(d):
            seen = w['ev_obs'] <= dd
            answered = seen & (w['ev_val'] != 3)
            known = w['sday'] <= dd
            ei, ej = w['ev_i'][answered], w['ev_j'][answered]
            g, var, q = infer(w['soft'], known, ei, ej, (w['ev_val'][answered] == 1).astype(float),
                              w['ev_day'][answered].astype(float), th, m_count, traits, generosity)
            n_all = np.bincount(w['ev_i'][seen], minlength=m_count)
            n_ans = np.bincount(ei, minlength=m_count)
            rows = np.where(w['ev_day'] == dd)[0]
            i, j = w['ev_i'][rows], w['ev_j'][rows]
            pe = (q[i] * q[j]).sum(-1)
            teq = (w['truth'][i] == w['truth'][j]).astype(float)
            both = known[i] & known[j]
            r[rows, 0:4] = pe
            r[rows, 4] = g[i]
            r[rows, 5] = var[i]
            r[rows, 6] = n_ans[i]
            r[rows, 7] = n_all[i]
            r[rows, 8:12] = teq                                    # hid: true agreement, diagnostic only
            r[rows, 12] = both.sum(1)
            r[rows, 13] = w['hid_person'][i, 0]                    # hid: true generosity, diagnostic only
        res.append(r)
    return np.vstack(res)


def linear(th, F):
    return th['c'] + F[:, 0:4] @ th['a'] + th['late'] * (day >= 35) + th['dday'] * (day / 60.0)


def predict(th, F):
    o = linear(th, F)
    p_acc = sig((o + F[:, 4]) / np.sqrt(1.0 + 0.3927 * F[:, 5]))
    p_ans = (F[:, 6] + th['alpha'] * th['r0']) / (F[:, 7] + th['alpha'])
    return p_ans * p_acc, p_acc, p_ans


def newton_offset(Xd, t, offset, ridge=1e-3):
    beta = np.zeros(Xd.shape[1])
    for _ in range(25):
        p = sig(Xd @ beta + offset)
        grad = Xd.T @ (t - p) - ridge * beta
        Hm = (Xd * (p * (1 - p))[:, None]).T @ Xd + ridge * np.eye(Xd.shape[1])
        step = np.linalg.solve(Hm, grad)
        beta += step
        if np.abs(step).max() < 1e-7:
            break
    return beta


def m_step(th, F, rows):
    """Re-estimate c, a_k and the time effects on answered training rows, holding each person's generosity as an offset."""
    Xd = np.column_stack([np.ones(rows.sum()), F[rows, 0:4], (day[rows] >= 35).astype(float), day[rows] / 60.0])
    beta = newton_offset(Xd, y[rows].astype(float), F[rows, 4])
    new = dict(th)
    new.update(c=float(beta[0]), a=beta[1:5].copy(), late=float(beta[5]), dday=float(beta[6]))
    return new


def report(label, p, ref=None, mask=None, store=True):
    m = test if mask is None else (test & mask)
    a = roc_auc_score(y[m], p[m])
    ll = log_loss(y[m], np.clip(p[m], 1e-6, 1 - 1e-6))
    bs = boot if mask is None else [b[mask[b]] for b in boot]
    vals = [roc_auc_score(y[b], p[b]) for b in bs if len(set(y[b])) == 2]
    msg = f'  {label:62} AUC {a:.4f} [{np.percentile(vals,2.5):.4f}-{np.percentile(vals,97.5):.4f}]  logloss {ll:.4f}'
    rec = {'auc': float(a), 'ci': [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))], 'logloss': float(ll)}
    if ref is not None:
        dl = [roc_auc_score(y[b], p[b]) - roc_auc_score(y[b], ref[b]) for b in bs if len(set(y[b])) == 2]
        msg += f'  vs current {np.mean(dl):+.4f} [{np.percentile(dl,2.5):+.4f}, {np.percentile(dl,97.5):+.4f}]'
        rec['delta'] = [float(np.mean(dl)), float(np.percentile(dl, 2.5)), float(np.percentile(dl, 97.5))]
    print(msg)
    if store:
        out[label] = rec
    return a


def stack(M, Cs=(0.003, 0.03, 0.3, 3.0)):
    best, best_ll = None, 9
    for C in Cs:
        m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000)).fit(M[fit_m], y[fit_m])
        ll = log_loss(y[val_m], m.predict_proba(M[val_m])[:, 1])
        if ll < best_ll:
            best, best_ll = C, ll
    return make_pipeline(StandardScaler(), LogisticRegression(C=best, max_iter=3000)).fit(M[train], y[train]).predict_proba(M)[:, 1]


# ================================================================ baseline and ceiling
print('\n' + '=' * 112 + '\n0. REFERENCE POINTS (held-out worlds)\n' + '=' * 112)
report('ceiling: true probability (hidden; diagnostic only)', hid_p)
p0 = stack(x19)
report('current model: logistic on the 19 policy features', p0)

# ================================================================ learn the parameters
print('\n' + '=' * 112 + '\n1. LEARNING THE MODEL PARAMETERS FROM TRAINING WORLDS\n' + '=' * 112)
seeds_w = np.array([w['seed'] for w in W])
fit_seeds = np.unique(seeds_w[seeds_w <= 1020])
val_seeds = np.unique(seeds_w[(seeds_w > 1020) & (seeds_w <= 1030)])
em_set, sel_set = fit_seeds[:8], val_seeds[:4]                 # a subset of worlds keeps the tuning loop fast
em_worlds = np.isin(seeds_w, em_set)
sel_worlds = np.isin(seeds_w, sel_set)
em_rows = np.isin(seed, em_set) & (ans == 1)
sel_rows = np.isin(seed, sel_set)
th = {'c': float(np.log(0.47 / 0.53)), 'a': np.zeros(4), 'late': 0.0, 'dday': 0.0, 'tau': 0.7, 'alpha': 4.0,
      'r0': float(ans[fit_m].mean())}
t0 = time.time()
best = None
for tau in (0.5, 0.75, 1.0, 1.3):
    cur = dict(th, tau=tau)
    for _ in range(3):                                         # hard-EM: infer latents, then refit effects
        F = run_pass(cur, em_worlds)
        cur = m_step(cur, F, em_rows)
    F = run_pass(cur, sel_worlds)
    for alpha in (2.0, 4.0, 8.0):
        cand = dict(cur, alpha=alpha)
        pv = np.clip(predict(cand, F)[0][sel_rows], 1e-6, 1 - 1e-6)
        ll = log_loss(y[sel_rows], pv)
        print(f'   tau {tau:<5} alpha {alpha:<4} validation logloss {ll:.5f}  AUC {roc_auc_score(y[sel_rows], pv):.4f}   [{time.time()-t0:.0f}s]')
        if best is None or ll < best[0]:
            best = (ll, cand)
th = best[1]
fit_worlds = seeds_w <= 1020
for _ in range(2):                                             # refine the effects on ALL fit worlds at the chosen tau
    F = run_pass(th, fit_worlds)
    th = m_step(th, F, fit_m & (ans == 1))
print(f'chosen: tau = {th["tau"]} (spread of generosity), alpha = {th["alpha"]} (prior strength of the response rate), r0 = {th["r0"]:.3f}   [{time.time()-t0:.0f}s]')
print('learned effect of agreeing on a field (log-odds), and time effects:')
for k in range(4):
    print(f'   {CORE_NAMES[k]:18} {th["a"][k]:+.3f}')
print(f'   after day 35       {th["late"]:+.3f}     per 60 days {th["dday"]:+.3f}     intercept {th["c"]:+.3f}')
out['theta'] = {'tau': th['tau'], 'alpha': th['alpha'], 'r0': th['r0'], 'c': th['c'], 'a': th['a'].tolist(),
                'late': th['late'], 'dday': th['dday']}

# ================================================================ evaluate
print('\n' + '=' * 112 + '\n2. HELD-OUT ACCURACY (AUC for "said yes"; 0.50 = chance)\n' + '=' * 112)
F = run_pass(th)
p_full, p_acc, p_ans = predict(th, F)
report('current model: logistic on the 19 policy features', p0, store=False)
report('probabilistic model, raw output (no extra fitting)', p_full, p0)
Fg = run_pass(th, traits=False)
report('  ablation: generosity only (unknown answers left at the prior)', predict(th, Fg)[0], p0)
Ft = run_pass(th, generosity=False)
report('  ablation: unknown-answer inference only (no generosity)', predict(th, Ft)[0], p0)
th_plain = dict(th)
Fn = run_pass(th, traits=False, generosity=False)
report('  ablation: neither (compatibility on known answers + response rate)', predict(th, Fn)[0], p0)
o = linear(th, F)
S = np.column_stack([o, F[:, 4], np.sqrt(F[:, 5]), F[:, 4] * np.sqrt(F[:, 5]), np.log(p_ans / (1 - p_ans)), F[:, 6], F[:, 7] - F[:, 6],
                     F[:, 0:4], np.log(np.clip(p_full, 1e-6, 1 - 1e-6) / (1 - np.clip(p_full, 1e-6, 1 - 1e-6)))])
p_s = stack(S)
report('probabilistic model + calibration layer', p_s, p0)
p_sx = stack(np.hstack([S, x19]))
report('probabilistic model + calibration layer + the 19 features', p_sx, p0)
best_label, best_p = max((('probabilistic model, raw', p_full), ('probabilistic + calibration', p_s), ('probabilistic + calibration + 19', p_sx)),
                         key=lambda t: roc_auc_score(y[val_m], t[1][val_m]))
print(f'best on validation worlds: {best_label}')
out['best'] = best_label

# ================================================================ probability analysis: is each hidden quantity real?
print('\n' + '=' * 112 + '\n3. PROBABILITY ANALYSIS: EVIDENCE FOR EACH HIDDEN QUANTITY (test worlds, observable data only)\n' + '=' * 112)
g_hat, n_ans = F[:, 4], F[:, 6]
has = n_ans >= 1
print('(a) Generosity. People with at least one past answer, split by the generosity the model inferred for them:')
qs = np.quantile(g_hat[train & has], [1 / 3, 2 / 3])
for lab, m in (('lowest third', g_hat <= qs[0]), ('middle third', (g_hat > qs[0]) & (g_hat <= qs[1])), ('highest third', g_hat > qs[1])):
    mm = test & has & m & (ans == 1)
    print(f'    inferred generosity {lab:14} n={int(mm.sum()):6d}   yes rate among those who answered {y[mm].mean()*100:5.1f}%')
print('(b) Response rate. Chance of answering at all, by what the person did before:')
n_all = F[:, 7]
sil = n_all - n_ans
for lab, m in (('no history', n_all == 0), ('answered every time (1+)', (n_all >= 1) & (sil == 0)), ('silent once', sil == 1), ('silent twice or more', sil >= 2)):
    mm = test & m
    print(f'    {lab:26} n={int(mm.sum()):6d}   answered {ans[mm].mean()*100:5.1f}%')
print('(c) Unknown soft answers. For pairs where the relationship goal is NOT known for both people, the model\'s inferred chance')
print('    that they agree, against how often such pairs said yes (among those who answered):')
unk_goal = (F[:, 0] > 1e-9) & (F[:, 0] < 1 - 1e-9)
pg = F[:, 0]
edges = np.quantile(pg[train & unk_goal], [0.1, 0.9])
for lab, m in (('lowest 10% (model thinks they differ)', pg <= edges[0]), ('middle 80%', (pg > edges[0]) & (pg < edges[1])),
               ('highest 10% (model thinks they agree)', pg >= edges[1])):
    mm = test & unk_goal & m & (ans == 1)
    print(f'    inferred P(same goal) {lab:38} n={int(mm.sum()):6d}  mean P {pg[mm].mean():.3f}  yes rate {y[mm].mean()*100:5.1f}%  [truth: same goal {F[mm, 8].mean()*100:5.1f}%]')
print('    (the "truth" column is a hidden-state check of the inference, not something the model sees)')
print('(d) Inferred generosity vs the hidden truth (diagnostic): correlation ' +
      ', '.join(f'{lab}: {np.corrcoef(g_hat[test & m], F[test & m, 13])[0,1]:.3f}' for lab, m in
                (('all rows', np.ones(len(y), bool)), ('1+ past answers', n_ans >= 1), ('3+ past answers', n_ans >= 3))))

# ================================================================ by history depth
print('\n' + '=' * 112 + '\n4. WHERE THE GAIN COMES FROM (test worlds)\n' + '=' * 112)
print(f'  {"past answers from this person":30} {"rows":>7} {"current":>9} {"best new":>9} {"ceiling":>9}')
out['by_history'] = {}
for lab, m in (('0', n_ans == 0), ('1', n_ans == 1), ('2', n_ans == 2), ('3+', n_ans >= 3)):
    mm = test & m
    r = (roc_auc_score(y[mm], p0[mm]), roc_auc_score(y[mm], best_p[mm]), roc_auc_score(y[mm], hid_p[mm]))
    out['by_history'][lab] = {'n': int(mm.sum()), 'current': float(r[0]), 'best': float(r[1]), 'ceiling': float(r[2])}
    print(f'  {lab:30} {int(mm.sum()):7d} {r[0]:9.4f} {r[1]:9.4f} {r[2]:9.4f}')
print(f'  {"soft fields known for both":30}')
for lab, m in (('0 of 4', F[:, 12] == 0), ('1-2 of 4', (F[:, 12] >= 1) & (F[:, 12] <= 2)), ('3-4 of 4', F[:, 12] >= 3)):
    mm = test & m
    print(f'  {lab:30} {int(mm.sum()):7d} {roc_auc_score(y[mm], p0[mm]):9.4f} {roc_auc_score(y[mm], best_p[mm]):9.4f} {roc_auc_score(y[mm], hid_p[mm]):9.4f}')

print('\nCalibration of the best new model (test worlds):')
pt = best_p[test]
cal = []
for b in np.array_split(np.argsort(pt), 10):
    cal.append((float(pt[b].mean() * 100), float(y[test][b].mean() * 100)))
    print(f'   predicted {cal[-1][0]:5.1f}%   observed {cal[-1][1]:5.1f}%')
out['calibration'] = cal
print(f'   spread of predictions (1st-99th percentile): {np.percentile(pt,1)*100:.1f}% to {np.percentile(pt,99)*100:.1f}%  (current model: {np.percentile(p0[test],1)*100:.1f}% to {np.percentile(p0[test],99)*100:.1f}%)')

# ================================================================ both say yes
print('\n' + '=' * 112 + '\n5. BOTH SAY YES: ARE THE TWO DIRECTIONS INDEPENDENT? (test worlds)\n' + '=' * 112)
off = np.cumsum([0] + [2 * w['n'] for w in W])
ra = np.concatenate([np.arange(off[i], off[i] + w['n']) for i, w in enumerate(W)])
rb = ra + np.concatenate([np.full(w['n'], w['n']) for w in W])
pt_pair = test[ra]
mutual = (y[ra] * y[rb])
print(f'  pairs in test worlds: {int(pt_pair.sum())}; both say yes {mutual[pt_pair].mean()*100:.2f}%')
ya, yb = y[ra][pt_pair], y[rb][pt_pair]
print(f'  P(B says yes | A said yes) = {yb[ya == 1].mean()*100:.1f}%    P(B says yes | A said no or was silent) = {yb[ya == 0].mean()*100:.1f}%')
for label, p in (('current model', p0), ('best new model', best_p)):
    res_a, res_b = (y[ra] - p[ra]), (y[rb] - p[rb])
    sd = np.sqrt(p[ra] * (1 - p[ra]) * p[rb] * (1 - p[rb]))
    rho_tr = float((res_a[~pt_pair & train[ra]] * res_b[~pt_pair & train[ra]]).mean() / sd[~pt_pair & train[ra]].mean())
    prod = p[ra] * p[rb]
    corr = np.clip(prod + rho_tr * sd, 1e-6, 1 - 1e-6)
    print(f'  {label:16} leftover correlation between the two answers (estimated on train) = {rho_tr:+.3f}')
    print(f'      product p_a*p_b:   AUC for both-yes {roc_auc_score(mutual[pt_pair], prod[pt_pair]):.4f}   mean predicted {prod[pt_pair].mean()*100:.2f}% vs observed {mutual[pt_pair].mean()*100:.2f}%')
    print(f'      with correlation:  AUC for both-yes {roc_auc_score(mutual[pt_pair], corr[pt_pair]):.4f}   mean predicted {corr[pt_pair].mean()*100:.2f}% vs observed {mutual[pt_pair].mean()*100:.2f}%')
    out['pair_' + label] = {'rho': rho_tr, 'auc_product': float(roc_auc_score(mutual[pt_pair], prod[pt_pair])),
                            'auc_corrected': float(roc_auc_score(mutual[pt_pair], corr[pt_pair]))}
ceil_pair = hid_p[ra] * hid_p[rb]
print(f'  ceiling (hidden, diagnostic): AUC for both-yes {roc_auc_score(mutual[pt_pair], ceil_pair[pt_pair]):.4f}')
json.dump(out, open(f'results/prob_model_{tag}.json', 'w'), indent=1, default=float)
print(f'\nsaved results/prob_model_{tag}.json')
