"""Can a better model raise the yes-model's AUC? Ceiling diagnosis, then new architectures on observable history.

Input : results/seq_rows.npz (collect_seq.py).  Output: printed tables + results/auc_models.json
Split by WORLD: fit seeds 1001-1020, validation seeds 1021-1030 (tuning), test seeds 1031-1040 (touched once per model).

Part 1 uses the simulator's hidden columns (hid_*) ONLY to measure ceilings. They are never inputs to the real models.
Part 2 builds models from observable inputs only:
  M0  logistic on the 19 policy features (current)
  M1  latent-trait model: a person's hidden generosity is inferred from their past answers AFTER adjusting for how
      compatible each past partner was (a yes to a poor match is stronger evidence than a yes to a good match)
  M2  gradient boosting on the same structured inputs
  M3  small neural network on the raw answer-history sequence
"""
import json
import sys
import warnings
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')
path = sys.argv[1] if len(sys.argv) > 1 else 'results/seq_rows.npz'
rng = np.random.default_rng(0)
d = np.load(path)
x19, now_eq, a_raw, c_raw = d['x19'].astype(float), d['now_eq'], d['a_raw'], d['c_raw']
h_out, h_eq, h_raw, h_day = d['h_out'], d['h_eq'], d['h_raw'], d['h_day']
scal, hid, y, ans, seed, variant = d['scal'].astype(float), d['hid'].astype(float), d['y'].astype(int), d['ans'].astype(int), d['seed'], d['variant']
day = scal[:, 0]
world = seed * 10 + variant
fit_m, val_m = seed <= 1020, (seed > 1020) & (seed <= 1030)
train = seed <= 1030
test = seed >= 1031
print(f'{path}: rows {len(y)} | train {int(train.sum())} | test {int(test.sum())} | yes rate {y.mean():.3f} | answered {ans.mean():.3f}')
print(f'soft fields known for BOTH people (average over 7 fields): {(now_eq > 0).mean()*100:.1f}%   '
      f'| rows where the person has at least one past answer: {((h_out > 0).sum(1) > 0).mean()*100:.1f}%')
tw = np.unique(world[test])
rt = {w: np.where(test & (world == w))[0] for w in tw}
boot = [np.concatenate([rt[w] for w in rng.choice(tw, len(tw))]) for _ in range(200)]
out = {}


def report(label, p, ref=None, store=True):
    a = roc_auc_score(y[test], p[test])
    ll = log_loss(y[test], np.clip(p[test], 1e-6, 1 - 1e-6))
    vals = [roc_auc_score(y[b], p[b]) for b in boot]
    msg = f'  {label:64} AUC {a:.4f} [{np.percentile(vals,2.5):.4f}-{np.percentile(vals,97.5):.4f}]  logloss {ll:.4f}'
    rec = {'auc': float(a), 'ci': [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))], 'logloss': float(ll)}
    if ref is not None:
        dl = [roc_auc_score(y[b], p[b]) - roc_auc_score(y[b], ref[b]) for b in boot]
        msg += f'  vs M0 {np.mean(dl):+.4f} [{np.percentile(dl,2.5):+.4f}, {np.percentile(dl,97.5):+.4f}]'
        rec['delta'] = [float(np.mean(dl)), float(np.percentile(dl, 2.5)), float(np.percentile(dl, 97.5))]
    print(msg)
    if store:
        out[label] = rec
    return a


def logit_fit(M, target=None, rows=None, Cs=(0.003, 0.03, 0.3, 3.0)):
    """Logistic regression with C picked on the validation worlds, then refit on all train worlds."""
    t = y if target is None else target
    rf = fit_m if rows is None else fit_m & rows
    rv = val_m if rows is None else val_m & rows
    rt_ = train if rows is None else train & rows
    best, best_ll = None, 9
    for C in Cs:
        m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000)).fit(M[rf], t[rf])
        ll = log_loss(t[rv], m.predict_proba(M[rv])[:, 1])
        if ll < best_ll:
            best, best_ll = C, ll
    return make_pipeline(StandardScaler(), LogisticRegression(C=best, max_iter=3000)).fit(M[rt_], t[rt_])


# ================================================================ Part 1: where is the ceiling?
print('\n' + '=' * 112 + '\nPART 1. CEILING DIAGNOSIS (hidden simulator quantities; diagnostic only, never used by a real model)\n' + '=' * 112)
p_true, p_accept, bias, resp, true_fit = hid[:, 0], hid[:, 1], hid[:, 2], hid[:, 3], hid[:, 4]
print('What a perfect model could reach (remaining noise is pair-specific chance that nobody can know):')
report('ceiling: the true probability of a yes', p_true)
m0 = logit_fit(x19)
p0 = m0.predict_proba(x19)[:, 1]
report('M0 current model (19 observable features)', p0)
print('\nHow much each HIDDEN factor would add if we somehow knew it (added to the 19 features):')
for label, extra in (('+ the person\'s true generosity (bias)', bias[:, None]),
                     ('+ the person\'s true response rate', resp[:, None]),
                     ('+ the true number of core soft fields that agree', true_fit[:, None]),
                     ('+ generosity and response rate', np.column_stack([bias, resp])),
                     ('+ all three', np.column_stack([bias, resp, true_fit]))):
    M = np.hstack([x19, extra])
    report('M0 ' + label, logit_fit(M).predict_proba(M)[:, 1], p0)

# ================================================================ Part 2: observable-only models
print('\n' + '=' * 112 + '\nPART 2. NEW MODELS FROM OBSERVABLE INPUTS ONLY\n' + '=' * 112)


def fit_feats(codes, dday):
    """Compatibility features of one introduction: same / different flags for the 7 soft fields, plus timing."""
    return np.concatenate([(codes == 1).astype(float), (codes == 2).astype(float),
                           (dday >= 35).astype(float)[..., None], (dday / 60.0)[..., None]], axis=-1)


F_now = fit_feats(now_eq, day)
F_hist = fit_feats(h_eq, h_day.astype(float))
mask_ans = (h_out == 1) | (h_out == 2)
Y_hist = (h_out == 1).astype(float)
n_ans = mask_ans.sum(1).astype(float)
n_sil = (h_out == 3).sum(1).astype(float)
n_all = (h_out > 0).sum(1).astype(float)
answered_rows = ans == 1


def offsets(model):
    sc, lr = model[0], model[-1]
    w = lr.coef_[0][:F_now.shape[1]] / sc.scale_[:F_now.shape[1]]
    c0 = lr.intercept_[0] - (lr.coef_[0][:F_now.shape[1]] * sc.mean_[:F_now.shape[1]] / sc.scale_[:F_now.shape[1]]).sum()
    return F_now @ w + c0, F_hist @ w + c0


def infer_bias(o_hist, tau):
    """MAP of a person's generosity given past answers and how compatible each past partner was (Newton steps)."""
    b = np.zeros(len(y))
    for _ in range(8):
        p = 1 / (1 + np.exp(-(o_hist + b[:, None])))
        grad = (mask_ans * (Y_hist - p)).sum(1) - b / tau ** 2
        hess = (mask_ans * p * (1 - p)).sum(1) + 1 / tau ** 2
        b = b + grad / hess
    p = 1 / (1 + np.exp(-(o_hist + b[:, None])))
    var = 1 / ((mask_ans * p * (1 - p)).sum(1) + 1 / tau ** 2)
    return b, var


def structured(tau, rounds=2):
    stage = logit_fit(F_now, rows=answered_rows)          # accept | answered, compatibility only
    for _ in range(rounds):
        o_now, o_hist = offsets(stage)
        b, var = infer_bias(o_hist, tau)
        stage = logit_fit(np.column_stack([F_now, b]), rows=answered_rows)   # refit with generosity as a covariate
    o_now, o_hist = offsets(stage)
    b, var = infer_bias(o_hist, tau)
    r2 = (n_ans + 2 * 0.75) / (n_all + 2)
    r6 = (n_ans + 6 * 0.75) / (n_all + 6)
    S = np.column_stack([o_now, b, np.sqrt(var), b * np.sqrt(var), np.log(r2 / (1 - r2)), np.log(r6 / (1 - r6)),
                         n_ans, n_sil, o_now * (n_ans > 0)])
    return S


best_tau, best_ll, best_S = None, 9, None
for tau in (0.4, 0.7, 1.0, 1.5):
    S = structured(tau)
    m = logit_fit(S)
    ll = log_loss(y[val_m], make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=3000)).fit(S[fit_m], y[fit_m]).predict_proba(S[val_m])[:, 1])
    if ll < best_ll:
        best_tau, best_ll, best_S = tau, ll, S
print(f'latent-trait model: prior spread of generosity chosen on validation worlds: tau = {best_tau}')
S = best_S
report('M0 current model (19 observable features)', p0, store=False)
p1 = logit_fit(S).predict_proba(S)[:, 1]
report('M1 latent-trait model (9 structured inputs)', p1, p0)
SX = np.hstack([S, x19])
p1b = logit_fit(SX).predict_proba(SX)[:, 1]
report('M1b latent-trait inputs + the 19 features', p1b, p0)

# how well does the inferred generosity track the true one? (diagnostic)
has = n_ans >= 1
print(f'   diagnostic: correlation of inferred generosity with the TRUE hidden bias: all rows {np.corrcoef(S[test, 1], bias[test])[0,1]:.3f}; '
      f'rows with 1+ past answers {np.corrcoef(S[test & has, 1], bias[test & has])[0,1]:.3f}; 3+ answers {np.corrcoef(S[test & (n_ans >= 3), 1], bias[test & (n_ans >= 3)])[0,1]:.3f}')
out['bias_recovery'] = {'all': float(np.corrcoef(S[test, 1], bias[test])[0, 1]),
                        'ge1': float(np.corrcoef(S[test & has, 1], bias[test & has])[0, 1])}

xgb = XGBClassifier(n_estimators=800, max_depth=3, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, min_child_weight=40,
                    reg_lambda=5.0, early_stopping_rounds=50, eval_metric='logloss', n_jobs=6, random_state=0, verbosity=0)
xgb.fit(SX[fit_m], y[fit_m], eval_set=[(SX[val_m], y[val_m])], verbose=False)
p2 = xgb.predict_proba(SX)[:, 1]
report('M2 gradient boosting on latent-trait inputs + 19 features', p2, p0)


def onehot(codes, k):
    return np.concatenate([(codes == i).astype(np.float32) for i in range(k)], axis=-1)


seq = np.concatenate([onehot(h_out[..., None], 4), onehot(h_eq, 3), (h_day[..., None] / 60.0).astype(np.float32)], axis=-1)
NN = np.hstack([seq.reshape(len(y), -1), onehot(now_eq, 3), onehot(a_raw, 5), onehot(c_raw, 5), x19, S]).astype(np.float32)
mu, sd = NN[fit_m].mean(0), NN[fit_m].std(0)
sd[sd < 1e-6] = 1
NNz = (NN - mu) / sd
best_nn, best_ll = None, 9
for hidden, alpha in (((32,), 1.0), ((64, 32), 1.0), ((64, 32), 10.0), ((16,), 10.0)):
    nn = MLPClassifier(hidden_layer_sizes=hidden, alpha=alpha, learning_rate_init=1e-3, batch_size=512, max_iter=40,
                       early_stopping=False, random_state=0)
    nn.fit(NNz[fit_m], y[fit_m])
    ll = log_loss(y[val_m], nn.predict_proba(NNz[val_m])[:, 1])
    print(f'   neural net {str(hidden):10} alpha {alpha:<5} validation logloss {ll:.4f}  val AUC {roc_auc_score(y[val_m], nn.predict_proba(NNz[val_m])[:, 1]):.4f}')
    if ll < best_ll:
        best_nn, best_ll = (hidden, alpha), ll
nn = MLPClassifier(hidden_layer_sizes=best_nn[0], alpha=best_nn[1], learning_rate_init=1e-3, batch_size=512, max_iter=40,
                   random_state=0).fit(NNz[train], y[train])
p3 = nn.predict_proba(NNz)[:, 1]
report(f'M3 small neural net on the raw history sequence {best_nn[0]} ({NN.shape[1]} inputs)', p3, p0)
p4 = (p1b + p2 + p3) / 3
report('M4 average of M1b, M2 and M3', p4, p0)

# ================================================================ Part 3: where does the gain come from?
print('\n' + '=' * 112 + '\nPART 3. BY AMOUNT OF HISTORY (test worlds)\n' + '=' * 112)
best_label, best_p = max((('M1b', p1b), ('M2', p2), ('M3', p3), ('M4', p4)), key=lambda t: roc_auc_score(y[test], t[1][test]))
print(f'best observable model: {best_label}')
print(f'  {"past answers from this person":32} {"rows":>7} {"M0":>8} {best_label:>8} {"ceiling":>9}')
out['by_history'] = {}
for lab, m in (('0', n_ans == 0), ('1', n_ans == 1), ('2', n_ans == 2), ('3+', n_ans >= 3)):
    mm = m & test
    a0, ab, ac = roc_auc_score(y[mm], p0[mm]), roc_auc_score(y[mm], best_p[mm]), roc_auc_score(y[mm], p_true[mm])
    out['by_history'][lab] = {'n': int(mm.sum()), 'm0': float(a0), 'best': float(ab), 'ceiling': float(ac)}
    print(f'  {lab:32} {int(mm.sum()):7d} {a0:8.4f} {ab:8.4f} {ac:9.4f}')
print('\nCalibration of the best model on test worlds:')
pt = best_p[test]
for b in np.array_split(np.argsort(pt), 8):
    print(f'   predicted {pt[b].mean()*100:5.1f}%   observed {y[test][b].mean()*100:5.1f}%')
out['best'] = best_label
json.dump(out, open('results/auc_models.json' if 'soft' not in path else 'results/auc_models_soft.json', 'w'), indent=1, default=float)
print('\nsaved')
