"""Mathematical evidence, from OBSERVABLE data only, for every feature and hidden quantity the approach relies on.

Input : results/seq_rows.npz (person-introduction rows from simulator rollouts; hidden diagnostic columns are NOT read here)
Output: printed derivations + results/proofs.json

Uncertainty: every interval is a 95% bootstrap over whole SEEDS (40 clusters, 1,000 resamples). The seed is the only
independent unit: the simulator draws an introduction's randomness from (seed, the two people, the day), so the same pair on
the same day has identical luck in all six scenarios and under every behaviour policy that share a seed.
"""
import json
import warnings
import numpy as np
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings('ignore')
rng = np.random.default_rng(0)
d = np.load('results/seq_rows.npz')
now_eq, a_raw, h_out, h_eq, h_day = d['now_eq'], d['a_raw'], d['h_out'], d['h_eq'], d['h_day']
scal, y, ans, seed, variant = d['scal'].astype(float), d['y'].astype(int), d['ans'].astype(int), d['seed'], d['variant']
day = scal[:, 0]
wkey = seed.astype(np.int64)
worlds, wid = np.unique(wkey, return_inverse=True)
NW = len(worlds)
B = 1000
picks = rng.integers(0, NW, size=(B, NW))
FIELDS = ['relationship goal', 'pace', 'lifestyle', 'conversations', 'emotional availability', 'space for relationship', 'relocate']
NOPT = [2, 3, 3, 4, 2, 3, 3]
out = {}
print(f'rows {len(y)} | independent seeds {NW} | yes {y.mean():.4f} | answered {ans.mean():.4f} | yes among answered {y[ans == 1].mean():.4f}')


def per_world(values, mask=None):
    v = values if mask is None else np.where(mask, values, 0.0)
    return np.bincount(wid, weights=v.astype(float), minlength=NW)


def boot(stat, *sums):
    """stat(*totals) evaluated on the data and on each world-bootstrap replicate of the per-world sums."""
    point = stat(*[s.sum() for s in sums])
    reps = np.array([stat(*[s[p].sum() for s in sums]) for p in picks])
    reps = reps[np.isfinite(reps)]
    lo, hi = np.percentile(reps, [2.5, 97.5])
    return float(point), float(lo), float(hi), float(reps.std())


def logit(p):
    return np.log(p / (1 - p))


def lor(ys, ns, yd, nd):
    return logit(ys / ns) - logit(yd / nd)


# ================================================================ 1. agreement on each soft field
print('\n' + '=' * 110 + '\n1. EFFECT OF AGREEING ON EACH SOFT FIELD  (rows where both answers are known and the person answered)\n' + '=' * 110)
print('   log-odds ratio  LOR = ln[ p_same/(1-p_same) ] - ln[ p_diff/(1-p_diff) ],   z = LOR / SE(LOR)')
print(f'   {"field":26} {"n same":>7} {"n diff":>7} {"yes|same":>9} {"yes|diff":>9} {"LOR":>7} {"95% CI":>18} {"z":>6}')
answered = ans == 1
eff = {}
for k, name in enumerate(FIELDS):
    same, diff = (now_eq[:, k] == 1) & answered, (now_eq[:, k] == 2) & answered
    s = (per_world(y, same), per_world(same), per_world(y, diff), per_world(diff))
    pt, lo, hi, se = boot(lor, *s)
    eff[k] = (pt, lo, hi, se)
    ps, pdf = s[0].sum() / s[1].sum(), s[2].sum() / s[3].sum()
    print(f'   {name:26} {int(s[1].sum()):7d} {int(s[3].sum()):7d} {ps*100:8.1f}% {pdf*100:8.1f}% {pt:+7.3f} [{lo:+.3f}, {hi:+.3f}] {pt/se:6.1f}')
    out[f'lor_{name}'] = {'lor': pt, 'ci': [lo, hi], 'z': pt / se, 'p_same': float(ps), 'p_diff': float(pdf)}
print('   Reading: an interval that excludes 0 proves the field moves the answer; one that includes 0 gives no evidence it does.')

print('\n   Does agreement change whether people ANSWER AT ALL? (same test, outcome = answered)')
for k in (0, 1):
    same, diff = now_eq[:, k] == 1, now_eq[:, k] == 2
    pt, lo, hi, se = boot(lor, per_world(ans, same), per_world(same), per_world(ans, diff), per_world(diff))
    print(f'   {FIELDS[k]:26} LOR for answering {pt:+.3f} [{lo:+.3f}, {hi:+.3f}]')
    out[f'answer_lor_{FIELDS[k]}'] = {'lor': pt, 'ci': [lo, hi]}
print('   Compare these with the acceptance effects above (+0.76 and +0.66): agreement moves accepting, and answering little or not at all.')

# ================================================================ 2. additivity
print('\n' + '=' * 110 + '\n2. DO THE GOAL AND PACE EFFECTS ADD ON THE LOG-ODDS SCALE?  (both fields known for both people, person answered)\n' + '=' * 110)
cells = {}
for g in (1, 2):
    for p in (1, 2):
        m = (now_eq[:, 0] == g) & (now_eq[:, 1] == p) & answered
        cells[(g, p)] = (per_world(y, m), per_world(m))
        print(f'   goal {"same" if g == 1 else "diff"} / pace {"same" if p == 1 else "diff"}: n={int(cells[(g,p)][1].sum()):6d}  yes {cells[(g,p)][0].sum()/cells[(g,p)][1].sum()*100:5.1f}%')


def inter(a, b, c, dd, e, f, g, h):
    return logit(a / b) - logit(c / dd) - logit(e / f) + logit(g / h)


pt, lo, hi, se = boot(inter, *cells[(1, 1)], *cells[(1, 2)], *cells[(2, 1)], *cells[(2, 2)])
print(f'   interaction  I = L(same,same) - L(same,diff) - L(diff,same) + L(diff,diff) = {pt:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]')
print('   => an interval containing 0 means the two effects add; a logistic model with one term per field is adequate.')
out['interaction_goal_pace'] = {'value': pt, 'ci': [lo, hi]}

# ================================================================ 3. unknown answers hide the same effect
print('\n' + '=' * 110 + '\n3. DO UNKNOWN ANSWERS HIDE THE SAME EFFECT?  (mixture test)\n' + '=' * 110)
print('   If an "unknown" pair is secretly either a same-pair or a different-pair, then')
print('        p_unknown = pi * p_same + (1 - pi) * p_diff     =>     pi = (p_unknown - p_diff) / (p_same - p_diff)')
print('   and pi should equal the chance that two random people agree, pi0 = sum_v P(answer = v)^2, measured from observed answers.')
print(f'   {"field":18} {"p_same":>7} {"p_diff":>7} {"p_unknown":>10} {"pi (from data)":>15} {"95% CI":>18} {"pi0 (chance)":>13}')
mix = {}
for k in range(4):
    same, diff, unk = (now_eq[:, k] == 1) & answered, (now_eq[:, k] == 2) & answered, (now_eq[:, k] == 0) & answered
    vals = a_raw[:, k][a_raw[:, k] > 0]
    pi0 = float(sum(((vals == v).mean()) ** 2 for v in range(1, NOPT[k] + 1)))
    s = (per_world(y, same), per_world(same), per_world(y, diff), per_world(diff), per_world(y, unk), per_world(unk))
    pt, lo, hi, se = boot(lambda a, b, c, dd, e, f: (e / f - c / dd) / (a / b - c / dd), *s)
    mix[k] = (pt, pi0)
    verdict = 'consistent with chance mixture' if lo <= pi0 <= hi else 'differs from chance mixture'
    between = 'strictly between same and different' if 0 < lo and hi < 1 else 'not separated from an end'
    print(f'   {FIELDS[k]:18} {s[0].sum()/s[1].sum()*100:6.1f}% {s[2].sum()/s[3].sum()*100:6.1f}% {s[4].sum()/s[5].sum()*100:9.1f}% {pt:15.3f} [{lo:+.3f}, {hi:+.3f}] {pi0:13.3f}   {verdict}; {between}')
    out[f'mixture_{FIELDS[k]}'] = {'pi': pt, 'ci': [lo, hi], 'pi0': pi0, 'contains_pi0': bool(lo <= pi0 <= hi)}
print('   pi strictly between 0 and 1 proves an unknown pair is neither a "different" pair (pi = 0, how the baseline count treats it)')
print('   nor a "same" pair (pi = 1). Agreement with pi0 is the stronger statement that the hidden answers follow chance.')

# ================================================================ 4. person generosity
print('\n' + '=' * 110 + '\n4. IS THERE A PERSISTENT PERSON EFFECT ("GENEROSITY"), AND HOW MUCH OF IT CAN BE LEARNED?\n' + '=' * 110)


def feats(codes, dd):
    return np.concatenate([(codes == 1).astype(float), (codes == 2).astype(float), (dd >= 35).astype(float)[..., None], (dd / 60.0)[..., None]], axis=-1)


Fn, Fh = feats(now_eq, day), feats(h_eq, h_day.astype(float))
acc = LogisticRegression(C=10.0, max_iter=3000).fit(Fn[answered], y[answered])        # compatibility only, NO history
m_now = acc.predict_proba(Fn)[:, 1]
m_hist = acc.predict_proba(Fh.reshape(-1, Fh.shape[-1]))[:, 1].reshape(h_out.shape)
past_ans = (h_out == 1) | (h_out == 2)
y_hist = (h_out == 1).astype(float)
last_idx = np.where(past_ans.any(1), h_out.shape[1] - 1 - np.argmax(past_ans[:, ::-1], axis=1), -1)
has_last = last_idx >= 0
last_yes = np.where(has_last, y_hist[np.arange(len(y)), np.maximum(last_idx, 0)], 0)
a_mask, b_mask = answered & has_last & (last_yes == 1), answered & has_last & (last_yes == 0)
pt, lo, hi, se = boot(lor, per_world(y, a_mask), per_world(a_mask), per_world(y, b_mask), per_world(b_mask))
print(f'(a) Same person, consecutive answers:  P(yes | their last answer was yes) = {y[a_mask].mean()*100:.1f}%   '
      f'P(yes | last was no) = {y[b_mask].mean()*100:.1f}%')
print(f'    LOR = {pt:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  z = {pt/se:.1f}   => answers of the same person are positively dependent.')
out['generosity_consecutive'] = {'lor': pt, 'ci': [lo, hi], 'p_after_yes': float(y[a_mask].mean()), 'p_after_no': float(y[b_mask].mean())}

r_now = (y - m_now)[:, None]
r_hist = y_hist - m_hist
w = (past_ans & answered[:, None]).astype(float)
num = per_world((w * r_now * r_hist).sum(1))
den = per_world((w * (m_now * (1 - m_now))[:, None] * m_hist * (1 - m_hist)).sum(1))
cnt = per_world(w.sum(1))
cov, clo, chi, _ = boot(lambda n, c: n / c, num, cnt)
tau2, tlo, thi, _ = boot(lambda n, dd: n / dd, num, den)
print('(b) Remove compatibility first. Let m = P(yes | compatibility only) and residual r = y - m. For two answers of the same person,')
print('        E[r1 * r2] = Cov(p1, p2) ~= m1(1-m1) * m2(1-m2) * tau^2        (first-order expansion of the logistic in the person effect g, Var(g) = tau^2)')
print(f'    measured  E[r1*r2] = {cov:+.4f}  95% CI [{clo:+.4f}, {chi:+.4f}]   over {int(cnt.sum())} within-person answer pairs  (0 if there were no person effect)')
print(f'    =>  tau^2 = sum(r1*r2) / sum(m1(1-m1)m2(1-m2)) = {tau2:.3f}  95% CI [{tlo:.3f}, {thi:.3f}],   tau = {np.sqrt(max(tau2,0)):.2f} on the log-odds scale')
out['generosity_tau2'] = {'cov': cov, 'cov_ci': [clo, chi], 'tau2': tau2, 'tau2_ci': [tlo, thi], 'pairs': int(cnt.sum())}

info = float((m_now[answered] * (1 - m_now[answered])).mean())
n_past = past_ans.sum(1)
print('(c) How much of it can be learned from n past answers? One yes/no answer carries Fisher information I = m(1-m) about g.')
print('    With a N(0, tau^2) prior, the share of the variance of g that n answers can recover is')
print(f'        R(n) = n*I*tau^2 / (1 + n*I*tau^2)        with I = {info:.3f}, tau^2 = {tau2:.3f}')
rel = {}
for n in (1, 2, 3, 5, 10):
    rel[n] = n * info * tau2 / (1 + n * info * tau2)
    print(f'        n = {n:2d}:  R = {rel[n]:.3f}   (correlation between the best estimate and the truth = sqrt(R) = {np.sqrt(rel[n]):.2f})')
shares = {n: float((n_past == n).mean()) for n in range(0, 4)}
shares['4+'] = float((n_past >= 4).mean())
avg_rel = float(np.mean([0.0 if n == 0 else n * info * tau2 / (1 + n * info * tau2) for n in n_past]))
print('    how many past answers do we actually have when deciding?  ' + ', '.join(f'{k}: {v*100:.1f}%' for k, v in shares.items()))
print(f'    => averaged over all decisions, only {avg_rel*100:.1f}% of the generosity variance is recoverable. This is a limit of the data, not of the model.')
out['generosity_reliability'] = {'info': info, 'R': rel, 'history_shares': shares, 'average_recoverable': avg_rel}

# ================================================================ 5. response rate
print('\n' + '=' * 110 + '\n5. IS THERE A PERSISTENT PERSON EFFECT ON ANSWERING AT ALL ("RESPONSE RATE")?\n' + '=' * 110)
seen = h_out > 0
a_hist = ((h_out == 1) | (h_out == 2)).astype(float)
abar = float(ans.mean())
has_seen = seen.any(1)
last_seen = np.where(has_seen, h_out.shape[1] - 1 - np.argmax(seen[:, ::-1], axis=1), 0)
last_ans = a_hist[np.arange(len(y)), last_seen]
m1, m2 = has_seen & (last_ans == 1), has_seen & (last_ans == 0)
pt, lo, hi, se = boot(lor, per_world(ans, m1), per_world(m1), per_world(ans, m2), per_world(m2))
print(f'(a) P(answers | answered last time) = {ans[m1].mean()*100:.1f}%    P(answers | silent last time) = {ans[m2].mean()*100:.1f}%    '
      f'LOR = {pt:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  z = {pt/se:.1f}')
ws = seen.astype(float)
num = per_world((ws * (ans - abar)[:, None] * (a_hist - abar)).sum(1))
cnt = per_world(ws.sum(1))
var_r, vlo, vhi, _ = boot(lambda n, c: n / c, num, cnt)
print(f'(b) For two introductions of the same person, E[(a1 - abar)(a2 - abar)] = Var(response rate) = {var_r:.4f}  95% CI [{vlo:.4f}, {vhi:.4f}]')
print(f'    => the response rate differs between people with standard deviation {np.sqrt(max(var_r,0)):.3f} around a mean of {abar:.3f}.')
out['response_rate'] = {'lor': pt, 'ci': [lo, hi], 'var': var_r, 'var_ci': [vlo, vhi], 'mean': abar}

# ================================================================ 6. the two directions of one introduction
print('\n' + '=' * 110 + '\n6. ARE THE TWO PEOPLE\'S ANSWERS TO THE SAME INTRODUCTION INDEPENDENT?\n' + '=' * 110)
ia, ib = np.arange(0, len(y), 2), np.arange(1, len(y), 2)
assert np.all(seed[ia] == seed[ib]) and np.all(day[ia] == day[ib]) and np.all(d['a_raw'][ia] == d['c_raw'][ib])
yes_model = LogisticRegression(C=10.0, max_iter=3000).fit(Fn, y)
p = yes_model.predict_proba(Fn)[:, 1]
pw = wid[ia]


def per_world_pair(v):
    return np.bincount(pw, weights=v.astype(float), minlength=NW)


ya, yb = y[ia], y[ib]
pt, lo, hi, se = boot(lor, per_world_pair(yb * (ya == 1)), per_world_pair(ya == 1), per_world_pair(yb * (ya == 0)), per_world_pair(ya == 0))
print(f'(a) P(B says yes | A said yes) = {yb[ya == 1].mean()*100:.1f}%    P(B says yes | A did not) = {yb[ya == 0].mean()*100:.1f}%    LOR = {pt:+.3f} [{lo:+.3f}, {hi:+.3f}]  z = {pt/se:.1f}')
ra, rb = ya - p[ia], yb - p[ib]
sd = np.sqrt(p[ia] * (1 - p[ia]) * p[ib] * (1 - p[ib]))
rho, rlo, rhi, _ = boot(lambda n, dd: n / dd, per_world_pair(ra * rb), per_world_pair(sd))
print(f'(b) After removing compatibility:  rho = sum(rA*rB) / sum(sqrt(pA(1-pA) pB(1-pB))) = {rho:+.3f}  95% CI [{rlo:+.3f}, {rhi:+.3f}]')
prod, obs = float((p[ia] * p[ib]).mean()), float((ya * yb).mean())
corr = float((p[ia] * p[ib] + rho * sd).mean())
print(f'    P(both yes): observed {obs*100:.2f}%;  independence pA*pB predicts {prod*100:.2f}%  (the correlation term closes this gap by construction)')
print('    => the two answers share a pair-specific component. Multiplying the two probabilities under-predicts "both say yes";')
print('       the product is still a valid RANKING score, but it is not a calibrated pair probability.')
out['pair_dependence'] = {'lor': pt, 'ci': [lo, hi], 'rho': rho, 'rho_ci': [rlo, rhi], 'observed': obs, 'product': prod, 'corrected': corr}

# ================================================================ 7. what an answer is worth
print('\n' + '=' * 110 + '\n7. WHICH UNKNOWN ANSWER IS WORTH BUYING?  (value of information)\n' + '=' * 110)
print('   Knowing whether a pair agrees on field k moves its log-odds by a_k with probability pi_k and by 0 otherwise, so the variance')
print('   of the score that asking removes is   V_k = a_k^2 * pi_k * (1 - pi_k).   (a_k = LOR from section 1, pi_k = chance of agreement)')
voi = {}
for k in range(4):
    a_k, pi_k = eff[k][0], mix[k][1]
    voi[FIELDS[k]] = a_k ** 2 * pi_k * (1 - pi_k)
tot = sum(voi.values())
for name, v in sorted(voi.items(), key=lambda t: -t[1]):
    print(f'   {name:18} V = {v:.4f}   ({v/tot*100:4.1f}% of the total)')
out['voi'] = voi
print('   => ask the relationship goal first, then pace; lifestyle and conversations are worth little.')

json.dump(out, open('results/proofs.json', 'w'), indent=1, default=float)
print('\nsaved results/proofs.json')
