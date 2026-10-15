"""The yes-model: how likely is one person to return yes to an introduction to another.

Inputs are only what is observable on the decision day. Weights live in model_weights.json, fitted offline on
simulator rollouts (pre-decision observations and observed answers; never hidden state). Inference is plain Python.

The two directions are scored separately and their product ranks pairs. It is not a calibrated pair probability,
because the two answers to one introduction are positively dependent.
"""
import json
import math
import os
from kit import OPTIONS
from .normalise import EMPTY_STATS, SOFT_CORE, SOFT_OTHER, member_stats

# Population rates used to shrink a person's own record; a person with no history gets exactly these.
# Computed on the six training pools of the supplied data (public_01-06): 558 of 700 responses answered, 267 of 558 yes.
PRIOR_ANSWER = 0.797
PRIOR_YES = 0.479
PRIOR_STRENGTH = 3.0
# Every feature the rollouts log. The shipped model uses the subset named in model_weights.json.
FEATURES = ['age_gap', 'same_zone',
            'eq_goal', 'ne_goal', 'eq_pace', 'ne_pace', 'eq_life', 'ne_life', 'eq_conv', 'ne_conv',
            'n_eq_other', 'n_unknown_core',
            'actor_p_answer', 'actor_p_yes', 'actor_n', 'partner_recv', 'partner_p_yes',
            'late', 'day']
# The terms of the simulator's response equation that are observable: agreement on the four core soft fields and the
# person's own record. Age gap, zone, the other soft fields and the day add nothing to held-out ranking.
SHIPPED_FEATURES = ['eq_goal', 'ne_goal', 'eq_pace', 'ne_pace', 'eq_life', 'ne_life', 'eq_conv', 'ne_conv',
                    'n_unknown_core', 'actor_p_answer', 'actor_p_yes', 'actor_n', 'partner_recv', 'partner_p_yes']
CORE_TAGS = (('goal', 'relationship_goal'), ('pace', 'relationship_pace'),
             ('life', 'lifestyle'), ('conv', 'conversations'))
WEIGHTS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'model_weights.json')
_MODEL = None


def load_model():
    global _MODEL
    if _MODEL is None:
        try:
            with open(WEIGHTS_PATH, encoding='utf-8') as handle:
                _MODEL = json.load(handle)
        except (OSError, ValueError):
            _MODEL = {}
    return _MODEL


def reload_model():
    global _MODEL
    _MODEL = None
    return load_model()


def shrunk_rates(st):
    """A person's answer rate and yes-given-answer rate so far, each pulled toward the population rate."""
    p_answer = (st['answered'] + PRIOR_STRENGTH * PRIOR_ANSWER) / (st['n'] + PRIOR_STRENGTH)
    p_yes = (st['yes'] + PRIOR_STRENGTH * PRIOR_YES) / (st['answered'] + PRIOR_STRENGTH)
    return p_answer, p_yes


def directional_features(actor, partner, stats, day, names=FEATURES):
    """Features describing how likely `actor` is to say yes to `partner`, using only what is observable now.

    Each core soft field has three states: same, different, or unknown (both flags zero). Unknown is not a mismatch.
    """
    fa, fb = actor['fields'], partner['fields']
    x = {'age_gap': abs(actor['age'] - partner['age']) / 10.0,
         'same_zone': 1.0 if actor['zone'] == partner['zone'] else 0.0}
    unknown = 0
    for tag, key in CORE_TAGS:
        va, vb = fa[key], fb[key]
        if va is None or vb is None:
            unknown += 1
            x['eq_' + tag] = x['ne_' + tag] = 0.0
        else:
            x['eq_' + tag] = 1.0 if va == vb else 0.0
            x['ne_' + tag] = 0.0 if va == vb else 1.0
    x['n_eq_other'] = sum(1 for k in SOFT_OTHER if fa[k] is not None and fa[k] == fb[k]) / 3.0
    x['n_unknown_core'] = unknown / 4.0
    sa = stats.get(actor['member_id'], EMPTY_STATS)
    sp = stats.get(partner['member_id'], EMPTY_STATS)
    p_answer, p_yes = shrunk_rates(sa)
    x['actor_p_answer'] = p_answer
    x['actor_p_yes'] = p_yes
    x['actor_n'] = min(sa['n'], 6) / 6.0
    x['partner_recv'] = (sp['recv_yes'] + PRIOR_STRENGTH * PRIOR_YES) / (sp['recv_n'] + PRIOR_STRENGTH)
    x['partner_p_yes'] = shrunk_rates(sp)[1]
    x['late'] = 1.0 if day >= 35 else 0.0
    x['day'] = day / 60.0
    return [x[name] for name in names]


def sigmoid(z):
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def logistic_z(spec, x):
    z = spec['intercept']
    for v, c, m, s in zip(x, spec['coef'], spec['mean'], spec['scale']):
        z += c * (v - m) / s
    return z


def logistic_p(spec, x):
    return sigmoid(logistic_z(spec, x))


def soft_match_count(a, b):
    """The reference-style score: number of core soft fields where both answers are known and equal."""
    return sum(1 for k in SOFT_CORE if a['fields'][k] is not None and a['fields'][k] == b['fields'][k])


def prior_rates(state):
    """P(yes) from each person's own shrunk record alone. Used only if trained weights cannot be loaded."""
    stats = member_stats(state)
    out = {}
    for m in state['members']:
        p_answer, p_yes = shrunk_rates(stats.get(m['member_id'], EMPTY_STATS))
        out[m['member_id']] = p_answer * p_yes
    return out


def spec_features(spec):
    """The feature names a fitted model expects, in order. Older weight files list them once at the top level."""
    return spec.get('features') or load_model().get('features', FEATURES)


def directional_p(spec, actor, partner, stats, day):
    return logistic_p(spec, directional_features(actor, partner, stats, day, spec_features(spec)))


def pair_values(state, edges, adapt=False):
    """Ranking value of each feasible pair: P(a says yes to b) x P(b says yes to a).
    With `adapt`, the four soft-field effects are corrected by what this episode's own replies show (see below)."""
    spec = load_model().get('logit')
    if not spec:
        prior = prior_rates(state)
        return [prior[a['member_id']] * prior[b['member_id']] for a, b in edges]
    stats = member_stats(state)
    day = state['day']
    if not adapt:
        return [directional_p(spec, a, b, stats, day) * directional_p(spec, b, a, stats, day) for a, b in edges]
    shift = field_adjustments(state, spec)
    names = spec_features(spec)
    out = []
    for a, b in edges:
        extra = sum(d * agreement(a, b, key) for d, (_, key) in zip(shift, CORE_TAGS))
        za = logistic_z(spec, directional_features(a, b, stats, day, names)) + extra
        zb = logistic_z(spec, directional_features(b, a, stats, day, names)) + extra
        out.append(sigmoid(za) * sigmoid(zb))
    return out


# ------------------------------------------------------------------ within-episode correction of soft-field effects
ADAPT_PRIOR_SD = 0.5     # how far this episode's soft-field effects may plausibly differ from the pooled ones
ADAPT_MIN_REPLIES = 30


def agreement(a, b, key):
    """+1 same answer, -1 different answers, 0 if either is unknown."""
    va, vb = a['fields'][key], b['fields'][key]
    if va is None or vb is None:
        return 0
    return 1 if va == vb else -1


def field_adjustments(state, spec):
    """Per core soft field, a correction d_k to the pooled effect, so that logit += d_k x agreement_k.

    Fitted on this episode's replies so far (yes vs not yes, the model's own target) by maximum a posteriori with a
    N(0, ADAPT_PRIOR_SD^2) prior on each d_k, the pooled model as a fixed offset. The offset ignores the person's
    history so that a reply cannot explain itself. Returns zeros until there are enough replies."""
    names = spec_features(spec)
    members = {m['member_id']: m for m in state['members']}
    ends = {i['introduction_id']: (i['user_a'], i['user_b'], i['assigned_day']) for i in state['introductions']}
    rows = []
    for e in state['feedback']:
        if e['event'] != 'introduction_response' or e['introduction_id'] not in ends:
            continue
        ua, ub, day = ends[e['introduction_id']]
        actor, other = (ua, ub) if e['member_id'] == ua else (ub, ua)
        if actor not in members or other not in members:
            continue
        x, y = members[actor], members[other]
        agree = [agreement(x, y, key) for _, key in CORE_TAGS]
        offset = logistic_z(spec, directional_features(x, y, {}, day, names))
        rows.append((agree, offset, 1 if e['value'] == 'yes' else 0))
    k = len(CORE_TAGS)
    d = [0.0] * k
    if len(rows) < ADAPT_MIN_REPLIES:
        return d
    precision = 1.0 / (ADAPT_PRIOR_SD ** 2)
    for _ in range(8):                                   # Newton steps on a concave objective
        grad = [-precision * d[j] for j in range(k)]
        hess = [[precision if i == j else 0.0 for j in range(k)] for i in range(k)]
        for agree, offset, y in rows:
            p = sigmoid(offset + sum(d[j] * agree[j] for j in range(k)))
            w = p * (1.0 - p)
            for i in range(k):
                if agree[i]:
                    grad[i] += (y - p) * agree[i]
                    for j in range(k):
                        if agree[j]:
                            hess[i][j] += w * agree[i] * agree[j]
        step = _solve(hess, grad)
        d = [d[j] + step[j] for j in range(k)]
        if max(abs(v) for v in step) < 1e-6:
            break
    return d


def _solve(a, b):
    """Solve a x = b for a small symmetric positive-definite matrix (Gaussian elimination)."""
    n = len(b)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for c in range(n):
        piv = max(range(c, n), key=lambda r: abs(m[r][c]))
        m[c], m[piv] = m[piv], m[c]
        for r in range(c + 1, n):
            f = m[r][c] / m[c][c]
            for j in range(c, n + 1):
                m[r][j] -= f * m[c][j]
    x = [0.0] * n
    for r in range(n - 1, -1, -1):
        x[r] = (m[r][n] - sum(m[r][j] * x[j] for j in range(r + 1, n))) / m[r][r]
    return x


def agreement_weights():
    """How much it is worth knowing whether two people agree on each core soft field:
    (effect of agreement on the log-odds of a yes)^2 x variance of 'they agree' before asking.
    Effects come from the trained yes-model; the chance that two people agree is 1 / (number of allowed answers)."""
    model = load_model()
    spec = model.get('logit')
    names = spec_features(spec) if spec else FEATURES
    out = {}
    for tag, key in CORE_TAGS:
        effect = 0.5
        if spec:
            ie, ine = names.index('eq_' + tag), names.index('ne_' + tag)
            effect = spec['coef'][ie] / spec['scale'][ie] - spec['coef'][ine] / spec['scale'][ine]
        p_agree = 1.0 / len(OPTIONS[key])
        out[key] = effect * effect * p_agree * (1.0 - p_agree)
    return out
