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
PRIOR_ANSWER = 0.78
PRIOR_YES = 0.47
PRIOR_STRENGTH = 3.0
FEATURES = ['age_gap', 'same_zone',
            'eq_goal', 'ne_goal', 'eq_pace', 'ne_pace', 'eq_life', 'ne_life', 'eq_conv', 'ne_conv',
            'n_eq_other', 'n_unknown_core',
            'actor_p_answer', 'actor_p_yes', 'actor_n', 'partner_recv', 'partner_p_yes',
            'late', 'day']
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


def directional_features(actor, partner, stats, day):
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
    return [x[name] for name in FEATURES]


def sigmoid(z):
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def logistic_p(spec, x):
    z = spec['intercept']
    for v, c, m, s in zip(x, spec['coef'], spec['mean'], spec['scale']):
        z += c * (v - m) / s
    return sigmoid(z)


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


def pair_values(state, edges):
    """Ranking value of each feasible pair: P(a says yes to b) x P(b says yes to a)."""
    spec = load_model().get('logit')
    if not spec:
        prior = prior_rates(state)
        return [prior[a['member_id']] * prior[b['member_id']] for a, b in edges]
    stats = member_stats(state)
    day = state['day']
    return [logistic_p(spec, directional_features(a, b, stats, day))
            * logistic_p(spec, directional_features(b, a, stats, day)) for a, b in edges]


def agreement_weights():
    """How much it is worth knowing whether two people agree on each core soft field:
    (effect of agreement on the log-odds of a yes)^2 x variance of 'they agree' before asking.
    Effects come from the trained yes-model; the chance that two people agree is 1 / (number of allowed answers)."""
    model = load_model()
    spec, names = model.get('logit'), model.get('features', FEATURES)
    out = {}
    for tag, key in CORE_TAGS:
        effect = 0.5
        if spec:
            ie, ine = names.index('eq_' + tag), names.index('ne_' + tag)
            effect = spec['coef'][ie] / spec['scale'][ie] - spec['coef'][ine] / spec['scale'][ine]
        p_agree = 1.0 / len(OPTIONS[key])
        out[key] = effect * effect * p_agree * (1.0 - p_agree)
    return out
