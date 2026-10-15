"""Wires the layers together for one request: clarification in the ask phase, allocation in the match phase.

The policy keeps no memory. Its state is rebuilt from the observation on every call, so nothing can carry over
between episodes and a decision can always be reproduced from the request alone.

A configuration is a small dict of switches, each of which is an ablation handle:
  ask          'baseline' hard-constraint bundles in list order | 'bayes' ordered by Bayesian chance of a match
               | 'value' ordered by possible partners | 'none' never ask
  soft_asks    'voi' also buy soft answers by value of information with leftover units
  match        'global' joint matching | 'greedy' best pair first | 'tiered' exact matcher with tiered weights
  score        'logit' the yes-model | 'soft' the reference soft-match count
  eps          constant added to every pair so that more introductions beat fewer
  skip_below   with ask='bayes': do not ask today anyone whose chance of any partner is below this
  scarcity     with match='tiered': among first-time members, favour those with few feasible partners left
  adapt        re-estimate the four soft-field effects from this episode's own replies
"""
from . import allocate, clarify

CONFIGS = {
    # the shipped default: Bayesian ask order with daily skipping, exact tiered matching with the scarcity bonus,
    # 14-feature yes-model
    'submission':   dict(ask='bayes', match='tiered', score='logit', skip_below=0.05, scarcity=True),
    # the default before the scarcity bonus was adopted (study_next.json compares against this)
    'submission_v1': dict(ask='bayes', match='tiered', score='logit', skip_below=0.05),
    # Round 1 submitted default
    'final':        dict(ask='baseline', match='global', score='logit', eps=0.02),
    # Round 2 allocation: exact matcher, tiered weights (cardinality, first-time coverage, predicted value)
    'tiered':       dict(ask='baseline', match='tiered', score='logit'),
    'tiered_nocov': dict(ask='baseline', match='tiered', score='logit', coverage=False),
    'tiered_novalue': dict(ask='baseline', match='tiered', score='none'),
    'exact_value':  dict(ask='baseline', match='tiered', score='logit', cardinality=False, coverage=False),
    # who to ask first: rule (potential partners) vs Bayesian chance of a match; optionally skip unlikely members
    'tiered_valueask':   dict(ask='value', match='tiered', score='logit'),
    'tiered_bayes':      dict(ask='bayes', match='tiered', score='logit'),
    'tiered_bayes_skip': dict(ask='bayes', match='tiered', score='logit', skip_below=0.05),
    # Round 2 candidates on top of the shipped default
    'sub_skip10':   dict(ask='bayes', match='tiered', score='logit', skip_below=0.10),
    'sub_scarce':   dict(ask='bayes', match='tiered', score='logit', skip_below=0.05, scarcity=True),
    'sub_adapt':    dict(ask='bayes', match='tiered', score='logit', skip_below=0.05, adapt=True),
    # wait for a better match: pairs below a bar are not introduced (optionally only until a release day)
    'wait_v_p25':   dict(ask='bayes', match='tiered', score='logit', skip_below=0.05, scarcity=True, min_value=0.107),
    'wait_v_p50':   dict(ask='bayes', match='tiered', score='logit', skip_below=0.05, scarcity=True, min_value=0.127),
    'wait_v_p50_r40': dict(ask='bayes', match='tiered', score='logit', skip_below=0.05, scarcity=True, min_value=0.127,
                           release_day=40),
    'wait_soft1':   dict(ask='bayes', match='tiered', score='logit', skip_below=0.05, scarcity=True, min_soft=1),
    'wait_soft2':   dict(ask='bayes', match='tiered', score='logit', skip_below=0.05, scarcity=True, min_soft=2),
    # ablations of the default, one switch at a time
    'final_noask':  dict(ask='none', match='global', score='logit', eps=0.02),
    'logit_greedy': dict(ask='baseline', match='greedy', score='logit'),
    'logit_global': dict(ask='baseline', match='global', score='logit', eps=0.02),
    'global_only':  dict(ask='baseline', match='global', score='soft', eps=1.0),
    'final_voi':    dict(ask='baseline', match='global', score='logit', eps=0.02, soft_asks='voi'),
}


def decide(request, cfg):
    state = request['state']
    memory = request.get('memory') or {}
    if request['phase'] == 'ask':
        return {'asks': clarify.choose_asks(state, cfg), 'memory': memory}
    return {'pairs': allocate.choose_pairs(state, cfg), 'memory': memory}
