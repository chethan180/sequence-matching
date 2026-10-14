"""Wires the layers together for one request: clarification in the ask phase, allocation in the match phase.

The policy keeps no memory. Its state is rebuilt from the observation on every call, so nothing can carry over
between episodes and a decision can always be reproduced from the request alone.

A configuration is a small dict of switches, each of which is an ablation handle:
  ask          'baseline' hard-constraint bundles in list order | 'none' never ask
  soft_asks    'voi' also buy soft answers by value of information with leftover units
  match        'global' joint matching | 'greedy' best pair first
  score        'logit' the yes-model | 'soft' the reference soft-match count
  eps          constant added to every pair so that more introductions beat fewer
"""
from . import allocate, clarify

CONFIGS = {
    # submitted default
    'final':        dict(ask='baseline', match='global', score='logit', eps=0.02),
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
