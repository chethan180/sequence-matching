"""Policy entrypoint: reads one JSON request on stdin, writes one JSON response on stdout, exits with code zero.

The work is done by the layers in the `smp` package (input handling, feasibility, clarification, the yes-model,
allocation, safety). This file only selects a mode and guards the output. Hidden simulator state is never read.

Modes
  submission                  the submitted policy, and the default when no argument is given
  final                       the Round 1 policy
  greedy / no_asks / random   the three reference baselines, unchanged
  any other key of CONFIGS    an ablation or research configuration
"""
import itertools
import json
import random
import sys
from kit import baseline_asks, baseline_match, eligibility
from smp import allocate, experimental, feasibility, model, normalise, pipeline, safety

DEFAULT_MODE = 'submission'
REFERENCE_MODES = ('greedy', 'no_asks', 'random')
CONFIGS = dict(experimental.CONFIGS)
CONFIGS.update(pipeline.CONFIGS)

# Names used by the analysis scripts.
PRIOR_ANSWER, PRIOR_YES, PRIOR_STRENGTH = model.PRIOR_ANSWER, model.PRIOR_YES, model.PRIOR_STRENGTH
FEATURES, SOFT_SIM, OTHER_SOFT = model.FEATURES, normalise.SOFT_CORE, normalise.SOFT_OTHER
member_stats, directional_features, logistic_p = normalise.member_stats, model.directional_features, model.logistic_p
load_model, reload_model, agreement_weights = model.load_model, model.reload_model, model.agreement_weights
feasible_edges, greedy_matching, global_matching = feasibility.feasible_edges, allocate.greedy_matching, allocate.global_matching
propensities, gbm_p, soft_matches = experimental.propensities, experimental.gbm_p, model.soft_match_count


def _reference(request, mode):
    """The supplied baselines, kept exactly as released."""
    state = request['state']
    memory = request.get('memory') or {}
    if request['phase'] == 'ask':
        return {'asks': [] if mode == 'no_asks' else baseline_asks(state), 'memory': memory}
    if mode != 'random':
        pairs = baseline_match(state)
    else:
        candidates = [m for m in state['members'] if m['available']]
        past = normalise.past_pairs(state)
        edges = [tuple(sorted((a['member_id'], b['member_id'])))
                 for a, b in itertools.combinations(candidates, 2)
                 if eligibility(a, b)['status'] == 'feasible'
                 and tuple(sorted((a['member_id'], b['member_id']))) not in past]
        random.Random(17 + state['day']).shuffle(edges)
        pairs, used = [], set()
        for pair in edges:
            if not used.intersection(pair):
                pairs.append(list(pair))
                used.update(pair)
    return {'pairs': pairs, 'memory': memory}


def decide_raw(request, mode=DEFAULT_MODE):
    """The decision of a mode with no validation or fallback. For tests and for proving refactors."""
    if mode in REFERENCE_MODES:
        return _reference(request, mode)
    return pipeline.decide(request, mode if isinstance(mode, dict) else CONFIGS[mode])


def decide(request, mode=DEFAULT_MODE):
    """The response that is sent: the mode's decision, validated, with fallbacks. Never raises."""
    if mode in REFERENCE_MODES:
        return safety.guarded(request, lambda r: _reference(r, mode))
    return safety.guarded(request, lambda r: decide_raw(r, mode), fallbacks=(lambda r: _reference(r, 'greedy'),))


def parse_mode(argv):
    """`--baseline NAME` (how the evaluator runs the reference policies) or `--mode NAME`; anything else is ignored.
    Parsed by hand: argparse alone adds tens of milliseconds to every call, and inference time is a tie-breaker."""
    found = {}
    for i, arg in enumerate(argv):
        for flag in ('--baseline', '--mode'):
            if arg == flag and i + 1 < len(argv):
                found[flag] = argv[i + 1]
            elif arg.startswith(flag + '='):
                found[flag] = arg[len(flag) + 1:]
    if found.get('--baseline') in REFERENCE_MODES:
        return found['--baseline']
    return found.get('--mode', DEFAULT_MODE)


def main(argv=None):
    mode = parse_mode(sys.argv[1:] if argv is None else argv)
    if mode not in REFERENCE_MODES and mode not in CONFIGS:
        mode = DEFAULT_MODE
    try:
        request = json.load(sys.stdin)
    except Exception:
        request = None
    response = decide(request, mode)
    try:
        text = json.dumps(response, allow_nan=False)
    except Exception:          # memory that cannot be serialised: send the actions with empty memory
        response['memory'] = {}
        text = json.dumps(response, allow_nan=False)
    sys.stdout.write(text + '\n')


if __name__ == '__main__':
    main()
