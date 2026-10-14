"""Domain-knowledge features suggested by the online-dating literature, built only from the observation.

Sources of the ideas (see DOMAIN_FEATURES.md):
  * desirability hierarchy (Bruch & Newman 2018): how others treat you, adjusted for how generous they are
  * reciprocal recommenders (Xia et al. 2015): interest and attractiveness measured from interactions
  * stated-preference demand / selectivity (Hitsch, Hortacsu & Ariely 2010; Fisman et al. 2006)
  * main effects beat pair effects (Joel, Eastwick & Finkel 2017): focus on the actor's and partner's own tendencies
  * behaviour over stated preference; response speed as engagement
  * ordinal / interaction versions of soft-trait agreement; recency-weighted history; segment priors
"""
import policy

DOMAIN_GROUPS = {
    'desirability': ['a_desir', 'p_desir', 'desir_gap', 'desir_absgap'],
    'demand_supply': ['a_demand', 'p_demand', 'demand_gap', 'a_supply', 'p_supply', 'mutual_pool_fit'],
    'range_fit': ['p_age_centrality', 'a_age_centrality', 'mutual_age_centrality'],
    'latency': ['a_latency_days', 'a_fast_frac', 'p_latency_days'],
    'trait_logic': ['pace_dist', 'space_dist', 'both_long_term', 'both_exploring', 'goal_long_vs_explore',
                    'emotional_dist', 'trait_agree_weighted'],
    'recency_segment': ['a_ewma_yes', 'a_gender_rate', 'a_ageband_rate', 'a_yes_vs_segment'],
}
DOMAIN_NAMES = [n for names in DOMAIN_GROUPS.values() for n in names]
DOMAIN_GROUP_OF = {n: g for g, names in DOMAIN_GROUPS.items() for n in names}
PACE = {'slow': 0, 'steady': 1, 'quick': 2}
SPACE = {'limited': 0, 'moderate': 1, 'ample': 2}


def _accepts(x, y):
    """Does x's stated age / gender / zone preference accept y?"""
    fx = x['fields']
    if fx['who_to_meet'] is None or fx['age_min'] is None or fx['acceptable_zones'] is None:
        return None
    return (y['gender'] in fx['who_to_meet'] and fx['age_min'] <= y['age'] <= fx['age_max']
            and y['zone'] in fx['acceptable_zones'])


def domain_context(state):
    members = state['members']
    complete = [m for m in members if all(m['fields'][k] is not None for k in
                ('who_to_meet', 'age_min', 'age_max', 'acceptable_zones'))]
    demand, supply = {}, {}
    for m in members:
        accepted_by = sum(1 for c in complete if c is not m and _accepts(c, m))
        demand[m['member_id']] = accepted_by / max(1, len(complete))
    for m in complete:
        accepts = sum(1 for o in members if o is not m and _accepts(m, o))
        supply[m['member_id']] = accepts / max(1, len(members))
    ends = {i['introduction_id']: i for i in state['introductions']}
    gen = {}
    answers = {}
    for e in sorted((e for e in state['feedback'] if e['event'] == 'introduction_response' and e['member_id']),
                    key=lambda e: (e['observed_day'], e['introduction_id'])):
        answers.setdefault(e['member_id'], []).append(e)
    for mid, es in answers.items():
        ans = [e for e in es if e['value'] is not None]
        gen[mid] = (sum(e['value'] == 'yes' for e in ans) + 3 * policy.PRIOR_YES) / (len(ans) + 3)
    desir_sum, desir_n = {}, {}
    for mid, es in answers.items():
        inv = ends
        for e in es:
            if e['value'] is None:
                continue
            i = inv.get(e['introduction_id'])
            if not i:
                continue
            other = i['user_b'] if e['member_id'] == i['user_a'] else i['user_a']
            desir_sum[other] = desir_sum.get(other, 0.0) + ((e['value'] == 'yes') - gen.get(mid, policy.PRIOR_YES))
            desir_n[other] = desir_n.get(other, 0) + 1
    latency = {}
    for mid, es in answers.items():
        lats = []
        for e in es:
            if e['value'] is not None and e['introduction_id'] in ends:
                lats.append(e['observed_day'] - ends[e['introduction_id']]['assigned_day'])
        latency[mid] = lats
    by_gender, by_band = {}, {}
    byid = {m['member_id']: m for m in members}
    for mid, es in answers.items():
        m = byid.get(mid)
        if not m:
            continue
        band = 0 if m['age'] < 28 else (1 if m['age'] < 36 else 2)
        for table, key in ((by_gender, m['gender']), (by_band, band)):
            t = table.setdefault(key, [0, 0])
            for e in es:
                if e['value'] is not None:
                    t[0] += 1
                    t[1] += e['value'] == 'yes'
    return {'demand': demand, 'supply': supply, 'gen': gen, 'desir_sum': desir_sum, 'desir_n': desir_n,
            'latency': latency, 'answers': answers, 'by_gender': by_gender, 'by_band': by_band}


def _desir(ctx, mid):
    return ctx['desir_sum'].get(mid, 0.0) / (ctx['desir_n'].get(mid, 0) + 3)


def _lat(ctx, mid):
    l = ctx['latency'].get(mid, [])
    return (sum(l) + 3 * 4.0) / (len(l) + 3), (sum(1 for v in l if v <= 2) + 3 * 0.25) / (len(l) + 3)


def domain_features(actor, partner, ctx):
    a, p = actor['member_id'], partner['member_id']
    fa, fb = actor['fields'], partner['fields']
    x = {}
    x['a_desir'], x['p_desir'] = _desir(ctx, a), _desir(ctx, p)
    x['desir_gap'] = x['p_desir'] - x['a_desir']
    x['desir_absgap'] = abs(x['desir_gap'])
    x['a_demand'], x['p_demand'] = ctx['demand'].get(a, 0.0), ctx['demand'].get(p, 0.0)
    x['demand_gap'] = x['p_demand'] - x['a_demand']
    x['a_supply'], x['p_supply'] = ctx['supply'].get(a, 0.0), ctx['supply'].get(p, 0.0)
    x['mutual_pool_fit'] = x['a_demand'] * x['p_demand']

    def centrality(x_, y_):
        lo, hi = x_['fields']['age_min'], x_['fields']['age_max']
        if lo is None or hi is None or hi <= lo:
            return 0.5
        return 1.0 - abs((y_['age'] - (lo + hi) / 2.0) / ((hi - lo) / 2.0))
    x['p_age_centrality'], x['a_age_centrality'] = centrality(actor, partner), centrality(partner, actor)
    x['mutual_age_centrality'] = x['p_age_centrality'] * x['a_age_centrality']
    x['a_latency_days'], x['a_fast_frac'] = _lat(ctx, a)
    x['p_latency_days'] = _lat(ctx, p)[0]
    pa, pb = fa['relationship_pace'], fb['relationship_pace']
    x['pace_dist'] = abs(PACE[pa] - PACE[pb]) if pa and pb else 1.0
    sa, sb = fa['space_for_relationship'], fb['space_for_relationship']
    x['space_dist'] = abs(SPACE[sa] - SPACE[sb]) if sa and sb else 1.0
    ga, gb = fa['relationship_goal'], fb['relationship_goal']
    x['both_long_term'] = float(ga == 'long_term' and gb == 'long_term')
    x['both_exploring'] = float(ga == 'exploring' and gb == 'exploring')
    x['goal_long_vs_explore'] = float(ga is not None and gb is not None and ga != gb)
    ea, eb = fa['emotional_availability'], fb['emotional_availability']
    x['emotional_dist'] = float(ea is not None and eb is not None and ea != eb)
    x['trait_agree_weighted'] = (2.0 * (ga == gb and ga is not None) + 1.0 * (pa == pb and pa is not None)
                                 - 2.0 * x['goal_long_vs_explore'] - 1.0 * (pa is not None and pb is not None and pa != pb))
    ans = [e for e in ctx['answers'].get(a, []) if e['value'] is not None]
    e_ = policy.PRIOR_YES
    for e in ans:
        e_ = 0.6 * e_ + 0.4 * (e['value'] == 'yes')
    x['a_ewma_yes'] = e_
    g = ctx['by_gender'].get(actor['gender'], [0, 0])
    x['a_gender_rate'] = (g[1] + 20 * policy.PRIOR_YES) / (g[0] + 20)
    band = 0 if actor['age'] < 28 else (1 if actor['age'] < 36 else 2)
    b = ctx['by_band'].get(band, [0, 0])
    x['a_ageband_rate'] = (b[1] + 20 * policy.PRIOR_YES) / (b[0] + 20)
    own = (sum(e['value'] == 'yes' for e in ans) + 3 * policy.PRIOR_YES) / (len(ans) + 3)
    x['a_yes_vs_segment'] = own - x['a_gender_rate']
    return [float(x[n]) for n in DOMAIN_NAMES]
