"""Merge every source file for 5 real (synthetic) members into one cleaned JSON, and compute the worked numbers.

Run from the repo root:  python exploration/build_five_people.py
Writes exploration/five_people.json
"""
import itertools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import kit
import policy

POOL = 'public_05'
IDS = ['syn_0cf9e38720b54f715da4f4930ee5b602', 'syn_4a30986e184de09da4d488e6dc42ec50',
       'syn_6386031efeb54abdfa96637beb180642', 'syn_d1ce45f75d93596c81ae980e72fec48f',
       'syn_f52d998f7b3adeed3355e5e1ddd5221d']
ALIAS = {mid: 'P%d' % (i + 1) for i, mid in enumerate(IDS)}
D = ROOT / 'data' / POOL


def jl(name):
    return [json.loads(line) for line in open(D / name, encoding='utf-8')]


state = json.load(open(D / 'state.json'))
members = {m['member_id']: m for m in jl('members.jsonl')}
quest = {q['member_id']: q for q in jl('questionnaires.jsonl')}
convo = {c['member_id']: c for c in jl('conversations.jsonl')}
intros = jl('introductions.jsonl')
feedback = jl('feedback.jsonl')


def label(mid):
    return ALIAS.get(mid, 'outside:' + mid[4:10])


# ---------------------------------------------------------------- people (one record each, all sources merged)
people = []
for mid in IDS:
    m = members[mid]
    fields = {}
    for k, v in m['fields'].items():
        fields[k] = {'value': v, 'status': m['field_status'][k], 'observed_day': m['field_observed_day'][k]}
    my_intros = [i for i in intros if mid in (i['user_a'], i['user_b'])]
    my_fb = [e for e in feedback if e['member_id'] == mid]
    people.append({
        'alias': ALIAS[mid], 'member_id': mid, 'pool_id': m['pool_id'],
        'profile': {'age': m['age'], 'gender': m['gender'], 'zone': m['zone'],
                    'arrived_day': m['arrived_day'], 'available_at_day_30': m['available']},
        'hard_fields': {k: fields[k] for k in kit.HARD},
        'soft_fields': {k: fields[k] for k in kit.SOFT},
        'questionnaire_text': quest[mid]['questionnaire_answers'],
        'conversation': [(x['role'], x['content']) for x in convo[mid]['messages']] if mid in convo else None,
        'introduction_ids': [i['introduction_id'] for i in my_intros],
        'own_feedback_events': [{k: e[k] for k in ('introduction_id', 'event', 'value', 'occurred_day', 'observed_day', 'missing_reason')}
                                for e in my_fb],
    })

# ---------------------------------------------------------------- introductions that touch any of the five
by_intro = {}
for e in feedback:
    by_intro.setdefault(e['introduction_id'], []).append(e)
intro_out = []
for i in intros:
    if i['user_a'] in ALIAS or i['user_b'] in ALIAS:
        evs = sorted(by_intro.get(i['introduction_id'], []), key=lambda e: (e['observed_day'], e['event']))
        intro_out.append({
            'introduction_id': i['introduction_id'], 'a': label(i['user_a']), 'b': label(i['user_b']),
            'assigned_day': i['assigned_day'], 'response_deadline_day': i['response_deadline_day'],
            'events': [{'who': label(e['member_id']) if e['member_id'] else 'pair', 'event': e['event'],
                        'value': e['value'], 'occurred_day': e['occurred_day'], 'observed_day': e['observed_day'],
                        'missing_reason': e['missing_reason']} for e in evs]})

# ---------------------------------------------------------------- learned response estimates (what policy.propensities does)
prop_all = policy.propensities(state)
seen = {}
for e in state['feedback']:
    if e['event'] == 'introduction_response' and e['member_id'] in ALIAS:
        s = seen.setdefault(e['member_id'], [0, 0, 0])
        s[0] += 1
        if e['value'] is not None:
            s[1] += 1
            s[2] += e['value'] == 'yes'
for p in people:
    t, a, y = seen.get(p['member_id'], (0, 0, 0))
    p['response_history'] = {'introductions_answered_or_timed_out': t, 'answered': a, 'yes': y,
                             'shrunk_p_answer': round((a + 3 * 0.78) / (t + 3), 4),
                             'shrunk_p_yes_given_answer': round((y + 3 * 0.47) / (a + 3), 4),
                             'p_yes': round(prop_all[p['member_id']], 4)}

# ---------------------------------------------------------------- pairwise table
by_id = {m['member_id']: m for m in state['members']}
pairs = []
for a, b in itertools.combinations(IDS, 2):
    e = kit.eligibility(by_id[a], by_id[b])
    comp = {}
    for k in kit.SOFT:
        va, vb = by_id[a]['fields'][k], by_id[b]['fields'][k]
        comp[k] = {'a': va, 'b': vb, 'counts': va is not None and va == vb}
    sim4 = sum(comp[k]['counts'] for k in policy.SOFT_SIM)
    pa, pb = prop_all[a], prop_all[b]
    pairs.append({
        'pair': [ALIAS[a], ALIAS[b]], 'eligibility': e['status'], 'fail_reasons': e['reasons'],
        'soft_comparison': comp, 'soft_score_7': sum(c['counts'] for c in comp.values()),
        'soft_matches_4': sim4,
        'p_yes_a': round(pa, 4), 'p_yes_b': round(pb, 4), 'p_mutual': round(pa * pb, 4),
        'learned_weight': round(pa * pb * (1 + 0.15 * sim4), 4)})

# ---------------------------------------------------------------- matchings among the five
ps = [p for p in pairs if p['eligibility'] == 'feasible']
id_of = {v: k for k, v in ALIAS.items()}
w_soft = [(p['soft_score_7'], id_of[p['pair'][0]], id_of[p['pair'][1]]) for p in ps]
w_soft_eps = [(p['soft_score_7'] + 1, id_of[p['pair'][0]], id_of[p['pair'][1]]) for p in ps]
w_learn = [(0.05 + p['learned_weight'], id_of[p['pair'][0]], id_of[p['pair'][1]]) for p in ps]


def show(matching):
    return [[ALIAS[a], ALIAS[b]] for a, b in matching]


results = {
    'greedy_by_soft_score': show(policy.greedy_matching(w_soft)),
    'global_by_soft_score_plus_1': show(policy.global_matching(w_soft_eps)),
    'greedy_by_learned_weight': show(policy.greedy_matching(w_learn)),
    'global_by_learned_weight': show(policy.global_matching(w_learn)),
}

out = {
    'meta': {
        'description': 'Five synthetic members from public_05 with every source file merged: members.jsonl (fields, status, '
                       'observed day), questionnaires.jsonl (text answers), conversations.jsonl (scripted chat), '
                       'introductions.jsonl and feedback.jsonl. Snapshot is day 30.',
        'pool_id': POOL, 'snapshot_day': 30, 'hard_fields': kit.HARD, 'soft_fields': kit.SOFT,
        'soft_fields_used_in_learned_score': policy.SOFT_SIM,
        'alias_to_member_id': {v: k for k, v in ALIAS.items()},
        'null_means': 'unknown, never zero or no'},
    'people': people, 'introductions_touching_these_people': intro_out,
    'pairs': pairs, 'matchings': results}
(ROOT / 'exploration' / 'five_people.json').write_text(json.dumps(out, indent=1), encoding='utf-8')

# ---------------------------------------------------------------- printed summary used to write the document
for p in people:
    pr = p['profile']
    print(p['alias'], pr, [(k, v['value']) for k, v in p['soft_fields'].items()])
    print('   hard:', {k: v['value'] for k, v in p['hard_fields'].items()})
    print('   hist:', p['response_history'])
print()
for p in pairs:
    print(p['pair'], p['eligibility'], p['fail_reasons'], 'soft7=', p['soft_score_7'], 'soft4=', p['soft_matches_4'],
          'pm=', p['p_mutual'], 'w=', p['learned_weight'])
    if p['eligibility'] == 'feasible':
        print('     ', {k: (c['a'], c['b'], c['counts']) for k, c in p['soft_comparison'].items()})
print()
print(json.dumps(results, indent=1))
print()
for i in intro_out:
    print(i['introduction_id'], i['a'], i['b'], 'day', i['assigned_day'])
    for e in i['events']:
        print('    ', e)
