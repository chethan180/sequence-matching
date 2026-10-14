"""Checks for the submitted policy: every response must be valid, whatever happens inside.

One invalid assessed episode makes an entry ineligible, so these tests concentrate on the safety layer:
the validators, the fallbacks, damaged requests, and complete episodes run against the simulator's own checks.
"""
import copy
import json
import random
import sys
import unittest
from evaluate import invoke, validate_response
from kit import SOFT, Simulator, eligibility, generate
import policy
from smp import pipeline, safety


def state_on_day(seed=5, day=12, variant='development'):
    sim = Simulator(generate(seed, 200, 'evaluation', variant))
    for _ in range(day):
        sim.resolve_asks(policy.decide({'phase': 'ask', 'state': sim.observe(), 'memory': None})['asks'])
        sim.advance(policy.decide({'phase': 'match', 'state': sim.observe(), 'memory': None})['pairs'])
    return sim, sim.observe()


def well_formed(test, response, phase):
    key = 'asks' if phase == 'ask' else 'pairs'
    test.assertEqual(set(response), {key, 'memory'})
    validate_response(response, phase)                       # the evaluator's own shape check
    json.dumps(response, allow_nan=False)


class ValidatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sim, cls.state = state_on_day()
        cls.members = cls.state['members']
        cls.available = [m for m in cls.members if m['available']]
        cls.busy = [m for m in cls.members if not m['available']]

    def test_asks_only_valid_ones_survive_in_order(self):
        a, b, c = [m['member_id'] for m in self.available[:3]]
        asks = [{'member_id': a, 'field': 'constraints'},
                {'member_id': 'nobody', 'field': 'constraints'},            # unknown member
                {'member_id': self.busy[0]['member_id'], 'field': SOFT[0]},  # unavailable member
                {'member_id': b, 'field': 'not_a_field'},                    # unknown field
                {'member_id': a, 'field': 'constraints'},                    # duplicate
                {'member_id': b, 'field': SOFT[0]},
                {'member_id': c},                                            # malformed
                'junk']
        self.assertEqual(safety.validate_asks(self.state, asks),
                         [{'member_id': a, 'field': 'constraints'}, {'member_id': b, 'field': SOFT[0]}])

    def test_asks_never_exceed_the_budget(self):
        ids = [m['member_id'] for m in self.available[:8]]
        kept = safety.validate_asks(self.state, [{'member_id': i, 'field': 'constraints'} for i in ids])
        self.assertEqual(len(kept), self.state['ask_budget_remaining'] // 3)
        state = dict(self.state, ask_budget_remaining=4)
        kept = safety.validate_asks(state, [{'member_id': ids[0], 'field': 'constraints'},
                                            {'member_id': ids[1], 'field': 'constraints'},
                                            {'member_id': ids[2], 'field': SOFT[1]}])
        self.assertEqual(sum(3 if k['field'] == 'constraints' else 1 for k in kept), 4)
        copy.deepcopy(self.sim).resolve_asks(safety.validate_asks(self.state, [{'member_id': i, 'field': f}
                                                                               for i in ids for f in ('constraints', SOFT[0])]))

    def test_pairs_only_valid_ones_survive(self):
        edges = policy.feasible_edges(self.state)
        self.assertTrue(edges)
        a, b = edges[0][0]['member_id'], edges[0][1]['member_id']
        infeasible = next([x['member_id'], y['member_id']] for x in self.available for y in self.available
                          if x is not y and eligibility(x, y)['status'] != 'feasible')
        done = self.state['introductions'][0]
        pairs = [[a, b],
                 [b, a],                                           # the same people again
                 infeasible,
                 [a, a], [a], 'junk', [a, 7],
                 [self.busy[0]['member_id'], a],                    # unavailable
                 [done['user_a'], done['user_b']],                  # already introduced
                 ['nobody', a]]
        self.assertEqual(safety.validate_pairs(self.state, pairs), [[a, b]])

    def test_validated_batch_is_accepted_by_the_simulator(self):
        rng = random.Random(1)
        ids = [m['member_id'] for m in self.members]
        junk = [[rng.choice(ids), rng.choice(ids)] for _ in range(400)]
        kept = safety.validate_pairs(self.state, junk)
        copy.deepcopy(self.sim).advance(kept)                       # raises ValueError if any pair were invalid


class FallbackTests(unittest.TestCase):
    def setUp(self):
        self.sim, self.state = state_on_day(day=6)
        self.real = pipeline.decide

    def tearDown(self):
        pipeline.decide = self.real

    def test_internal_error_falls_back_to_the_reference_rule(self):
        def broken(request, cfg):
            raise RuntimeError('boom')
        pipeline.decide = broken
        for phase in ('ask', 'match'):
            request = {'phase': phase, 'state': self.state, 'memory': None}
            response = policy.decide(request)
            well_formed(self, response, phase)
            key = 'asks' if phase == 'ask' else 'pairs'
            self.assertEqual(response[key], policy.decide(request, 'greedy')[key])

    def test_invalid_output_is_repaired(self):
        ids = [m['member_id'] for m in self.state['members']]

        def nonsense(request, cfg):
            if request['phase'] == 'ask':
                return {'asks': [{'member_id': i, 'field': 'constraints'} for i in ids], 'memory': {}}
            return {'pairs': [[a, b] for a in ids[:40] for b in ids[:40]], 'memory': {}}
        pipeline.decide = nonsense
        sim = copy.deepcopy(self.sim)
        sim.resolve_asks(policy.decide({'phase': 'ask', 'state': sim.observe(), 'memory': None})['asks'])
        sim.advance(policy.decide({'phase': 'match', 'state': sim.observe(), 'memory': None})['pairs'])

    def test_damaged_requests_never_raise(self):
        for request in (None, {}, [], 'x', {'phase': 'match'}, {'phase': 'ask', 'state': None},
                        {'phase': 'match', 'state': {}}, {'phase': 'nonsense', 'state': self.state}):
            response = policy.decide(request)
            phase = 'match' if isinstance(request, dict) and request.get('phase') == 'match' else 'ask'
            well_formed(self, response, phase)

    def test_corrupted_observations_never_raise(self):
        rng = random.Random(3)
        for trial in range(120):
            state = copy.deepcopy(self.state)
            for _ in range(rng.randint(1, 4)):
                member = rng.choice(state.get('members') or [{}])
                fields = member.get('fields') if isinstance(member.get('fields'), dict) else {}
                choice = rng.randrange(9)
                if choice == 0:
                    member.pop(rng.choice(['fields', 'field_status', 'age', 'zone', 'available', 'gender']), None)
                elif choice == 1 and fields:
                    fields[rng.choice(list(fields))] = rng.choice(['??', 123, [], None, {'a': 1}])
                elif choice == 2:
                    state.pop(rng.choice(['feedback', 'introductions', 'members', 'day', 'ask_budget_remaining']), None)
                elif choice == 3:
                    state.setdefault('feedback', []).append({'event': 'introduction_response'})
                elif choice == 4:
                    state['ask_budget_remaining'] = rng.choice([-5, 'x', None, 10 ** 9])
                elif choice == 5:
                    state.setdefault('introductions', []).append({'introduction_id': 'x'})
                elif choice == 6 and fields:
                    fields.pop(rng.choice(list(fields)))
                elif choice == 7:
                    member['field_status'] = {}
                else:
                    state.setdefault('members', []).append({'member_id': 'ghost'})
            for phase in ('ask', 'match'):
                well_formed(self, policy.decide({'phase': phase, 'state': state, 'memory': None}), phase)


class DamagedRecordTests(unittest.TestCase):
    def test_one_damaged_member_does_not_stop_the_others(self):
        sim, state = state_on_day(day=6)
        clean_pairs = policy.decide({'phase': 'match', 'state': state, 'memory': None})['pairs']
        self.assertTrue(clean_pairs)
        in_pairs = {i for p in clean_pairs for i in p}
        victim = next(m for m in state['members'] if m['available'] and m['member_id'] not in in_pairs)
        for damage in ('fields', 'age', 'zone'):
            broken = copy.deepcopy(state)
            target = next(m for m in broken['members'] if m['member_id'] == victim['member_id'])
            target.pop(damage)
            pairs = policy.decide({'phase': 'match', 'state': broken, 'memory': None})['pairs']
            self.assertTrue(pairs)
            self.assertNotIn(victim['member_id'], [i for p in pairs for i in p])
            asks = policy.decide({'phase': 'ask', 'state': broken, 'memory': None})['asks']
            self.assertNotIn(victim['member_id'], [a['member_id'] for a in asks])
            copy.deepcopy(sim).advance(pairs)

    def test_well_formed_state_is_passed_through_untouched(self):
        from smp.normalise import clean_state
        _, state = state_on_day(day=3)
        cleaned, dropped = clean_state(state)
        self.assertIs(cleaned, state)
        self.assertEqual(dropped, 0)


class EpisodeTests(unittest.TestCase):
    def test_empty_population(self):
        state = Simulator(generate(42, 0)).observe()
        self.assertEqual(policy.decide({'phase': 'ask', 'state': state, 'memory': None}), {'asks': [], 'memory': {}})
        self.assertEqual(policy.decide({'phase': 'match', 'state': state, 'memory': None}), {'pairs': [], 'memory': {}})

    def test_full_episodes_are_valid_and_need_no_repair(self):
        """The simulator raises on any invalid action, and the safety layer should have nothing to do."""
        before = len(safety.EVENTS)
        for variant in ('development', 'sparse', 'cold_start'):
            sim = Simulator(generate(77, 200, 'evaluation', variant))
            for _ in range(60):
                sim.resolve_asks(policy.decide({'phase': 'ask', 'state': sim.observe(), 'memory': None})['asks'])
                sim.advance(policy.decide({'phase': 'match', 'state': sim.observe(), 'memory': None})['pairs'])
            self.assertGreater(len(sim.introductions), 0)
        self.assertEqual(len(safety.EVENTS), before)

    def test_decisions_are_reproducible(self):
        _, state = state_on_day(day=9)
        for phase in ('ask', 'match'):
            request = {'phase': phase, 'state': state, 'memory': None}
            self.assertEqual(policy.decide(copy.deepcopy(request)), policy.decide(copy.deepcopy(request)))

    def test_real_process_without_arguments_is_the_submitted_policy(self):
        """The container entrypoint passes no arguments."""
        _, state = state_on_day(day=4)
        for phase in ('ask', 'match'):
            request = {'schema_version': '1.0.0', 'phase': phase, 'state': state, 'memory': None}
            response, seconds = invoke([sys.executable, 'policy.py'], request)
            self.assertEqual(response, policy.decide(request))
            self.assertLess(seconds, 10)

    def test_unknown_mode_falls_back_to_the_default(self):
        request = {'schema_version': '1.0.0', 'phase': 'match', 'state': Simulator(generate(42, 0)).observe(), 'memory': None}
        response, _ = invoke([sys.executable, 'policy.py', '--mode', 'does_not_exist'], request)
        self.assertEqual(response, {'pairs': [], 'memory': {}})


if __name__ == '__main__':
    unittest.main()
