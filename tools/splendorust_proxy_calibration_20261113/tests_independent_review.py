"""Pinned historical regression evidence through the new scientific checker.

These are model-free checker tests, NOT new timing or player qualification.
Only explicit profile/schema/allowance annotations are projected for the new
checker. Every action, seed, callback, numerical value and search count remains
the original saved value. The historically rejected request stays rejected.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import unittest

from tools import future_training_rules_20261021 as guard
from tools.external_strength_screen_20261018_02 import bridge
from tools.splendorust_decision_quality_20261027_v2 import audit as historical
from . import adapters, qualification

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = {
    'rejected64': ('outputs/splendorust_decision_quality_20261027_01/qualification_rows/0002.json', 51072,
                   'd2789734dfe5f8975a4301c8def8731d8187432fe1ccc89552f088513dd9368d'),
    'accepted64': ('outputs/splendorust_decision_quality_20261027_02/records/0024.json', 76757,
                   'd69ffc1f2cc9d76456c7489b6867ff7b38eddf4d03e8c9a8ca857b89acba0dd0'),
    'accepted256': ('outputs/splendorust_decision_quality_20261027_02/records/0025.json', 205635,
                    'e6f98da269ab8900afebdfa7c7b024e80fad2a43c719931fa96be3ddc35644a8'),
    'accepted1024': ('outputs/splendorust_decision_quality_20261027_02/records/0026.json', 720994,
                     '322981ab35c9cb0f89ac2f764dbc5a7f49dcf3cb18af5784fdf61f417d6de2ca'),
}


def read_fixture(name):
    path, size, sha = FIXTURES[name]; raw = (ROOT/path).read_bytes()
    if len(raw) != size or hashlib.sha256(raw).hexdigest() != sha:
        raise ValueError('Independent historical regression fixture bytes changed')
    return json.loads(raw)


def explicit_checker_projection(row):
    original = row['clock']['result']; result = deepcopy(original); quota = result['simulations_quota']
    profile = adapters.profile(quota)
    fixture_sha = hashlib.sha256(json.dumps(profile, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    result.update(profile_id=profile['profile_id'], profile_sha256=fixture_sha, wrapper_id=profile['wrapper_id'])
    result['diagnostics']['profile_id'] = profile['profile_id']
    result['research'].update(schema=adapters.SCHEMA, internal_seconds=adapters.INTERNAL_SECONDS)
    if historical.scientific_core(result) != historical.scientific_core(original) or result['research']['phases'] != original['research']['phases']:
        raise ValueError('Regression projection changed scientific evidence')
    identity = dict(quota=quota, seed_label='A', profile_id=profile['profile_id'], profile_sha256=fixture_sha,
                    wrapper_id=profile['wrapper_id'], native_binary_sha256=profile['native_binary_sha256'])
    return result, identity


class IndependentScientificRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runtime_sources = guard.verify_runtime_sources()

    def test_retained_old_checker_failure_has_valid_search_statistics(self):
        row = read_fixture('rejected64')
        self.assertEqual(row['clock']['status'], 'illegal'); self.assertFalse(row['clock']['accepted'])
        self.assertIn('root value must use acting-player signed-output conversion', row['clock']['detail'])
        result, identity = explicit_checker_projection(row)
        receipt = qualification.validate_result(result, row['observer'], **identity)
        self.assertEqual(receipt['status'], 'passed')
        self.assertEqual(receipt['simulations'], 64)
        self.assertEqual(read_fixture('rejected64'), row)

    def test_complete_saved64_and256_science_passes_new_checker(self):
        for name in ('accepted64', 'accepted256'):
            row = read_fixture(name); self.assertTrue(row['clock']['accepted'])
            result, identity = explicit_checker_projection(row)
            receipt = qualification.validate_result(result, row['observer'], **identity)
            self.assertEqual(receipt['status'], 'passed'); self.assertEqual(receipt['simulations'], identity['quota'])
            self.assertEqual(result['action'], row['action'])

    def test_saved1024_root_uses_actual_visited_q_and_dynamic_fpu_mean(self):
        row = read_fixture('accepted1024'); result = row['clock']['result']; phase = result['diagnostics']['phases'][0]
        annotation = result['research']['phases'][0]
        raw83 = qualification.validate_callback_capture(phase, annotation['callback_bits'])
        receipt = qualification.validate_root_statistics(annotation['telemetry'], bridge.encode_observation(row['observer']),
            bridge.main_actions(row['observer']), phase['action'], 1024, raw83)
        self.assertLessEqual(receipt['dynamic_fpu_absolute_residual'], receipt['dynamic_fpu_absolute_rounding_bound'])
        self.assertEqual(receipt['initial_network_win_credit'], (raw83[81] + 1.) / 2.)

    def test_corrupted_saved_q_and_callbacks_cannot_pass(self):
        row = read_fixture('accepted256'); original, identity = explicit_checker_projection(row)
        for field in ('unvisited_q', 'aggregate_value', 'callback_bits', 'seed', 'visit_total'):
            result = deepcopy(original); telemetry = result['research']['phases'][0]['telemetry']
            if field == 'unvisited_q':
                index = next(i for i, n in enumerate(telemetry['visits']) if n == 0); telemetry['q'][index] = 0.
            elif field == 'aggregate_value': telemetry['root_value'] = qualification.f32(telemetry['root_value'] + .01)
            elif field == 'callback_bits': result['research']['phases'][0]['callback_bits'][0]['input_float32_sha256'] = '0'*64
            elif field == 'seed': result['research']['seed'] += 1
            else: telemetry['visits'][next(i for i, n in enumerate(telemetry['visits']) if n > 0)] -= 1
            with self.subTest(field=field), self.assertRaises(ValueError):
                qualification.validate_result(result, row['observer'], **identity)

    def test_reference_seeds_do_not_reuse_source_or_control_seeds(self):
        row = read_fixture('accepted64'); observer = row['observer']
        seeds = {s: adapters.observer_seed(observer, s) for s in adapters.SEED_LABELS}
        self.assertEqual(seeds['A'], row['clock']['result']['research']['seed'])
        self.assertEqual(len(set(seeds.values())), 4)
        for label in ('B','R','S'):
            self.assertNotEqual(seeds[label], historical.seed_for(observer, 'B'))


if __name__ == '__main__': unittest.main()
