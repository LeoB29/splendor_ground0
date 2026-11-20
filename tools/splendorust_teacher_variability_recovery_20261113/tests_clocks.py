"""Model-free regression checks for post-reboot, non-extending watchdogs."""
from copy import deepcopy
import unittest
from unittest.mock import patch

from . import clocks as k


def original():
    start, diagnostic = 1_000_000., 1_000_400.
    return dict(schema='teacher_variability_original_deadlines_v1',
        original_freeze_sha256=k.ORIGINAL_FREEZE_SHA,
        original_unix_origin=start, global_unix_deadline=start+86400,
        model_unix_deadline=start+64800, diagnostic_unix_deadline=diagnostic+43200,
        original_diagnostic_unix_origin=diagnostic,
        original_monotonic_origin=2_000_000., original_monotonic_deadline=2_086_400.,
        original_diagnostic_monotonic_origin=2_000_400., original_diagnostic_monotonic_deadline=2_043_600.,
        original_bindings=[dict(path='outputs/original/'+name, bytes=100, sha256='a'*64)
                           for name in ('freeze.json','launcher_claim.json','claim.json','diagnostic_owner.json')])


class ClockTests(unittest.TestCase):
    def setUp(self):
        self.original = original()
        self.patch = patch.object(k, 'original_deadlines', return_value=deepcopy(self.original))
        self.patch.start(); self.addCleanup(self.patch.stop)
        self.wall = 1_014_000.
        self.mono = 100.
        self.boot = self.wall-self.mono
        self.root = k.fresh_global_limits(now_mono=self.mono, now_unix=self.wall, boot_unix=self.boot)

    def stage(self, role, seconds=None, delta=0., wall_delta=None):
        return k.stage_limits(self.root, k.ROLE_CAPS[role] if seconds is None else seconds, role,
                              now_mono=self.mono+delta, now_unix=self.wall+(delta if wall_delta is None else wall_delta),
                              boot_unix=self.boot)

    def test_reboot_does_not_reuse_old_uptime(self):
        self.assertLess(self.root['monotonic_origin'], self.original['original_monotonic_origin'])
        self.assertEqual(self.root['unix_deadline'], self.original['global_unix_deadline'])
        self.assertEqual(self.root['monotonic_deadline'], self.mono+self.original['global_unix_deadline']-self.wall)
        self.assertTrue(k.check_deadlines(self.root, now_mono=self.mono+30, now_unix=self.wall+30, boot_unix=self.boot))
        self.assertFalse(self.root['scope']['per_decision_clocks_rebased'])

    def test_diagnostic_preserves_original_stage_cutoff(self):
        stage = self.stage('diagnostic')
        self.assertEqual(stage['unix_deadline'], self.original['diagnostic_unix_deadline'])
        self.assertEqual(stage['monotonic_deadline'], self.mono+self.original['diagnostic_unix_deadline']-self.wall)
        self.assertLess(stage['unix_deadline']-self.wall, 12*3600)

    def test_model_cutoff_uses_original_origin(self):
        for role in ('admission','reproduction'):
            delta = self.original['model_unix_deadline']-self.wall-900
            stage = self.stage(role, delta=delta)
            self.assertEqual(stage['unix_deadline'], self.original['model_unix_deadline'])
            self.assertEqual(stage['monotonic_deadline'], self.mono+self.original['model_unix_deadline']-self.wall)

    def test_analysis_allowed_after_model_cutoff_but_not_global(self):
        delta = self.original['model_unix_deadline']-self.wall+300
        stage = self.stage('analysis', delta=delta)
        self.assertGreater(stage['unix_deadline'], self.original['model_unix_deadline'])
        near_end = self.original['global_unix_deadline']-self.wall-60
        stage = self.stage('acceptance', delta=near_end)
        self.assertEqual(stage['unix_deadline'], self.original['global_unix_deadline'])

    def test_overdue_diagnostic_cannot_restart_stage(self):
        delta = self.original['diagnostic_unix_deadline']-self.wall
        with self.assertRaisesRegex(ValueError, 'already expired'):
            self.stage('diagnostic', delta=delta)

    def test_overdue_models_cannot_restart_allowance(self):
        delta = self.original['model_unix_deadline']-self.wall
        for role in k.MODEL_ROLES:
            with self.assertRaises(ValueError): self.stage(role, delta=delta)

    def test_each_deadline_is_independently_enforced(self):
        stage = self.stage('admission', seconds=100)
        with self.assertRaises(ValueError):
            k.check_deadlines(stage, now_mono=self.mono+100, now_unix=self.wall+99.9, boot_unix=self.boot)
        with self.assertRaises(ValueError):
            k.check_deadlines(stage, now_mono=self.mono+10, now_unix=self.wall+100, boot_unix=self.boot)

    def test_wall_rollback_fails_closed(self):
        with self.assertRaisesRegex(ValueError, 'moved backwards'):
            k.check_deadlines(self.root, now_mono=self.mono+100, now_unix=self.wall+99, boot_unix=self.boot)

    def test_small_sampling_skew_cannot_extend_anchored_cutoff(self):
        stage = self.stage('diagnostic', delta=100, wall_delta=99.9)
        self.assertEqual(stage['monotonic_deadline'], self.mono+self.original['diagnostic_unix_deadline']-self.wall)
        self.assertEqual(stage['unix_deadline'], self.original['diagnostic_unix_deadline'])

    def test_wall_forward_jump_does_not_refund_time(self):
        stage = self.stage('diagnostic', delta=100, wall_delta=200)
        self.assertEqual(stage['monotonic_deadline'], self.mono+self.original['diagnostic_unix_deadline']-self.wall-100)
        with self.assertRaises(ValueError):
            k.check_deadlines(self.root, now_mono=self.mono+100,
                             now_unix=self.original['global_unix_deadline'], boot_unix=self.boot)

    def test_second_reboot_is_rejected(self):
        with self.assertRaises(ValueError):
            k.check_deadlines(self.root, now_mono=1, now_unix=self.wall+100, boot_unix=self.wall+99)
        with self.assertRaisesRegex(ValueError, 'Boot changed'):
            k.check_deadlines(self.root, now_mono=self.mono+100, now_unix=self.wall+100, boot_unix=self.boot+10)

    def test_metadata_mutations_and_extensions_are_rejected(self):
        mutations = [lambda p: p.update(unix_deadline=p['unix_deadline']+1),
                     lambda p: p.update(monotonic_deadline=p['monotonic_deadline']+1),
                     lambda p: p['scope'].update(original_deadlines_extended=True),
                     lambda p: p['original_deadlines'].update(model_unix_deadline=self.wall+64800),
                     lambda p: p['recovery_anchor'].update(unix=self.wall-1)]
        for mutate in mutations:
            trial = deepcopy(self.root); mutate(trial)
            with self.assertRaises(ValueError):
                k.check_deadlines(trial, now_mono=self.mono, now_unix=self.wall, boot_unix=self.boot)

    def test_original_claim_changes_block_stage(self):
        changed = deepcopy(self.original); changed['original_bindings'][0]['sha256'] = 'b'*64
        with patch.object(k, 'original_deadlines', return_value=changed):
            with self.assertRaisesRegex(ValueError, 'claims unchanged'):
                self.stage('analysis')

    def test_fixture_is_short_and_checkable(self):
        fixture = k.fixture_limits(30, now_mono=self.mono, now_unix=self.wall, boot_unix=self.boot)
        self.assertEqual(fixture['role'], 'fixture')
        self.assertEqual(fixture['unix_deadline'], self.wall+30)
        self.assertTrue(k.check_deadlines(fixture, now_mono=self.mono+29, now_unix=self.wall+29, boot_unix=self.boot))
        with self.assertRaises(ValueError):
            k.fixture_limits(31, now_mono=self.mono, now_unix=self.wall, boot_unix=self.boot)

    def test_bounds_roles_and_nonfinite_values_rejected(self):
        for bad in (0, -1, float('inf'), float('nan'), True, 7201):
            with self.assertRaises(ValueError): self.stage('admission', seconds=bad)
        with self.assertRaises(ValueError): self.stage('qualification', seconds=1)
        with self.assertRaises(ValueError):
            k.fresh_global_limits(now_mono=self.mono, now_unix=self.wall, boot_unix=self.original['original_unix_origin']-1)
        with self.assertRaises(ValueError):
            k.fresh_global_limits(now_mono=self.mono, now_unix=self.original['global_unix_deadline'], boot_unix=self.boot)


if __name__ == '__main__':
    unittest.main()
