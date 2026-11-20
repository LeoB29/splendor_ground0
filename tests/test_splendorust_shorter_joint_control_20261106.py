import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from tools.splendorust_shorter_joint_20261106 import common as c, control
from tools.splendorust_shorter_joint_20261106.worker import Boundary


class DeadlineAndPreservationTests(unittest.TestCase):
    def test_either_clock_at_deadline_stops(self):
        control.check_clock(100., 1000., 20., 119.9, 1019.9, 'test')
        for mono, wall in [(120., 1001.), (101., 1020.), (99., 1001.), (101., 999.)]:
            with self.subTest(mono=mono, wall=wall), self.assertRaises(ValueError):
                control.check_clock(100., 1000., 20., mono, wall, 'test')

    def test_evidence_is_exclusive_and_cannot_escape(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)/'new_evidence'
            with patch.object(c, 'OUT', root):
                c.save(root/'receipt.json', {'first': True})
                with self.assertRaises(FileExistsError): c.save(root/'receipt.json', {'replaced': True})
                with self.assertRaises(ValueError): c.save(root/'../outside.json', {})
                with self.assertRaises(ValueError): c.atomic_json(root/'receipt.json', {})
                self.assertEqual(json.loads((root/'receipt.json').read_text()), {'first': True})

    def test_input_byte_mutation_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'source.json'
            path.write_text('first')
            descriptor = c.artifact(path)
            c.verify([descriptor])
            path.write_text('other')
            with self.assertRaises(ValueError): c.verify([descriptor])

    def test_only_two_prespecified_broad_joint_fits(self):
        self.assertEqual(c.fits(), [dict(fit_id=f'r{rep}_broad_joint', repetition=rep,
                          breadth='broad', scope='joint') for rep in (0, 1)])

    def test_freeze_cannot_promote_or_launch_other_games(self):
        spec = control.contract()
        self.assertEqual(spec['new_games'], 0)
        self.assertEqual(spec['maximum_optimizer_steps'], 2*96+2+2)
        self.assertEqual(spec['per_loss_gradient_probes'], 0)
        self.assertFalse(spec['automatic_promotion'])
        self.assertFalse(spec['resource_forecasts_may_stop'])
        self.assertEqual(spec['available_ram_reserve_bytes'], 0)

    def test_torch_absolute_path_descriptors_are_not_omitted(self):
        previous = c.read(c.ROOT/'outputs/splendorust_policy_head_20261104_01/freeze.json')
        entries = control.runtime_bindings(previous)
        actual = {(x['path'], x['bytes'], x['sha256']) for x in entries}
        self.assertGreater(len(previous['torch_runtime']), 1)
        for item in previous['torch_runtime']:
            self.assertIn((item['absolute_path'], item['bytes'], item['sha256']), actual)

    def test_worker_does_not_reset_aggregate_evaluator_clock(self):
        claim = {'monotonic_origin': 0., 'unix_origin': 1000.}
        owner = {'monotonic_origin': 7200., 'unix_origin': 8200., 'maximum_seconds': 100.,
                 'evaluation_clock': {'monotonic_origin': 1., 'unix_origin': 1001.}}
        boundary = Boundary(claim, owner, Path('unused'), True)
        boundary.last_deep = 7201.
        with patch('time.monotonic', return_value=7201.), patch('time.time', return_value=8201.):
            with self.assertRaisesRegex(ValueError, 'aggregate evaluation'): boundary()


if __name__ == '__main__': unittest.main()
