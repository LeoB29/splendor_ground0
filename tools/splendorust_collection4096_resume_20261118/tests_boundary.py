"""Model-free failure-boundary tests against the preserved shutdown evidence."""
from copy import deepcopy
import unittest
from . import boundary


class SavedBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = boundary.inspect_boundary()

    def test_exact_saved_failure(self):
        report=self.report
        self.assertEqual(report['completed_games'],672)
        self.assertEqual(report['completed_replays'],32)
        self.assertEqual(report['selected_replays'][32]['game_id'],'g0666')
        self.assertEqual(report['selected_replays'][32]['ply'],40)
        self.assertEqual(len(report['selected_replays'][32:]),16)
        self.assertTrue(boundary.validate_boundary(report))

    def test_cannot_relabel_failed_attempt_complete(self):
        report=deepcopy(self.report);report['original_attempt_incomplete']=False
        with self.assertRaisesRegex(ValueError,'fixed failure inventory'):
            boundary.validate_boundary(report)

    def test_cannot_omit_the_interrupted_verification(self):
        report=deepcopy(self.report);report['interrupted_verification_decisions']=0
        with self.assertRaisesRegex(ValueError,'fixed failure inventory'):
            boundary.validate_boundary(report)

    def test_tampered_original_artifact_is_rejected(self):
        report=deepcopy(self.report);report['artifacts'][0]['sha256']='0'*64
        with self.assertRaises(ValueError):boundary.validate_boundary(report)


if __name__=='__main__':unittest.main()
