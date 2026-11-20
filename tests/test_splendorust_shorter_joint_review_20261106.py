"""Independent regressions for the isolated saved-only reporting worker.

Run the actual worker dispatcher in a fresh Python process with tiny receipts
and a stub numerical renderer. No tensor runtime, model or fitting is invoked.
"""
import os
from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]

CHILD = r'''
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import types
from unittest.mock import patch
import psutil
from tools.splendorust_shorter_joint_20261106 import common as c, worker, analysis
from tools import future_training_rules_20261021 as guard

assert not any(k == 'torch' or k.startswith('torch.') for k in sys.modules)
assert 'tools.splendorust_shorter_joint_20261106.training' not in sys.modules
scenario = sys.argv[1]
with tempfile.TemporaryDirectory() as temporary:
    out = Path(temporary) / 'evidence'
    folder = out / 'analysis' / 'worker'
    folder.mkdir(parents=True)
    with patch.object(c, 'OUT', out), patch.object(c, 'PLAN', out / 'plan.json'):
        c.save(c.PLAN, {'fits': c.fits(), 'worker_inputs': []})
        identity = {'pid': os.getpid(), 'created': psutil.Process().create_time()}
        origin, wall = time.monotonic(), time.time()
        claim = {**identity, 'monotonic_origin': origin, 'unix_origin': wall,
                 'plan': c.artifact(c.PLAN)}
        c.save(out / 'claim.json', claim)
        owner = {'identity': identity, 'windows_job_assigned': True, 'fit': None,
                 'mode': 'analysis', 'plan_sha256': c.sha(c.PLAN),
                 'global_claim': c.artifact(out / 'claim.json'),
                 'monotonic_origin': origin, 'unix_origin': wall,
                 'maximum_seconds': 60., 'evaluation_clock': None}
        c.save(folder.parent / 'owner.json', owner)
        audit_passed = scenario != 'audit_failed'
        c.save(out / 'audit' / 'execution.json', {
            'status': 'passed' if audit_passed else 'failed',
            'worker_exit_code': 0 if audit_passed else 1,
            'cleanup_errors': []})
        c.save(out / 'audit' / 'worker' / 'result.json', {
            'status': 'passed' if audit_passed else 'failed'})
        calls = []
        def renderer(plan, folders, output, *, boundary):
            boundary()
            assert output == out / 'report'
            assert set(folders) == {x['fit_id'] for x in c.fits()}
            assert not any(k == 'torch' or k.startswith('torch.') for k in sys.modules)
            assert 'tools.splendorust_shorter_joint_20261106.training' not in sys.modules
            calls.append('saved-only')
            return {'status': 'synthetic_saved_report', 'models': 0}
        if scenario == 'tensor_poisoned':
            sys.modules['torch'] = types.ModuleType('torch')
        argv = ['worker', '--mode', 'analysis', '--plan-sha256', c.sha(c.PLAN),
                '--claim-sha256', c.sha(out / 'claim.json'), '--output', str(folder)]
        with patch.object(sys, 'argv', argv), patch.object(guard, 'verify_runtime_sources', return_value={'fixture': True}) as source_check, patch.object(analysis, 'run_analysis', side_effect=renderer):
            result = worker.main()
        receipt = c.read(folder / 'result.json')
        assert source_check.call_count == 1
        if scenario == 'passed':
            assert result == 0 and receipt['status'] == 'passed'
            assert calls == ['saved-only']
        else:
            assert result == 1 and receipt['status'] == 'failed' and not calls
            assert (folder / 'failure.json').exists()
            expected = ('must not import Torch' if scenario == 'tensor_poisoned'
                        else 'Saved scientific audit passed')
            assert expected in receipt['error']
        assert 'tools.splendorust_shorter_joint_20261106.training' not in sys.modules
print(json.dumps({'scenario': scenario, 'checks_passed': True, 'models': 0}))
'''


class IsolatedAnalysisReviewTests(unittest.TestCase):
    def run_scenario(self, scenario):
        env = os.environ.copy()
        env.pop('KMP_DUPLICATE_LIB_OK', None)
        env.update(PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1',
                   MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                   NUMEXPR_NUM_THREADS='1')
        result = subprocess.run(
            [sys.executable, '-B', '-c', CHILD, scenario], cwd=ROOT, env=env,
            capture_output=True, text=True, timeout=60,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('"checks_passed": true', result.stdout)

    def test_actual_analysis_dispatch_stays_without_tensor_runtime(self):
        self.run_scenario('passed')

    def test_failed_scientific_audit_never_reaches_renderer(self):
        self.run_scenario('audit_failed')

    def test_existing_tensor_runtime_refuses_rendering_without_override(self):
        self.run_scenario('tensor_poisoned')


if __name__ == '__main__':
    unittest.main()
