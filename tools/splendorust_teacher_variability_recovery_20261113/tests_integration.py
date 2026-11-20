"""Model-free counterexamples for recovery custody and single-retry admission."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from . import common as c, control, worker


class RetryCustodyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='teacher_recovery_contract_')
        self.root = Path(self.temp.name)
        self.patch = patch.multiple(c, OUT=self.root, PLAN=self.root/'freeze.json',
            MATRIX=self.root/'matrix.json', SUFFIX=self.root/'suffix.json')
        self.patch.start()
        self.identity = dict(pid=12001, created=1794600100.25)
        self.specs = [dict(request_id='original_boundary', quota=2048, pool=3, mode='split'),
                      dict(request_id='original_following', quota=2048, pool=3, mode='split')]
        self.write(c.PLAN, {'fixed': 'recovery-freeze'})
        self.write(c.MATRIX, [{}]*c.PREFIX_COUNT+self.specs)
        self.write(c.SUFFIX, self.specs)
        self.write(self.root/'jobs/diagnostic/owner.json', dict(identity=self.identity))
        self.token = dict(request=self.specs[0], original_request_index=c.PREFIX_COUNT,
            new_allowance_seconds=120, original_attempt_remains_incomplete=True,
            freeze_sha256=c.sha(c.PLAN), identity=self.identity, time='2026-11-14T00:00:00+00:00')
        self.token_path = self.root/'ambiguous_retry_consumed.json'
        self.write(self.token_path,self.token)
        self.dispatch = self.root/'jobs/diagnostic/worker/dispatch'
        self.rows = [dict(spec=s, role='diagnostic', deadline_seconds=120,
            identity=self.identity, freeze_sha256=c.sha(c.PLAN), time='2026-11-14T00:00:01+00:00',
            monotonic_before_dispatch=100.+i) for i,s in enumerate(self.specs)]
        for i,row in enumerate(self.rows): self.write(self.dispatch/f'{i:05d}.json',row)

    def tearDown(self):
        self.patch.stop(); self.temp.cleanup()

    @staticmethod
    def write(path,value):
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value),encoding='utf-8')

    def test_exact_token_and_dispatch_sequence_pass(self):
        result=worker.check_retry({},lambda:None)
        self.assertTrue(result['only_boundary_request_may_repeat'])
        self.assertTrue(result['original_attempt_incomplete'])
        self.assertEqual(result['consumption'],c.artifact(self.token_path))

    def test_consumption_receipt_cannot_be_overwritten(self):
        before=self.token_path.read_bytes()
        with self.assertRaises(FileExistsError):c.save(self.token_path,self.token)
        self.assertEqual(before,self.token_path.read_bytes())

    def test_token_wrong_freeze_or_boundary_rejected(self):
        for change in (dict(freeze_sha256='0'*64),dict(original_request_index=c.PREFIX_COUNT+1),
                       dict(request=self.specs[1]),dict(new_allowance_seconds=121),
                       dict(original_attempt_remains_incomplete=False)):
            with self.subTest(change=change):
                self.write(self.token_path,dict(self.token,**change))
                with self.assertRaises(ValueError):worker.check_retry({},lambda:None)

    def test_token_must_precede_first_dispatch(self):
        self.write(self.token_path,dict(self.token,time='2026-11-14T00:00:02+00:00'))
        with self.assertRaises(ValueError):worker.check_retry({},lambda:None)

    def test_reused_pid_with_different_birth_rejected(self):
        for target in ('token','dispatch'):
            with self.subTest(target=target):
                wrong=dict(self.identity,created=self.identity['created']+1)
                self.write(self.token_path,dict(self.token,identity=wrong) if target=='token' else self.token)
                self.write(self.dispatch/'00001.json',dict(self.rows[1],identity=wrong) if target=='dispatch' else self.rows[1])
                with self.assertRaises(ValueError):worker.check_retry({},lambda:None)

    def test_wrong_dispatch_spec_clock_role_or_freeze_rejected(self):
        for change in (dict(spec=self.specs[0]),dict(deadline_seconds=121),
                       dict(role='admission'),dict(freeze_sha256='f'*64)):
            with self.subTest(change=change):
                self.write(self.dispatch/'00001.json',dict(self.rows[1],**change))
                with self.assertRaises(ValueError):worker.check_retry({},lambda:None)

    def test_missing_or_extra_dispatch_rejected(self):
        missing=self.dispatch/'00001.json';missing.unlink()
        with self.assertRaises(ValueError):worker.check_retry({},lambda:None)
        self.write(missing,self.rows[1]);self.write(self.dispatch/'00002.json',self.rows[1])
        with self.assertRaises(ValueError):worker.check_retry({},lambda:None)

    def test_writes_outside_recovery_refused(self):
        outside=self.root.parent/(self.root.name+'_outside.json')
        with self.assertRaises(ValueError):c.save(outside,{'bad':True})
        self.assertFalse(outside.exists())

    def test_duplicate_json_keys_cannot_disguise_evidence(self):
        malformed=self.root/'malformed.json';malformed.write_text('{"allowed":true,"allowed":false}')
        with self.assertRaises(ValueError):c.read(malformed)


class CleanupIntegrationTests(unittest.TestCase):
    def test_cleanup_waits_for_exact_absence(self):
        instant=[0.]
        def tick(seconds):instant[0]+=seconds
        with patch.object(control.time,'monotonic',side_effect=lambda:instant[0]), \
             patch.object(control.time,'sleep',side_effect=tick), \
             patch.object(c,'check_deadlines') as deadline, \
             patch.object(c,'exact_alive',side_effect=[True,True,False]):
            proof=control.observe_absence([{'pid':1,'created':1.}],{'bound':'fixture'},seconds=1.)
        self.assertTrue(proof['exact_owned_absence']);self.assertEqual(proof['samples'],3)
        self.assertEqual(deadline.call_count,3);self.assertAlmostEqual(proof['elapsed_seconds'],.1)

    def test_surviving_exact_identity_is_not_success(self):
        instant=[0.]
        def tick(seconds):instant[0]+=seconds
        with patch.object(control.time,'monotonic',side_effect=lambda:instant[0]), \
             patch.object(control.time,'sleep',side_effect=tick), \
             patch.object(c,'check_deadlines'),patch.object(c,'exact_alive',return_value=True):
            proof=control.observe_absence([{'pid':1,'created':1.}],{},seconds=.1)
        self.assertFalse(proof['exact_owned_absence'])

    def test_cleanup_cannot_extend_original_deadline(self):
        with patch.object(c,'check_deadlines',side_effect=ValueError('original deadline elapsed')):
            with self.assertRaisesRegex(ValueError,'original deadline'):
                control.observe_absence([{'pid':1,'created':1.}],{})


if __name__=='__main__':unittest.main()
