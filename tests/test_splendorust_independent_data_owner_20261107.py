"""Original-clock and exact study contract counterexamples, with no models."""
import unittest
import os
from pathlib import Path
import tempfile
import time
import threading
from unittest.mock import patch
import psutil
from tools.splendorust_independent_data_20261107 import common as c, runtime, owner


class OriginalClockTests(unittest.TestCase):
    def setUp(self):
        self.claim={'monotonic_origin':100.,'unix_origin':1000.}
        self.phase={'monotonic':120.,'unix_time':1020.,'monotonic_deadline':180.,'unix_deadline':1080.}
        self.worker={'monotonic_origin':130.,'unix_origin':1030.,'monotonic_deadline':170.,'unix_deadline':1070.}

    def check(self, mono=150.,wall=1050.):
        runtime.check_deadlines(self.claim,self.phase,self.worker,mono,wall)

    def test_valid_inside_all_three_original_clocks(self):
        self.check()

    def test_either_clock_expiry_rejects_completed_worker(self):
        for mono,wall in ((170.,1050.),(150.,1070.),(150.,999.),(99.,1050.)):
            with self.subTest(mono=mono,wall=wall),self.assertRaises(ValueError): self.check(mono,wall)

    def test_fresh_worker_clock_cannot_extend_original_phase(self):
        self.worker.update(monotonic_deadline=9999.,unix_deadline=9999.)
        with self.assertRaises(ValueError): self.check(180.,1080.)

    def test_fresh_phase_cannot_extend_original_model_cutoff(self):
        delta=c.MODEL_SECONDS
        self.phase.update(monotonic_deadline=delta+999.,unix_deadline=delta+1999.)
        self.worker.update(monotonic_deadline=delta+999.,unix_deadline=delta+1999.)
        with self.assertRaises(ValueError): self.check(100.+delta,1000.+delta)

    def test_no_host_memory_or_forecast_gate_and_fixed_effort(self):
        plan=owner.contract()
        self.assertFalse(plan['available_ram_gate'])
        self.assertFalse(plan['resource_forecasts_may_stop'])
        self.assertEqual(plan['max_optimizer_calls'],390)
        self.assertEqual(plan['examples_per_fit'],3072)
        self.assertEqual(plan['updates_per_fit'],96)
        self.assertEqual(plan['training_games'],1024)
        self.assertEqual(plan['validation_games']+plan['test_games'],128)
        self.assertEqual(len(plan['fits']),4)
        self.assertEqual({x['fit_id'] for x in plan['fits']},
                         {'r0_narrow_joint','r0_broad_joint','r1_narrow_joint','r1_broad_joint'})
        self.assertFalse(plan['automatic_promotion'])

    def test_strength_sample_never_extends_or_substitutes(self):
        for invalid in (-10,1,200,True):
            with self.subTest(offset=invalid),self.assertRaises(ValueError):
                owner.segment_id('main_block',0,invalid)
        with self.assertRaises(ValueError): owner.segment_id('replay',1,0)
        with self.assertRaises(ValueError): owner.segment_id('main_block',True,0)
        self.assertEqual(len({owner.segment_id('main_block',rep,start)
            for rep in (0,1) for start in range(0,200,10)}),40)

    def test_partial_or_skipped_test_suite_cannot_admit_study(self):
        with self.assertRaises(ValueError):owner.validate_test_receipt(dict(exit_code=0,models=0,games=0))
        proof=dict(schema='independent_data_complete_pytest_v1',full_suite=True,exit_code=0,models=0,games=0,
            runner=c.artifact(c.ROOT/'tools/splendorust_independent_data_20261107/preflight.py'),
            suite_summary=dict(tests_passed=280,tests_collected=280,failures=0,skipped=0,
                               tensor_runtime_loaded=False,models=0,games=0))
        owner.validate_test_receipt(proof)
        for key,value in (('tests_passed',9),('skipped',1),('tensor_runtime_loaded',True)):
            changed=dict(proof,suite_summary=dict(proof['suite_summary'],**{key:value}))
            with self.subTest(key=key),self.assertRaises(ValueError):owner.validate_test_receipt(changed)


@unittest.skipUnless(os.name=='nt','Windows owning-job integration')
class OwnedProcessIntegrationTests(unittest.TestCase):
    def exercise(self,mode):
        with tempfile.TemporaryDirectory(prefix='splendor-independent-owner-') as temporary:
            folder=Path(temporary).resolve()
            with patch.object(c,'OUT',folder),patch.dict(os.environ,{'INDEPENDENT_DATA_PROCESS_PROBE':'1'}):
                mono,wall=time.monotonic(),time.time()
                claim=dict(pid=os.getpid(),created=psutil.Process().create_time(),
                           monotonic_origin=mono,unix_origin=wall)
                c.save(folder/'claim.json',claim)
                phase=c.save(folder/'stages/qualification_start.json',dict(stage='qualification',
                    monotonic=mono,unix_time=wall,monotonic_deadline=mono+30,unix_deadline=wall+30))
                call=lambda:owner.run_job(claim,phase,'a'*64,c.PACKAGE+'.process_probe',mode,
                    folder/'probe',seconds=20,rss=c.GiB,model=False)
                if mode=='passed':call()
                else:
                    with self.assertRaises(ValueError):call()
                receipt=c.read(folder/'probe/execution.json')
                self.assertEqual(receipt['status'],'passed' if mode=='passed' else 'failed')
                self.assertEqual(receipt['cleanup_errors'],[])
                self.assertEqual(receipt['retries'],0)
                self.assertGreaterEqual(len(receipt['observed_owned_identities']),1)
                self.assertEqual(receipt['worker_exit_code']==0,mode=='passed')
                from tools.splendorust_gpu_probe_20261105.control import require_absent
                for identity in receipt['observed_owned_identities']:require_absent(identity)

    def test_actual_owned_process_exit_and_cleanup(self):self.exercise('passed')
    def test_failure_is_preserved_with_no_retry(self):self.exercise('failed')


class ConcurrentBoundaryTests(unittest.TestCase):
    def test_eight_game_threads_do_not_race_status_publication(self):
        mono,wall=time.monotonic(),time.time()
        claim=dict(pid=os.getpid(),created=psutil.Process().create_time(),monotonic_origin=mono,unix_origin=wall)
        phase=dict(monotonic=mono,unix_time=wall,monotonic_deadline=mono+30,unix_deadline=wall+30)
        own=dict(monotonic_origin=mono,unix_origin=wall,monotonic_deadline=mono+30,
                 unix_deadline=wall+30,maximum_rss_bytes=8*c.GiB,mode='games')
        boundary=runtime.Boundary(claim,phase,own,c.OUT/'test_fixture',True)
        barrier=threading.Barrier(8);published=[];errors=[]
        def record(*args):time.sleep(.02);published.append(args)
        def call():
            try:barrier.wait(timeout=5);boundary()
            except BaseException as exc:errors.append(exc)
        with patch.object(c,'status',side_effect=record):
            threads=[threading.Thread(target=call) for _ in range(8)]
            for thread in threads:thread.start()
            for thread in threads:thread.join(timeout=5)
        self.assertFalse(errors)
        self.assertTrue(all(not t.is_alive() for t in threads))
        self.assertEqual(len(published),1)


if __name__=='__main__': unittest.main()
