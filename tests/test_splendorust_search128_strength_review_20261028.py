"""Synthetic reader checks only; no learned models, games or historical full audits."""
import ast
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from tools.splendorust_search128_strength_20261028.review_runtime import ReviewRuntime, identity
from tools.splendorust_search128_strength_20261028 import independent_review as reader, schedule

class ReaderRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.out = self.root/'out'; self.out.mkdir()
        self.r = ReviewRuntime(self.root,self.out)
    def tearDown(self): self.temp.cleanup()
    def file(self,name='a.json',raw=b'{"v":1}'):
        path=self.root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
        return path,dict(path=path.relative_to(self.root).as_posix(),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
    def test_parallel_full_bytes_then_identity_cache(self):
        jobs=[self.file(f'x/{i}.json',str(i).encode()) for i in range(35)]
        self.r.verify_many(jobs)
        for _,item in jobs:self.r.verify(item)
        self.assertEqual(self.r.metrics['hashed_reads'],35)
        self.assertEqual(self.r.metrics['cached_verifications'],35)
        self.assertEqual(self.r.revalidate_identities()['files'],35)
    def test_semantic_read_always_rehashes_actual_bytes(self):
        path,item=self.file();self.r.verify(item)
        self.assertEqual(self.r.read_path(path),{'v':1})
        self.assertEqual(self.r.metrics['hashed_reads'],2)
    def test_modified_cached_file_rejected(self):
        path,item=self.file();self.r.verify(item);path.write_bytes(b'{"v":2}')
        with self.assertRaisesRegex(ValueError,'identity changed'):self.r.verify(item)
    def test_mutation_restored_mtime_still_rejected(self):
        path,item=self.file();self.r.verify(item);old=path.stat();time.sleep(.01)
        path.write_bytes(b'{"v":2}');os.utime(path,ns=(old.st_atime_ns,old.st_mtime_ns))
        with self.assertRaises(ValueError):self.r.read_path(path)
    def test_replaced_inode_rejected(self):
        path,item=self.file();self.r.verify(item);other=self.root/'other';other.write_bytes(path.read_bytes())
        os.replace(other,path)
        with self.assertRaises(ValueError):self.r.verify(item)
    def test_identity_fields_include_dev_inode_size_times(self):
        path,_=self.file();s=path.stat()
        self.assertEqual(identity(s)[:5],(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns))
    def test_conflicting_expected_pins_fail_before_reuse(self):
        path,item=self.file();self.r.verify(item)
        wrong=dict(item,sha256='0'*64)
        with self.assertRaisesRegex(ValueError,'Contradictory expected'):self.r.verify(wrong)
    def test_parallel_conflicting_pins_fail(self):
        path,item=self.file()
        with self.assertRaisesRegex(ValueError,'Contradictory expected'):
            self.r.verify_many([(path,item),(path,dict(item,sha256='0'*64))])
    def test_hash_mismatch_thread_failure_cancels(self):
        jobs=[self.file(f'{i}.json',b'x') for i in range(18)]
        jobs[3]=(jobs[3][0],dict(jobs[3][1],sha256='0'*64))
        with self.assertRaises(ValueError):self.r.verify_many(jobs)
        self.assertTrue(self.r._cancel.is_set())
        self.assertFalse(any(t.name.startswith('saved-hash') for t in threading.enumerate()))
    def test_parallel_identical_pin_threads_safe(self):
        path,item=self.file()
        with ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(lambda _:self.r.verify(item),range(20)))
        self.assertEqual(list(self.r.consumed.values()),[item])
    def test_deadline_has_no_acceptance(self):
        _,item=self.file();self.r.start-=3601
        with self.assertRaisesRegex(ValueError,'deadline'):self.r.verify(item)
        self.assertFalse((self.out/'independent_completion_review.json').exists())
    def test_wall_clock_deadline_not_monotonic_reset(self):
        self.r.wall-=3601
        with self.assertRaisesRegex(ValueError,'deadline'):self.r.guard()
    def test_rss_resource_cap_fails_closed(self):
        with patch.object(self.r._resource_process,'memory_info') as memory:
            memory.return_value.rss=2*1024**3+1
            with self.assertRaisesRegex(ValueError,'RSS'):self.r.guard()
    def test_disk_reserve_fails_closed(self):
        with patch('tools.splendorust_search128_strength_20261028.review_runtime.shutil.disk_usage') as usage:
            usage.return_value.free=5*1024**3-1
            with self.assertRaisesRegex(ValueError,'disk'):self.r.guard()
    def test_final_identity_recheck_detects_change(self):
        path,item=self.file();self.r.verify(item);path.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'changed before acceptance'):self.r.revalidate_identities()
    def test_wrong_tree_inventory_rejected(self):
        self.file('out/a.json')
        with self.assertRaisesRegex(ValueError,'Exact preserved tree'):self.r.exact_tree(self.out,[])

    def test_parallel_queue_is_bounded(self):
        source=Path(reader.__file__).with_name('review_runtime.py').read_text()
        self.assertIn('ThreadPoolExecutor(max_workers=8',source)
        self.assertIn('range(min(16, len(unique)))',source)

    def test_shared_final_deadline_is_not_reset(self):
        self.r.shared_deadlines=(time.monotonic()-.1,time.time()+100)
        with self.assertRaisesRegex(ValueError,'shared deadline'):self.r.guard()

    def test_shared_wall_deadline_is_not_reset(self):
        self.r.shared_deadlines=(time.monotonic()+100,time.time()-.1)
        with self.assertRaisesRegex(ValueError,'shared deadline'):self.r.guard()

    def test_identity_receipt_detects_write_after_component(self):
        path,pin=self.file();self.r.verify(pin);proof=reader.identity_receipt(self.r)
        self.assertEqual(reader.verify_identity_receipt(self.r,proof)['files'],1)
        path.write_bytes(b'new bytes')
        with self.assertRaisesRegex(ValueError,'between component and acceptance'):
            reader.verify_identity_receipt(self.r,proof)

    def test_identity_receipt_rejects_duplicate(self):
        _,pin=self.file();self.r.verify(pin);proof=reader.identity_receipt(self.r)
        proof['files']*=2
        with self.assertRaisesRegex(ValueError,'Duplicate final file'):reader.verify_identity_receipt(self.r,proof)

    def clock_fixture(self):
        now,wall=time.monotonic(),time.time()
        claim=dict(freeze_sha256='f',monotonic_origin=now-100,unix_origin=wall-100)
        final=dict(cap_seconds=14400,monotonic=now-10,unix_time=wall-10,
                   monotonic_deadline=now+1000,unix_deadline=wall+1000)
        clock=dict(freeze_sha256='f',review_monotonic_origin=now,review_unix_origin=wall,
                   deadline_monotonic=now+1000,deadline_unix=wall+1000)
        completion=dict(status='completed_verified',independent_acceptance_deadline_monotonic=now+1000,
                        independent_acceptance_deadline_unix=wall+1000)
        for name,value,key in (('guardian_claim.json',claim,'guardian_claim_sha256'),('completion.json',completion,'completion_sha256')):
            path=self.out/name;path.write_text(json.dumps(value))
            clock[key]=hashlib.sha256(path.read_bytes()).hexdigest()
        docs={'guardian_claim.json':claim,'stages/final_start.json':final,
              'postrun/review_owner_clock.json':clock,'completion.json':completion}
        return docs

    def test_clock_clamps_to_original_final_not_fresh_hour(self):
        docs=self.clock_fixture()
        with patch.object(reader,'OUT',self.out),patch.object(self.r,'read',side_effect=lambda name:docs[name]):
            saved=reader.bind_shared_clock(self.r,'f')
        self.assertEqual(self.r.shared_deadlines,(saved['deadline_monotonic'],saved['deadline_unix']))
        self.assertLess(saved['deadline_monotonic']-saved['review_monotonic_origin'],3600)

    def test_owner_clock_extension_rejected(self):
        docs=self.clock_fixture();docs['postrun/review_owner_clock.json']['deadline_monotonic']+=1
        with patch.object(reader,'OUT',self.out),patch.object(self.r,'read',side_effect=lambda name:docs[name]),self.assertRaisesRegex(ValueError,'extended'):
            reader.bind_shared_clock(self.r,'f')

    def test_owner_clock_claim_hash_must_match_actual_bytes(self):
        docs=self.clock_fixture();docs['postrun/review_owner_clock.json']['guardian_claim_sha256']='0'*64
        with patch.object(reader,'OUT',self.out),patch.object(self.r,'read',side_effect=lambda name:docs[name]),self.assertRaisesRegex(ValueError,'claim/completion bytes'):
            reader.bind_shared_clock(self.r,'f')

    def test_publish_checks_both_sides(self):
        target=self.out/'receipt.json';calls=[]
        def guard():
            calls.append(1)
            if len(calls)==2:raise ValueError('deadline during save')
        with patch.object(self.r,'guard',side_effect=guard),self.assertRaisesRegex(ValueError,'deadline during save'):
            reader.guarded_publish(self.r,target,{'accepted':True})
        self.assertTrue(target.exists());self.assertEqual(len(calls),2)

    def test_receipt_publication_keeps_original_evidence_cap(self):
        target=self.out/'receipt.json';self.r.output_evidence_bytes=16*reader.GiB-1024**2
        with self.assertRaisesRegex(ValueError,'original evidence cap'):
            reader.guarded_publish(self.r,target,{'accepted':True})
        self.assertFalse(target.exists())

    def test_reader_schedule_independently_matches_full406(self):
        expected=[reader.expected_game(i,stage) for stage,n in (('operational',6),('main',400)) for i in range(n)]
        self.assertEqual(expected,schedule.make_schedule())
        self.assertEqual(len({g['seed_decimal'] for g in expected}),203)
        self.assertEqual(expected[6]['game_index'],0)
        self.assertEqual(expected[-1]['challenger_quota'],128)

    def test_reader_statistics_refuses_operational_contamination(self):
        with self.assertRaisesRegex(ValueError,'exclude all six'):
            reader.check_statistics([dict(stage='operational')]*400,{}, {})

    def test_original_raw_10s_clock_accepts_late_completed(self):
        result=dict(action={'type':'pass'},diagnostics={})
        row=dict(completed=True,clock=dict(status='ok',clock='perf_counter_ns',deadline_seconds=10.,accepted=True,
            elapsed_seconds=1.7,receipt_elapsed_seconds=1.3,received_before_deadline=True,result=result),
            action=result['action'],trace={},after={})
        self.assertIsNone(reader.check_clock(row))
        row['clock']['deadline_seconds']=1.
        with self.assertRaisesRegex(ValueError,'ten-second'):reader.check_clock(row)

    def test_only_actual_research_guard_is_administrative(self):
        row=dict(completed=False,action=None,trace=None,after=None,
            clock=dict(clock='perf_counter_ns',deadline_seconds=10.,status='worker_error',accepted=False,
                       elapsed_seconds=9.2,receipt_elapsed_seconds=9.1,received_before_deadline=True,result=None,
                       detail='TimeoutError: STRENGTH_INTERNAL_DEADLINE:9s'))
        self.assertEqual(reader.check_clock(row),'research_internal_hard_guard')
        row['clock']['detail']='bad action'
        with self.assertRaisesRegex(ValueError,'explicit research hard guard'):reader.check_clock(row)

    def test_no_model_or_game_imports_in_reader(self):
        for path in (Path(reader.__file__),Path(reader.__file__).with_name('review_runtime.py')):
            tree=ast.parse(path.read_text())
            imports=[n.module or '' for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
            imports += [a.name for n in ast.walk(tree) if isinstance(n,ast.Import) for a in n.names]
            self.assertFalse(any(any(bad in x for bad in ('torch','numpy','rules','adapters','worker','subprocess')) for x in imports),imports)

    def test_missing_clock_does_not_silently_start_fresh_budget(self):
        with patch.object(self.r,'read',side_effect=FileNotFoundError('original owner clock missing')):
            with self.assertRaises(FileNotFoundError):reader.bind_shared_clock(self.r,'f')

    def test_acceptance_write_overrun_leaves_invalidation(self):
        component=dict(status='passed_saved_evidence_component',accepted=False,freeze_sha256='f',shared_owner_clock={},
            reviewer_pid=999999999,reviewer_created=1.,reviewer_source=dict(sha256='s'),
            identity_inventory=dict(path='identity.json'),result={})
        terminal=dict(freeze_sha256='f',exit_code=0,observed_by='owned Process handle',cleanup_reason=None,
            reader_pid=999999999,reader_created=1.,script_sha256='s',component_sha256='s')
        def read(name):return component if name.endswith('component.json') else terminal if name.endswith('os_exit.json') else {}
        original=reader.save_new
        def save(path,value):
            original(path,value)
            if path.name=='independent_completion_review.json':self.r.shared_deadlines=(time.monotonic()-1,time.time()-1)
        with patch.object(reader,'OUT',self.out),patch.object(reader,'Review',return_value=self.r), \
             patch.object(reader,'bind_shared_clock',return_value={}),patch.object(reader,'source_authentication'), \
             patch.object(self.r,'read',side_effect=read),patch.object(self.r,'pin',return_value=dict(sha256='s')), \
             patch.object(self.r,'verify'),patch.object(self.r,'read_path',return_value={}), \
             patch.object(reader,'verify_identity_receipt',return_value={}),patch.object(reader,'save_new',side_effect=save):
            self.assertEqual(reader.accept_exit('f'),1)
        self.assertTrue((self.out/'postrun/independent_completion_review.json').exists())
        self.assertTrue((self.out/'postrun/acceptance_invalidated.json').exists())


if __name__=='__main__':unittest.main(verbosity=2)
