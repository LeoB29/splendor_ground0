"""Counterexamples for stopped owners, fixed work, preservation and dual clocks."""
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
from . import common as c, worker, launcher


class Clocks(unittest.TestCase):
    def setUp(self):
        self.limits=dict(monotonic_origin=10,unix_origin=100,monotonic_deadline=20,unix_deadline=110)
    def test_inside_both(self): c.check_deadlines(self.limits,mono=15,wall=105)
    def test_monotonic_expired(self):
        with self.assertRaises(ValueError): c.check_deadlines(self.limits,mono=20,wall=105)
    def test_wall_expired(self):
        with self.assertRaises(ValueError): c.check_deadlines(self.limits,mono=15,wall=110)
    def test_reboot_or_clock_rollback(self):
        for mono,wall in [(9,105),(15,99)]:
            with self.subTest(mono=mono,wall=wall),self.assertRaises(ValueError):c.check_deadlines(self.limits,mono=mono,wall=wall)
    def test_inherited_minimum(self):
        limits=c.original_limits();reader=c.read(c.SOURCE/'reader_clock.json')
        self.assertEqual(limits['monotonic_deadline'],reader['monotonic_deadline'])
        self.assertEqual(limits['unix_deadline'],reader['unix_deadline'])


class WritesAndInventory(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.base=Path(self.temp.name)
        self.source=self.base/'source';self.source.mkdir();self.out=self.base/'out';self.out.mkdir()
        self.patches=[patch.object(c,'SOURCE',self.source),patch.object(c,'OUT',self.out)]
        for p in self.patches:p.start()
        self.item=self.source/'evidence.json';self.item.write_text('{"accepted":true}')
    def tearDown(self):
        for p in reversed(self.patches):p.stop()
        self.temp.cleanup()
    def plan(self,items=None):
        path=self.out/'source_inventory.jsonl'
        path.write_text(''.join(json.dumps(x)+'\n' for x in (items if items is not None else [c.artifact(self.item)])))
        return {'source_inventory':c.artifact(path)}
    def test_save_cannot_touch_source(self):
        with self.assertRaises(ValueError):c.save(self.source/'new.json',{})
    def test_save_exclusive(self):
        p=self.out/'new.json';c.save(p,{'v':1})
        with self.assertRaises(FileExistsError):c.save(p,{'v':2})
    def test_manifest_passes(self):
        self.assertEqual(c.verify_source_inventory(self.plan())['files'],1)
    def test_mutation_rejected(self):
        p=self.plan();self.item.write_text('{"accepted":false}')
        with self.assertRaises(ValueError):c.verify_source_inventory(p)
    def test_added_file_rejected(self):
        p=self.plan();(self.source/'extra').write_text('x')
        with self.assertRaises(ValueError):c.verify_source_inventory(p)
    def test_missing_file_rejected(self):
        p=self.plan();self.item.unlink()
        with self.assertRaises(FileNotFoundError):c.verify_source_inventory(p)
    def test_duplicate_rejected(self):
        item=c.artifact(self.item)
        with self.assertRaises(ValueError):c.verify_source_inventory(self.plan([item,item]))
    def test_outside_source_rejected(self):
        p=self.out/'outside';p.write_text('x')
        with self.assertRaises(ValueError):c.verify_source_inventory(self.plan([c.artifact(p)]))
    def test_manifest_hash_rejected(self):
        p=self.plan();(self.out/'source_inventory.jsonl').write_text('')
        with self.assertRaises(ValueError):c.verify_source_inventory(p)


class FixedDispatch(unittest.TestCase):
    def test_only_saved_modes(self):
        for mode in ('fit','games','replay','resume_partial','qualification','prediction','retry'):
            with self.subTest(mode=mode),self.assertRaises(ValueError):worker.dispatch(mode,{},Path('.'),lambda:None)
    def test_three_only(self):
        self.assertEqual(c.JOBS,('independent_mixed_terminal_r1','reader_mixed_terminal_r0','final_acceptance'))
    def test_no_model_runtime(self):self.assertNotIn('torch',sys.modules)
    def test_source_stop_shape_matches_saved_failure(self):c.assert_stopped()
    def test_source_terminal_receipt_refused(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(c,'SOURCE',Path(tmp)),patch.object(c,'SOURCE_IDENTITIES',()):
            (Path(tmp)/'completion.json').write_text('{}')
            with self.assertRaises(ValueError):c.assert_stopped()


class Launcher(unittest.TestCase):
    def test_launcher_clock_expiry_cleans_owned_child(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(c,'OUT',Path(tmp)):
            folder=Path(tmp)/'deadline_fixture';mono,wall=time.monotonic(),time.time()
            limits=dict(monotonic_origin=mono,unix_origin=wall,monotonic_deadline=mono+1,unix_deadline=wall+1)
            with self.assertRaises(ValueError):
                launcher.run_child([sys.executable,'-B','-c','import time; time.sleep(30)'],
                                   folder,metadata={'fixture':True},limits=limits)
            value=c.read(folder/'actual_exit.json')
            self.assertIn('dual deadlines',value['error']);self.assertTrue(value['actual_exit_observed'])
            self.assertTrue(value['exact_process_absent']);self.assertEqual(value['cleanup_errors'],[])
    def test_post_start_receipt_failure_cleans_owned_child(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(c,'OUT',Path(tmp)):
            folder=Path(tmp)/'receipt_failure_fixture';real_save=c.save
            def fail_child_receipt(path,value):
                if Path(path).name=='child.json':raise OSError('injected child receipt failure')
                return real_save(path,value)
            with patch.object(c,'save',side_effect=fail_child_receipt),self.assertRaises(ValueError):
                launcher.run_child([sys.executable,'-B','-c','import time; time.sleep(30)'],
                                   folder,metadata={'fixture':True})
            value=c.read(folder/'actual_exit.json')
            self.assertIn('injected child receipt failure',value['error'])
            self.assertTrue(value['actual_exit_observed']);self.assertTrue(value['exact_process_absent'])
            self.assertIsNotNone(value['exit_code']);self.assertEqual(value['cleanup_errors'],[])
    def test_actual_nonzero_exit_is_recorded_not_hidden(self):
        # Deliberately failed non-scientific child; retained only as a unit fixture.
        with tempfile.TemporaryDirectory() as tmp,patch.object(c,'OUT',Path(tmp)):
            folder=Path(tmp)/'non_scientific_exit_fixture'
            code,receipt=launcher.run_child([sys.executable,'-B','-c','import time; time.sleep(.5); raise SystemExit(7)'],
                folder,metadata={'fixture':True,'model_calls':0,'new_games':0})
            self.assertEqual(code,7);value=c.read(c.resolve(receipt))
            self.assertEqual(value['exit_code'],7);self.assertTrue(value['actual_exit_observed'])
            self.assertTrue(value['exact_process_absent'])
    def test_worker_has_one_runtime_verification_call(self):
        import ast
        tree=ast.parse(Path(worker.__file__).read_text())
        calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='verify_runtime_sources']
        self.assertEqual(len(calls),1)


if __name__=='__main__':unittest.main()
