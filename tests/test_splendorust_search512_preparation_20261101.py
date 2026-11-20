"""Frozen-evidence provenance and model-free preparation regressions."""
import json,math,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from tools.splendorust_search512_20261101 import common as c,preparation as p


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.directory=tempfile.TemporaryDirectory()
        self.root=Path(self.directory.name).resolve();self.out=self.root/'outputs/study'
        self.root_patch=patch.object(c,'ROOT',self.root);self.out_patch=patch.object(c,'OUT',self.out)
        self.root_patch.start();self.out_patch.start()
    def tearDown(self):
        self.root_patch.stop();self.out_patch.stop();self.directory.cleanup()
    def test_exclusive_publication_preserves_first_receipt(self):
        target=self.out/'evidence.json';first=c.save(target,{'decision':1})
        with self.assertRaises(FileExistsError):c.save(target,{'decision':2})
        self.assertEqual(c.read(target),{'decision':1});c.verify([first])
    def test_atomic_receipt_is_complete_and_never_overwritten(self):
        target=self.out/'claim.json';proof=c.atomic_save(target,{'owner':17})
        self.assertFalse(target.with_suffix('.json.pending').exists());c.verify([proof])
        with self.assertRaises(ValueError):c.atomic_save(target,{'owner':18})
    def test_output_boundary_rejects_outside_write(self):
        target=self.root/'frozen_original.json'
        with self.assertRaises(ValueError):c.save(target,{'changed':True})
        self.assertFalse(target.exists())
    def test_tampered_input_is_rejected(self):
        target=self.out/'source.json';proof=c.save(target,{'quota':256})
        target.write_text('{"quota":512}',encoding='utf-8')
        with self.assertRaises(ValueError):c.verify([proof])
    def test_noncanonical_relative_artifact_is_rejected(self):
        proof=c.save(self.out/'source.json',{'quota':256})
        proof['path']='outputs/study/../study/source.json'
        with self.assertRaises(ValueError):c.verify([proof])
    def test_prepare_repetition_requires_equal_content(self):
        path=self.out/'plan.json';first=p.fixed(path,{'games':400})
        self.assertEqual(p.fixed(path,{'games':400}),first)
        with self.assertRaises(ValueError):p.fixed(path,{'games':200})
    def test_nonfinite_science_cannot_be_serialized(self):
        for value in (float('nan'),float('inf'),-float('inf')):
            with self.assertRaises(ValueError):c.canonical({'value':value})
    def test_review_rejects_unapproved_or_changed_source(self):
        source=c.save(self.out/'source.json',{'route':'control'})
        path=self.out/'review.json'
        c.save(path,{'approved':False,'blocking_findings':['bad quota'],'inputs':[source]})
        with self.assertRaises(ValueError):p.require_review(path)
        accepted=self.out/'approved.json'
        c.save(accepted,{'approved':True,'blocking_findings':[],'inputs':[source]})
        self.assertTrue(p.require_review(accepted)['approved'])
        (self.out/'source.json').write_text('{}',encoding='utf-8')
        with self.assertRaises(ValueError):p.require_review(accepted)
    def test_passed_tests_must_bind_sources_and_real_log(self):
        source=c.save(self.out/'source.json',{'revision':1})
        log=c.save(self.out/'log.json',{'passed':5})
        receipt=self.out/'tests.json'
        c.save(receipt,{'exit_code':0,'models':0,'games':0,'inputs':[source],'log':log})
        self.assertEqual(p.require_tests(receipt)['exit_code'],0)
        (self.out/'log.json').write_text('tampered',encoding='utf-8')
        with self.assertRaises(ValueError):p.require_tests(receipt)
    def test_local_dependency_closure_includes_relative_and_function_imports(self):
        package=self.root/'tools/test_package';package.mkdir(parents=True)
        (package/'__init__.py').write_text('',encoding='utf-8')
        entry=package/'entry.py';helper=package/'helper.py';nested=package/'nested.py'
        entry.write_text('from . import helper\ndef f():\n    from .nested import answer\n',encoding='utf-8')
        helper.write_text('x=1\n',encoding='utf-8');nested.write_text('answer=42\n',encoding='utf-8')
        self.assertEqual(set(p.python_dependencies([entry])),{entry,helper,nested,package/'__init__.py'})
    def test_seed_receipt_cannot_reuse_an_unrelated_schedule_or_statistics_seed(self):
        from tools.splendorust_search512_20261101 import design
        saved=dict(status='passed',collisions=[],domain=design.DOMAIN,new_setup_count=203,
            bootstrap_seed=design.BOOTSTRAP_SEED,prior_seed_reuse=False,checked=[])
        p.verify_seed_receipt(saved)
        for key,value in [('domain','old_study'),('new_setup_count',200),('bootstrap_seed',42),
            ('prior_seed_reuse',True),('collisions',['used_seed'])]:
            with self.subTest(key=key),self.assertRaises(ValueError):p.verify_seed_receipt(dict(saved,**{key:value}))


class LiveProvenanceTests(unittest.TestCase):
    def test_accepted_predecessor_is_saved_only_and_actual_exits_verified(self):
        prior,receipt=p.accepted_predecessor()
        self.assertTrue(receipt['approved'])
        self.assertEqual(prior['only_scientific_change'],'simulations_128_to_256')
        self.assertEqual(receipt['guardian_actual_exit']['exit_code'],0)
    def test_new_search_budget_has_finite_resource_and_verification_reserves(self):
        self.assertEqual(c.ROUTE_QUOTAS,{'control':256,'candidate':512})
        self.assertLess(c.COLLECTION_CUTOFF_SECONDS,c.JOB_SECONDS)
        self.assertEqual(c.JOB_SECONDS-c.COLLECTION_CUTOFF_SECONDS,c.STAGES['final'])
        self.assertLess(c.PREFINAL_EVIDENCE_BYTES,c.MAX_EVIDENCE_BYTES)
        self.assertEqual(c.MAX_WORKER_LIFECYCLES,814)
        self.assertEqual(c.MAX_GAMES,406)
        self.assertEqual(c.MAX_REPLAY_DECISIONS,96)


if __name__=='__main__':unittest.main()
