"""Regression checks for saved evidence routing and amended acceptance."""
import copy
import unittest
from unittest.mock import patch
from tools.splendorust_collection4096_resume_20261118_v2 import common as c, corpus, audit
from tools.splendorust_collection4096_resume_20261118_v2.collection import common as cc


class SegmentTests(unittest.TestCase):
    def test_fixed_segments_preserve_all_predecessor_paths(self):
        expected={0:('original_passed',c.ORIGINAL_OUT/'collection/cells/c000'),
            40:('original_passed',c.ORIGINAL_OUT/'collection/cells/c040'),
            41:('original_recovered',c.ORIGINAL_OUT/'collection/cells/c041'),
            42:('previous_passed',c.PREVIOUS_OUT/'collection/cells/c042'),
            134:('previous_passed',c.PREVIOUS_OUT/'collection/cells/c134'),
            135:('partial_resumed',cc.OUT/'recovery_games'),
            136:('resume_new',cc.OUT/'cells/c136'),207:('resume_new',cc.OUT/'cells/c207')}
        for index,value in expected.items():
            with self.subTest(index=index):self.assertEqual(corpus.cell_source(f'c{index:03d}'),value)

    def test_ambiguous_and_out_of_schedule_ids_rejected(self):
        for ident in ['c41','c0041','c208','c-01','../c135','C135',135,True,None]:
            with self.subTest(ident=ident),self.assertRaises((ValueError,TypeError)):
                corpus.cell_source(ident)

    def test_two_recovered_cells_keep_distinct_first_proofs(self):
        self.assertEqual(corpus.first_audit_path('c041'),c.PREVIOUS_OUT/'collection/recovery_audit/worker/saved_audit.json')
        self.assertEqual(corpus.first_audit_path('c135'),cc.OUT/'recovery_audit/worker/saved_audit.json')
        self.assertEqual(corpus.first_audit_path('c134'),c.PREVIOUS_OUT/'collection/cell_audits/c134/worker/saved_audit.json')

    def test_only_unstarted_full_cells_released_after_supplement(self):
        jobs=audit.expected_jobs()
        self.assertEqual(len(jobs),151)
        self.assertEqual([j['mode'] for j in jobs[:5]],['saved_boundary','recovery_qualification',
            'audit_recovery_qualification','recovery_games','supplemental_audit'])
        self.assertTrue(all(j['phase']=='supplemental' and j['seconds']==7200 for j in jobs[:5]))
        game_jobs=[j for j in jobs if j['mode']=='games']
        self.assertEqual([j['path'] for j in game_jobs],[f'collection/cells/c{i:03d}' for i in range(136,208)])
        self.assertTrue(all(j['seconds']==14400 for j in game_jobs))
        self.assertEqual(jobs[-2]['mode'],'corpus');self.assertEqual(jobs[-1]['mode'],'selection')


class AuthenticationTests(unittest.TestCase):
    def test_only_exact_roundtrip_logical_digest_is_excluded(self):
        raw={'path':'same','bytes':100,'sha256':'a'*64};stored={'path':'same','bytes':50,'sha256':'b'*64}
        value={'nested':[{'lossless_storage':{'raw':raw,'stored':stored,'exact_byte_roundtrip':True}}]}
        self.assertEqual(corpus._logical_raw_artifacts(value),[raw])
        for mutation in ('no_roundtrip','different_path','same_digest'):
            bad=copy.deepcopy(value);receipt=bad['nested'][0]['lossless_storage']
            if mutation=='no_roundtrip':receipt['exact_byte_roundtrip']=False
            elif mutation=='different_path':receipt['stored']['path']='other'
            else:receipt['stored']['sha256']=raw['sha256']
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):corpus._logical_raw_artifacts(bad)

    def test_authentication_binds_all_physical_artifacts_and_rows(self):
        artifact=lambda name:dict(path=name,bytes=1,sha256=(name[0]*64))
        raw=artifact('raw');stored=dict(raw,sha256='s'*64)
        document=artifact('doc');row=artifact('move');target=artifact('target')
        cell={'cell_id':'c134','specifications':[{'seed_decimal':str(i)} for i in range(16)]}
        proof=dict(status='passed',full_cohort_verified=True,cell_id='c134',games=16,
            model_calls=0,training_updates=0,seed_decimals=[str(i) for i in range(16)],
            journal=dict(journal=stored,lossless_storage=dict(raw=raw,stored=stored,exact_byte_roundtrip=True)),
            game_proofs=[dict(document=document,target=target)])
        checked=[]
        with patch.object(c,'verify',side_effect=lambda items,**kw:checked.extend(items)), \
             patch.object(cc,'read',return_value={'rows':[row]}):
            self.assertEqual(corpus.authenticate_passed_proof(proof,cell),proof)
        self.assertNotIn(raw,checked)
        for item in (stored,document,target,row):self.assertIn(item,checked)
        for field,value in [('full_cohort_verified',False),('status','failed'),('games',15),('model_calls',1)]:
            bad=dict(proof,**{field:value})
            with self.subTest(field=field),self.assertRaises(ValueError):corpus.authenticate_passed_proof(bad,cell)

    def test_saved_prior_failure_and_all_successful_jobs_are_authenticated(self):
        observer=c.ROOT/'outputs/splendorust_collection4096_recovery_launch_20261116_01/observed_launcher_exit.json'
        plan=dict(previous_freeze=c.artifact(c.PREVIOUS_OUT/'freeze.json'),inputs=[c.artifact(observer)])
        # Before freeze construction, use the identical preserved scientific plan.
        with patch.object(cc,'PLAN',c.PREVIOUS_OUT/'collection/plan.json'):
            result=audit.verify_previous_prefix(plan)
        self.assertEqual(len(result['successful_actual_exits']),191)
        self.assertEqual(result['previous_attempt_status'],'incomplete')
        self.assertEqual(c.read(c.resolve(result['failed_execution']))['status'],'failed')
        self.assertFalse(c.read(c.resolve(result['failed_operation_cleanup']))['success'])
        bad=copy.deepcopy(plan);bad['previous_freeze']['sha256']='0'*64
        with self.assertRaises(ValueError):audit.verify_previous_prefix(bad)


if __name__=='__main__':unittest.main()
