"""Model-free regression tests for the separate CPU4 trainer and evidence contract."""
from copy import deepcopy
from pathlib import Path
import ast
import math
import sys
import types
import unittest
from unittest.mock import patch,Mock
from . import fit


def valid_metadata(size=256,rep=0,step=445):
    one,roots=(445,14218) if size==256 else (889,28426)
    return dict(study_id=fit.c.STUDY,fit_id=f'g{size}_r{rep}_mixed',game_count=size,repetition=rep,
        seed=fit.sampling.SEEDS[rep],checkpoint_step=step,breadth='mixed',scope='joint',qualification_only=False,
        source_training_games=size,fixed_final_step=1782,batch_size=32,parameter_count=fit.accepted.PARAMETERS,
        model_cpu=4,original_reference_cpu=0,one_pass_step=one,training_records=roots,last_batch_size=10,
        retained_checkpoint_steps=[0,one-1,one,1781,1782],endpoint='onepass' if step==one else 'matched' if step==1782 else 'boundary',
        historical_data_relabelled=False,historical_warm_start_sha256=fit.accepted.PARENT_SHA256,
        model_source_sha256=fit.accepted.MODEL_SOURCE_SHA256,
        optimizer_recipe={k:v for k,v in fit.accepted.RECIPE.items() if k not in ('epochs','shuffle','example_weight','batch_reduction')},
        training_initializer='original_EF8A_and_fresh_AdamW_same_seed',checkpoint_selection=False,best_seed_selection=False,
        cpu0_training_executed=False,value_target_contract=fit.targets.contract('mixed'),
        batch_reduction='mean_actual_batch_size_equal_record_weights',record_presentations=roots,
        selection_sha256='1'*64,schedule_sha256='2'*64,value_target_sources_sha256='3'*64,
        corpus_cache=fit.c.artifact(fit.c.CACHE_ACCEPTANCE),**fit.IDENTITY)


class TrainerContract(unittest.TestCase):
    def setUp(self):
        self.verify=patch.object(fit.c,'verify',return_value=None);self.verify.start()
    def tearDown(self):self.verify.stop()

    def test_original_math_exact_except_partial_admission(self):
        result=fit.source_equivalence()
        self.assertEqual(result['optimizer_calls'],0)
        self.assertEqual(result['actual_partial_range'],[1,32])

    def test_partial_admission_all_genuine_lengths(self):
        tree=ast.parse(Path(fit.__file__).read_text())
        function=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='step')
        admission=function.body[4].value.args[0]
        expression=compile(ast.Expression(admission),'<actual admission>','eval')
        for size in range(35):
            self.assertEqual(eval(expression,{'batch':{'tokens':[0]*size},'BATCH':32}),1<=size<=32)

    def test_valid_endpoint_metadata(self):
        for size,one in ((256,445),(512,889)):
            for rep in (0,1):
                for step in (0,one-1,one,1781,1782):
                    meta=valid_metadata(size,rep,step);meta['record_presentations']=0 if step==0 else min(meta['training_records'],step*32)
                    self.assertEqual(fit.validate_checkpoint_metadata(meta),meta)

    def test_reject_mutated_training_contract(self):
        changes={'game_count':True,'repetition':True,'checkpoint_step':444.0,'seed':5,'fit_id':'g256_r0_terminal',
            'scope':'policy','breadth':'terminal','model_cpu':0,'fixed_final_step':1783,'last_batch_size':32,
            'retained_checkpoint_steps':[0,445,1782],'endpoint':'matched','historical_data_relabelled':True,
            'historical_warm_start_sha256':'f'*64,'model_source_sha256':'f'*64,'checkpoint_selection':True,
            'best_seed_selection':True,'cpu0_training_executed':True,'optimizer_recipe':{},'value_target_contract':{},
            'training_initializer':'continued_model','batch_reduction':'fixed_denominator32',
            'record_presentations':999999,'selection_sha256':'bad','ruleset_id':'legacy',
            'catalogue_sha256':'f'*64,'turn_contract':'other'}
        for key,value in changes.items():
            with self.subTest(key=key),self.assertRaises((ValueError,KeyError,TypeError)):
                fit.validate_checkpoint_metadata(dict(valid_metadata(),**{key:value}))

    def test_actual_partial_qualification_metadata(self):
        meta=valid_metadata(step=1);meta.update(qualification_only=True,qualification_batch_size=10,
            endpoint='qualification',record_presentations=10)
        fit.validate_checkpoint_metadata(meta)
        for size in (0,9,11,32,True):
            with self.subTest(size=size),self.assertRaises(ValueError):
                fit.validate_checkpoint_metadata(dict(meta,qualification_batch_size=size))

    def test_update_comparison_requires_complete_evidence(self):
        update={key:0 for key in fit.UPDATE_KEYS};update['seconds']=1
        expected={**update,'seconds':20,'step':1782}
        fit.compare_update(update,expected)
        for key in fit.UPDATE_KEYS-{'seconds'}:
            altered=deepcopy(update);altered[key]=1
            with self.subTest(key=key),self.assertRaises(ValueError):fit.compare_update(altered,expected)
        with self.assertRaises(ValueError):fit.compare_update({},expected)

    def test_partial_saved_losses_use_real_ten_row_denominator(self):
        examples=[dict(game_index=i,ply=2,split='train',root_input_sha256='1'*64,row_sha256='2'*64,
            mask=[True,True]+[False]*79,policy=[1.]+[0.]*80,value=([.5,-.5] if i<5 else [1.,-1.])) for i in range(10)]
        outputs=[[0.]*83 for _ in examples]
        result=fit.saved_batch_statistics(examples,outputs)
        self.assertAlmostEqual(result['policy_mean_unweighted'],math.log(2.))
        self.assertEqual(result['value_mean_unweighted'],.625)
        self.assertAlmostEqual(result['loss'],math.log(2.)+.625)
        self.assertEqual(fit.saved_batch_statistics(examples*3,outputs*3),fit.old.saved_batch_statistics(examples*3,outputs*3))
        for count in (0,33):
            with self.assertRaises(ValueError):fit.saved_batch_statistics([examples[0]]*count,[[0.]*83]*count)

    def test_saved_prediction_routes_exist_and_are_bound(self):
        for model_id in ('parent','r1_mixed_joint'):
            item,packet=fit._saved_prediction(model_id)
            self.assertEqual(packet['model_id'],model_id)
            self.assertEqual(len(packet['rows']),32)
            self.assertEqual(item['sha256'],fit.c.sha(fit.c.resolve(item)))

    def test_no_old_engine_calls(self):
        tree=ast.parse(Path(fit.__file__).read_text())
        forbidden=[]
        for node in ast.walk(tree):
            if isinstance(node,ast.Call) and ast.unparse(node.func) in ('old.engine','old.run_fit','old.run_qualification'):
                forbidden.append(ast.unparse(node))
        self.assertEqual(forbidden,[])


class PlacementAndRng(unittest.TestCase):
    def test_seed_uses_exact_original_repetition_seed(self):
        numpy=types.SimpleNamespace(random=types.SimpleNamespace(seed=Mock()))
        torch=types.SimpleNamespace(manual_seed=Mock())
        with patch.dict(sys.modules,{'numpy':numpy,'torch':torch}),patch.object(fit.random,'seed') as pyseed:
            for rep in (0,1):
                seed=fit.seed_rng(rep)
                self.assertEqual(seed,fit.sampling.SEEDS[rep]);pyseed.assert_called_with(seed)
                numpy.random.seed.assert_called_with(seed);torch.manual_seed.assert_called_with(seed)
            with self.assertRaises(ValueError):fit.seed_rng(True)

    def test_cpu4_only_one_thread_and_one_guard_call(self):
        class Process:
            pid=100
            affinity=[0,1,2,3,4,5,6,7]
            def cpu_affinity(self,value=None):
                if value is not None:self.affinity=value
                return self.affinity
            def nice(self,value):self.priority=value
            def create_time(self):return 42.
        process=Process()
        psutil=types.SimpleNamespace(Process=lambda:process,NORMAL_PRIORITY_CLASS=32)
        torch=types.SimpleNamespace(set_num_threads=Mock(),set_num_interop_threads=Mock())
        with patch.object(fit,'_STARTUP',None),patch.object(fit,'_ENGINE',None),patch.object(fit.c,'save'), \
            patch.object(fit.guard,'verify_runtime_sources',return_value=deepcopy(fit.guard.SOURCE_HASHES)) as guard, \
            patch.dict(sys.modules,{'psutil':psutil,'torch':torch}):
            engine=fit.engine('unused');fit.engine('unused2')
            guard.assert_called_once();self.assertEqual(process.affinity,[4]);self.assertEqual(engine['cpu_affinity'],[4])
            torch.set_num_threads.assert_called_once_with(1);torch.set_num_interop_threads.assert_called_once_with(1)

    def test_preverified_guard_never_reexecutes_and_rejects_changed_receipt(self):
        with patch.object(fit,'_STARTUP',None),patch.object(fit.guard,'verify_runtime_sources') as guard:
            receipt=deepcopy(fit.guard.SOURCE_HASHES)
            self.assertEqual(fit.startup(receipt),receipt);fit.startup(receipt);guard.assert_not_called()
            with self.assertRaises(ValueError):fit.startup({'bad':'receipt'})
        with patch.object(fit,'_STARTUP',None):
            with self.assertRaises(ValueError):fit.startup({'bad':'receipt'})
            self.assertIsNone(fit._STARTUP)


if __name__=='__main__':unittest.main()
