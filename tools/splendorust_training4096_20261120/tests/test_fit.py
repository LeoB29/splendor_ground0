"""No-model counterexamples for exact exposure, namespace and checkpoint scope."""
from copy import deepcopy
import ast
import inspect
import sys
import pytest

from tools.splendorust_training4096_20261120 import fit


def test_numerical_functions_are_direct_accepted_aliases():
    result=fit.source_equivalence()
    assert result['status']=='unchanged_accepted_numeric_functions'
    assert fit.recorded_step is fit.numeric.recorded_step
    assert fit.loss_components is fit.old.loss_components
    assert 'torch' not in sys.modules


def test_prepared_schedule_complete_and_exact_exposure():
    for name in fit.c.FIT_IDS:
        spec,order=fit.load_order(name)
        assert len(order)==227203
        assert sum(len(fit.batch_indices(order,s)) for s in range(1,7102))==227203
        assert len(fit.batch_indices(order,7101))==3
        assert len(set(order))==(227203 if spec['arm']=='larger' else 57022)


@pytest.mark.parametrize('step',[0,-1,7102,True,1.0])
def test_changed_update_index_rejected(step):
    with pytest.raises(ValueError):fit.batch_indices(list(range(fit.ROOTS)),step)


def test_missing_final_position_rejected():
    with pytest.raises(ValueError):fit.batch_indices(list(range(fit.ROOTS-1)),7101)


def test_all_three_pass_boundary_batches_are_actual_crossings():
    assert fit.QUALIFICATION_BATCHES == (('full32',1),('final3',7101),('cross1',1782),('cross2',3564),('cross3',5346))
    for number,(_,step) in enumerate(fit.QUALIFICATION_BATCHES[2:],1):
        boundary=number*57022
        assert (step-1)*32 < boundary < step*32
    assert 2+2*len(fit.QUALIFICATION_BATCHES)==12 <=16


def test_source_identity_cannot_collapse_equal_game_numbers():
    prior=dict(source='prior_20261107',game_index=0,ply=0,row_sha256='a'*64)
    fresh=dict(prior,source='fresh_20261115')
    assert fit.batch_source([prior])!=fit.batch_source([fresh])
    assert fit.batch_source([prior,fresh])!=fit.batch_source([fresh,prior])


@pytest.fixture
def meta(monkeypatch):
    monkeypatch.setattr(fit.c,'verify',lambda *a,**k:None)
    value=fit.metadata('larger_r0',dict(path='accepted-cache',bytes=1,sha256='a'*64),'b'*64)
    value.update(checkpoint_step=7101,record_presentations=227203)
    return value


def test_exact_scientific_checkpoint_metadata(meta):
    assert fit.validate_checkpoint_metadata(meta)==meta


@pytest.mark.parametrize('field,value',[
    ('checkpoint_step',7100),('checkpoint_step',True),('record_presentations',227200),
    ('training_record_presentations',228088),('last_batch_size',30),('scope','policy'),
    ('historical_data_relabelled',True),('checkpoint_selection',True),('best_seed_selection',True),
    ('training_initializer','continued_old_optimizer'),('seed',202611072),('model_cpu',4),
    ('model_cpu',False),('repetition',False),
    ('qualification_only','false'),('ruleset_id','legacy'),
    ('retained_checkpoint_steps',[0,3551,7101]),('value_target_contract',{}),
])
def test_changed_checkpoint_recipe_or_exposure_rejected(meta,field,value):
    changed=deepcopy(meta);changed[field]=value
    with pytest.raises((ValueError,TypeError,KeyError)):fit.validate_checkpoint_metadata(changed)


def test_final_three_disposable_checkpoint_is_not_scientific(meta):
    value=deepcopy(meta);value.update(qualification_only=True,qualification_case='final3',
        qualification_batch_size=3,checkpoint_step=1,record_presentations=3)
    assert fit.validate_checkpoint_metadata(value)['qualification_only'] is True
    value['qualification_only']=False
    with pytest.raises(ValueError):fit.validate_checkpoint_metadata(value)


def test_disposable_full_batch_cannot_claim_partial_denominator(meta):
    value=deepcopy(meta);value.update(qualification_only=True,qualification_case='cross1',
        qualification_batch_size=3,checkpoint_step=1,record_presentations=3)
    with pytest.raises(ValueError):fit.validate_checkpoint_metadata(value)


def test_schedule_replacement_and_missing_source_cannot_pass(meta):
    meta['schedule']=dict(meta['schedule'],sha256='0'*64)
    with pytest.raises(ValueError):fit.validate_checkpoint_metadata(meta)


def test_startup_guard_called_once_with_shared_receipt(monkeypatch):
    monkeypatch.setattr(fit,'_STARTUP',None)
    calls=[]
    monkeypatch.setattr(fit.guard,'verify_runtime_sources',lambda:calls.append(1) or deepcopy(fit.guard.SOURCE_HASHES))
    assert fit.startup()==fit.guard.SOURCE_HASHES
    assert fit.startup(fit.guard.SOURCE_HASHES)==fit.guard.SOURCE_HASHES
    assert calls==[1]
    with pytest.raises(ValueError):fit.startup({'changed':'source'})


def test_preverified_worker_guard_is_not_called_again(monkeypatch):
    monkeypatch.setattr(fit,'_STARTUP',None)
    def forbidden():raise AssertionError('second runtime source call')
    monkeypatch.setattr(fit.guard,'verify_runtime_sources',forbidden)
    assert fit.startup(fit.guard.SOURCE_HASHES)==fit.guard.SOURCE_HASHES


@pytest.mark.parametrize('entry', ['fit','qualification'])
def test_opened_assessment_blocks_before_model_or_loader(monkeypatch,tmp_path,entry):
    monkeypatch.setattr(fit.c,'OUT',tmp_path)
    (tmp_path/'assessment_opened.json').write_text('{}',encoding='utf-8')
    def forbidden(*a,**k):raise AssertionError('Model startup must not run')
    monkeypatch.setattr(fit,'engine',forbidden)
    with pytest.raises(ValueError,match='after assessment opens'):
        if entry=='fit':fit.run_fit('larger_r0',tmp_path/'worker')
        else:fit.run_qualification(tmp_path/'worker')


@pytest.fixture
def saved_update():
    path=fit.c.ORIGINAL_OUT/'fits/r0_mixed_joint/worker/updates/1782.json'
    return fit.c.read(path)


def test_accepted_saved_update_can_change_elapsed_time_only(saved_update):
    actual={k:deepcopy(saved_update[k]) for k in fit.numeric.UPDATE_KEYS}
    actual['seconds']+=10
    fit.compare_update(actual,saved_update)


@pytest.mark.parametrize('mutation', ['missing_gradient','changed_gradient','changed_output','changed_loss','changed_denominator'])
def test_saved_update_corruption_rejected(saved_update,mutation):
    actual={k:deepcopy(saved_update[k]) for k in fit.numeric.UPDATE_KEYS}
    if mutation=='missing_gradient':actual.pop('gradient_sha256')
    if mutation=='changed_gradient':actual['gradient_sha256'][next(iter(actual['gradient_sha256']))]='0'*64
    if mutation=='changed_output':actual['outputs83'][0][0]+=0.25
    if mutation=='changed_loss':actual['loss']+=0.25
    if mutation=='changed_denominator':actual['batch_denominator']=32
    with pytest.raises(ValueError):fit.compare_update(actual,saved_update)


@pytest.fixture
def saved_trees():
    zero=fit.c.read(fit.c.resolve(fit.source_checkpoint(0,0)))
    final=fit.c.read(fit.c.resolve(fit.source_checkpoint(0)))
    return (fit.c.read(fit.c.resolve(zero['tensor_manifest'])),
            fit.c.read(fit.c.resolve(final['tensor_manifest'])))


def test_saved_state_inventory_authenticates_actual_architecture_without_torch(saved_trees):
    zero,final=saved_trees
    before=fit._validate_saved_tensor_inventory(zero,{'checkpoint_step':0},zero)
    after=fit._validate_saved_tensor_inventory(final,{'checkpoint_step':1782},zero)
    assert before['optimizer']['state']=={}
    assert len(after['optimizer']['state'])==len(after['state_dict'])
    assert 'torch' not in sys.modules


@pytest.mark.parametrize('component',['state_dict','optimizer','rng'])
def test_missing_saved_state_component_rejected(saved_trees,component):
    zero,final=saved_trees
    changed=deepcopy(final);changed.pop(component)
    with pytest.raises(ValueError):fit._validate_saved_tensor_inventory(changed,{'checkpoint_step':1782},zero)


def test_saved_step_and_architecture_corruption_rejected(saved_trees):
    zero,final=saved_trees
    with pytest.raises(ValueError,match='step bytes'):
        fit._validate_saved_tensor_inventory(final,{'checkpoint_step':1781},zero)
    changed=deepcopy(final)
    changed['state_dict']['dict'][0][1]['shape'][0]+=1
    with pytest.raises(ValueError):fit._validate_saved_tensor_inventory(changed,{'checkpoint_step':1782},zero)


@pytest.mark.parametrize('node',[
    {'type':'int','value':True},
    {'float64':'000000000000f87f'},
    {'dict':[[{'type':'str','value':'same'},{'type':'int','value':1}],
             [{'type':'str','value':'same'},{'type':'int','value':2}]]},
    {'tensor':True,'dtype':'torch.float32','shape':[2],'bytes':4,'sha256':'a'*64},
    {'tensor':True,'dtype':'torch.float16','shape':[2],'bytes':4,'sha256':'a'*64},
])
def test_malformed_saved_tensor_nodes_rejected(node):
    with pytest.raises(ValueError):fit._decode_tensor_tree(node)


def test_saved_verifiers_never_deserialize_tensor_archives():
    for fn in (fit.verify_fit,fit.verify_qualification,fit.checkpoint_manifest,
               fit.verify_checkpoint_payload_qualification):
        tree=ast.parse(inspect.getsource(fn))
        calls=[n.func.id for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)]
        assert 'checkpoint_payload' not in calls
        assert not any(isinstance(n,(ast.Import,ast.ImportFrom)) and
                       ('torch' in ast.unparse(n)) for n in ast.walk(tree))
    assert 'torch' not in sys.modules


def test_payload_receipt_requires_every_checkpoint_and_exact_archive(monkeypatch):
    descriptor={'path':'checkpoint.json','bytes':2,'sha256':'a'*64}
    manifest={'checkpoint':{'path':'weights.pt','bytes':10,'sha256':'b'*64},
              'tensor_manifest':{'path':'tensors.json','bytes':10,'sha256':'c'*64},
              'tree_sha256':{'state_dict':'d'*64,'optimizer':'e'*64,'rng':'f'*64}}
    row=dict(manifest=descriptor,actual_payload_deserialized=True,**manifest)
    proof=dict(status='all_checkpoint_payloads_verified',optimizer_calls=0,model_forward_calls=0,checkpoints=[row])
    monkeypatch.setattr(fit.c,'verify',lambda *a,**k:None)
    monkeypatch.setattr(fit.c,'read',lambda *a:proof)
    monkeypatch.setattr(fit,'checkpoint_manifest',lambda *a,**k:({},manifest))
    assert fit.verify_checkpoint_payload_qualification(descriptor,[descriptor])==proof
    proof['checkpoints']=[]
    with pytest.raises(ValueError):fit.verify_checkpoint_payload_qualification(descriptor,[descriptor])
    proof['checkpoints']=[dict(row,checkpoint=dict(manifest['checkpoint'],sha256='0'*64))]
    with pytest.raises(ValueError):fit.verify_checkpoint_payload_qualification(descriptor,[descriptor])
