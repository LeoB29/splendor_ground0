from copy import deepcopy
import math
import struct
import sys
import pytest
from tools.splendorust_value_targets_20261110 import targets, sampling, fit, corpus, offline
from tools.splendorust_independent_data_20261107 import corpus as original_corpus
from tools.splendorust_independent_data_20261107.collection import data as source_data

def test_unchanged_optimizer_loss_call_rng_math():
    assert fit.source_equivalence()['status']=='unchanged_joint_update_math_verified'
    assert 'torch' not in sys.modules

@pytest.mark.parametrize('z,q,expected',[(1.,1.,1.),(-1.,0.,-1.),(0.,.5,0.),(1.,.5,1/1.837),(-1.,1.,(-1+.837)/1.837)])
def test_signed_mixture_and_opponent_orientation(z,q,expected):
    value=targets.mixture([z,-z],q)
    assert value==[expected,-expected]
    assert targets.mixture([-z,z],1-q)==[-expected,expected]

@pytest.mark.parametrize('terminal,q', [([True,-1],.5),([1,0],.5),([1.,-1.],True),([1.,-1.],math.nan),([0.,0.],1.01),([.5,-.5],.5)])
def test_invalid_outcome_or_root_credit_rejected(terminal,q):
    with pytest.raises(ValueError):targets.mixture(terminal,q)

@pytest.fixture(scope='module')
def actual():
    selection=sampling.prepare_selection();game=selection['games'][0]
    examples=original_corpus.load_selected_game(game,selection['source_receipt'])['examples'][:2]
    document=source_data.read_artifact(game['source_game'])
    return selection,game,examples,document

def test_exact_saved_broad_schedule_and_seeds(actual):
    selection=actual[0];old=sampling.source_selection(selection)
    assert selection['cells']==[x for x in old['cells'] if x['breadth']=='broad']
    assert len(selection['terminal_controls'])==2
    assert all(x['roots']==3072 and x['optimizer_steps']==96 for x in selection['cells'])

@pytest.mark.parametrize('i',[0,1])
def test_actual_saved_root_q_both_actor_seats(actual,i):
    _,game,examples,document=actual
    example=examples[i];item=document['rows'][example['ply']];row=source_data.read_artifact(item)
    value=targets.attach(example,row,item,game['spec'])
    assert value['actor']==i
    assert value['tokens']==example['tokens'] and value['policy']==example['policy'] and value['mask']==example['mask']
    assert value['terminal_value']==example['value']
    assert value['value']==targets.mixture(example['value'],row['training_capture']['native_capture']['root_value'])
    assert value['root_q_signed']==2*value['root_q_win_credit']-1

@pytest.mark.parametrize('change',[
    lambda x:x.__setitem__('split','offline'),
    lambda x:x.__setitem__('actor',1-x['actor']),
    lambda x:x.__setitem__('row_sha256','f'*64),
    lambda x:x['policy'].__setitem__(0,x['policy'][0]+.001),
    lambda x:x['tokens'].__setitem__(0,x['tokens'][0]+.001),
])
def test_saved_root_misbindings_rejected(actual,change):
    _,game,examples,document=actual
    example=deepcopy(examples[0]);change(example);item=document['rows'][examples[0]['ply']]
    with pytest.raises((ValueError,KeyError)):targets.attach(example,source_data.read_artifact(item),item,game['spec'])

@pytest.mark.parametrize('change',[
    lambda x:x['cells'][0]['batches'][0].reverse(),
    lambda x:x['cells'].pop(),
    lambda x:x['value_target_contract'].__setitem__('root_q_weight',.5),
    lambda x:x['terminal_controls'][0]['final_checkpoint'].__setitem__('sha256','f'*64),
])
def test_selection_cannot_tune_or_substitute(actual,change):
    selection=deepcopy(actual[0]);change(selection)
    with pytest.raises(ValueError):sampling.validate_selection(selection)

def metadata():
    return dict(**sampling.fit_identity(0,'broad'),**sampling.IDENTITY,study_id=sampling.STUDY,checkpoint_step=96,
        qualification_only=False,historical_data_relabelled=False,historical_warm_start_sha256=fit.accepted.PARENT_SHA256,
        training_records=3072,record_presentations=3072,fixed_final_step=96,complete_passes=1,source_training_games=1024,
        roots_per_game=3,parameter_count=fit.accepted.PARAMETERS,diagnostic_gradient_probes=0,checkpoint_selection=False,
        seed=sampling.SEEDS[0],value_target_contract=targets.contract(),value_target_sources_sha256='a'*64)

@pytest.mark.parametrize('key,value',[('checkpoint_step',96.),('value_target_contract',{}),('value_target_sources_sha256',''),
    ('repetition',False),('source_training_games',192),('roots_per_game',16)])
def test_strict_changed_checkpoint_metadata(key,value):
    meta=metadata();fit.validate_checkpoint_metadata(meta);meta[key]=value
    with pytest.raises(ValueError):fit.validate_checkpoint_metadata(meta)

def row(game,split,value,ply=0):
    f=lambda x:struct.unpack('<f',struct.pack('<f',x))[0]
    return dict(game_index=game,ply=ply,split=split,root_input_sha256=sampling.digest([game,ply,'root']),
        row_sha256=sampling.digest([game,ply,'row']),legal_mask81=[True,True]+[False]*79,
        policy_target81=[1.,0.]+[0.]*79,value_target2=[1.,-1.],output83=[0.]*81+[f(value),f(-value)])

def test_mixed_terminal_offline_primary_and_independent_arithmetic():
    bundles={model:{'train':[], 'validation':[row(0,'validation',.8 if model.startswith('r') else .4),row(1,'validation',.5)],
        'offline':[row(2,'offline',.7 if model.startswith('r') else .3),row(3,'offline',.4)]} for model in offline.MODEL_IDS}
    result=offline.analyze_predictions(bundles,draws=100,seed=123,strict=False,
        expected_games={'validation':[0,1],'offline':[2,3]})
    assert result['primary']['contrast']=='mixed_minus_terminal' and result['primary']['metric']=='value'
    assert result['primary']['difference']<0
    assert result['checkpoint_selection'] is False and result['offline_strength_admission_gate'] is False
    assert offline.verify_arithmetic(bundles,result)['contrasts']==42
    bad=deepcopy(result);bad['contrasts'][0]['difference']+=.01
    with pytest.raises(ValueError):offline.verify_arithmetic(bundles,bad)

def test_training_loss_uses_actual_float32_continuous_target():
    r=row(0,'train',.2);r['value_target2']=[.123456789,-.123456789]
    loss=targets.prediction_losses(r)
    assert loss['value']==(r['output83'][81]-targets.f32(.123456789))**2
    assert loss['policy']==math.log(2)

def test_prediction_change_diagnostics_are_game_weighted_and_descriptive():
    bundles={model:{'train':[], 'validation':[row(0,'validation',.8 if model.startswith('r') else .4),row(1,'validation',.5)],
        'offline':[row(2,'offline',.7 if model.startswith('r') else .3),row(3,'offline',.4)]} for model in offline.MODEL_IDS}
    result=offline.prediction_changes(bundles)
    assert result['descriptive_only'] and result['raw_policy_is_not_searched_action'] and not result['strength_claim']
    assert result['models']['r0_broad_joint']['mean_game']['mean_absolute_value_drift']>0
    assert result['models']['terminal_r0_broad_joint']['mean_game']['mean_absolute_value_drift']==0

def test_saved_search_error_separate_from_official_outcome_metric():
    bundles={model:{'train':[], 'validation':[row(0,'validation',.8),row(1,'validation',.5)],
        'offline':[row(2,'offline',.7),row(3,'offline',.4)]} for model in offline.MODEL_IDS}
    labels=[{k:r[k] for k in ('game_index','ply','split','row_sha256','root_input_sha256')}|
        {'root_q_signed':r['output83'][81]} for split in offline.SPLITS for r in bundles['parent'][split]]
    result=offline.search_value_errors(bundles,labels)
    assert result['models']['parent']['mean_game_search_value_mse']==0.
    assert result['objective_related_metric_not_independent_outcome_validation'] and not result['strength_claim']
    assert offline.strict_prediction(bundles['parent']['validation'][0])['value']>0.
