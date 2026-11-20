"""Model-free counterexamples for the fixed-compute sampling study."""
from copy import deepcopy
from unittest.mock import patch
import pytest

from tools.splendorust_sampling_stability_20261027 import audit as a, common as c, design
from tools.splendorust_decision_quality_20261027_v2 import audit as old
from test_splendorust_decision_quality_20261027_v2_audit import synthetic_result


@pytest.fixture(scope='module')
def panel():
    return c.read(c.ROOT/'outputs/decision_quality_preparation_20261027_01/panel.json')


@pytest.fixture(scope='module')
def source(panel):
    descriptor=panel['roots'][0]
    row=c.read(c.ROOT/descriptor['source']['path'])
    clock=c.artifact(c.ROOT/descriptor['source_worker_clock_file'])
    return descriptor,row,dict(inputs=[descriptor['source'],clock])


def matrix():
    return c.read(c.OUT/'prepared_request_matrix.json')


def sampling_result(source,pool=3,label='A'):
    result,observer,_=synthetic_result(source)
    pid=f'sampling_puct64_pool{pool}_20261027_v1'
    identity=dict(profile_id=pid,profile_sha256='1'*64,wrapper_id=pid+'_1s',native_binary_sha256='2'*64)
    result.update({key:value for key,value in identity.items() if key!='native_binary_sha256'})
    result['diagnostics'].update(profile_id=pid,observer_seed=a.seed_for(observer,label))
    result['research'].update(schema=a.RESEARCH_SCHEMA,internal_seconds=.9,world_pool=pool,
        seed_label=label,seed=a.seed_for(observer,label))
    return result,observer,identity


def check(result,observer,identity,pool=3):
    return a.validate_research_result(result,observer,quota=64,world_pool=pool,
        seed_label=result['research']['seed_label'],**identity)


def test_matrix_has_all288_cells_and_exact12_repeats(panel):
    value=matrix()
    assert value==design.build_matrix()
    assert a.validate_matrix(value,panel)==dict(status='passed',requests=324,sampling_cells=288,repeat_cells=12)


@pytest.mark.parametrize('label',list('ABCDEF'))
def test_observer_only_seed_contract_and_legacy_abc_exact(panel,label):
    observer=panel['roots'][0]['observer']
    assert a.seed_for(observer,label)==design.seed(observer,label)
    if label in 'ABC': assert a.seed_for(observer,label)==old.seed_for(observer,label)


@pytest.mark.parametrize('change',[
    lambda m:m['requests'][24].update(world_pool=6),
    lambda m:m['requests'][25].update(seed_family='G',seed_label='G'),
    lambda m:m['requests'][100].update(complete_response_seconds=30.),
    lambda m:m['requests'][100].update(quota=256),
    lambda m:m['requests'][312].update(repeat_of=24),
    lambda m:m['requests'][23].update(root_id='root_00'),
    lambda m:m['requests'][95].update(stage='research_main'),
    lambda m:m['requests'].pop(),
])
def test_matrix_counterexamples_rejected(panel,change):
    value=matrix();change(value)
    with pytest.raises((ValueError,KeyError)):a.validate_matrix(value,panel)


@pytest.mark.parametrize('pool,label',[(3,'A'),(3,'F'),(6,'A'),(6,'F')])
def test_finite_science_same64_work_explicit_sampling_identity(source,pool,label):
    result,observer,identity=sampling_result(source,pool,label)
    checked=check(result,observer,identity,pool)
    assert checked['simulations']==64 and checked['neural_callbacks']<=65


@pytest.mark.parametrize('field,value',[
    ('world_pool',6),('world_pool',True),('internal_seconds',29.),('quota',256),
    ('schema','splendorust_saved_decision_research_v1'),('seed',1),
])
def test_sampling_annotations_cannot_relabel_pool_work_clock_or_rng(source,field,value):
    result,observer,identity=sampling_result(source)
    result['research'][field]=value
    with pytest.raises(ValueError):check(result,observer,identity)


def test_profile_requires_exact_pool_and_short_clock():
    pid='sampling_puct64_pool6_20261027_v1'
    profile=dict(profile_id=pid,profile_sha256='1'*64,wrapper_id=pid+'_1s',iterations=64,
        world_pool=6,internal_budget_seconds=.9,native_binary_sha256='2'*64)
    freeze={'research_profiles':{'64-6':profile}}
    assert a.profile_identity(freeze,64,6)['profile_id']==pid
    for field,value in [('world_pool',3),('internal_budget_seconds',29.),('iterations',256),('wrapper_id',pid+'_30s')]:
        bad=deepcopy(freeze);bad['research_profiles']['64-6'][field]=value
        with pytest.raises(ValueError):a.profile_identity(bad,64,6)


def test_previous_actual_value_semantics_kept_without_model_or_history_relabel():
    row=c.read(c.ROOT/'outputs/splendorust_decision_quality_20261027_02/records/0024.json')
    result=deepcopy(row['clock']['result']);pid='sampling_puct64_pool3_20261027_v1'
    identity=dict(profile_id=pid,profile_sha256='1'*64,wrapper_id=pid+'_1s',native_binary_sha256='2'*64)
    result.update({key:value for key,value in identity.items() if key!='native_binary_sha256'})
    result['diagnostics']['profile_id']=pid
    result['research'].update(schema=a.RESEARCH_SCHEMA,world_pool=3,internal_seconds=.9,native_binary_sha256='2'*64)
    checked=check(result,row['observer'],identity)
    stat=checked['root_statistic_checks'][0]
    assert stat['initial_network_win_credit']!=stat['post_search_root_node_value']
    assert a.scientific_core(result)==old.scientific_core(row['clock']['result'])
    assert row['clock']['deadline_seconds']==30.  # Test fixture projection only.


def prefix_records(value):
    records=[]
    for spec in value['requests']:
        result={'cell':[spec['root_id'],spec['world_pool'],spec['seed_family']]}
        records.append(dict(spec=spec,completed=True,clock=dict(status='ok',request_id=f"r{spec['sequence']}",result=result),
            worker_clock_file=f"temp/r{spec['sequence']}.json"))
    return records


def test_full_inventory_and_replays_keep_pool_in_identity(panel):
    value=matrix();records=prefix_records(value)
    checked=lambda row,*args,**kw:dict(complete=True,status='passed')
    loaded={root['root_id']:{} for root in panel['roots']}
    with patch.object(a,'audit_record',side_effect=checked):
        result=a.audit_prefix(records,{},value,panel,saved_roots=loaded)
        assert result['complete'] and result['completed_requests']==324
        records[312]['clock']['result']['cell'][1]=6
        with pytest.raises(ValueError,match='exact full research reproduction'):
            a.audit_prefix(records,{},value,panel,saved_roots=loaded)


def test_rejected_one_second_response_remains_failed_science(source):
    _,row,_=source
    receipt=deepcopy(row['clock']);receipt.update(status='illegal',accepted=False,detail='test checker rejection')
    a.check_clock(receipt,limit=1.)
    assert receipt['result']==row['clock']['result'] and receipt['accepted'] is False
    receipt['deadline_seconds']=30.
    with pytest.raises(ValueError):a.check_clock(receipt,limit=1.)
