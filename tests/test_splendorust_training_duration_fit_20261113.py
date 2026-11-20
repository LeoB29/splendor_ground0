"""Model-free tests; actual continuation/parity runs only in frozen qualification."""
from copy import deepcopy
import sys
import pytest
from tools.splendorust_training_duration_20261113 import fit


def test_exact_numerical_calls_and_actual_fixture_source():
    assert fit.source_equivalence()['status']=='unchanged_continuation_math_and_actual_fixture_route'
    assert fit.recorded_step is fit.old.recorded_step
    assert fit.selected_batch is fit.old.selected_batch
    assert fit.make_optimizer is fit.old.make_optimizer
    assert fit.restore_rng is fit.old.restore_rng


@pytest.fixture(scope='module')
def schedules():
    refs=[[i//60,i%60] for i in range(57022)]
    selection={'cells':[dict(fit_id=f'r{r}_mixed_joint',root_ids=refs) for r in (0,1)]}
    return selection,[fit.build_schedules(selection,r) for r in (0,1)]


def test_all_passes_complete_distinct_and_last30(schedules):
    selection,both=schedules
    for rep,schedule in enumerate(both):
        assert fit.build_schedules(selection,rep)==schedule
        assert list(schedule)==['2','3','4']
        for batches in schedule.values():
            assert len(batches)==1782 and len(batches[-1])==30
            assert all(len(batch)==32 for batch in batches[:-1])
            assert sorted(i for batch in batches for i in batch)==list(range(57022))
    assert both[0]!=both[1]


@pytest.mark.parametrize('change',[
    lambda x:x['2'][-1].append(0),
    lambda x:x['2'][0].__setitem__(0,x['2'][0][1]),
    lambda x:x['2'][0].__setitem__(0,True),
    lambda x:x.__setitem__('3',deepcopy(x['2'])),
    lambda x:x.pop('4'),
])
def test_missing_duplicated_padded_or_adaptive_orders_rejected(schedules,change):
    value=deepcopy(schedules[1][0]);change(value)
    with pytest.raises(ValueError):fit.validate_schedules(value)


@pytest.mark.parametrize('rep',[True,2,-1,0.0,'0'])
def test_nonfixed_seed_rejected(schedules,rep):
    with pytest.raises(ValueError):fit.build_schedules(schedules[0],rep)


def metadata(index=3564,qualification=False):
    return dict(study_id=fit.c.STUDY,fit_id='r0_mixed_duration',repetition=0,
        seed=fit.sampling.SEEDS[0],breadth='mixed',scope='joint',**fit.IDENTITY,
        qualification_only=qualification,checkpoint_step=index,training_records=57022,
        source_training_games=1024,batch_size=32,last_batch_size=30,fixed_final_step=7128,
        continuation_starts_at_step=1782,parameter_count=fit.accepted.PARAMETERS,
        complete_passes=index//1782,record_presentations=index//1782*57022+index%1782*32,
        checkpoint_selection=False,best_seed_selection=False,diagnostic_passes=[1,2,4],
        original_optimizer_and_rng_restored=True,historical_data_relabelled=False,
        historical_warm_start_sha256=fit.accepted.PARENT_SHA256,value_target_contract=fit.targets.contract('mixed'),
        batch_reduction='mean_actual_batch_size_equal_record_weights',selection_sha256='a'*64,
        schedule_sha256='b'*64,value_target_sources_sha256='c'*64,
        source_one_pass_checkpoint={'path':'fixture1','bytes':1,'sha256':'d'*64},
        corpus_cache={'path':'fixture2','bytes':1,'sha256':'e'*64})


@pytest.mark.parametrize('step',[3563,3564,5345,5346,7127,7128])
def test_absolute_steps_cover_all_final30_boundaries(monkeypatch,step):
    monkeypatch.setattr(fit.c,'verify',lambda *args,**kwargs:None)
    fit.validate_checkpoint_metadata(metadata(step))


@pytest.mark.parametrize('key,value',[
    ('checkpoint_step',1782),('checkpoint_step',3564.0),('qualification_only',1),
    ('repetition',False),('seed',0),('breadth','terminal'),('scope','policy'),
    ('training_records',3072),('last_batch_size',32),('complete_passes',4),
    ('record_presentations',114048),('checkpoint_selection',True),('best_seed_selection',True),
    ('diagnostic_passes',[1,2,3,4]),('original_optimizer_and_rng_restored',False),
    ('historical_data_relabelled',True),('value_target_contract',{}),
    ('schedule_sha256','bad'),('batch_reduction','divide_by_32'),
])
def test_wrong_optimizer_continuity_and_metadata_rejected(monkeypatch,key,value):
    monkeypatch.setattr(fit.c,'verify',lambda *args,**kwargs:None)
    meta=metadata();fit.validate_checkpoint_metadata(meta);meta[key]=value
    with pytest.raises(ValueError):fit.validate_checkpoint_metadata(meta)


@pytest.mark.parametrize('step',[1782,1783])
def test_qualification_checkpoints_are_disposable_and_separate(monkeypatch,step):
    monkeypatch.setattr(fit.c,'verify',lambda *args,**kwargs:None)
    fit.validate_checkpoint_metadata(metadata(step,True))


@pytest.mark.parametrize('n',[30,32])
def test_unchanged_loss_uses_actual_batch_denominator(n):
    row=dict(game_index=0,ply=0,split='train',root_input_sha256='a'*64,row_sha256='b'*64,
        mask=[True,True]+[False]*79,policy=[1.,0.]+[0.]*79,value=[1.,-1.])
    expected=fit.old.saved_batch_statistics([row]*n,[[0.]*83]*n)
    import math
    assert expected['loss']==pytest.approx(1+math.log(2))


def test_exact_update_comparison_ignores_only_elapsed_time():
    actual={'seconds':1,'outputs83':[[1.]],'gradient_sha256':{'x':'a'},'loss':.2}
    expected={**actual,'seconds':2,'step':1783}
    fit.compare_update(actual,expected)
    for key,value in [('outputs83',[[1.0000001]]),('gradient_sha256',{'x':'b'}),('loss',.2000001)]:
        changed={**expected,key:value}
        with pytest.raises(ValueError):fit.compare_update(actual,changed)


def test_schedule_construction_does_not_consume_python_rng(schedules):
    import random
    before=random.getstate();fit.build_schedules(schedules[0],0)
    assert before==random.getstate()
