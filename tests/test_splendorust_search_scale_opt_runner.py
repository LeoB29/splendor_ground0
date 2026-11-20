"""Model-free predecessor, resource-gate and no-retry orchestration checks."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from tools.splendorust_search_scale_opt_20261024 import runner


def acceptance_records():
    return {
        'freeze.json': {'inputs': [], 'job_seconds': 5400},
        'completion.json': {'status':'completed_verified','failure':None,'freeze_sha256':'bound'},
        'execution.json': {'status':'completed','exit_code':0,'cleanup_errors':[],
                           'freeze_sha256':'bound','charged_job_seconds':500},
        'qualification_outcome.json': {'status':'completed_verified','optimized32_qualified':True,
            'optimized64_qualified':True,'strength_games_launched':0,'next_strength_comparison':'64versus32'},
        'sampler_qualification.json': {'status':'passed','polls':20},
        'qualification32_summary.json': {'qualified':True},
        'ops32_summary.json': {'qualified':True,'games':6},
        'qualification64_summary.json': {'qualified':True},
        'ops64_summary.json': {'qualified':True,'games':6},
        'verification_summary.json': {'status':'passed'},
        'all_game_replay.json': {'status':'passed','games':12},
        'repeated_decision_summary.json': {'status':'passed','requests':24},
        'all_worker_log_integrity.json': {'status':'passed'},
    }


def mock_acceptance(monkeypatch, records):
    monkeypatch.setattr(runner.c,'read',lambda path: records[path.name])
    monkeypatch.setattr(runner.c,'file_hash',lambda path:'bound')
    monkeypatch.setattr(runner.c,'artifact',lambda path:{'path':path.name,'sha256':'bound','bytes':1})
    verified=[]
    monkeypatch.setattr(runner.c,'verify',lambda inputs:verified.append(inputs))
    return verified


def test_complete_qualification_requires_outer_cleanup_and_all_exact_gates(monkeypatch):
    records=acceptance_records()
    verified=mock_acceptance(monkeypatch,records)
    result=runner.qualification_acceptance()
    assert result['status']=='passed' and result['optimized_quotas']==[32,64]
    assert result['saved_timing_rerun_required'] is False
    assert result['actual_entrypoint_requalification_required'] is True
    assert len(result['evidence'])==13 and verified==[[]]


@pytest.mark.parametrize('name,field,value',[
    ('completion.json','status','incomplete'),
    ('completion.json','failure','preserved failure'),
    ('execution.json','status','cleanup_failed'),
    ('execution.json','cleanup_errors',['unconfirmed process']),
    ('execution.json','charged_job_seconds',5400),
    ('execution.json','freeze_sha256','different'),
    ('qualification_outcome.json','optimized64_qualified',False),
    ('qualification_outcome.json','optimized32_qualified',False),
    ('ops64_summary.json','qualified',False),
    ('qualification64_summary.json','qualified',False),
    ('all_game_replay.json','games',11),
    ('repeated_decision_summary.json','requests',23),
])
def test_any_incomplete_or_failed_predecessor_blocks_comparison(monkeypatch,name,field,value):
    records=acceptance_records(); records[name][field]=value
    mock_acceptance(monkeypatch,records)
    with pytest.raises(ValueError): runner.qualification_acceptance()


def test_prepare_stops_before_any_new_evidence_or_freeze_if_predecessor_not_admitted(monkeypatch):
    events=[]
    def rejected():
        events.append('acceptance_gate')
        raise ValueError('not admitted')
    monkeypatch.setattr(runner,'qualification_acceptance',rejected)
    monkeypatch.setattr(runner.c,'save',lambda *args:pytest.fail('No preparation write allowed'))
    monkeypatch.setattr(runner,'verify_runtime_sources',lambda:pytest.fail('Gate must be first'))
    with pytest.raises(ValueError,match='not admitted'): runner.prepare()
    assert events==['acceptance_gate']


def forecast(**overrides):
    values=dict(observed_seconds=60.,observed_variable_bytes=1024*1024,current_bytes=5*1024*1024,
        independent_seconds=2.,fixed_final_seconds=30.,final_evidence_bytes=1024*1024,
        remaining_collection_seconds=14400.)
    values.update(overrides)
    return runner.operational_gate(**values)


def test_forecast_uses_conservative_factor_and_all_three_constraints():
    result=forecast()
    assert result['passed'] and result['outcomes_used'] is False
    assert result['arithmetic_factor']==pytest.approx(45.)
    assert result['projected_collection_seconds']==pytest.approx(2700.)
    assert result['projected_prefinal_bytes']==pytest.approx(50*1024*1024)
    assert result['final_arithmetic_factor']==pytest.approx(46.35)
    assert result['projected_final_seconds']==pytest.approx(122.7)
    assert result['projected_final_evidence_bytes']==1024*1024
    assert result['projected_total_evidence_bytes']==pytest.approx(51*1024*1024)


@pytest.mark.parametrize('change',[
    {'remaining_collection_seconds':2700.},
    {'observed_seconds':320.},
    {'observed_variable_bytes':10*1024*1024},
    {'independent_seconds':40.},
    {'fixed_final_seconds':1710.},
    {'final_evidence_bytes':64*1024*1024},
])
def test_forecast_failure_does_not_expand_cap(change):
    result=forecast(**change)
    assert result['passed'] is False


@pytest.mark.parametrize('change',[{'observed_seconds':float('nan')},{'current_bytes':-1},
    {'observed_variable_bytes':True},{'final_evidence_bytes':float('inf')}])
def test_forecast_rejects_invalid_measurements(change):
    with pytest.raises(ValueError,match='finite measured'): forecast(**change)


class Envelope:
    def __init__(self): self.stage=None; self.events=[]
    def begin(self,stage): self.stage=stage; self.events.append(('begin',stage))
    def finish(self): self.events.append(('finish',self.stage))
    def admit(self): pass
    def guard(self): pass
    def remaining(self): return 100.
    def timeout(self,requested): return requested


def test_no_main_games_when_operational_margin_or_forecast_fails(monkeypatch):
    envelope=Envelope(); calls=[]
    monkeypatch.setattr(runner,'qualify',lambda *args:calls.append('qualified') or {})
    def failed(*args):
        calls.append('operational_failed')
        raise ValueError('frozen margin/forecast failed')
    monkeypatch.setattr(runner,'operational',failed)
    monkeypatch.setattr(runner,'play_game',lambda *args:pytest.fail('Main must not launch'))
    monkeypatch.setattr(runner,'final_checks',lambda *args:pytest.fail('Incomplete collection cannot be accepted'))
    with pytest.raises(ValueError,match='frozen margin'): runner.run_stages(envelope,{},[])
    assert calls==['qualified','operational_failed']
    assert [stage for event,stage in envelope.events if event=='begin']==['qualification','operational']


def test_main_is_exactly200_and_startup_clock_not_reset(monkeypatch):
    envelope=Envelope(); envelope.begin('qualification')
    schedule=runner.schedules.make_schedule(); played=[]; completed=[]
    monkeypatch.setattr(runner,'qualify',lambda *args:{'forecast_seconds':1})
    monkeypatch.setattr(runner,'operational',lambda *args:None)
    monkeypatch.setattr(runner,'play_game',lambda spec,*args:played.append(spec))
    monkeypatch.setattr(runner,'final_checks',lambda *args:completed.append(True) or {'verified':True})
    result=runner.run_stages(envelope,{},schedule,qualification_started=True)
    assert len(played)==200 and all(spec['stage']=='main' for spec in played)
    assert len(completed)==1 and result=={'verified':True}
    assert envelope.events.count(('begin','qualification'))==1


def test_no_move_dispatched_if_second_guard_shortens_allowance(monkeypatch,tmp_path):
    state=runner.original_qualification.reference._initial()
    child=SimpleNamespace(challenger_quota=64,_counter=0,output_dir=tmp_path,
                          request=lambda *args,**kwargs:pytest.fail('Clamped request dispatched'))
    envelope=Envelope()
    monkeypatch.setattr(envelope,'timeout',lambda requested:.999)
    monkeypatch.setattr(runner.c,'save',lambda *args:None)
    with pytest.raises(ValueError,match='shortened the next move allowance'):
        runner.decision(child,state,envelope,'challenger')


def test_role_quota_cannot_silently_use_reference16(monkeypatch,tmp_path):
    child=SimpleNamespace(challenger_quota=16)
    with pytest.raises(ValueError,match='Role/quota binding'):
        runner.decision(child,None,Envelope(),'opponent')


def test_new_parent_class_is_exact_qualified_sampler_alias():
    from tools.splendorust_timing_opt_20261024.common import CandidateStudyWorker
    assert runner.c.CandidateStudyWorker is CandidateStudyWorker


def test_existing_attributable_failure_policy_is_not_replaced():
    from tools.splendorust_search_scale_20261024.audit import classify_failure
    assert runner.audit.classify_failure is classify_failure


def test_operational_hash_measurement_reads_every_file_twice(monkeypatch,tmp_path):
    folder=tmp_path/'closed'; folder.mkdir()
    (folder/'a.json').write_bytes(b'123')
    (folder/'b.gz').write_bytes(b'45678')
    monkeypatch.setattr(runner.c,'ROOT',tmp_path)
    real=runner.c.file_hash; calls=[]; boundaries=[]
    def measured(path):
        calls.append(path.name)
        return real(path)
    monkeypatch.setattr(runner.c,'file_hash',measured)
    measured=runner.measure_seal_cost([folder],lambda:boundaries.append(True))
    assert calls==['a.json','b.gz','a.json','b.gz']
    assert measured['passes']==2 and measured['files']==2 and measured['bytes']==8
    assert measured['seconds']>=0 and len(boundaries)==4


def test_final_byte_forecast_charges_full_logs_and_every_reproduction(monkeypatch,tmp_path):
    monkeypatch.setattr(runner.c,'ROOT',tmp_path)
    monkeypatch.setattr(runner.c,'OUT',tmp_path/'evidence')
    schedule=runner.schedules.make_schedule()
    def write(path,size):
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(b'x'*size)
    for spec in (item for item in schedule if item['stage']=='operational'):
        folder=runner.game_folder(spec)
        write(folder/'moves'/'0000.json',90)
        write(folder/'online_replay.json',20)
        for role in ('challenger','opponent'):
            write(folder/f'log_integrity_{role}.json',30)
            worker=runner.c.OUT/'workers'/f"operational_g{spec['game_index']}_{role}"
            write(worker/'inputs'/'request-0001.json',40)
            write(worker/'request-0001.json',50)
            write(worker/'startup.json',10)
            write(worker/'stdout.gz',100)
            write(worker/'stderr.gz',200)
    result=runner.final_evidence_forecast(schedule)
    assert result['outcomes_used'] is False and result['observed_workers']==12
    assert result['single_reproduction_worker_bound_bytes']==400
    assert result['projected_reproduction_bytes']==48*(400+90)
    assert result['fixed_analysis_and_control_bytes']==8*1024*1024
    assert result['projected_replay_control_bytes']==206*20
    assert result['projected_log_control_bytes']==462*30
    assert result['projected_sealed_files']==206*14+48*5
    raw=sum(result[key] for key in ('projected_reproduction_bytes','projected_seal_bytes',
        'projected_replay_control_bytes','projected_log_control_bytes','fixed_analysis_and_control_bytes'))
    assert result['projected_final_evidence_bytes']==runner.math.ceil(1.35*raw)
    assert result['projected_final_evidence_bytes']<result['reserve_bytes']
