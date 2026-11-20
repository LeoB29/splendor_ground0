from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import ast,json,os,subprocess,sys
import pytest
from tools.splendorust_search256_strength_20261029_02 import common as c,runner,guardian,independent_review,schedule,design


def fixture_inputs():
    return dict(game_costs=[dict(seconds=100.,independent_seconds=5.,bytes=10*c.MiB,files=400) for _ in range(6)],
        worker_costs=[dict(initialization=4.,graph=.5,closure=.5) for _ in range(12)],hash_seconds=[.5]*6,
        runtime_seconds=2.,ram_wait_used=c.PRIOR_RAM_WAIT_SECONDS+60.,current_bytes=200*c.MiB,current_files=3000,
        monotonic_origin=c.PRIOR_ORIGIN,unix_origin=c.PRIOR_WALL_START,monotonic=c.PRIOR_ORIGIN+1000.,unix_time=c.PRIOR_WALL_START+1000.)


def test_forecast_covers_fixed400_48_and_full_final_reader():
    result=runner.resource_projection(fixture_inputs())
    assert result['passed'] and result['operational_games']==6 and result['remaining_games']==400
    assert result['replay_decisions_maximum']==48
    assert set(result['phase_projected_seconds'])=={'main','main_verification'}
    assert result['projected_saved_final_seconds']>c.INDEPENDENT_REVIEW_SECONDS
    assert result['outcomes_used'] is False and result['timing_used_for_admission'] is False
    assert result['projected_collection_seconds']>sum(result['phase_projected_seconds'].values())


@pytest.mark.parametrize('mutation',['main','verification','disk','monotonic','wall','nan','negative','five_ops'])
def test_forecast_rejects_invalid_or_insufficient_resources(mutation):
    value=fixture_inputs()
    if mutation=='main':
        for row in value['game_costs']:row['seconds']=170.
    elif mutation=='verification':
        for row in value['game_costs']:row['independent_seconds']=40.
    elif mutation=='disk':
        for row in value['game_costs']:row['bytes']=c.GiB
    elif mutation=='monotonic':value['monotonic']=value['monotonic_origin']+c.COLLECTION_CUTOFF_SECONDS-1
    elif mutation=='wall':value['unix_time']=value['unix_origin']+c.COLLECTION_CUTOFF_SECONDS-1
    elif mutation=='nan':value['runtime_seconds']=float('nan')
    elif mutation=='negative':value['ram_wait_used']=-1
    elif mutation=='five_ops':value['game_costs'].pop()
    if mutation in ('nan','negative','five_ops'):
        with pytest.raises(ValueError):runner.resource_projection(value)
    else:assert runner.resource_projection(value)['passed'] is False


def test_forecast_scales_with_cost_without_using_game_outcomes():
    value=fixture_inputs();before=runner.resource_projection(value)
    value['game_costs'][0].update(winner=1,reason='natural',seconds=120.)
    after=runner.resource_projection(value)
    assert after['phase_projected_seconds']['main']-before['phase_projected_seconds']['main']==1.5*400*20
    del value['game_costs'][0]['winner'];del value['game_costs'][0]['reason']
    assert runner.resource_projection(value)==after


def test_fixed_schedule_and_lifecycle_coverage():
    specs=schedule.make_schedule();assert len(specs)==406
    assert len({runner.game_folder(x) for x in specs})==406
    inventory=design.lifecycle_inventory()
    for route in ('frozen','research'):
        item=inventory['qualification_'+route]
        assert item['ordinary_requests']==24 and item['startup_requests']==2
    for spec in specs:
        for role in ('challenger','opponent'):
            item=inventory[design.game_worker_name(spec,role)]
            assert item['route']==spec[role+'_mode'] and item['ordinary_requests']==2000
    assert c.MAX_WORKERS==862 and c.MAX_GRAPH_BUILDS==862
    assert 2+2*406+48==862 and 11*862==9482


def test_main_admission_requires_worker_creation_before_all_initialization():
    permit=dict(unix_time=100.,perf_counter_ns=1000)
    placement=dict(supervisor=dict(monotonic_ns=1200))
    assert runner.validate_main_startup(permit,dict(identity=dict(created=101.)),placement)
    for created in (99.,100.):
        with pytest.raises(ValueError,match='process creation'):
            runner.validate_main_startup(permit,dict(identity=dict(created=created)),placement)
    with pytest.raises(ValueError,match='placement'):
        runner.validate_main_startup(permit,dict(identity=dict(created=101.)),dict(supervisor=dict(monotonic_ns=999)))


@pytest.mark.parametrize('clock',['monotonic','unix_time'])
def test_main_admission_cannot_predate_passed_operational_gate(clock):
    rows=[dict(monotonic=i,unix_time=1000+i) for i in range(4)]
    assert runner.validate_main_admission_time(rows[3],rows[0],rows[1],rows[2])
    rows[3][clock]=rows[0][clock]-1
    with pytest.raises(ValueError,match='precede'):
        runner.validate_main_admission_time(rows[3],rows[0],rows[1],rows[2])


def test_all_model_stages_cutoff_and_final_requires_no_live_workers(monkeypatch,tmp_path):
    monkeypatch.setattr(c,'OUT',tmp_path);monkeypatch.setattr(c,'ROOT',tmp_path)
    monkeypatch.setattr(runner.time,'monotonic',lambda:c.PRIOR_ORIGIN+100.);monkeypatch.setattr(runner.time,'time',lambda:c.PRIOR_WALL_START+100.)
    e=runner.Envelope(dict(monotonic_origin=c.PRIOR_ORIGIN,unix_origin=c.PRIOR_WALL_START));e.guard=lambda:None
    for stage in c.STAGES:
        e.begin(stage);assert e.deadline<=e.end
        if stage!='final':assert e.deadline<=e.start+c.COLLECTION_CUTOFF_SECONDS
    e.resident={1:object()}
    with pytest.raises(ValueError,match='no resident'):e.begin('final')


def test_external_model_clock_and_savedonly_final():
    cutoff=c.COLLECTION_CUTOFF_SECONDS
    assert guardian.model_clock_open(0,10,cutoff-1,cutoff+9,final_started=False,owned_children=[])
    assert not guardian.model_clock_open(0,10,cutoff,cutoff+9,final_started=False,owned_children=[])
    assert not guardian.model_clock_open(0,10,cutoff-1,cutoff+10,final_started=False,owned_children=[])
    assert guardian.model_clock_open(0,10,cutoff+100,cutoff+110,final_started=True,owned_children=[])
    with pytest.raises(ValueError,match='model/helper'):
        guardian.model_clock_open(0,10,cutoff+100,cutoff+110,final_started=True,owned_children=[object()])


def test_reader_cap_clamps_to_original_final_global_clocks():
    claim=dict(monotonic_origin=c.PRIOR_ORIGIN,unix_origin=c.PRIOR_WALL_START)
    final=dict(monotonic_deadline=c.PRIOR_ORIGIN+c.JOB_SECONDS-10,unix_deadline=c.PRIOR_WALL_START+c.JOB_SECONDS-20)
    assert independent_review.review_deadlines(claim,final,c.PRIOR_ORIGIN+100.,c.PRIOR_WALL_START+100.)==(c.PRIOR_ORIGIN+10900,c.PRIOR_WALL_START+10900)
    assert independent_review.review_deadlines(claim,final,c.PRIOR_ORIGIN+c.JOB_SECONDS-100,c.PRIOR_WALL_START+c.JOB_SECONDS-100)==(final['monotonic_deadline'],final['unix_deadline'])


def test_child_environment_import_smoke():
    source=Path(guardian.__file__).read_text()
    assert 'SPLENDOR_EXPLORATION' not in source and source.count('SPLENDOR_SEARCH256_STRENGTH_PHASE=c.PHASE')==3
    env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',SPLENDOR_SEARCH256_STRENGTH_PHASE='strength',PYTHONPATH=str(c.ROOT/'src')+os.pathsep+str(c.ROOT))
    command="import sys; from tools.splendorust_search256_strength_20261029_02 import runner,worker,guardian; from tools.splendorust_search128_repeat_20261029 import common; assert 'torch' not in sys.modules"
    result=subprocess.run([sys.executable,'-B','-c',command],cwd=c.ROOT,env=env,capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stderr


def test_saved_readback_is_not_model_launch_and_retains_all_checks():
    source=Path(runner.__file__).read_text();tree=ast.parse(source)
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='verify_saved_attempt')
    calls={n.func.id for n in ast.walk(function) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)}
    assert not calls&{'make_worker','ordinary','decision','collect_games','reproduce_decision'}
    text=ast.get_source_segment(source,function)
    assert 'run_models=False' in text and 'validate_main_startup' in text
    assert 'verify_resource_forecast' in text and 'verify_closed_worker' in text and 'verify_runtime' in text
    assert 'Exact completed verification phase inventory' in text and 'Exact complete owned worker inventory' in text
    assert 'select_candidate' not in source and 'confirmation' not in source
    independent=Path(independent_review.__file__).read_text()
    assert 'writing=False' in independent and 'requires_actual_reader_and_guardian_exit_before_external_acceptance=True' in independent


@pytest.mark.parametrize('fail_forecast',[False,True])
def test_controller_gate_precedes_main_and_failed_gate_preserves_incomplete(monkeypatch,tmp_path,fail_forecast):
    events=[];saved={};specs={stage:schedule.make_schedule(stage) for stage in ('operational','main')}
    class Envelope:
        def __init__(self,claim):self.start=0.;self.wall_start=0.;self.stage=None
        def enable_evidence_accounting(self):pass
        def begin(self,stage):self.stage=stage;events.append(('begin',stage))
        def finish(self):events.append(('finish',self.stage))
        def admit(self):pass
    monkeypatch.setattr(c,'OUT',tmp_path)
    monkeypatch.setattr(c,'file_hash',lambda p:'freeze')
    monkeypatch.setattr(c,'read',lambda p:dict(freeze_sha256='freeze') if p.name=='watchdog_claim.json' else {})
    monkeypatch.setattr(c,'save',lambda p,value:saved.setdefault(p.name,deepcopy(value)))
    monkeypatch.setattr(runner,'Envelope',Envelope)
    monkeypatch.setattr(runner,'verify_runtime_sources',lambda:{})
    monkeypatch.setattr(runner,'verify_runtime',lambda *args:None)
    monkeypatch.setattr(runner,'switch_placement',lambda *args:None)
    monkeypatch.setattr(runner,'load_inputs',lambda *args:(specs,{}))
    monkeypatch.setattr(runner,'qualify',lambda *args:events.append(('qualify',None)))
    def collect(rows,envelope,freeze,games):
        if rows[0]['stage']=='main':assert 'main_admission.json' in saved and ('forecast',None) in events
        events.append(('collect',rows[0]['stage'],len(rows)));games.extend(rows)
    def forecast(*args):
        events.append(('forecast',None))
        if fail_forecast:raise ValueError('test resource gate')
    monkeypatch.setattr(runner,'collect_games',collect)
    monkeypatch.setattr(runner,'forecast_resources',forecast)
    monkeypatch.setattr(runner,'main_admission_bindings',lambda:{'fixed':True})
    monkeypatch.setattr(runner,'verify_phase',lambda *args,**kwargs:events.append(('replays',args[0],kwargs['run_models'])))
    monkeypatch.setattr(runner,'final_checks',lambda *args:dict(complete=True))
    result=runner.run('freeze')
    if fail_forecast:
        assert result==1 and saved['completion.json']['status']=='incomplete'
        assert 'failure.json' in saved and 'main_admission.json' not in saved
        assert not any(event[:2]==('collect','main') for event in events)
    else:
        assert result==0 and saved['completion.json']['actual_games']==406
        assert ('collect','main',400) in events and ('replays','main',True) in events
        assert saved['main_admission.json']['sample_size']==400 and saved['main_admission.json']['no_optional_stopping'] is True


def test_main_guard_loss_continues_sample_but_operational_guard_loss_stops(monkeypatch):
    def play(spec,*args):return dict(spec),[],0.,dict(saved_failure=True)
    monkeypatch.setattr(runner,'play_game',play)
    games=[];runner.collect_games(schedule.make_schedule('main'),None,None,games)
    assert len(games)==400
    with pytest.raises(ValueError,match='Operational'):
        runner.collect_games(schedule.make_schedule('operational'),None,None,[])


def test_infrastructure_failure_stops_without_replacing_game(monkeypatch):
    calls=[]
    def play(spec,*args):calls.append(spec);raise RuntimeError('saved infrastructure fault')
    monkeypatch.setattr(runner,'play_game',play)
    with pytest.raises(RuntimeError):runner.collect_games(schedule.make_schedule('main'),None,None,[])
    assert calls==schedule.make_schedule('main')[:1]


@pytest.mark.parametrize('mutation',[None,'exit','cleanup','freeze','active'])
def test_independent_reader_requires_clean_actual_owner_exits(monkeypatch,mutation):
    receipts={name:dict(status='completed',exit_code=0,freeze_sha256='freeze',cleanup_errors=[],charged_job_seconds=10.) for name in ('execution.json','watchdog_execution.json')}
    receipts.update({'launch.json':dict(pid=101,created=1.),'guardian_owner.json':dict(watchdog=dict(pid=102,created=2.))})
    if mutation=='exit':receipts['execution.json']['exit_code']=1
    elif mutation=='cleanup':receipts['watchdog_execution.json']['cleanup_errors']=['unconfirmed']
    elif mutation=='freeze':receipts['execution.json']['freeze_sha256']='other'
    monkeypatch.setattr(c,'read',lambda p:receipts[p.name])
    def process(pid):
        if mutation=='active':return SimpleNamespace(create_time=lambda:1. if pid==101 else 2.)
        raise independent_review.psutil.NoSuchProcess(pid)
    monkeypatch.setattr(independent_review.psutil,'Process',process)
    if mutation is None:assert len(independent_review.check_successful_owners('freeze'))==2
    else:
        with pytest.raises(ValueError):independent_review.check_successful_owners('freeze')


def test_cleanup_clamps_one_wait_budget_to_remaining_reader_allowance(monkeypatch):
    monkeypatch.setattr(guardian.time,'monotonic',lambda:100.)
    monkeypatch.setattr(guardian.time,'time',lambda:1000.)
    result=guardian.cleanup_owned(None,None,[],monotonic_origin=0.,unix_origin=900.,extra_deadlines=(101.,1002.))
    assert result['effective_allowance_seconds']==1 and result['errors']==[]


def test_amendment_carries_wait_and_cannot_restart_qualification(monkeypatch,tmp_path):
    monkeypatch.setattr(c,'OUT',tmp_path);monkeypatch.setattr(c,'ROOT',tmp_path)
    monkeypatch.setattr(runner.time,'monotonic',lambda:c.PRIOR_ORIGIN+3600.)
    monkeypatch.setattr(runner.time,'time',lambda:c.PRIOR_WALL_START+3600.)
    e=runner.Envelope(dict(monotonic_origin=c.PRIOR_ORIGIN,unix_origin=c.PRIOR_WALL_START))
    assert e.waited==c.PRIOR_RAM_WAIT_SECONDS and e.admitted is False
    e.guard=lambda:None;e.begin('qualification')
    assert e.deadline==c.PRIOR_ORIGIN+7200 and e.wall_deadline==c.PRIOR_WALL_START+7200
    assert e.remaining()==3600
    assert e.prior_worker_count==0  # The disclosed prior2 do not consume local862 slots.
    assert c.MAX_WORKERS+c.PRIOR_WORKER_TREES==864


@pytest.mark.parametrize('clock',['monotonic_origin','unix_origin'])
def test_fresh_origin_rejected_in_controller_reader_and_forecast(clock):
    claim=dict(monotonic_origin=c.PRIOR_ORIGIN,unix_origin=c.PRIOR_WALL_START);claim[clock]+=1
    with pytest.raises(ValueError,match='nonresettable'):runner.Envelope(claim)
    with pytest.raises(ValueError,match='nonresettable'):
        independent_review.review_deadlines(claim,{},0,0)
    inputs=fixture_inputs();inputs[clock]+=1
    with pytest.raises(ValueError,match='nonresettable'):runner.resource_projection(inputs)


def test_forecast_cannot_discard_consumed_ram_wait_or_old_evidence():
    inputs=fixture_inputs();inputs['ram_wait_used']=0
    with pytest.raises(ValueError,match='cumulative RAM'):runner.resource_projection(inputs)
    assert c.MAX_EVIDENCE_BYTES+c.PRIOR_EVIDENCE_BYTES==24*c.GiB
    assert c.PREFINAL_EVIDENCE_BYTES+c.PRIOR_EVIDENCE_BYTES==16*c.GiB


def test_external_watchdog_honors_old_qualification_clock():
    mono,wall=c.PRIOR_ORIGIN,c.PRIOR_WALL_START
    assert guardian.qualification_clock_open(mono,wall,mono+7199,wall+7199,qualification_finished=False)
    assert not guardian.qualification_clock_open(mono,wall,mono+7200,wall+7199,qualification_finished=False)
    assert not guardian.qualification_clock_open(mono,wall,mono+7199,wall+7200,qualification_finished=False)
    assert guardian.qualification_clock_open(mono,wall,mono+7201,wall+7201,qualification_finished=True)


@pytest.mark.parametrize('mutation',[None,'active_decision','games','guardian_exit','live_identity','reboot'])
def test_only_closed_predispatch_failure_is_admissible(monkeypatch,mutation):
    from tools.splendorust_search256_strength_20261029_02 import preparation
    descriptors={name:dict(path=name) for name in ('prior_attempt','prior_preservation','prior_guardian_claim','prior_qualification_start')}
    expected=dict(descriptors)
    old_claim=dict(monotonic_origin=c.PRIOR_ORIGIN,unix_origin=c.PRIOR_WALL_START)
    preserved=dict(status='failed_attempt_preserved_and_closed',actual_guardian_exit_code=1,actual_games=0,
        actual_256_scientific_decisions=0,failed_active_decision=False,original_frozen_attempt_remains_incomplete=True,
        freeze=descriptors['prior_attempt'],original_clock=old_claim,
        identities=[dict(pid=i,created=float(i),alive=False) for i in range(15)])
    if mutation=='active_decision':preserved['failed_active_decision']=True
    elif mutation=='games':preserved['actual_games']=1
    elif mutation=='guardian_exit':preserved['actual_guardian_exit_code']=None
    monkeypatch.setattr(preparation,'recovery_contract',lambda:expected)
    monkeypatch.setattr(c,'verify',lambda x:None)
    monkeypatch.setattr(c,'read',lambda p:preserved if p.name=='prior_preservation' else old_claim)
    def process(pid):
        if mutation=='live_identity':return SimpleNamespace(create_time=lambda:float(pid))
        raise guardian.psutil.NoSuchProcess(pid)
    monkeypatch.setattr(guardian.psutil,'Process',process)
    monkeypatch.setattr(guardian.time,'monotonic',lambda:c.PRIOR_ORIGIN+(-1 if mutation=='reboot' else 1))
    monkeypatch.setattr(guardian.time,'time',lambda:c.PRIOR_WALL_START+1)
    if mutation is None:assert guardian.verify_recovery_origin(dict(recovery=expected))==(c.PRIOR_ORIGIN,c.PRIOR_WALL_START)
    else:
        with pytest.raises(ValueError):guardian.verify_recovery_origin(dict(recovery=expected))


@pytest.mark.parametrize('quota',[128,256])
@pytest.mark.parametrize('operation',['graph_parity','ordinary'])
def test_actual_new_runner_registration_new_transport_and_real_publication(tmp_path,monkeypatch,quota,operation):
    import threading
    from tools import splendorust_search256_accounting_20261029 as fixed
    from tools.splendorust_search128_resume_20261029 import accounting as old
    from tools.splendorust_search256_strength_20261029_02 import transport
    from tools.splendorust_boundary_timing_20261025.child_profile import publish_profile
    prior=c.PRIOR_ATTEMPT_OUT/'workers/qualification_frozen'
    payload=c.read(prior/('startup_inputs/request-00000001.json' if operation=='graph_parity' else 'inputs/request-00000003.json'))
    graph=c.read(prior/'boundary_timing_identity.json')['graph_identity_sha256']
    monkeypatch.setattr(c,'ROOT',tmp_path);monkeypatch.setattr(c,'OUT',tmp_path/'study')
    e=runner.Envelope(dict(monotonic_origin=c.PRIOR_ORIGIN,unix_origin=c.PRIOR_WALL_START))
    e.evidence=dict(workers={});e.time_guard=lambda:None
    folder=c.OUT/'workers/fixture';e.register_worker(folder)
    account=e.worker_accounting[folder]
    assert type(account) is fixed.WorkerAccounting and not isinstance(account,old.WorkerAccounting)
    folder.mkdir(parents=True)
    child=SimpleNamespace(output_dir=folder,_log_lock=threading.Lock(),_logs={n:SimpleNamespace() for n in fixed.LOG_NAMES},
        max_log_bytes=128*c.MiB,_closed=False,_counter=0,challenger_quota=quota,process=SimpleNamespace(pid=101),
        placement_mode='isolated',recovery_accounting=account,recovery_envelope=e,
        _placement_freeze={'unchanged_graph_backend_identity':{'graph_identity_sha256':graph}})
    for name in fixed.LOG_NAMES:(folder/name).write_bytes(b'')
    account.bind_child(child);account.finish_initialization()
    calls=[]
    def request(request_payload,timeout,validate=None):
        calls.append((request_payload,timeout));child._counter+=1
        return dict(status='ok',request_id='request-00000001',deadline_seconds=timeout,result={'metadata_fixture':True})
    child.request=request
    monkeypatch.setattr(transport.placement,'verify_runtime',lambda *args,**kwargs:{'metadata_fixture':True})
    timeout=60. if operation=='graph_parity' else 10.
    receipt=transport.request(child,payload,timeout)
    assert calls==[(payload,timeout)] and account.active_request_id=='request-00000001'
    publication=deepcopy(account._publication);assert publication['expected']['quota']==quota
    publish_profile(folder,receipt['request_id'],publication['expected'],batch=operation=='graph_parity')
    assert account.finish_request(require_published=True)['state']=='idle'
    child._closed=True;assert account.close()['state']=='closed'


@pytest.mark.parametrize('quota',[True,256.,'256',255,257,512])
def test_fixed_accountant_still_rejects_unknown_or_coerced_quota(quota):
    from tools import splendorust_search256_accounting_20261029 as fixed
    with pytest.raises(fixed.AccountingError,match='scientific quota'):
        fixed.publication_identity('request-00000001',{'operation':'graph_parity'},pid=101,quota=quota)
