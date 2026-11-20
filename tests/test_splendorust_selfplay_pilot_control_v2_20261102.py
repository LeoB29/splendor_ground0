"""Model-free control regressions; no games, native models, or optimizer calls."""
import json
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import pytest

from tools.splendorust_selfplay_pilot_v2_20261102 import qualification_control as run
from tools.splendorust_selfplay_pilot_v2_20261102 import qualification_watchdog as watch
from tools.splendorust_selfplay_pilot_v2_20261102 import transport, worker, adapters, common as c


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value),encoding='utf-8')


def workspace(monkeypatch,tmp_path):
    monkeypatch.setattr(c,'ROOT',tmp_path)
    monkeypatch.setattr(c,'OUT',tmp_path/'output')
    c.OUT.mkdir()
    return c.OUT


def test_plan_reuses_corrected_real_counterexamples_and_three_full_suites():
    plan=run.build_plan()
    assert len(plan['loader_cases']['cases'])==15
    assert plan['actual_loader_ordinary_requests']==32
    assert plan['suite']['scientific_decisions']==201 and plan['suite']['graph_cases']==33
    assert plan['fresh_replay_decisions']==12 and plan['pilot_data'] is False


def test_lifecycle_limits_are_sequential_prefixes_without_retries():
    inventory=run.lifecycle_inventory()
    assert set(inventory)=={'capture_off','capture_on','replay_on'}
    assert sum(x['ordinary_requests'] for x in inventory.values())==8044
    for x in inventory.values():
        assert x['route']=='candidate' and x['quota']==512
        assert x['startup_requests']==2 and x['ordinary_first_request_id']=='request-00000003'
    assert inventory['capture_off']['capture_targets'] is False
    assert run.contract()['automatic_retries']==run.contract()['game_replacements']==0


@pytest.mark.parametrize('key,wrong',[('job_seconds',7201),('seconds_per_move',1),('teacher_simulations',256),('optimizer_steps',2)])
def test_contract_rejects_execution_drift(key,wrong):
    freeze={**run.contract(),'profiles':{'candidate':{'16':{},'512':{}}}}
    run.validate_freeze_contract(freeze)
    freeze[key]=wrong
    with pytest.raises(ValueError):run.validate_freeze_contract(freeze)


def fake_idle(folder,counter=18):
    identity={'pid':123,'created':45.}
    ready={'status':'ok','result':{'pid':123}}
    write(folder/'initialization.json',{**ready,'identity':identity,'command':['python'],'absolute_deadline':1000.})
    return SimpleNamespace(output_dir=folder,identity=identity,initialization=ready,
        process=SimpleNamespace(pid=123,poll=lambda:None),_ps=SimpleNamespace(pid=123,create_time=lambda:45.),
        _closed=False,_counter=counter,_failure=None,_lock=SimpleNamespace(locked=lambda:False),absolute_deadline=1000.)


def test_live_idle_handoff_uses_real_unaugmented_ready_shape(monkeypatch,tmp_path):
    out=workspace(monkeypatch,tmp_path);child=fake_idle(out/'workers/capture_on')
    env=SimpleNamespace(start=0.,wall_start=100.,deadline=1500.)
    run.advance_idle_worker(child,env)
    saved=c.read(out/'worker_phase_transitions/capture_on.json')
    assert saved['worker_identity']==child.identity and saved['next_request_id']=='request-00000019'
    assert child.absolute_deadline==1500. and 'identity' not in child.initialization


@pytest.mark.parametrize('counter',[0,2,17,19])
def test_handoff_rejects_wrong_prefix_before_mutating_deadline(monkeypatch,tmp_path,counter):
    out=workspace(monkeypatch,tmp_path);child=fake_idle(out/'workers/capture_on',counter)
    with pytest.raises(ValueError):run.advance_idle_worker(child,SimpleNamespace(start=0.,wall_start=100.,deadline=1500.))
    assert child.absolute_deadline==1000.


def test_handoff_publication_failure_keeps_original_deadline(monkeypatch,tmp_path):
    out=workspace(monkeypatch,tmp_path);child=fake_idle(out/'workers/capture_on')
    monkeypatch.setattr(c,'save',Mock(side_effect=OSError('full disk')))
    with pytest.raises(OSError):run.advance_idle_worker(child,SimpleNamespace(start=0.,wall_start=100.,deadline=1500.))
    assert child.absolute_deadline==1000.


@pytest.mark.parametrize('mutate',[lambda x:x.identity.update(created=46.),lambda x:setattr(x,'_failure','timeout'),lambda x:setattr(x,'_lock',SimpleNamespace(locked=lambda:True))])
def test_bad_identity_or_active_request_cannot_handoff(monkeypatch,tmp_path,mutate):
    out=workspace(monkeypatch,tmp_path);child=fake_idle(out/'workers/capture_on');mutate(child)
    with pytest.raises(ValueError):run.advance_idle_worker(child,SimpleNamespace(start=0.,wall_start=100.,deadline=1500.))
    assert child.absolute_deadline==1000.


def test_stage_limits_use_both_original_clocks(monkeypatch,tmp_path):
    workspace(monkeypatch,tmp_path)
    monkeypatch.setattr(run.time,'monotonic',lambda:5000.)
    monkeypatch.setattr(run.time,'time',lambda:15000.)
    env=run.Envelope({'monotonic_origin':0.,'unix_origin':10000.})
    monkeypatch.setattr(env,'guard',lambda:None)
    env.begin('collection')
    assert env.deadline==7200. and env.wall_deadline==17200.
    assert env.remaining()==2200.
    env.finish=lambda:None
    env.begin('final')
    assert env.deadline==6800. and env.wall_deadline==16800.


def test_saved_only_final_forbids_live_models(monkeypatch,tmp_path):
    workspace(monkeypatch,tmp_path)
    env=run.Envelope({'monotonic_origin':0.,'unix_origin':10000.})
    env.resident={(123,45.):object()}
    with pytest.raises(ValueError,match='no resident'):env.begin('final')
    with pytest.raises(ValueError,match='cannot own'):watch.model_clock_open(0,0,100,100,final_started=True,owned_children=[1])


def off_fixture(out):
    folder=out/'workers/capture_off';observer={'observation':{'example':1}}
    receipt={'request_id':'request-00000003','result':{'action':'saved'}}
    write(folder/'initialization.json',{'identity':{'pid':321,'created':22.}})
    value=dict(schema='selfplay_capture_publication_v1',request_id=receipt['request_id'],pid=321,
        observer_sha256=c.digest(observer),result_sha256=c.digest(receipt['result']),capture_targets=False,
        training_capture=dict(schema='selfplay_root_capture_disabled_v1',enabled=False,
            observer_sha256=c.digest(observer),quota=512,training_eligible=False))
    write(folder/(receipt['request_id']+'_training_capture.json'),value)
    return folder,observer,receipt,value


def test_actual_off_capture_publication_is_bound_to_raw_result(monkeypatch,tmp_path):
    out=workspace(monkeypatch,tmp_path);folder,observer,receipt,value=off_fixture(out)
    assert transport.verify_capture_publication(folder,receipt,observer,{})==value
    receipt['result']['action']='substituted'
    with pytest.raises(ValueError):transport.verify_capture_publication(folder,receipt,observer,{})


@pytest.mark.parametrize('field,bad',[('pid',322),('request_id','request-00000004'),('capture_targets',True),('observer_sha256','0'*64)])
def test_capture_publication_cannot_change_identity(monkeypatch,tmp_path,field,bad):
    out=workspace(monkeypatch,tmp_path);folder,observer,receipt,value=off_fixture(out)
    value[field]=bad;write(folder/(receipt['request_id']+'_training_capture.json'),value)
    with pytest.raises(ValueError):transport.verify_capture_publication(folder,receipt,observer,{})


def test_capture_fixture_comparison_does_not_ignore_scientific_callbacks(monkeypatch,tmp_path):
    workspace(monkeypatch,tmp_path)
    base={'spec':{},'before':{},'observer':{},'action':{},'trace':{'callbacks':[1]},'after':{},'clock':{'result':{'callbacks':[1]}}}
    off=[deepcopy(base) for _ in range(16)];on=deepcopy(off);on[3]['trace']['callbacks']=[2]
    monkeypatch.setattr(run,'fixture_rows',lambda name:off if name=='capture_off' else on)
    monkeypatch.setattr(run,'validate_row',lambda *a,**kw:True)
    with pytest.raises(ValueError,match='exact science trace'):
        run.verify_capture_fixtures(SimpleNamespace(guard=lambda:None),{}, {'loader_cases':{}},writing=True)


def test_replay_selection_ignores_outcomes_and_timing(monkeypatch,tmp_path):
    out=workspace(monkeypatch,tmp_path);descriptors=[]
    for index in range(4):
        spec={'game_index':index};folder=run.game_folder(spec)
        rows=[{'ply':ply,'observer':{'x':ply},'clock':{'receipt_elapsed_seconds':ply},'action':{'x':ply}} for ply in range(9)]
        for row in rows:write(folder/'moves'/f"{row['ply']:06d}.json",row)
        path=folder/'game.json';write(path,{'spec':spec,'winner':0,'rows':rows});descriptors.append(c.artifact(path))
    env=SimpleNamespace(guard=lambda:None)
    first=run.select_replays(descriptors,env)
    for descriptor in descriptors:
        path=c.ROOT/descriptor['path'];game=c.read(path);game['winner']=1
        for row in game['rows']:row['clock']['receipt_elapsed_seconds']=1000-row['ply']
        write(path,game)
    second=run.select_replays([c.artifact(c.ROOT/d['path']) for d in descriptors],env)
    assert [(s['game_index'],s['ply'],s['third']) for s in first]==[(s['game_index'],s['ply'],s['third']) for s in second]
    assert len(first)==12 and len({(s['game_index'],s['third']) for s in first})==12


def test_parent_and_qualified_profiles_are_literal_original_bytes():
    original=c.read(c.ACCEPTED_OUT/'freeze.json')['profiles']['candidate']
    for quota in (16,512):
        assert c.artifact(adapters.profile_path(quota,route='candidate'))==original[str(quota)]
        assert adapters.profile(quota,route='candidate')['iterations']==quota


def test_bad_route_does_not_admit_an_unintended_player():
    for route in ['control','research','candidate_1024','']:
        with pytest.raises(ValueError):adapters.route_quota(route)


def test_original_training_and_deadlock_labels_never_pass_game_loader():
    from tools.splendorust_selfplay_pilot_v2_20261102 import data
    for reason in ['forfeit','cutoff','deadlock','natural']:
        document={'schema':data.GAME_SCHEMA,**data.IDENTITY,'spec':{'x':1},'reason':reason}
        with pytest.raises(ValueError,match='official natural'):data.load_game(document,{'x':1})


def learned_fixture(out,*,completed=True):
    folder=out/'fit';directory=folder/'learned_entrypoint'
    write(folder/'owner.json',{'monotonic':90.,'owner_monotonic_start':90.,'request_qpc_start':90.,
        'unix_time':900.,'clock_contract':run.clocks.runtime_contract()})
    write(folder/'checkpoint_manifest.json',{'fixture':'synthetic-checkpoint-identity-only'})
    manifest=c.artifact(folder/'checkpoint_manifest.json')
    start=dict(schema='selfplay_learned_request_start_v2',request_index=0,case_id='a',mode='eager',
        candidate_manifest=manifest,**run.clocks.start_fields(reading={'request_qpc':100.,'owner_monotonic':100.,'unix':1000.}))
    path=directory/'request_000.started.json';write(path,start)
    response=dict(schema='selfplay_learned_response_v2',request_index=0,case_id='a',mode='eager',
        start=c.artifact(path),candidate_manifest=manifest,
        **run.clocks.completion_fields(start,reading={'request_qpc':101.,'owner_monotonic':101.,'unix':1001.}))
    if completed:write(directory/'request_000.completed.json',response)
    env=SimpleNamespace(deadline=200.,wall_deadline=1100.)
    return folder,env,{'learned_entrypoint':{'case_ids':['a','b','c']},'clock_contract':run.clocks.runtime_contract()}


def test_parent_observes_full_learned_response_before_ack(monkeypatch,tmp_path):
    out=workspace(monkeypatch,tmp_path);folder,env,freeze=learned_fixture(out)
    monkeypatch.setattr(run.time,'monotonic',lambda:102.)
    monkeypatch.setattr(run.time,'perf_counter',lambda:102.)
    monkeypatch.setattr(run.time,'time',lambda:1002.)
    observed=[];run.observe_learned_requests(folder,{'pid':999,'created':88.},env,freeze,observed)
    assert len(observed)==1 and observed[0]['receipt_elapsed_seconds']==2.
    assert observed[0]['parent_observed_qpc']==102.
    assert observed[0]==c.read(folder/'learned_entrypoint/request_000.parent_accepted.json')


def test_ontime_child_claim_cannot_override_late_parent_receipt(monkeypatch,tmp_path):
    out=workspace(monkeypatch,tmp_path);folder,env,freeze=learned_fixture(out)
    monkeypatch.setattr(run.time,'monotonic',lambda:102.)
    monkeypatch.setattr(run.time,'perf_counter',lambda:110.001)
    monkeypatch.setattr(run.time,'time',lambda:1002.)
    with pytest.raises(ValueError,match='parent_request_qpc_deadline'):
        run.observe_learned_requests(folder,{'pid':999,'created':88.},env,freeze,[])
    assert not (folder/'learned_entrypoint/request_000.parent_accepted.json').exists()
    rejected=c.read(folder/'learned_entrypoint/request_000.parent_rejected.json')
    assert rejected['branch']=='parent_request_qpc_deadline'
    assert rejected['observation']==dict(request_qpc=110.001,owner_monotonic=102.,unix=1002.)
    assert rejected['retry'] is False and rejected['original_request_clock_preserved'] is True


@pytest.mark.parametrize('mono,wall',[(110.,1002.),(102.,1010.)])
def test_missing_learned_complete_response_stops_at_either_real_clock(monkeypatch,tmp_path,mono,wall):
    out=workspace(monkeypatch,tmp_path);folder,env,freeze=learned_fixture(out,completed=False)
    monkeypatch.setattr(run.time,'monotonic',lambda:102.);monkeypatch.setattr(run.time,'perf_counter',lambda:mono);monkeypatch.setattr(run.time,'time',lambda:wall)
    with pytest.raises(ValueError,match='pending_(request_qpc|wall)_window'):
        run.observe_learned_requests(folder,{'pid':999,'created':88.},env,freeze,[])


def test_no_unacknowledged_second_learned_decision(monkeypatch,tmp_path):
    out=workspace(monkeypatch,tmp_path);folder,env,freeze=learned_fixture(out)
    write(folder/'learned_entrypoint/request_001.started.json',{})
    with pytest.raises(ValueError,match='No next learned'):
        run.observe_learned_requests(folder,{},env,freeze,[])


def test_learned_allowance_is_never_shortened_at_stage_end(monkeypatch,tmp_path):
    out=workspace(monkeypatch,tmp_path);folder,env,freeze=learned_fixture(out)
    env.deadline=105.
    with pytest.raises(ValueError,match='full_request_fits_original_owner_clock'):
        run.observe_learned_requests(folder,{},env,freeze,[])


def test_saved_handoff_accepts_actual_slightly_earlier_effective_deadline(monkeypatch,tmp_path):
    out=workspace(monkeypatch,tmp_path);child=fake_idle(out/'workers/capture_on')
    child.absolute_deadline=999.999997
    init=c.read(child.output_dir/'initialization.json');init['absolute_deadline']=child.absolute_deadline
    write(child.output_dir/'initialization.json',init)
    monkeypatch.setattr(run.time,'monotonic',lambda:1002.)
    monkeypatch.setattr(run.time,'time',lambda:1102.)
    monkeypatch.setattr(run.time,'perf_counter_ns',lambda:1002000000000)
    env=SimpleNamespace(start=0.,wall_start=100.,deadline=1500.)
    run.advance_idle_worker(child,env)
    write(out/'stages/qualification_start.json',{'monotonic':0.,'monotonic_deadline':1000.})
    write(out/'stages/collection_start.json',{'monotonic':1001.,'monotonic_deadline':1500.})
    write(child.output_dir/'parent_profiles/request-00000019.json',{'request_start_ns':1003000000000})
    assert run.verify_phase_transition(env)
    init['absolute_deadline']=1000.;write(child.output_dir/'initialization.json',init)
    with pytest.raises(ValueError):run.verify_phase_transition(env)


def test_fit_command_uses_full_wrapper_and_dedicated_fixture_plan(monkeypatch,tmp_path):
    out=workspace(monkeypatch,tmp_path)
    freeze={'fit_plan':{'path':'wrapper.json'},'fit_parity_fixtures':{'path':'fixtures.json'},
        'training_plan':{'path':'wrong_flat_recipe.json'},'graph_parity_plan':{'path':'wrong_old_plan.json'}}
    write(out/'freeze.json',freeze)
    command,env=run.fit_command_environment(SimpleNamespace(deadline=100.,wall_deadline=1000.),freeze,out/'fit')
    assert command[command.index('--plan')+1]==str(tmp_path/'wrapper.json')
    assert command[command.index('--fixtures')+1]==str(tmp_path/'fixtures.json')
    assert env['STUDY_RESEARCH_FREEZE_SHA256']==c.file_hash(out/'freeze.json')
    assert env['SELFPLAY_MONOTONIC_DEADLINE']=='100.' or env['SELFPLAY_MONOTONIC_DEADLINE']=='100.0'






def test_actual_supervised_ordinary_preserves_saved512_science_and_separate_capture():
    saved=c.read(c.ACCEPTED_OUT/'games/operational/g000/moves/000000.json')
    original=saved['clock']['result'];marker={'fixture':'capture is not a scientific response field'}
    action=run.rules.deserialize_action(saved['action'])
    instance=SimpleNamespace(choose=lambda observer,diagnostic_capture:(action,deepcopy(saved['trace'])),
        last_training_capture=marker,request_deadline=None)
    identity={k:original[k] for k in ('profile_id','profile_sha256','wrapper_id')}
    context=dict(route='candidate',adapters={'search':instance},execution_identities={512:identity},
        diagnostic_capture=False,request_started=None)
    result=worker._ordinary(context,{'player':'search','observer':saved['observer']})
    assert result==original and context['training_capture']==marker
    assert 'training_capture' not in result and instance.request_deadline is None
    assert run.actual_after(saved['before'],saved['action'])==saved['after']


def test_real_accounting_accepts_capture_publication_before_original_boundary_barrier(tmp_path):
    import threading
    from tools.splendorust_selfplay_pilot_v2_20261102.accounting import WorkerAccounting,LOG_NAMES
    from tools.splendorust_boundary_timing_20261025.child_profile import publish_profile
    saved=c.read(c.ACCEPTED_OUT/'games/operational/g000/moves/000000.json')
    folder=tmp_path/'workers/capture_on';account=WorkerAccounting(folder);folder.mkdir(parents=True)
    child=SimpleNamespace(output_dir=folder,_log_lock=threading.Lock(),_logs={x:None for x in LOG_NAMES},
        max_log_bytes=c.MAX_LOG_BYTES,_closed=False)
    account.bind_child(child);account.finish_initialization()
    profile=deepcopy(saved['child_profile']);rid=saved['clock']['request_id']
    payload={'player':'search','observer':saved['observer']}
    begun=account.begin_request(rid,payload,pid=profile['pid'],quota=512,graph_identity_sha256=transport.GRAPH_IDENTITY)
    # This fixture publishes a reconstructed request with its own JSON field
    # ordering; bind that exact synthetic wire, retaining scientific evidence.
    profile['request_raw_sha256']=begun['publication']['expected']['request_raw_sha256']
    capture=folder/(rid+'_training_capture.json')
    worker._save(capture,{'schema':'synthetic_publication_only','complete':True})
    target=publish_profile(folder,rid,profile,batch=False)
    result=account.finish_request(require_published=True)
    assert result['state']=='idle' and result['metadata_bytes']==capture.stat().st_size+target.stat().st_size
    assert not list(folder.glob('*.pending'))
    child._closed=True;assert account.close()['state']=='closed'




def test_supervisor_imports_and_actual_shared_collator_remain_model_free():
    import subprocess,os
    script="""import sys
from tools.splendorust_selfplay_pilot_v2_20261102 import qualification_control,qualification_watchdog,worker,transport,data,training
example=dict(tokens=[0.]*1488,mask=[True]*81,policy=[1./81]*81)
batch=data.collate_arrays([example],[1.])
assert batch['tokens'].shape==(1,31,48)
assert 'torch' not in sys.modules
print('model-free-import-and-collation-passed')
"""
    env=os.environ.copy();env.update(PYTHONPATH=str(c.ROOT/'src')+os.pathsep+str(c.ROOT),PYTHONDONTWRITEBYTECODE='1',
        STUDY_SELFPLAY_PHASE='qualification',SPLENDOR_SEARCH512_STRENGTH_PHASE='strength')
    result=subprocess.run([sys.executable,'-B','-c',script],cwd=c.ROOT,env=env,capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stderr
    assert 'model-free-import-and-collation-passed' in result.stdout


def test_saved_capture_proof_is_compared_without_duplicate_exclusive_write(monkeypatch,tmp_path):
    out=workspace(monkeypatch,tmp_path)
    base=dict(spec={},before={},observer={},action={},trace={},after={},clock={'result':{}},training_capture={})
    rows=[deepcopy(base) for _ in range(16)]
    monkeypatch.setattr(run,'fixture_rows',lambda name:deepcopy(rows))
    monkeypatch.setattr(run,'validate_row',lambda *args,**kwargs:None)
    monkeypatch.setattr(run.data,'qualify_captured_rows',lambda *args,**kwargs:{'passed':True})
    env=SimpleNamespace(guard=lambda:None)
    proof=run.verify_capture_fixtures(env,{}, {'loader_cases':{}},writing=True)
    before=(out/'capture_qualification.json').read_bytes()
    assert run.verify_capture_fixtures(env,{}, {'loader_cases':{}},writing=False)==proof
    assert (out/'capture_qualification.json').read_bytes()==before


def test_forecast_readback_keeps_original_clock_and_does_not_republish(monkeypatch,tmp_path):
    out=workspace(monkeypatch,tmp_path)
    for name in ('capture_off','capture_on'):write(out/'qualification'/f'{name}.json',{})
    monkeypatch.setattr(run.time,'monotonic',lambda:100.)
    monkeypatch.setattr(run.time,'time',lambda:1100.)
    env=SimpleNamespace(start=0.,wall_start=1000.,waited=60.,accounted_bytes=lambda:1234)
    monkeypatch.setattr(run.design,'source_forecast',lambda freeze,snapshot,source_costs:dict(snapshot=deepcopy(snapshot),affordable=True))
    first=run.capture_forecast(env,{},writing=True)
    before=(out/'collection_forecast.json').read_bytes()
    assert run.capture_forecast(env,{},writing=False)==first
    assert (out/'collection_forecast.json').read_bytes()==before
    first['snapshot']['original_monotonic_origin']=1.
    write(out/'collection_forecast.json',first)
    with pytest.raises(ValueError,match='original owner'):run.capture_forecast(env,{},writing=False)


@pytest.mark.parametrize('failure_at',[None,'game','fit','assessment'])
def test_full_controller_preserves_failures_and_closes_before_training(monkeypatch,tmp_path,failure_at):
    from tools.splendorust_selfplay_pilot_v2_20261102.assessment import control as assess
    out=workspace(monkeypatch,tmp_path);events=[];live={}
    write(out/'freeze.json',{});sha=c.file_hash(out/'freeze.json')
    write(out/'watchdog_claim.json',{'freeze_sha256':sha})
    class Envelope:
        def __init__(self,claim):self.start=run.time.monotonic();self.wall_start=run.time.time();self.stage=None
        def enable_evidence_accounting(self):pass
        def begin(self,stage):self.stage=stage;events.append(('stage',stage))
        def finish(self):pass
        def admit(self):pass
    monkeypatch.setattr(run,'Envelope',Envelope)
    for name in ('verify_runtime','switch_placement','collect_fixtures','verify_capture_fixtures','advance_idle_worker','verify_block_forecast'):
        monkeypatch.setattr(run,name,lambda *args,**kwargs:None)
    monkeypatch.setattr(run.placement,'configure_execution',lambda *args:None)
    monkeypatch.setattr(run,'verify_runtime_sources',lambda:[])
    specs=[{'game_index':i} for i in range(4)]
    monkeypatch.setattr(run,'load_inputs',lambda *args:(specs,{}))
    monkeypatch.setattr(run.design,'initial_states',lambda:[{'spec':s,'initial':{}} for s in specs])
    monkeypatch.setattr(run.design,'collection_blocks',lambda:[dict(worker='capture_on',block_index=0,game_indices=list(range(4)))])
    def qualify(name,workers,*args):
        assert not workers;workers[name]=SimpleNamespace(_closed=False);live.update(workers)
    def close(workers,*args):workers.clear();live.clear()
    monkeypatch.setattr(run,'qualify',qualify);monkeypatch.setattr(run,'close_workers',close)
    monkeypatch.setattr(run,'capture_forecast',lambda *args,**kwargs:{'passed':True})
    def game(child,spec,*args):
        events.append(('game',spec['game_index']))
        if failure_at=='game':raise ValueError('saved game failure')
        return {}
    monkeypatch.setattr(run,'play_game',game)
    monkeypatch.setattr(run.data,'build_manifest',lambda *args,**kwargs:{})
    monkeypatch.setattr(run,'select_replays',lambda *args:list(range(12)))
    monkeypatch.setattr(run,'reproduce',lambda *args:events.append(('replay',12)))
    def fit(*args):
        assert not live;events.append(('fit',))
        if failure_at=='fit':raise ValueError('saved fitting failure')
        write(out/'fit/fit_result.json',{'checkpoint_manifest':{'fixed':True}})
    monkeypatch.setattr(run,'run_fit',fit);monkeypatch.setattr(run,'verify_fit',lambda *args:None)
    def assessment(env,freeze,candidate):
        assert not live and candidate=={'fixed':True};events.append(('assessment',))
        if failure_at=='assessment':raise ValueError('saved assessment failure')
    monkeypatch.setattr(assess,'run',assessment)
    monkeypatch.setattr(run,'final_checks',lambda *args:{'passed':True})
    monkeypatch.setattr(run.transport,'close_and_audit',lambda *args:None)
    assert run.run(sha)==(0 if failure_at is None else 1)
    assert sum(event[0]=='game' for event in events)==(1 if failure_at=='game' else 4)
    assert sum(event[0]=='fit' for event in events)==(0 if failure_at=='game' else 1)
    assert (out/'failure.json').exists()==(failure_at is not None)
    assert c.read(out/'completion.json')['automatic_retries']==0
