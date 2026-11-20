"""Model-free recovery orchestration regressions using the saved clean RAM-stop state."""
import ast
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from tools.splendorust_search256_recovery_20261030_02 import common as c,runner,transport,schedule,design


def forecast_inputs():
    return dict(game_costs=[dict(seconds=110.,independent_seconds=4.,bytes=10*c.MiB,files=400) for _ in range(6)],
        worker_costs=[dict(initialization=4.,graph=.5,closure=.5) for _ in range(12)],hash_seconds=[.5]*6,
        runtime_seconds=2.,ram_wait_used=c.PRIOR_RAM_WAIT_SECONDS+60.,current_bytes=200*c.MiB,current_files=3000,
        prior_files=93000,monotonic_origin=c.PRIOR_ORIGIN,unix_origin=c.PRIOR_WALL_START,
        monotonic=c.PRIOR_ORIGIN+45000.,unix_time=c.PRIOR_WALL_START+45000.)


def test_recovery_forecast_includes106full_games48replays_and400readbacks():
    value=runner.recovery_projection(forecast_inputs())
    assert value['passed'] and value['remaining_games']==106 and value['retained_completed_games']==294
    assert value['replay_decisions_maximum']==48 and value['all_main_saved_readbacks']==400
    assert value['projected_saved_final_seconds']>c.INDEPENDENT_REVIEW_SECONDS
    assert value['projected_collection_seconds']>value['projected_main_seconds']+value['projected_verification_seconds']
    assert value['projected_final_bytes']>93000*2048 and value['main_deadline_not_reset']


@pytest.mark.parametrize('mutation',['main_mono','main_wall','global','verification','storage','nan','prior_wait','prior_files','five_ops'])
def test_recovery_forecast_cannot_refresh_time_or_omit_bounds(mutation):
    values=forecast_inputs()
    if mutation=='main_mono':values['monotonic']=c.ORIGINAL_MAIN_MONOTONIC_DEADLINE-1
    elif mutation=='main_wall':values['unix_time']=c.ORIGINAL_MAIN_UNIX_DEADLINE-1
    elif mutation=='global':values['unix_time']=c.PRIOR_WALL_START+c.JOB_SECONDS-1
    elif mutation=='verification':values['game_costs'][0]['independent_seconds']=40.
    elif mutation=='storage':values['current_bytes']=c.PREFINAL_EVIDENCE_BYTES
    elif mutation=='nan':values['runtime_seconds']=float('nan')
    elif mutation=='prior_wait':values['ram_wait_used']=c.PRIOR_RAM_WAIT_SECONDS-1
    elif mutation=='prior_files':values['prior_files']=-1
    elif mutation=='five_ops':values['game_costs'].pop()
    if mutation in ('nan','prior_wait','prior_files','five_ops'):
        with pytest.raises(ValueError):runner.recovery_projection(values)
    else:assert runner.recovery_projection(values)['passed'] is False


def test_projection_uses_costs_not_outcomes():
    values=forecast_inputs();first=runner.recovery_projection(values)
    values['game_costs'][0].update(winner=0,reason='natural',score=1.)
    assert runner.recovery_projection(values)==first
    values['game_costs'][0]['seconds']+=10.
    assert runner.recovery_projection(values)['projected_main_seconds']-first['projected_main_seconds']==1.5*106*10


def test_original_clocks_clip_recovery_qualification_and_main(monkeypatch,tmp_path):
    monkeypatch.setattr(c,'OUT',tmp_path);monkeypatch.setattr(c,'ROOT',tmp_path)
    mono=c.ORIGINAL_MAIN_MONOTONIC_DEADLINE-1000;wall=c.ORIGINAL_MAIN_UNIX_DEADLINE-1000
    monkeypatch.setattr(runner.time,'monotonic',lambda:mono);monkeypatch.setattr(runner.time,'time',lambda:wall)
    e=runner.Envelope(dict(monotonic_origin=c.PRIOR_ORIGIN,unix_origin=c.PRIOR_WALL_START,
        owner_started_monotonic=mono-1,owner_started_unix_time=wall-1));e.guard=lambda:None
    assert e.waited==c.PRIOR_RAM_WAIT_SECONDS
    for stage in ('recovery_qualification','main'):
        e.begin(stage)
        assert e.deadline==c.ORIGINAL_MAIN_MONOTONIC_DEADLINE
        assert e.wall_deadline==c.ORIGINAL_MAIN_UNIX_DEADLINE
    for key in ('monotonic_origin','unix_origin'):
        claim=dict(monotonic_origin=c.PRIOR_ORIGIN,unix_origin=c.PRIOR_WALL_START);claim[key]+=1
        with pytest.raises(ValueError,match='nonresettable'):runner.Envelope(claim)


def test_saved_only_final_rejects_resident_models(monkeypatch):
    e=runner.Envelope(dict(monotonic_origin=c.PRIOR_ORIGIN,unix_origin=c.PRIOR_WALL_START))
    e.resident={1:object()}
    with pytest.raises(ValueError,match='no resident'):e.begin('final')


@pytest.mark.parametrize('publication_error',[False,True])
def test_exhausted_wait_is_fully_charged_and_original_error_survives(monkeypatch,tmp_path,publication_error):
    monkeypatch.setattr(c,'OUT',tmp_path);monkeypatch.setattr(c,'ROOT',tmp_path)
    clock=iter([100.,102.,103.])
    monkeypatch.setattr(runner.time,'monotonic',lambda:next(clock))
    monkeypatch.setattr(runner.time,'time',lambda:1000.)
    monkeypatch.setattr(runner.psutil,'virtual_memory',lambda:SimpleNamespace(available=0))
    e=runner.Envelope(dict(monotonic_origin=c.PRIOR_ORIGIN,unix_origin=c.PRIOR_WALL_START))
    e.guard=lambda:None;e.admitted=True;e.stage='main';e.waited=c.RAM_WAIT_SECONDS-1
    if publication_error:
        def fail(*args):raise OSError('test receipt publication failure')
        monkeypatch.setattr(c,'save',fail)
    with pytest.raises(ValueError,match='Cumulative frozen RAM wait exceeded') as caught:e.admit()
    assert e.waited==c.RAM_WAIT_SECONDS+2
    if publication_error:
        assert any('receipt publication failure' in s for s in caught.value.__notes__)
    else:
        saved=json.loads((tmp_path/'memory_wait_failure.json').read_text())
        assert saved['prior_cumulative_wait_seconds']==c.RAM_WAIT_SECONDS-1
        assert saved['wait_seconds']==3 and saved['cumulative_wait_seconds']==c.RAM_WAIT_SECONDS+2
        assert saved['decision_input_not_published'] and saved['decision_clock_not_started']
    assert not (tmp_path/'memory_admissions').exists()


def test_extended_wait_budget_preserves_prior_charge_and60second_stability(monkeypatch,tmp_path):
    monkeypatch.setattr(c,'OUT',tmp_path);monkeypatch.setattr(c,'ROOT',tmp_path)
    clock=[100.]
    monkeypatch.setattr(runner.time,'monotonic',lambda:clock[0])
    monkeypatch.setattr(runner.time,'time',lambda:1000.+clock[0])
    monkeypatch.setattr(runner.time,'sleep',lambda seconds:clock.__setitem__(0,clock[0]+seconds))
    monkeypatch.setattr(runner.psutil,'virtual_memory',lambda:SimpleNamespace(available=c.INITIAL_AVAILABLE_RAM_BYTES))
    e=runner.Envelope(dict(monotonic_origin=c.PRIOR_ORIGIN,unix_origin=c.PRIOR_WALL_START));e.guard=lambda:None
    assert e.waited>=1800 and c.RAM_WAIT_SECONDS==10800
    e.admit()
    assert e.waited==c.PRIOR_RAM_WAIT_SECONDS+60 and e.admitted
    saved=json.loads((tmp_path/'memory_admissions/000001.json').read_text())
    assert saved['wait_seconds']==60 and saved['stable_seconds']==60


def test_qualification_uses_guardian_start_not_later_runner_start(monkeypatch,tmp_path):
    monkeypatch.setattr(c,'OUT',tmp_path);monkeypatch.setattr(c,'ROOT',tmp_path)
    start=c.PRIOR_ORIGIN+45000.;wall=c.PRIOR_WALL_START+45000.
    monkeypatch.setattr(runner.time,'monotonic',lambda:start+7000.)
    monkeypatch.setattr(runner.time,'time',lambda:wall+7000.)
    e=runner.Envelope(dict(monotonic_origin=c.PRIOR_ORIGIN,unix_origin=c.PRIOR_WALL_START,
        owner_started_monotonic=start,owner_started_unix_time=wall));e.guard=lambda:None
    e.begin('recovery_qualification')
    assert e.deadline==start+7200 and e.wall_deadline==wall+7200 and e.remaining()==200.


def test_actual_tree_bound_includes_two_interrupted_replacements_without_dropping_replays():
    allowed=design.lifecycle_inventory();new_games=schedule.make_schedule('main')[294:]
    assert len(new_games)==106 and len(allowed)==2*106+2*48
    assert c.PRIOR_WORKER_TREES==608 and c.MAX_WORKERS==260 and c.MAX_COMBINED_WORKERS==868
    assert c.PRIOR_WORKER_TREES+2*len(new_games)+48==868
    assert not any(name.startswith('qualification_') for name in allowed)
    for spec in new_games:
        for role in ('challenger','opponent'):
            entry=allowed[design.game_worker_name(spec,role)]
            assert entry['entrypoint_request_id']==('request-00000002' if spec['game_index']==294 else None)
            assert entry['ordinary_first_request_id']==('request-00000003' if spec['game_index']==294 else 'request-00000002')


def saved_prefix():
    folder=c.PREVIOUS_OUT/'games/main/research/p147_g294'
    rows=[c.read(p) for p in sorted((folder/'moves').glob('*.json'))]
    assert len(rows)==50
    return rows,runner.decode_training_snapshot(rows[-1]['after']),schedule.make_schedule('main')[294]


def prepare_continuation_fixture(monkeypatch,tmp_path):
    rows,state,spec=saved_prefix()
    monkeypatch.setattr(c,'OUT',tmp_path)
    monkeypatch.setattr(c,'ROOT',tmp_path)
    monkeypatch.setattr(c,'artifact',lambda p:dict(path=str(p),sha256='fixture',bytes=1))
    (tmp_path/'freeze.json').write_text('{}')
    child=SimpleNamespace(route='frozen',_counter=2,output_dir=tmp_path/'workers/main_research_g294_opponent')
    return rows,state,spec,child


def test_clean_unstarted_boundary_can_be_committed_exactly_once(monkeypatch,tmp_path):
    rows,state,spec,child=prepare_continuation_fixture(monkeypatch,tmp_path)
    marker=runner.consume_continuation_boundary(spec,rows,state,child,{})
    target=tmp_path/'clean_continuation_consumed.json'
    assert target.exists() and marker['path']==str(target)
    value=json.loads(target.read_text())
    assert value['previous_request_started'] is False and value['new_decision_retries']==0
    assert value['new_request_id']=='request-00000003' and value['previous_next_request_id']=='request-00000027'
    assert value['complete_response_seconds']==10. and value['internal_seconds']==9.
    with pytest.raises(ValueError,match='Single clean'):runner.consume_continuation_boundary(spec,rows,state,child,{})
    assert json.loads(target.read_text())==value


@pytest.mark.parametrize('mutation',[None,'clock_before_commitment','admission_after_commitment','changed_state','new_retry','wrong_route','previous_timeout'])
def test_saved_clean_boundary_reader_rejects_chronology_or_retry_tampering(monkeypatch,tmp_path,mutation):
    rows,state,spec,child=prepare_continuation_fixture(monkeypatch,tmp_path)
    # Use the real accepted state and old closure evidence, with only a synthetic
    # first-response envelope: full science/transport checks have separate tests.
    mono=c.ORIGINAL_MAIN_MONOTONIC_DEADLINE-1000.;wall=c.ORIGINAL_MAIN_UNIX_DEADLINE-1000.
    monkeypatch.setattr(runner.time,'monotonic',lambda:mono)
    monkeypatch.setattr(runner.time,'time',lambda:wall)
    monkeypatch.setattr(runner.time,'perf_counter_ns',lambda:200)
    commitment=runner.consume_continuation_boundary(spec,rows,state,child,{})
    row=dict(spec,ply=50,seat=0,player_role='opponent',route='frozen',quota=128,completed=True,
        before=deepcopy(rows[-1]['after']),observer=runner.rules.observation_payload(state),
        clock=dict(request_id='request-00000003',status='ok'),parent_profile=dict(request_start_ns=300))
    permit=dict(perf_counter_ns=100,monotonic=mono-1,unix_time=wall-1)
    outcome=dict(status='recorded',commitment=commitment,new_move=c.artifact(runner.game_folder(spec)/'moves/0050.json'),
        completed=True,clock_status='ok',new_decision_retries=0,automatic_retries=0,previous_request_not_classified_as_timeout=True)
    if mutation=='clock_before_commitment':row['parent_profile']['request_start_ns']=199
    elif mutation=='admission_after_commitment':permit['perf_counter_ns']=201
    elif mutation=='changed_state':row['before']['state']['turn_index']=49
    elif mutation=='wrong_route':row['route']='research'
    elif mutation=='previous_timeout':outcome['previous_request_not_classified_as_timeout']=False
    elif mutation=='new_retry':(tmp_path/'retry_authorization_consumed.json').write_text('{}')
    c.save(runner.game_folder(spec)/'moves/0050.json',row)
    c.save(tmp_path/'recovery_main_admission.json',permit)
    c.save(tmp_path/'clean_continuation.json',outcome)
    check=lambda:runner.verify_clean_continuation({'schedules':{'main':schedule.make_schedule('main')}},{},SimpleNamespace(guard=lambda:None),row)
    if mutation is None:assert check()['new_decision_retries']==0
    else:
        with pytest.raises(ValueError):check()


@pytest.mark.parametrize('mutation',['short_prefix','failed_prefix','other_game','other_request','other_route','other_state','extra_pending','bad_cleanup','wrong_failure'])
def test_continuation_rejects_any_unproved_boundary(monkeypatch,tmp_path,mutation):
    rows,state,spec,child=prepare_continuation_fixture(monkeypatch,tmp_path)
    if mutation=='short_prefix':rows=rows[:-1]
    elif mutation=='failed_prefix':rows=deepcopy(rows);rows[0]['completed']=False
    elif mutation=='other_game':spec=dict(spec,game_index=295)
    elif mutation=='other_request':child._counter=3
    elif mutation=='other_route':child.route='research'
    elif mutation=='other_state':state=runner.decode_training_snapshot(rows[-2]['after'])
    elif mutation=='extra_pending':
        original=Path.rglob
        def extra(path,pattern):
            if path==c.PREVIOUS_OUT/'workers/main_research_g294_opponent' and pattern=='*.pending':
                return iter([path/'request-00000027_child_profile.pending'])
            return original(path,pattern)
        monkeypatch.setattr(Path,'rglob',extra)
    elif mutation in ('bad_cleanup','wrong_failure'):
        original=c.read
        def changed(path):
            value=original(path)
            if mutation=='bad_cleanup' and Path(path).name=='cleanup.json':value['errors']=['test preservation failure']
            if mutation=='wrong_failure' and Path(path).name=='failure.json':value['error']='unknown interruption'
            return value
        monkeypatch.setattr(c,'read',changed)
    with pytest.raises(ValueError):runner.consume_continuation_boundary(spec,rows,state,child,{})
    assert not (tmp_path/'clean_continuation_consumed.json').exists()


def test_transport_science_readers_clocks_unchanged_except_execution_namespace():
    original=c.ROOT/'tools/splendorust_search256_recovery_20261030/transport.py'
    expected=original.read_text().replace('tools.splendorust_search256_recovery_20261030.worker','tools.splendorust_search256_recovery_20261030_02.worker')
    assert Path(transport.__file__).read_text()==expected


@pytest.mark.parametrize('active',[False,True])
def test_retained_phase_transition_requires_idle_qualified_workers(monkeypatch,tmp_path,active):
    monkeypatch.setattr(c,'OUT',tmp_path);monkeypatch.setattr(c,'ROOT',tmp_path)
    children={role:SimpleNamespace(_counter=2,_closed=False,recovery_accounting=SimpleNamespace(state='request' if active else 'idle'),
        output_dir=tmp_path/'workers'/role,process=SimpleNamespace(pid=i),absolute_deadline=100.)
        for i,role in enumerate(('challenger','opponent'),1)}
    env=SimpleNamespace(stage='main',effective_deadline=lambda:c.ORIGINAL_MAIN_MONOTONIC_DEADLINE)
    if active:
        with pytest.raises(ValueError,match='No active'):runner.transition_retained_workers(children,env)
        assert not (tmp_path/'retained_worker_phase_transition.json').exists()
    else:
        runner.transition_retained_workers(children,env)
        assert all(child.absolute_deadline==c.ORIGINAL_MAIN_MONOTONIC_DEADLINE for child in children.values())


@pytest.mark.parametrize('failure',['none','preflight','qualification','qualified','continuation','later'])
def test_controller_never_replaces_or_retries_failed_continuation(monkeypatch,tmp_path,failure):
    from tools.splendorust_search256_recovery_20261030_02 import verification
    events=[];saved={};specs={s:schedule.make_schedule(s) for s in ('operational','main')}
    class Env:
        def __init__(self,claim):self.start=0.;self.wall_start=0.;self.stage=None;self.source_runtime_seconds=0.
        def enable_evidence_accounting(self):pass
        def begin(self,stage):self.stage=stage;events.append(('stage',stage))
        def finish(self):pass
        def admit(self):pass
    monkeypatch.setattr(c,'OUT',tmp_path);monkeypatch.setattr(c,'file_hash',lambda p:'freeze')
    owner=dict(freeze_sha256='freeze',monotonic_origin=c.PRIOR_ORIGIN,unix_origin=c.PRIOR_WALL_START,
        owner_started_monotonic=c.PRIOR_ORIGIN+100,owner_started_unix_time=c.PRIOR_WALL_START+100)
    monkeypatch.setattr(c,'read',lambda p:owner if p.name in ('watchdog_claim.json','guardian_claim.json') else {})
    monkeypatch.setattr(c,'save',lambda p,v:saved.setdefault(p.name,deepcopy(v)))
    monkeypatch.setattr(c,'artifact',lambda p:{'path':str(p),'sha256':'fixture','bytes':1})
    monkeypatch.setattr(runner,'Envelope',Env);monkeypatch.setattr(runner,'verify_runtime_sources',lambda:{})
    monkeypatch.setattr(runner,'verify_runtime',lambda *a:None);monkeypatch.setattr(runner,'switch_placement',lambda *a:None)
    monkeypatch.setattr(runner,'load_inputs',lambda *a:(specs,{}))
    monkeypatch.setattr(verification,'load_context',lambda *a:{})
    monkeypatch.setattr(verification,'verify_prior_admission',lambda *a:None)
    monkeypatch.setattr(verification,'verify_prefix',lambda *a:{'saved_prefix':True})
    def forecast(*a,name):
        events.append(('forecast',name))
        if failure==name:raise RuntimeError('saved resource failure')
        return {'passed':True}
    def qualify(spec,envelope,freeze,plan,workers):
        events.append(('qualification',spec['game_index']))
        if failure=='qualification':raise RuntimeError('saved qualification failure')
    def play(spec,*a,**kw):
        events.append(('game',spec['game_index']))
        assert 'recovery_main_admission.json' in saved and ('forecast','qualified') in events
        if spec['game_index']==294:assert kw['prefix']=={'saved_prefix':True} and 'retained_workers' in kw
        else:assert not kw
        if failure=='continuation' or failure=='later' and spec['game_index']==295:raise RuntimeError('saved infrastructure failure')
        return dict(spec),[],0.,{'hard_guard_loss':True}
    monkeypatch.setattr(runner,'forecast_recovery',forecast);monkeypatch.setattr(runner,'qualify_retained_workers',qualify)
    monkeypatch.setattr(runner,'transition_retained_workers',lambda *a:None);monkeypatch.setattr(runner,'play_game',play)
    monkeypatch.setattr(runner,'verify_phase',lambda *a,**kw:events.append(('verification',kw['run_models'])))
    monkeypatch.setattr(verification,'final_checks',lambda *a,**kw:{'complete':True},raising=False)
    code=runner.run('freeze');game_ids=[e[1] for e in events if e[0]=='game']
    if failure=='none':
        assert code==0 and game_ids==list(range(294,400)) and ('verification',True) in events
        assert saved['completion.json']['newly_completed_games']==106
    else:
        assert code==1 and saved['completion.json']['status']=='incomplete' and 'failure.json' in saved
        assert game_ids==([294] if failure=='continuation' else [294,295] if failure=='later' else [])
    assert saved['completion.json']['automatic_retries']==0
