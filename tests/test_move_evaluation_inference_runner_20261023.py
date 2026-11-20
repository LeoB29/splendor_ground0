"""Model-free fixed-order, shared-clock, provenance and cleanup regressions."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from tools.move_evaluation_inference_20261023 import runner as r, launch as l


def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value),encoding='utf-8')


@pytest.fixture
def sandbox(tmp_path,monkeypatch):
    c=r.c
    root=tmp_path/'workspace';out=root/'outputs/inference'
    out.mkdir(parents=True)
    for key,value in {'ROOT':root,'OUT':out,'ORIGINAL':root/'outputs/original',
        'PREVIOUS':root/'outputs/previous','PREP':root/'preparation',
        'STATUS':root/'outputs/inference_status'}.items():monkeypatch.setattr(c,key,value)
    def artifact(path):
        path=Path(path).resolve();data=path.read_bytes()
        return {'path':path.relative_to(root).as_posix(),'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)}
    monkeypatch.setattr(c,'artifact',artifact)
    monkeypatch.setattr(r.psutil,'boot_time',lambda:10.)
    return c,root,out


def plan(mono=50.,wall=950.):
    return {**r.new_study_plan(),'monotonic_origin':mono,'unix_origin':wall,'boot_time':10.}


def freeze_plan(c,value=None):
    put(c.OUT/'study_plan.json',value or plan())
    item=c.artifact(c.OUT/'study_plan.json')
    freeze={'study_plan':item,'inputs':[item],'topology':{'synthetic':True},
            'job_seconds':900,'final_reserve_seconds':120}
    put(c.OUT/'freeze.json',freeze)
    return freeze,c.file_hash(c.OUT/'freeze.json')


@pytest.fixture
def orchestration(sandbox,monkeypatch):
    c,root,out=sandbox
    from tools.move_evaluation_inference_20261023 import checks
    import tools.future_training_rules_20261021 as runtime
    events=[];control={}
    _,sha=freeze_plan(c)
    monkeypatch.setenv('DIAGNOSTIC_START_MONOTONIC','100')
    monkeypatch.setenv('DIAGNOSTIC_START_UNIX','1000')
    monkeypatch.setattr(runtime,'verify_runtime_sources',lambda:events.append(('startup_guard',)) or {'synthetic':'a'*64})
    monkeypatch.setattr(r,'verify_runtime',lambda freeze:events.append(('runtime',)))
    monkeypatch.setattr(r,'processor_topology',lambda:{'synthetic':True})
    put(c.ORIGINAL/'entrypoint_qualification.json',{'fixture':'accepted_original'})
    class Envelope:
        def __init__(self,launch,study):
            assert study['monotonic_origin']==50.
            self.end=1000.;self.work_end=880.;self.waited=60.
        def guard(self,deadline,**kwargs):
            events.append(('guard',deadline))
            if control.get('final_guard_failure') and len([x for x in events if x[0]=='panel_check']):
                raise ValueError('final resource boundary')
    monkeypatch.setattr(r,'Envelope',Envelope)
    def perform(name,mode,payload,limit,deadline,envelope,freeze):
        events.append(('request',name,mode,deepcopy(payload),limit,deadline))
        if control.get('request_failure')==name:raise ValueError('synthetic request failed')
        science={'fixture':'accepted_original'} if mode.startswith('qualify_') else {'panel':'same_fixed_science'}
        value={'result':{'scientific':science,'measurement':{'name':name}},'complete_response_seconds':1.}
        c.save(out/'measurements'/(name+'.json'),value)
        return value
    monkeypatch.setattr(r,'perform',perform)
    def fixture(a,b,original,measurement):
        events.append(('fixture_check',))
        assert a==b==original and measurement['name']=='qualification_timed'
        if control.get('fixture_exception'):raise ValueError('fixture scientific mismatch')
        return {'status':'failed' if control.get('fixture_failed') else 'passed'}
    def panel(reference,plain,timed,measurement):
        events.append(('panel_check',))
        assert reference==plain==timed and measurement['name']=='probe_timed'
        if control.get('panel_exception'):raise ValueError('panel count mismatch')
        return {'status':'failed' if control.get('panel_failed') else 'passed'}
    monkeypatch.setattr(checks,'validate_fixture',fixture)
    monkeypatch.setattr(checks,'validate_panel',panel)
    return c,events,control,sha


@pytest.mark.parametrize('failure',['fixture_exception','fixture_failed'])
def test_failed_actual_model_qualification_blocks_every_probe(orchestration,failure):
    c,events,control,sha=orchestration;control[failure]=True
    with pytest.raises(ValueError):r.run(sha)
    assert [e[1] for e in events if e[0]=='request']==['qualification_plain','qualification_timed']
    saved=c.read(c.OUT/'failure.json')
    assert saved['verified_stages']==[] and saved['automatic_retry'] is False
    assert not (c.OUT/'completion.json').exists()


def test_exact_five_requests_and_fixture_gate_precede_panel(orchestration):
    c,events,control,sha=orchestration;r.run(sha)
    requests=[e for e in events if e[0]=='request']
    assert [x[1] for x in requests]==['qualification_plain','qualification_timed','probe_reference','probe_plain','probe_timed']
    assert events.index(('fixture_check',))<events.index(requests[2])
    assert all(x[4:]==(120,880.) for x in requests)
    assert all(x[3]=={'operation':'entrypoint_checks'} for x in requests[:2])
    assert all(x[3]=={'operation':'public_inference_panel','repetitions':8} for x in requests[2:])
    assert events.count(('startup_guard',))==1 and events.count(('runtime',))==2
    result=c.read(c.OUT/'completion.json')
    assert result['status']=='completed_inference_diagnosis_only'
    assert result['verified_stages']==['qualification','panel']
    assert result['stage_b_started'] is False and result['full_selection_requests']==0
    assert result['optimization_enabled'] is False and result['full_18_position_qualification'] is False
    assert result['original_120_second_qualification']=='failed_unchanged'


@pytest.mark.parametrize('failure',['panel_exception','panel_failed','final_guard_failure'])
def test_failed_panel_or_final_authentication_never_creates_completion(orchestration,failure):
    c,events,control,sha=orchestration;control[failure]=True
    with pytest.raises(ValueError):r.run(sha)
    assert not (c.OUT/'completion.json').exists()
    saved=c.read(c.OUT/'failure.json')
    assert saved['verified_stages']==(['qualification','panel'] if failure=='final_guard_failure' else ['qualification'])
    assert saved['stage_b_started'] is False


@pytest.mark.parametrize('failed',['qualification_plain','qualification_timed','probe_reference','probe_plain','probe_timed'])
def test_request_failure_preserves_last_completed_boundary_without_retry(orchestration,failed):
    c,events,control,sha=orchestration;control['request_failure']=failed
    ordered=['qualification_plain','qualification_timed','probe_reference','probe_plain','probe_timed']
    with pytest.raises(ValueError,match='request failed'):r.run(sha)
    assert [e[1] for e in events if e[0]=='request']==ordered[:ordered.index(failed)+1]
    saved=c.read(c.OUT/'failure.json')
    assert saved['completed_requests']==ordered[:ordered.index(failed)]
    assert saved['verified_stages']==([] if failed.startswith('qualification') else ['qualification'])


def test_existing_launch_cannot_be_overwritten(orchestration):
    c,events,control,sha=orchestration;r.run(sha)
    original=(c.OUT/'launch.json').read_bytes()
    with pytest.raises(FileExistsError):r.run(sha)
    assert (c.OUT/'launch.json').read_bytes()==original


@pytest.mark.parametrize('now,expected',[(5,10),(12,8),(19.5,.5)])
def test_request_timeout_has_no_grace(now,expected):
    assert r.clamp_timeout(10,20,clock=lambda:now)==expected


@pytest.mark.parametrize('now',[20,21])
def test_expired_request_never_starts(now):
    with pytest.raises(ValueError,match='deadline'):r.clamp_timeout(10,20,clock=lambda:now)


@pytest.fixture
def envelope(sandbox,monkeypatch):
    c,root,out=sandbox
    clock=SimpleNamespace(mono=100.,wall=1000.,ram=3*c.GiB,disk=6*c.GiB)
    def sleep(seconds):clock.mono+=seconds;clock.wall+=seconds
    monkeypatch.setattr(r,'time',SimpleNamespace(monotonic=lambda:clock.mono,time=lambda:clock.wall,sleep=sleep))
    monkeypatch.setattr(r.psutil,'virtual_memory',lambda:SimpleNamespace(available=clock.ram))
    monkeypatch.setattr(r.shutil,'disk_usage',lambda path:SimpleNamespace(free=clock.disk))
    return c,r.Envelope({'monotonic':100.,'unix_time':1000.},plan()),clock


def test_initial_ram_stability_then_fast_admission(envelope):
    c,e,clock=envelope;e.admit(e.work_end);e.admit(e.work_end)
    first=c.read(c.OUT/'memory_admissions/001.json');second=c.read(c.OUT/'memory_admissions/002.json')
    assert first['wait_seconds']==60 and first['fast_path'] is False
    assert second['fast_path'] is True and second['cumulative_wait_seconds']==60


def test_ram_wait_cannot_consume_final_work_reserve(envelope):
    c,e,clock=envelope
    with pytest.raises(ValueError,match='deadline'):e.admit(clock.mono+59)
    assert not (c.OUT/'memory_admissions/001.json').exists()


def test_ram_wait_is_charged_and_bounded(envelope):
    c,e,clock=envelope;clock.ram=0
    with pytest.raises(ValueError,match='Cumulative memory'):e.admit(e.work_end)
    assert clock.mono==400 and not (c.OUT/'memory_admissions/001.json').exists()


@pytest.mark.parametrize('damage',['mono_work','wall_work','disk','memory'])
def test_live_guards_fail_closed_before_reserved_finalization(envelope,damage):
    c,e,clock=envelope
    if damage=='mono_work':clock.mono=e.work_end
    elif damage=='wall_work':clock.wall=e.wall_start+(e.work_end-e.start)
    elif damage=='disk':clock.disk=5*c.GiB-1
    else:clock.ram=2*c.GiB-1
    with pytest.raises(ValueError):e.guard(e.work_end)


def test_forward_shared_wall_jump_is_not_refunded_by_fresh_stage_clock(envelope):
    c,e,clock=envelope
    # A launch close to the shared deadline cannot restart a fresh 15 minutes.
    e.plan['unix_origin']=clock.wall-r.STUDY_SECONDS+200
    clock.wall+=80
    assert clock.mono<e.work_end
    with pytest.raises(ValueError,match='three-hour'):e.guard(e.work_end)


def test_shared_monotonic_deadline_clamps_fresh_stage_and_reserve(envelope):
    c,e,clock=envelope
    late=r.Envelope({'monotonic':10800.,'unix_time':11700.},plan())
    assert late.end==10850 and late.work_end==10730
    clock.mono=10800;clock.wall=11700
    with pytest.raises(ValueError,match='deadline'):late.guard(late.work_end)


def test_request_deadline_is_clamped_to_shorter_wall_allowance(envelope):
    c,e,clock=envelope
    clock.wall=e.wall_start+(e.work_end-e.start)-17
    assert e.effective_deadline(e.work_end)==clock.mono+17


def test_wall_clock_read_cannot_refund_elapsed_time(envelope,monkeypatch):
    c,e,clock=envelope
    # Only the final wall sample advances time; the conservative mono sample precedes it.
    calls=[]
    def wall():
        calls.append(True)
        if len(calls)==3:
            clock.mono+=7;clock.wall+=7
        return clock.wall
    monkeypatch.setattr(r.time,'time',wall)
    effective=e.effective_deadline(e.work_end)
    assert clock.mono==107 and effective==e.work_end-7


@pytest.mark.parametrize('damage',['extra','cap','retry','origin','reboot','bool'])
def test_plan_limits_origins_and_boot_are_immutable(sandbox,damage):
    value=plan()
    if damage=='extra':value['extension']=1
    elif damage=='cap':value['maximum_wall_seconds']=10801
    elif damage=='retry':value['automatic_retries']=1
    elif damage=='origin':value['monotonic_origin']=float('nan')
    elif damage=='reboot':value['boot_time']=9.
    else:value['stage_b_requires_closed_stage_a']=1
    with pytest.raises(ValueError):r.validate_study_plan(value)


@pytest.mark.parametrize('damage',['file','descriptor','membership'])
def test_plan_must_match_its_frozen_artifact(sandbox,damage):
    c,root,out=sandbox;freeze,_=freeze_plan(c)
    if damage=='file':put(out/'study_plan.json',{**plan(),'monotonic_origin':51.})
    elif damage=='descriptor':freeze['study_plan']['sha256']='0'*64
    else:freeze['inputs']=[]
    with pytest.raises(ValueError,match='bound'):r.read_study_plan(freeze)


def test_failed_prepare_preserves_origin_and_cannot_be_retried(sandbox):
    c,root,out=sandbox
    put(c.PREVIOUS/'freeze.json',{'wrong':'freeze'})
    with pytest.raises(ValueError,match='Previous reuse'):r.prepare()
    old=(out/'study_plan.json').read_bytes()
    with pytest.raises(ValueError,match='one-shot'):r.prepare()
    assert (out/'study_plan.json').read_bytes()==old and not (out/'freeze.json').exists()


@pytest.mark.parametrize('damage',[None,'run','admin','report','collision','missing','extra','duplicate'])
def test_closed_reuse_inventory_inherits_every_file_but_not_mutable_handoffs(sandbox,damage):
    c,root,out=sandbox
    partial=c.PREVIOUS/'workers/r00/continuations.jsonl.gz'
    partial.parent.mkdir(parents=True);partial.write_bytes(b'closed evidence')
    script=root/'outputs/move_evaluation_reuse_20261023_status_01/verify_closeout.py'
    report=root/'docs/engineering/result.md'
    put(script,{'script':True});put(report,{'reported':True})
    handoff=root/'AGENT_CURRENT.md';handoff.write_text('old')
    run=[c.artifact(partial)];admin=[c.artifact(script),c.artifact(report)]
    value={'run_inventory':run,'run_files':1,'run_bytes':run[0]['bytes'],
           'inspection_source':admin[0],'report':admin[1],'continuity':[c.artifact(handoff)]}
    handoff.write_text('new authorized handoff');inputs={}
    if damage=='run':partial.write_bytes(b'tampered')
    elif damage=='admin':script.write_text('tampered')
    elif damage=='report':report.write_text('tampered')
    elif damage=='collision':inputs[run[0]['path']]={**run[0],'sha256':'0'*64}
    elif damage=='missing':value.update(run_inventory=[],run_files=0,run_bytes=0)
    elif damage=='extra':(c.PREVIOUS/'added').write_text('extra')
    elif damage=='duplicate':value.update(run_inventory=run*2,run_files=2,run_bytes=run[0]['bytes']*2)
    if damage is None:
        r.inherit_closed_evidence(value,inputs)
        assert inputs=={a['path']:a for a in run+admin}
    else:
        with pytest.raises(ValueError):r.inherit_closed_evidence(value,inputs)


@pytest.mark.parametrize('finish,cleanup_failure',[(True,False),(False,False),(False,True)])
def test_owned_watchdog_closes_children_and_charges_both_clocks(sandbox,monkeypatch,finish,cleanup_failure):
    c,root,out=sandbox;_,sha=freeze_plan(c,plan(100.,1000.))
    clock=SimpleNamespace(mono=110.,wall=1010.);events=[]
    def sleep(seconds):clock.mono+=899;clock.wall+=901
    fake_time=SimpleNamespace(monotonic=lambda:clock.mono,time=lambda:clock.wall,sleep=sleep)
    monkeypatch.setattr(l,'time',fake_time);monkeypatch.setattr(r,'time',fake_time)
    monkeypatch.setattr(l.sys,'argv',['launch','--freeze-sha256',sha])
    monkeypatch.setattr(l.psutil,'Process',lambda *args:SimpleNamespace(create_time=lambda:99.))
    class Child:
        pid=123;returncode=0 if finish else None
        def poll(self):return self.returncode
        def kill(self):events.append('kill');self.returncode=-1
        def wait(self,timeout):events.append('wait');return self.returncode
    child=Child()
    def popen(command,**kwargs):
        assert command[4]=='tools.move_evaluation_inference_20261023.runner'
        assert kwargs['env']['DIAGNOSTIC_START_MONOTONIC']=='110.0'
        assert kwargs['env']['OMP_NUM_THREADS']=='1'
        return child
    monkeypatch.setattr(l.subprocess,'Popen',popen)
    class Job:
        def __init__(self,pid):assert pid==123
        def close(self):
            events.append('job_close')
            if cleanup_failure:raise OSError('synthetic cleanup failure')
    monkeypatch.setattr(l,'_WindowsJob',Job)
    l.main()
    value=c.read(out/'execution.json')
    assert events[0]=='job_close' and events[-1]=='wait'
    assert value['status']==('cleanup_failed' if cleanup_failure else 'completed' if finish else 'watchdog_deadline')
    assert value['charged_owned_job_seconds']==(0 if finish else 901)
    assert value['remaining_owned_job_seconds']==(4500 if finish else 3599)
    assert ('kill' in events)==(not finish)
    before=(out/'execution.json').read_bytes()
    with pytest.raises(ValueError,match='relaunch'):l.main()
    assert (out/'execution.json').read_bytes()==before


def test_shared_study_expiry_prevents_child_creation_and_preserves_receipt(sandbox,monkeypatch):
    c,root,out=sandbox;_,sha=freeze_plan(c,plan(100.,1000.))
    fake_time=SimpleNamespace(monotonic=lambda:10800.,time=lambda:11800.,sleep=lambda _:None)
    monkeypatch.setattr(l,'time',fake_time);monkeypatch.setattr(r,'time',fake_time)
    monkeypatch.setattr(l.sys,'argv',['launch','--freeze-sha256',sha])
    monkeypatch.setattr(l.subprocess,'Popen',lambda *a,**kw:pytest.fail('must not create child'))
    with pytest.raises(ValueError,match='three-hour'):l.main()
    assert c.read(out/'execution.json')['status']=='watchdog_study_deadline'
    assert not (out/'launch.json').exists()


@pytest.mark.parametrize('mode',r.MODES)
def test_model_free_worker_metadata_is_authoritative(mode):
    value=r.implementation_metadata(mode)
    assert value['optimization_enabled'] is False and value['prediction_cache'] is False
    assert value['active_opponent']==('original_reference' if mode=='probe_reference' else 'factored_inference')
    r.check_implementation(value,mode)
    changed={**value,'prediction_cache':True}
    with pytest.raises(ValueError,match='implementation'):r.check_implementation(changed,mode)


@pytest.mark.parametrize('damage',[None,'initialization','response'])
def test_perform_checks_implementation_before_payload_and_after_receipt(sandbox,monkeypatch,damage):
    c,root,out=sandbox;events=[];metadata=r.implementation_metadata('probe_reference')
    class Child:
        def __init__(self,module,folder,**kwargs):
            assert module=='tools.move_evaluation_inference_20261023.worker'
            assert kwargs['affinity']==[0] and kwargs['max_rss_bytes']==2*c.GiB
            assert kwargs['initialization_timeout']<=60
            self.folder=folder;initial=deepcopy(metadata)
            if damage=='initialization':initial['active_opponent']='factored_inference'
            c.save(folder/'implementation.json',initial)
            self.initialization={'elapsed_seconds':1.,'telemetry':{'process_tree':[]}}
        def __enter__(self):return self
        def __exit__(self,*args):events.append('closed')
        def request(self,payload,timeout):
            events.append('request');assert timeout<=120
            impl=deepcopy(metadata)
            if damage=='response':impl['prediction_cache']=True
            receipt={'status':'ok','accepted':True,'detail':None,'request_id':'request-00000001',
                'receipt_elapsed_seconds':2.,'result':{'scientific':{},'implementation':impl},
                'telemetry':{'peak_sampled_rss_bytes':100,'process_tree':[]}}
            c.save(self.folder/'request-00000001.json',receipt)
            return receipt
    monkeypatch.setattr(c,'StudyWorker',Child)
    env=SimpleNamespace(admit=lambda deadline:None,guard=lambda deadline:None,
                        effective_deadline=lambda deadline:deadline)
    if damage is None:
        value=r.perform('probe_reference','probe_reference',{},120,1e12,env,{'native_binary':{}})
        assert value['result']['implementation']==metadata
    else:
        with pytest.raises(ValueError,match='implementation'):
            r.perform('probe_reference','probe_reference',{},120,1e12,env,{'native_binary':{}})
        assert not (out/'measurements/probe_reference.json').exists()
    assert events[-1]=='closed' and ('request' in events)==(damage!='initialization')
