"""Model-free orchestration, admission and owned-watchdog regression tests."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from tools.move_evaluation_reuse_20261023 import runner as r, launch as l


def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value),encoding='utf-8')


@pytest.fixture
def sandbox(tmp_path,monkeypatch):
    c=r.c
    root=tmp_path/'workspace';out=root/'outputs/timing'
    out.mkdir(parents=True)
    for name,value in {'ROOT':root,'OUT':out,'OLD':root/'outputs/original',
                       'PREVIOUS':root/'outputs/previous','PREP':root/'preparation'}.items():
        monkeypatch.setattr(c,name,value)
    def artifact(path):
        path=Path(path).resolve();data=path.read_bytes()
        return {'path':path.relative_to(root).as_posix(),'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)}
    monkeypatch.setattr(c,'artifact',artifact)
    return c,root,out


@pytest.fixture
def orchestration(sandbox,monkeypatch):
    c,root,out=sandbox
    from tools.future_training_rules_20261021 import verify_runtime_sources
    import tools.future_training_rules_20261021 as runtime
    from tools.move_evaluation_profile_20261023 import analyze
    from tools.move_evaluation_reuse_20261023 import checks
    events=[];control={}
    monkeypatch.setattr(runtime,'verify_runtime_sources',lambda:{'synthetic':'a'*64})
    monkeypatch.setattr(r,'verify_runtime',lambda freeze:events.append(('runtime',)))
    monkeypatch.setattr(r,'processor_topology',lambda:{'synthetic':True})
    monkeypatch.setattr(c,'selector_payload',lambda row:{'id':row['id']})
    monkeypatch.setenv('DIAGNOSTIC_START_MONOTONIC','0')
    monkeypatch.setenv('DIAGNOSTIC_START_UNIX','0')
    put(out/'freeze.json',{'topology':{'synthetic':True}})
    put(c.OLD/'entrypoint_qualification.json',{'status':'entrypoint_counterexamples_passed','fixture':'fixed'})
    for index in (0,6,12):
        put(c.PREVIOUS/'measurements'/f'r{index:02d}_plain.json',
            {'result':{'scientific':{'previous':True,'root':f'r{index:02d}'}},'complete_response_seconds':200.})
    put(c.PREP/'candidate_lock.json',{'roots':[{'id':str(i),'fit':161001+i//6} for i in range(13)]})
    class Envelope:
        def __init__(self,launch):self.end=3600;self.work_end=3300
        def guard(self,deadline,**kwargs):events.append(('guard',deadline))
    monkeypatch.setattr(r,'Envelope',Envelope)
    def perform(name,mode,payload,limit,deadline,envelope,freeze):
        events.append(('request',name,mode,limit,deadline))
        if mode.startswith('qualify_'):science={'status':'entrypoint_counterexamples_passed','fixture':'fixed'}
        elif mode=='audit':science={'status':'failed' if control.get('audit_failure')==name else 'passed'}
        else:science={'root':name.split('/')[0],'trace':{'path':f'outputs/timing/workers/{name}/continuations.jsonl.gz','sha256':'a'*64,'bytes':1}}
        value={'result':{'scientific':science,'measurement':{'name':name}},'complete_response_seconds':1.}
        c.save(out/'measurements'/(name.replace('/','_')+'.json'),value)
        return value
    monkeypatch.setattr(r,'perform',perform)
    def fixture(*args):
        events.append(('fixture_check',))
        if control.get('fixture_failure'):raise ValueError('fixture count mismatch')
        assert args[0]==args[1]==args[2]
        return {'status':'passed'}
    def selection(measurement,path):
        name=measurement['name'];events.append(('count_check',name))
        if control.get('count_failure')==name:raise ValueError('full count mismatch')
        return {'status':'passed','source':name}
    def compare(first,second):
        events.append(('compare_previous' if first.get('previous') else 'compare_pair',second.get('root')))
        if first.get('previous') and control.get('previous_failure') in (True,second.get('root')):
            raise ValueError('previous scientific mismatch')
        return {'exact':True}
    monkeypatch.setattr(checks,'validate_fixture',fixture)
    monkeypatch.setattr(checks,'validate_selection',selection)
    monkeypatch.setattr(analyze,'compare_scientific_runs',compare)
    monkeypatch.setattr(analyze,'inspect_old_prefix',lambda *args:events.append(('old_prefix',)) or {'records':296})
    return c,events,control,c.file_hash(out/'freeze.json')


def test_fixture_failure_blocks_all_full_roots(orchestration):
    c,events,control,sha=orchestration;control['fixture_failure']=True
    with pytest.raises(ValueError,match='fixture count'):r.run(sha)
    assert [e[1] for e in events if e[0]=='request']==['qualification_plain','qualification_timed']
    failure=c.read(c.OUT/'failure.json')
    assert failure['fully_completed_root_indices']==[] and not failure['automatic_retry']
    assert not (c.OUT/'completion.json').exists()


def test_fixed_order_checks_each_timed_root_before_audit_and_next_root(orchestration):
    c,events,control,sha=orchestration;r.run(sha)
    requests=[e for e in events if e[0]=='request']
    assert [e[1] for e in requests]==['qualification_plain','qualification_timed',
        'r00/plain','r00/timed','r00/audit','r06/plain','r06/timed','r06/audit','r12/plain','r12/timed','r12/audit']
    assert events.index(('fixture_check',))<events.index(requests[2])
    for index in (0,6,12):
        root=f'r{index:02d}'
        count=events.index(('count_check',root+'/timed'))
        audit=next(e for e in requests if e[1]==root+'/audit')
        assert count<events.index(audit)
        assert audit[3:]==(600,3300)
        for mode in ('plain','timed'):
            req=next(e for e in requests if e[1]==root+'/'+mode)
            assert req[3:]==(600,2640)
    for index in (0,6,12):
        reference=events.index(('compare_previous',f'r{index:02d}'))
        audit=events.index(next(e for e in requests if e[1]==f'r{index:02d}/audit'))
        assert reference<audit
    completed=c.read(c.OUT/'completion.json')
    assert completed['status']=='completed_exact_reuse_diagnosis_only'
    assert [x['index'] for x in completed['roots']]==[0,6,12]
    assert completed['original_120_second_qualification']=='failed_unchanged'
    assert completed['new_baseline_selection_requests']==0 and not completed['full_18_position_qualification']
    assert all(row['ordinary_within_120_seconds'] for row in completed['roots'])


@pytest.mark.parametrize('failure',['counts','previous','audit'])
def test_first_root_failure_stops_before_next_root(orchestration,failure):
    c,events,control,sha=orchestration
    control[{'counts':'count_failure','previous':'previous_failure','audit':'audit_failure'}[failure]]= \
        True if failure=='previous' else 'r00/'+('audit' if failure=='audit' else 'timed')
    with pytest.raises(ValueError):r.run(sha)
    names=[e[1] for e in events if e[0]=='request']
    assert 'r06/plain' not in names
    assert ('r00/audit' in names)==(failure=='audit')
    assert c.read(c.OUT/'failure.json')['fully_completed_root_indices']==[]
    assert not (c.OUT/'completed_roots/r00.json').exists()


def test_later_audit_failure_preserves_only_previously_verified_root(orchestration):
    c,events,control,sha=orchestration;control['audit_failure']='r06/audit'
    with pytest.raises(ValueError,match='full replay failed'):r.run(sha)
    assert c.read(c.OUT/'failure.json')['fully_completed_root_indices']==[0]
    assert (c.OUT/'completed_roots/r00.json').exists()
    assert not (c.OUT/'completed_roots/r06.json').exists()
    assert 'r12/plain' not in [e[1] for e in events if e[0]=='request']


@pytest.mark.parametrize('root',[6,12])
def test_each_fit_requires_its_own_frozen_reference_before_replay(orchestration,root):
    c,events,control,sha=orchestration;control['previous_failure']=f'r{root:02d}'
    with pytest.raises(ValueError,match='previous scientific mismatch'):r.run(sha)
    assert f'r{root:02d}/audit' not in [e[1] for e in events if e[0]=='request']
    assert c.read(c.OUT/'failure.json')['fully_completed_root_indices']==([0] if root==6 else [0,6])


@pytest.mark.parametrize('now,expected',[(5,10),(12,8),(19.5,.5)])
def test_request_timeout_is_clamped_without_grace(now,expected):
    assert r.clamp_timeout(10,20,clock=lambda:now)==expected


@pytest.mark.parametrize('now',[20,21])
def test_no_request_is_admitted_at_or_after_deadline(now):
    with pytest.raises(ValueError,match='deadline'):r.clamp_timeout(10,20,clock=lambda:now)


@pytest.fixture
def envelope(sandbox,monkeypatch):
    c,root,out=sandbox
    clock=SimpleNamespace(mono=100.,wall=1000.,ram=3*c.GiB,disk=6*c.GiB)
    def sleep(seconds):clock.mono+=seconds;clock.wall+=seconds
    monkeypatch.setattr(r,'time',SimpleNamespace(monotonic=lambda:clock.mono,time=lambda:clock.wall,sleep=sleep))
    monkeypatch.setattr(r,'psutil',SimpleNamespace(virtual_memory=lambda:SimpleNamespace(available=clock.ram)))
    monkeypatch.setattr(r.shutil,'disk_usage',lambda path:SimpleNamespace(free=clock.disk))
    return c,r.Envelope({'monotonic':100.,'unix_time':1000.}),clock


def test_initial_ram_stability_and_later_fast_admission_are_bounded(envelope):
    c,e,clock=envelope;e.admit(e.work_end)
    first=c.read(c.OUT/'memory_admissions/001.json')
    assert first['wait_seconds']==60 and not first['fast_path']
    e.admit(e.work_end)
    second=c.read(c.OUT/'memory_admissions/002.json')
    assert second['fast_path'] and second['cumulative_wait_seconds']==60


def test_memory_wait_cannot_spend_final_work_reserve(envelope):
    c,e,clock=envelope
    with pytest.raises(ValueError,match='deadline'):e.admit(clock.mono+59)
    assert not (c.OUT/'memory_admissions/001.json').exists()


def test_low_ram_wait_has_finite_cumulative_limit(envelope):
    c,e,clock=envelope;clock.ram=0
    with pytest.raises(ValueError,match='Cumulative memory'):e.admit(e.work_end)
    assert clock.mono==400 and not (c.OUT/'memory_admissions/001.json').exists()


@pytest.mark.parametrize('damage',['monotonic','wall','disk','memory'])
def test_resource_guards_fail_closed(envelope,damage):
    c,e,clock=envelope
    if damage=='monotonic':clock.mono=e.work_end
    elif damage=='wall':clock.wall=1000+3600
    elif damage=='disk':clock.disk=5*c.GiB-1
    else:clock.ram=2*c.GiB-1
    with pytest.raises(ValueError):e.guard(e.work_end)


@pytest.mark.parametrize('deadline_offset',[3300,2640])
def test_forward_wall_jump_cannot_spend_protected_stage_time(envelope,deadline_offset):
    c,e,clock=envelope
    clock.wall=1000+deadline_offset
    assert clock.mono==100 and clock.wall<1000+c.JOB_SECONDS
    with pytest.raises(ValueError,match='wall stage'):e.guard(e.start+deadline_offset)


@pytest.mark.parametrize('damage',[None,'partial_changed','admin_changed','report_changed','collision','missing_run','extra_run','duplicate'])
def test_closed_timing_run_and_report_are_inherited_except_live_handoffs(sandbox,damage):
    c,root,out=sandbox
    partial=c.PREVIOUS/'workers/r06/plain/continuations.jsonl.gz'
    partial.parent.mkdir(parents=True);partial.write_bytes(b'preserved partial')
    admin=root/'outputs/move_evaluation_timing_20261023_status_01/verify_closeout.py'
    put(admin,{'inspection':True})
    report=root/'docs/engineering/timing_result.md'
    put(report,{'reported':True})
    handoff=root/'AGENT_CURRENT.md';handoff.write_text('old administrative status')
    run=[c.artifact(partial)];administrative=[c.artifact(admin),c.artifact(report)]
    value={'run_inventory':run,'run_files':1,'run_bytes':run[0]['bytes'],
           'inspection_source':administrative[0],'report':administrative[1],'continuity':[c.artifact(handoff)]}
    handoff.write_text('legitimate updated continuity')
    inputs={}
    if damage=='partial_changed':partial.write_bytes(b'changed partial')
    elif damage=='admin_changed':put(admin,{'inspection':False})
    elif damage=='report_changed':put(report,{'reported':False})
    elif damage=='collision':inputs[run[0]['path']]={**run[0],'sha256':'0'*64}
    elif damage=='missing_run':value.update(run_inventory=[],run_files=0,run_bytes=0)
    elif damage=='extra_run':(c.PREVIOUS/'unexpected.txt').write_text('added')
    elif damage=='duplicate':value.update(run_inventory=run*2,run_files=2,run_bytes=run[0]['bytes']*2)
    if damage is None:
        r.inherit_closed_evidence(value,inputs)
        assert inputs=={item['path']:item for item in run+administrative}
        assert 'AGENT_CURRENT.md' not in inputs
    else:
        with pytest.raises(ValueError):r.inherit_closed_evidence(value,inputs)


@pytest.mark.parametrize('finish,cleanup_failure',[(True,False),(False,False),(False,True)])
def test_watchdog_closes_owned_job_and_preserves_execution(sandbox,monkeypatch,finish,cleanup_failure):
    c,root,out=sandbox;put(out/'freeze.json',{'synthetic':True});sha=c.file_hash(out/'freeze.json')
    clock=SimpleNamespace(mono=0.,wall=1000.);events=[]
    def sleep(seconds):clock.mono+=3601;clock.wall+=3601
    monkeypatch.setattr(l,'time',SimpleNamespace(monotonic=lambda:clock.mono,time=lambda:clock.wall,sleep=sleep))
    monkeypatch.setattr(l.sys,'argv',['launch','--freeze-sha256',sha])
    monkeypatch.setattr(l.psutil,'Process',lambda *args:SimpleNamespace(create_time=lambda:99.))
    class Child:
        pid=123
        returncode=0 if finish else None
        def poll(self):return self.returncode
        def kill(self):events.append('kill');self.returncode=-1
        def wait(self,timeout):events.append('wait');return self.returncode
    child=Child()
    def popen(command,**kwargs):
        assert command[4]=='tools.move_evaluation_reuse_20261023.runner'
        assert kwargs['env']['DIAGNOSTIC_START_MONOTONIC']=='0.0'
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
    assert bool(value['cleanup_errors'])==cleanup_failure
    assert ('kill' in events)==(not finish)


@pytest.mark.parametrize('reference,current,ratio,within',[(200.,100.,.5,True),(200.,120.,.6,True),
    (200.,120.001,.600005,False),(100.,150.,1.5,False)])
def test_latency_ratios_are_observational_and_not_a_full_qualification(reference,current,ratio,within):
    value=r.latency_observation(reference,current)
    assert value['current_over_reference_ratio']==pytest.approx(ratio)
    assert value['observed_reduction_fraction']==pytest.approx(1-ratio)
    assert value['ordinary_within_120_seconds'] is within
    assert not value['causal_speedup_claim'] and not value['full_18_position_qualification']


@pytest.mark.parametrize('bad',[0,-1,float('nan'),float('inf'),True])
def test_latency_requires_actual_finite_positive_completed_times(bad):
    with pytest.raises(ValueError):r.latency_observation(200,bad)
    with pytest.raises(ValueError):r.latency_observation(bad,100)


@pytest.mark.parametrize('mode',['qualify_plain','qualify_timed','plain','timed','audit'])
def test_implementation_distinguishes_optimized_requests_and_original_audit(mode):
    value=r.implementation_metadata(mode)
    assert value['active_opponent']==('original_reference' if mode=='audit' else 'per_decision_reuse')
    assert value['cross_decision_cache'] is False
    r.check_implementation(value,mode)


@pytest.mark.parametrize('damage',['opponent','policy','optimization','cache','bool','extra'])
def test_implementation_policy_or_cache_change_is_rejected(damage):
    value=r.implementation_metadata('audit')
    if damage=='opponent':value['active_opponent']='per_decision_reuse'
    elif damage=='policy':value['scientific_policy_id']='changed'
    elif damage=='optimization':value['optimization_id']='other'
    elif damage=='cache':value['cross_decision_cache']=True
    elif damage=='bool':value['scientific_policy_unchanged']=1
    else:value['unexpected']=True
    with pytest.raises(ValueError,match='policy mismatch'):r.check_implementation(value,'audit')


@pytest.mark.parametrize('mode',['audit','timed'])
def test_actual_worker_initializer_metadata_matches_runner_with_fake_models(sandbox,monkeypatch,mode):
    c,root,out=sandbox
    from tools.move_evaluation_reuse_20261023 import worker,greedy_adapter
    from tools.move_evaluation_20261023 import greedy_adapter as original
    import tools.corrected_planner_qualification_20261018 as package
    import tools.future_training_rules_20261021 as runtime
    calls=[]
    monkeypatch.setattr(runtime,'verify_runtime_sources',lambda:calls.append('rules') or {'synthetic':'a'*64})
    fakeq=SimpleNamespace(initialize=lambda:{'models':'synthetic'})
    monkeypatch.setitem(sys.modules,'tools.corrected_planner_qualification_20261018.local',fakeq)
    monkeypatch.setattr(package,'local',fakeq,raising=False)
    monkeypatch.setattr(original,'GreedyOpponent',lambda folder:SimpleNamespace(kind='original'))
    monkeypatch.setattr(greedy_adapter,'GreedyOpponent',lambda folder:SimpleNamespace(kind='optimized'))
    folder=out/'workers'/mode
    monkeypatch.setenv('STUDY_WORKER_DIR',str(folder));monkeypatch.setenv('STUDY_MODE',mode)
    ctx=worker.initialize()
    assert ctx['implementation']==r.implementation_metadata(mode)==c.read(folder/'implementation.json')
    assert ctx['implementation']['scientific_policy_id']==original.PROFILE_ID
    assert ctx['opponent'].kind==('original' if mode=='audit' else 'optimized')
    assert calls==['rules']


@pytest.mark.parametrize('damage',[None,'initialization','response'])
def test_perform_requires_reference_implementation_before_audit_and_at_receipt(sandbox,monkeypatch,damage):
    c,root,out=sandbox;events=[]
    metadata=r.implementation_metadata('audit')
    class FakeWorker:
        def __init__(self,module,folder,**kwargs):
            assert module=='tools.move_evaluation_reuse_20261023.worker'
            assert kwargs['affinity']==[0] and kwargs['max_rss_bytes']==2*c.GiB
            self.folder=folder
            initial=deepcopy(metadata)
            if damage=='initialization':initial['active_opponent']='per_decision_reuse'
            c.save(folder/'implementation.json',initial)
            self.initialization={'elapsed_seconds':1.,'telemetry':{'process_tree':[
                {'pid':1,'created':1.,'cpu_seconds':1.,'role':'python'}]}}
        def __enter__(self):return self
        def __exit__(self,*args):events.append('closed')
        def request(self,payload,timeout):
            events.append('requested')
            impl=deepcopy(metadata)
            if damage=='response':impl['active_opponent']='per_decision_reuse'
            receipt={'status':'ok','accepted':True,'detail':None,'request_id':'request-00000001',
                'receipt_elapsed_seconds':2.,'result':{'scientific':{'status':'passed'},'implementation':impl},
                'telemetry':{'peak_sampled_rss_bytes':100,'process_tree':[
                    {'pid':1,'created':1.,'cpu_seconds':2.,'role':'python'}]}}
            c.save(self.folder/'request-00000001.json',receipt)
            return receipt
    monkeypatch.setattr(c,'StudyWorker',FakeWorker)
    envelope=SimpleNamespace(admit=lambda deadline:None,guard=lambda deadline:None)
    if damage is None:
        value=r.perform('r00/audit','audit',{},600,1e12,envelope,{'native_binary':{}})
        assert value['result']['implementation']==metadata
        assert value['implementation']==c.artifact(out/'workers/r00/audit/implementation.json')
    else:
        with pytest.raises(ValueError,match='policy mismatch'):
            r.perform('r00/audit','audit',{},600,1e12,envelope,{'native_binary':{}})
        assert not (out/'measurements/r00_audit.json').exists()
    assert events[-1]=='closed'
    assert ('requested' in events)==(damage!='initialization')


