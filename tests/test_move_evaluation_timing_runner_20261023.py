"""Model-free orchestration, admission and owned-watchdog regression tests."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.move_evaluation_timing_20261023 import runner as r, launch as l


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
    from tools.move_evaluation_timing_20261023 import checks
    events=[];control={}
    monkeypatch.setattr(runtime,'verify_runtime_sources',lambda:{'synthetic':'a'*64})
    monkeypatch.setattr(r,'verify_runtime',lambda freeze:events.append(('runtime',)))
    monkeypatch.setattr(r,'processor_topology',lambda:{'synthetic':True})
    monkeypatch.setattr(c,'selector_payload',lambda row:{'id':row['id']})
    monkeypatch.setenv('DIAGNOSTIC_START_MONOTONIC','0')
    monkeypatch.setenv('DIAGNOSTIC_START_UNIX','0')
    put(out/'freeze.json',{'topology':{'synthetic':True}})
    put(c.OLD/'entrypoint_qualification.json',{'status':'entrypoint_counterexamples_passed','fixture':'fixed'})
    put(c.PREVIOUS/'measurements/r00_plain.json',{'result':{'scientific':{'previous':True}}})
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
        if first.get('previous') and control.get('previous_failure'):raise ValueError('previous scientific mismatch')
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
    assert events.index(('compare_previous','r00'))<events.index(requests[4])
    completed=c.read(c.OUT/'completion.json')
    assert completed['status']=='completed_timing_diagnosis_only'
    assert [x['index'] for x in completed['roots']]==[0,6,12]
    assert completed['original_120_second_qualification']=='failed_unchanged'


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


@pytest.mark.parametrize('damage',[None,'partial_changed','admin_changed','collision','missing_run','extra_run','duplicate'])
def test_closed_profile_run_and_admin_evidence_are_inherited_except_live_handoffs(sandbox,damage):
    c,root,out=sandbox
    partial=c.PREVIOUS/'workers/r06/plain/continuations.jsonl.gz'
    partial.parent.mkdir(parents=True);partial.write_bytes(b'preserved partial')
    admin=root/'outputs/move_evaluation_profile_20261023_status_01/operator_stop.json'
    put(admin,{'stopped':True})
    handoff=root/'AGENT_CURRENT.md';handoff.write_text('old administrative status')
    run=[c.artifact(partial)];administrative=[c.artifact(admin)]
    value={'run_inventory':run,'run_files':1,'run_bytes':run[0]['bytes'],
           'administrative_evidence':administrative,'continuity':[c.artifact(handoff)]}
    handoff.write_text('legitimate updated continuity')
    inputs={}
    if damage=='partial_changed':partial.write_bytes(b'changed partial')
    elif damage=='admin_changed':put(admin,{'stopped':False})
    elif damage=='collision':inputs[run[0]['path']]={**run[0],'sha256':'0'*64}
    elif damage=='missing_run':value.update(run_inventory=[],run_files=0,run_bytes=0)
    elif damage=='extra_run':(c.PREVIOUS/'unexpected.txt').write_text('added')
    elif damage=='duplicate':value['administrative_evidence']*=2
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
        assert command[4]=='tools.move_evaluation_timing_20261023.runner'
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
