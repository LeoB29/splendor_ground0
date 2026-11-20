"""Model-free Stage B ordering, inherited budgets, receipts and watchdog tests."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.move_evaluation_inference_opt_20261023 import runner as r,launch as l


def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value),encoding='utf-8')


@pytest.fixture
def sandbox(tmp_path,monkeypatch):
    c=r.c;root=tmp_path/'repo';out=root/'outputs/opt';out.mkdir(parents=True)
    for key,value in {'ROOT':root,'OUT':out,'SOURCE':root/'outputs/failed_a',
        'STAGE_A':root/'outputs/stage_a','STAGE_A_STATUS':root/'outputs/stage_a_status',
        'ORIGINAL':root/'outputs/original','PREVIOUS':root/'outputs/reuse','PREP':root/'prep',
        'PLAN_PATH':root/'outputs/failed_a/study_plan.json'}.items():monkeypatch.setattr(c,key,value)
    def artifact(path):
        path=Path(path).resolve();data=path.read_bytes()
        return {'path':path.relative_to(root).as_posix(),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
    monkeypatch.setattr(c,'artifact',artifact)
    monkeypatch.setattr(r.psutil,'boot_time',lambda:10.)
    plan={**r.new_study_plan(),'monotonic_origin':50.,'unix_origin':950.,'boot_time':10.}
    put(c.PLAN_PATH,plan);monkeypatch.setattr(c,'PLAN_SHA',c.file_hash(c.PLAN_PATH))
    prior={'owned_seconds':600.,'ram_wait_seconds':60.,'evidence_bytes':100,'execution':{},'closeout':{}}
    freeze={'study_plan':artifact(c.PLAN_PATH),'inputs':[artifact(c.PLAN_PATH)],'prior_stage_a':prior,
        'job_seconds':3600,'final_reserve_seconds':300,'request_seconds':600,'gate_request_seconds':120,
        'initialization_seconds':60,'audit_reserve_seconds_per_root':660,'topology':{'test':True},
        'native_binary':{},'torch_runtime':[]}
    put(out/'freeze.json',freeze);sha=c.file_hash(out/'freeze.json')
    put(out/'watchdog_claim.json',{'monotonic_origin':100.,'unix_origin':1000.,'freeze_sha256':sha})
    return c,root,plan,prior,freeze,sha


@pytest.fixture
def orchestration(sandbox,monkeypatch):
    c,root,plan,prior,freeze,sha=sandbox
    from tools.move_evaluation_inference_opt_20261023 import checks
    from tools.move_evaluation_profile_20261023 import analyze
    import tools.future_training_rules_20261021 as policy
    events=[];control={}
    monkeypatch.setattr(r,'verify_runtime',lambda freeze:None)
    monkeypatch.setattr(r,'load_prior',lambda:prior)
    monkeypatch.setattr(r,'processor_topology',lambda:{'test':True})
    monkeypatch.setattr(policy,'verify_runtime_sources',lambda:events.append(('startup',)) or {})
    monkeypatch.setenv('DIAGNOSTIC_START_MONOTONIC','100');monkeypatch.setenv('DIAGNOSTIC_START_UNIX','1000')
    put(c.ORIGINAL/'entrypoint_qualification.json',{'fixture':'exact'})
    put(c.PREP/'candidate_lock.json',{'roots':[{'id':str(i),'fit':161001+i//6} for i in range(13)]})
    monkeypatch.setattr(c,'selector_payload',lambda row:{'root':row['id']})
    for i in (0,6,12):
        put(c.PREVIOUS/'measurements'/f'r{i:02d}_plain.json',
            {'result':{'scientific':{'previous':True,'root':f'r{i:02d}'}},'complete_response_seconds':200.})
    class Env:
        def __init__(self,launch,plan,prior):self.end=3700.;self.work_end=3400.;self.waited=120.
        def guard(self,deadline,**kw):events.append(('guard',deadline))
    monkeypatch.setattr(r,'Envelope',Env)
    def perform(name,mode,payload,limit,deadline,envelope,freeze):
        events.append(('request',name,mode,limit,deadline))
        if control.get('request')==name:raise ValueError('request failure')
        if mode.startswith('qualify_'):science={'fixture':'exact'}
        elif mode.startswith('probe_'):science={'panel':'exact'}
        elif mode=='audit':science={'status':'failed' if control.get('audit')==name else 'passed'}
        else:science={'root':name.split('/')[0],'trace':{'path':f'outputs/opt/{name}/continuations.jsonl.gz','sha256':'a'*64,'bytes':1}}
        value={'result':{'scientific':science,'measurement':{'name':name}},'complete_response_seconds':100.}
        c.save(c.OUT/'measurements'/(name.replace('/','_')+'.json'),value);return value
    monkeypatch.setattr(r,'perform',perform)
    def fixture(a,b,original,m):
        events.append(('fixture',));assert a==b==original
        return {'status':'failed' if control.get('fixture') else 'passed'}
    def panel(ref,a,b,m):
        events.append(('panel',));assert ref==a==b
        return {'status':'failed' if control.get('panel') else 'passed'}
    def selection(science,measurement):
        events.append(('counts',science['root']))
        return {'status':'failed' if control.get('counts')==science['root'] else 'passed'}
    def compare(a,b):
        events.append(('previous' if a.get('previous') else 'pair',b['root']))
        if a.get('previous') and control.get('previous')==b['root']:raise ValueError('previous mismatch')
        return {'exact':True}
    monkeypatch.setattr(checks,'validate_fixture',fixture);monkeypatch.setattr(checks,'validate_panel',panel)
    monkeypatch.setattr(checks,'validate_selection',selection);monkeypatch.setattr(analyze,'compare_scientific_runs',compare)
    monkeypatch.setattr(analyze,'inspect_old_prefix',lambda *args:{'complete_records':296})
    return c,events,control,sha


@pytest.mark.parametrize('gate',['fixture','panel'])
def test_failed_gate_blocks_all_full_roots(orchestration,gate):
    c,events,control,sha=orchestration;control[gate]=True
    with pytest.raises(ValueError):r.run(sha)
    names=[e[1] for e in events if e[0]=='request']
    assert len(names)==(2 if gate=='fixture' else 5) and not any(n.startswith('r00') for n in names)
    assert c.read(c.OUT/'failure.json')['fully_completed_root_indices']==[]
    assert not (c.OUT/'completion.json').exists()


def test_two_gates_then_exact_trio_with_counts_reference_and_replay_before_next_case(orchestration):
    c,events,control,sha=orchestration;r.run(sha)
    req=[e for e in events if e[0]=='request']
    assert [x[1] for x in req]==['qualification_plain','qualification_timed','probe_reference','probe_plain','probe_timed']+[
        f'r{i:02d}/{m}' for i in (0,6,12) for m in ('plain','timed','audit')]
    assert all(x[3:]==(120,3400.) for x in req[:5])
    assert events.index(('fixture',))<events.index(req[2]) and events.index(('panel',))<events.index(req[5])
    for i in (0,6,12):
        key=f'r{i:02d}';audit=next(e for e in req if e[1]==key+'/audit')
        assert events.index(('counts',key))<events.index(audit) and events.index(('previous',key))<events.index(audit)
        assert audit[3:]==(600,3400.)
        assert all(x[3:]==(600,2740.) for x in req if x[1] in (key+'/plain',key+'/timed'))
    result=c.read(c.OUT/'completion.json')
    assert result['cumulative_ram_wait_seconds']==120 and result['stage_b_ram_wait_seconds']==60
    assert result['prior_stage_a']['owned_seconds']==600 and result['optimized_implementations_evaluated']==1
    assert result['original_120_second_qualification']=='failed_unchanged' and not result['full_18_position_qualification']
    assert result['observational_latency_only'] is True and len(result['roots'])==3


@pytest.mark.parametrize('failure',['counts','previous','audit'])
@pytest.mark.parametrize('index',[0,6,12])
def test_failed_root_stops_and_preserves_exact_last_verified_prefix(orchestration,failure,index):
    c,events,control,sha=orchestration;key=f'r{index:02d}'
    control[failure]=key+('/audit' if failure=='audit' else '')
    with pytest.raises(ValueError):r.run(sha)
    prefix=[i for i in (0,6,12) if i<index]
    assert c.read(c.OUT/'failure.json')['fully_completed_root_indices']==prefix
    assert not (c.OUT/'completion.json').exists()
    assert not any(e[0]=='request' and e[1].startswith(f'r{i:02d}') for i in (0,6,12) if i>index for e in events)


def test_direct_run_cannot_supply_an_unowned_new_clock(orchestration,monkeypatch):
    c,events,control,sha=orchestration;monkeypatch.setenv('DIAGNOSTIC_START_MONOTONIC','101')
    with pytest.raises(ValueError,match='origins'):r.run(sha)
    assert not any(e[0]=='request' for e in events)


@pytest.fixture
def envelope(sandbox,monkeypatch):
    c,root,plan,prior,freeze,sha=sandbox;clock=SimpleNamespace(mono=100.,wall=1000.,ram=3*c.GiB,disk=6*c.GiB)
    def sleep(n):clock.mono+=n;clock.wall+=n
    monkeypatch.setattr(r,'time',SimpleNamespace(monotonic=lambda:clock.mono,time=lambda:clock.wall,sleep=sleep))
    monkeypatch.setattr(r.psutil,'virtual_memory',lambda:SimpleNamespace(available=clock.ram))
    monkeypatch.setattr(r.shutil,'disk_usage',lambda path:SimpleNamespace(free=clock.disk))
    return c,r.Envelope({'monotonic':100.,'unix_time':1000.},plan,prior),clock


def test_stage_b_ram_stability_is_added_to_stage_a_wait(envelope):
    c,e,t=envelope;e.admit(e.work_end);e.admit(e.work_end)
    first=c.read(c.OUT/'memory_admissions/001.json');second=c.read(c.OUT/'memory_admissions/002.json')
    assert first['wait_seconds']==60 and first['prior_stage_a_wait_seconds']==60 and first['cumulative_wait_seconds']==120
    assert second['fast_path'] and second['cumulative_wait_seconds']==120


def test_no_new_300_second_ram_allowance(envelope):
    c,e,t=envelope;t.ram=0
    with pytest.raises(ValueError,match='Cumulative'):e.admit(e.work_end)
    assert t.mono==340 and not (c.OUT/'memory_admissions/001.json').exists()


@pytest.mark.parametrize('damage',['mono','wall','shared_wall','disk','evidence'])
def test_inherited_and_stage_limits_fail_closed(envelope,damage):
    c,e,t=envelope
    if damage=='mono':t.mono=e.work_end
    elif damage=='wall':t.wall=e.wall_start+(e.work_end-e.start)
    elif damage=='shared_wall':e.plan['unix_origin']=t.wall-10800+300
    elif damage=='disk':t.disk=5*c.GiB-1
    else:e.prior['evidence_bytes']=c.GiB
    with pytest.raises(ValueError):e.guard(e.work_end)


def test_wall_remaining_time_clamps_each_request(envelope):
    c,e,t=envelope;t.wall=e.wall_start+(e.work_end-e.start)-17
    assert e.effective_deadline(e.work_end)==117


def test_shared_study_deadline_limits_fresh_stage(envelope):
    c,e,t=envelope
    late=r.Envelope({'monotonic':10800.,'unix_time':11700.},e.plan,e.prior)
    assert late.end==10850 and late.work_end==10550


@pytest.mark.parametrize('mode',['probe_reference','audit'])
@pytest.mark.parametrize('damage',[None,'initialization','response','artifact_mutation'])
def test_eligibility_is_bound_before_request_and_after_response(sandbox,monkeypatch,mode,damage):
    c,root,plan,prior,freeze,sha=sandbox;events=[]
    metadata=r.implementation_metadata(mode)
    eligibility={'schema':'entity_prebound_reference_dispatch_v1','active_opponent':'original_reference','optimization_enabled':False}
    class Worker:
        def __init__(self,module,folder,**kwargs):
            assert module=='tools.move_evaluation_inference_opt_20261023.worker' and kwargs['affinity']==[0]
            self.folder=folder;c.save(folder/'implementation.json',metadata)
            value=deepcopy(eligibility)
            if damage=='initialization':value['optimization_enabled']=True
            self.bound=c.save(folder/'eligibility.json',value)
            self.initialization={'elapsed_seconds':1,'telemetry':{'process_tree':[]}}
        def __enter__(self):return self
        def __exit__(self,*args):events.append('closed')
        def request(self,payload,timeout):
            events.append('request');bound=deepcopy(self.bound)
            if damage=='response':bound['sha256']='0'*64
            if damage=='artifact_mutation':put(self.folder/'eligibility.json',{'changed':True})
            receipt={'status':'ok','accepted':True,'detail':None,'request_id':'request-00000001','receipt_elapsed_seconds':1,
                'result':{'implementation':metadata,'eligibility':bound},'telemetry':{'process_tree':[],'peak_sampled_rss_bytes':100}}
            c.save(self.folder/'request-00000001.json',receipt);return receipt
    monkeypatch.setattr(c,'StudyWorker',Worker)
    env=SimpleNamespace(admit=lambda d:None,effective_deadline=lambda d:d,guard=lambda d:None)
    if damage is None:
        result=r.perform(mode,mode,{},120 if mode.startswith('probe') else 600,1e12,env,freeze)
        assert result['eligibility']==result['result']['eligibility']
    else:
        with pytest.raises(ValueError):r.perform(mode,mode,{},120 if mode.startswith('probe') else 600,1e12,env,freeze)
        assert not (c.OUT/'measurements'/f'{mode}.json').exists()
    assert events[-1]=='closed' and ('request' in events)==(damage!='initialization')


def test_supervisor_hashes_native_runtime_without_importing_torch(sandbox,monkeypatch):
    c,root,plan,prior,freeze,sha=sandbox
    from tools.move_evaluation_inference_opt_20261023 import inference
    prefix=root/'python';torch=prefix/'Lib/site-packages/torch';path=torch/'lib/native.dll'
    path.parent.mkdir(parents=True);path.write_bytes(b'pinned native implementation')
    monkeypatch.setattr(r.sys,'prefix',str(prefix))
    monkeypatch.setattr(inference,'RUNTIME_HASHES',{'lib/native.dll':hashlib.sha256(path.read_bytes()).hexdigest()})
    value=r.torch_runtime();assert value[0]['absolute_path']==str(path.resolve())
    path.write_bytes(b'changed runtime')
    with pytest.raises(ValueError,match='Pinned Torch'):r.torch_runtime()


def test_optimized_eligibility_must_use_exact_frozen_runtime_paths(sandbox,monkeypatch):
    c,root,plan,prior,freeze,sha=sandbox
    from tools.move_evaluation_inference_opt_20261023 import inference
    monkeypatch.setattr(inference,'validate_eligibility_receipt',lambda value:value)
    path=c.OUT/'eligibility.json';put(path,{'runtime_inputs':[{'absolute_path':'different'}]})
    with pytest.raises(ValueError,match='exact frozen Torch'):r.check_eligibility(path,'plain',[])


@pytest.mark.parametrize('finish,cleanup_failure',[(True,False),(False,False),(False,True)])
def test_watchdog_charges_both_stages_and_closes_owned_tree(sandbox,monkeypatch,finish,cleanup_failure):
    c,root,plan,prior,freeze,sha=sandbox
    (c.OUT/'watchdog_claim.json').unlink()  # Synthetic fixture only, never production evidence.
    clock=SimpleNamespace(mono=100.,wall=1000.);events=[]
    def sleep(n):clock.mono+=3601;clock.wall+=3602
    fake=SimpleNamespace(monotonic=lambda:clock.mono,time=lambda:clock.wall,sleep=sleep)
    monkeypatch.setattr(l,'time',fake);monkeypatch.setattr(r,'time',fake)
    monkeypatch.setattr(l,'load_prior',lambda:prior)
    monkeypatch.setattr(l.sys,'argv',['launch','--freeze-sha256',sha])
    monkeypatch.setattr(l.psutil,'Process',lambda *a:SimpleNamespace(create_time=lambda:99.))
    class Child:
        pid=123;returncode=0 if finish else None
        def poll(self):return self.returncode
        def kill(self):events.append('kill');self.returncode=-1
        def wait(self,timeout):events.append('wait')
    child=Child()
    monkeypatch.setattr(l.subprocess,'Popen',lambda *a,**k:child)
    class Job:
        def __init__(self,pid):assert pid==123
        def close(self):
            events.append('job_close')
            if cleanup_failure:raise OSError('close failed')
    monkeypatch.setattr(l,'_WindowsJob',Job)
    l.main();value=c.read(c.OUT/'execution.json')
    assert events[0]=='job_close' and events[-1]=='wait'
    assert value['charged_owned_job_seconds']==(0 if finish else 3602)
    assert value['cumulative_owned_job_seconds']==(600 if finish else 4202)
    assert value['remaining_owned_job_seconds']==(3900 if finish else 298)
    assert value['status']==('cleanup_failed' if cleanup_failure else 'completed' if finish else 'watchdog_deadline')
    with pytest.raises(ValueError,match='relaunch'):l.main()


@pytest.mark.parametrize('expired',['owned_mono','owned_wall','study_wall'])
def test_slow_prior_authentication_cannot_spawn_after_work_deadline(sandbox,monkeypatch,expired):
    c,root,plan,prior,freeze,sha=sandbox;(c.OUT/'watchdog_claim.json').unlink()
    clock=SimpleNamespace(mono=100.,wall=1000.)
    fake=SimpleNamespace(monotonic=lambda:clock.mono,time=lambda:clock.wall,sleep=lambda n:None)
    monkeypatch.setattr(l,'time',fake);monkeypatch.setattr(r,'time',fake)
    def slow_prior():
        if expired=='owned_mono':clock.mono+=3300
        elif expired=='owned_wall':clock.wall+=3300
        else:clock.wall=plan['unix_origin']+10800-300
        return prior
    monkeypatch.setattr(l,'load_prior',slow_prior)
    monkeypatch.setattr(l.sys,'argv',['launch','--freeze-sha256',sha])
    monkeypatch.setattr(l.subprocess,'Popen',lambda *a,**k:pytest.fail('expired startup must not spawn'))
    with pytest.raises(ValueError):l.main()
    assert c.read(c.OUT/'execution.json')['exit_code'] is None
