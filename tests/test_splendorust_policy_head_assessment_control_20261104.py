"""Saved real evidence and model-free assessment integration regressions."""
import ast,copy,inspect,json,os,subprocess,sys,tempfile,threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import pytest
from tools.splendorust_policy_head_20261104.assessment import common as c,control,adapters,checkpoint,qualification,scientific,transport,worker,graph

OLD=c.ROOT/'outputs/splendorust_exploration128_20261029_02/workers/qualification_frozen'
CANDIDATE=c.ROOT/'outputs/splendorust_selfplay_qualification_20261101_01/fit/checkpoint_manifest.json'

@pytest.fixture
def local_out(monkeypatch):
    base=c.parent.OUT/'prelaunch';base.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='assessment_test_',dir=base) as name:
        path=Path(name).resolve();assert path.is_relative_to(base.resolve())
        monkeypatch.setattr(c,'OUT',path)
        yield path

def test_no_models_on_import():
    code="import sys;from tools.splendorust_policy_head_20261104.assessment import control,worker,checkpoint;assert 'torch' not in sys.modules"
    result=subprocess.run([sys.executable,'-B','-c',code],cwd=c.ROOT,env={**os.environ,'PYTHONPATH':str(c.ROOT/'src')+os.pathsep+str(c.ROOT)},capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stderr

def test_exact_parent_reference_export():
    result=checkpoint.verify_parent_export()
    assert result['bytes']==19123540 and result['tensors']==81
    assert result['model_bin']['sha256']=='57f6e227f8ac0382b7fa67dba6b58ec6663d1f635c7b9f583937867cd6692deb'


@pytest.mark.parametrize('change',['order','shape','dtype','member','bytes','sha'])
def test_export_rejects_bad_tensor_bindings(change):
    entries=checkpoint.parent_entries()
    if change=='order':entries[0],entries[1]=entries[1],entries[0]
    elif change=='shape':entries[0]['shape']=[1]
    elif change=='dtype':entries[0]['dtype']='torch.float16'
    elif change=='member':entries[0]['zip_member']=entries[1]['zip_member']
    elif change=='bytes':entries[0]['bytes']-=4
    else:entries[0]['sha256']='0'*64
    with pytest.raises(ValueError):list(checkpoint.tensor_bytes(checkpoint.training.PARENT_CHECKPOINT,entries))


@pytest.mark.parametrize('quota',[True,0,64,256,512,128.0])
def test_only_matched128_and_reference16(quota):
    with pytest.raises(ValueError):adapters.profile_id(quota,route='candidate')


def test_parent_parity_will_not_accept_changed_science():
    suite=copy.deepcopy(c.read(OLD/'entrypoint_clock.json')['result'])
    suite.update(execution_schema=adapters.EXECUTION_SCHEMA,execution_route='control')
    suite['fixtures'][0]['results']['search']['diagnostics']['inference_calls']+=1
    with pytest.raises(ValueError):qualification.compare_control_parity(suite,qualification.build_plan())

def test_saved_native_reference_path_cannot_be_swapped(local_out):
    clock=copy.deepcopy(c.read(OLD/'entrypoint_clock.json'));clock['result'].update(execution_schema=adapters.EXECUTION_SCHEMA,execution_route='control')
    fake=local_out/'wrong.bin';fake.write_bytes(b'wrong')
    freeze={'profiles':{'control':{str(q):c.artifact(adapters.profile_path(q,route='control')) for q in (16,128)}},'checkpoint_bindings':{'control':{'native_reference':c.artifact(fake)}}}
    with pytest.raises(ValueError,match='same bound checkpoint'):qualification.verify_entrypoint(OLD,clock,qualification.build_plan(),freeze)

def test_graph_uses_exact_accepted_forward_implementation():
    from tools.splendorust_inference_opt_20261025 import graph as original
    assert graph._make_forward is original._make_forward
    a=ast.parse(inspect.getsource(graph._build_graph));b=ast.parse(inspect.getsource(original._build_graph))
    a.body[0].args.args=[x for x in a.body[0].args.args if x.arg!='checkpoint_sha256']
    class Rename(ast.NodeTransformer):
        def visit_Name(self,node):
            if node.id=='CHECKPOINT_SHA256':node.id='checkpoint_sha256'
            return node
    b=Rename().visit(b)
    assert ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False)
    assert 'checkpoint_sha256' in inspect.signature(graph.build_graph).parameters
    assert inspect.getsource(graph._weight_digest)==inspect.getsource(original._weight_digest)



def actual_child(folder):
    stored=c.read(OLD/'initialization.json');c.save(folder/'initialization.json',stored)
    extra={'identity','command','affinity','topology','thread_environment'}
    live={k:v for k,v in stored.items() if k not in extra}
    identity=stored['identity']
    return SimpleNamespace(output_dir=folder,identity=identity,initialization=live,_closed=False,_counter=2,_failure=None,_lock=threading.Lock(),
        process=SimpleNamespace(pid=identity['pid'],poll=lambda:None),_ps=SimpleNamespace(pid=identity['pid'],create_time=lambda:identity['created']),absolute_deadline=1000.0)

def test_actual_unaugmented_ready_shape_idle_handoff(local_out):
    child=actual_child(local_out/'workers/operational_g000_control')
    assert 'identity' not in child.initialization
    envelope=SimpleNamespace(start=0.,wall_start=10.,deadline=1500.)
    control.advance_idle_worker(child,envelope)
    proof=c.read(local_out/'worker_phase_transitions/operational_g000_control.json')
    assert child.absolute_deadline==1500. and proof['completed_startup_requests']==2 and proof['next_request_id']=='request-00000003'


@pytest.mark.parametrize('remaining_global,arrival_seconds,accepted',[(20.,.5,True),(3.,.5,True),(3.,3.1,False)])
def test_real_inherited_receiver_converts_duration_not_clock_origins(monkeypatch,remaining_global,arrival_seconds,accepted):
    from tools.local_player_qualification_128 import clock as inherited
    import queue
    qpc=10_000_000_000_000_000
    monotonic=1000.
    fake_time=SimpleNamespace(monotonic=lambda:monotonic,perf_counter_ns=lambda:qpc,
        get_clock_info=lambda name:SimpleNamespace(resolution=.015625,implementation='QueryPerformanceCounter()' if name=='perf_counter' else 'GetTickCount64()'))
    monkeypatch.setattr(inherited,'time',fake_time)
    child=object.__new__(transport.PlacedProfiledStudyWorker)
    child._boundary_context=None;child.absolute_deadline=monotonic+remaining_global
    child._sample=lambda:{'cpu_seconds':0.};child._check_resource_guard=lambda:None
    child._failure=child._failure_detail=None;child._queue=queue.Queue()
    child._queue.put(('line',qpc+int(arrival_seconds*1e9),b'{"type":"response","request_id":"request-00000003","result":{"fixture":true}}\n'))
    receipt=child._receive(qpc,10.,'response','request-00000003',None)
    assert receipt['accepted'] is accepted
    assert receipt['deadline_seconds']==min(10.,remaining_global-.015625)
    assert receipt['receipt_elapsed_seconds']==arrival_seconds


def test_idle_handoff_saved_reader_keeps_qpc_stage_and_wall_domains_separate(local_out,monkeypatch):
    name='operational_g000_control';child=actual_child(local_out/'workers'/name)
    mono,wall,qpc=1100.,1_800_000_000.,10_000_000_000_000_000
    monkeypatch.setattr(control,'time',SimpleNamespace(monotonic=lambda:mono,time=lambda:wall,perf_counter_ns=lambda:qpc))
    envelope=SimpleNamespace(start=0.,wall_start=wall-mono,deadline=1500.)
    control.advance_idle_worker(child,envelope)
    c.save(local_out/'stages/qualification_start.json',{'monotonic_deadline':1000.,'unix_deadline':wall-100.})
    c.save(local_out/'stages/operational_start.json',{'monotonic_deadline':1500.})
    c.save(local_out/'operational_admission.json',{'monotonic':1001.})
    c.save(child.output_dir/'initialization_deadlines.json',dict(worker_identity=child.identity,
        stage_monotonic_deadline=1000.,stage_unix_deadline=wall-100.,absolute_deadline=1000.))
    c.save(child.output_dir/'parent_profiles/request-00000002.json',{'capture_snapshot_ns':qpc-1000})
    c.save(child.output_dir/'parent_profiles/request-00000003.json',{'request_start_ns':qpc+1000})
    value=control.verify_phase_transition(name,envelope)
    assert value['monotonic']==mono and value['perf_counter_ns']==qpc and value['unix_time']==wall
    path=local_out/'worker_phase_transitions'/f'{name}.json'
    value['perf_counter_ns']=int(mono*1e9);path.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='before first ordinary clock'):control.verify_phase_transition(name,envelope)


def test_actual_cross_process_clock_samples_compare_only_same_provider():
    import time
    before={'qpc':time.perf_counter_ns(),'monotonic':time.monotonic_ns(),'wall':time.time_ns()}
    code="import time,json;print(json.dumps({'qpc':time.perf_counter_ns(),'monotonic':time.monotonic_ns(),'wall':time.time_ns()}))"
    result=subprocess.run([sys.executable,'-B','-c',code],capture_output=True,text=True,timeout=20)
    after={'qpc':time.perf_counter_ns(),'monotonic':time.monotonic_ns(),'wall':time.time_ns()}
    assert result.returncode==0,result.stderr
    observed=json.loads(result.stdout)
    for name in before:assert before[name]<=observed[name]<=after[name]



@pytest.mark.parametrize('change',['missing','wrongpid','wrongcreate','active','failed','counter','closed'])
def test_bad_handoff_fails_before_mutating_deadline(local_out,change):
    child=actual_child(local_out/'workers/operational_g000_control');before=child.absolute_deadline
    if change=='missing':child.identity={}
    elif change=='wrongpid':child.identity={**child.identity,'pid':1}
    elif change=='wrongcreate':child.identity={**child.identity,'created':1.}
    elif change=='active':child._lock.acquire()
    elif change=='failed':child._failure='timeout'
    elif change=='counter':child._counter=3
    else:child._closed=True
    with pytest.raises((ValueError,KeyError)):control.advance_idle_worker(child,SimpleNamespace(start=0.,wall_start=0.,deadline=1500.))
    assert child.absolute_deadline==before

def test_existing_proof_is_recomputed_not_rewritten(local_out):
    path=local_out/'proof.json';control.save_or_verify(path,{'passed':True},True);before=path.stat().st_mtime_ns
    control.save_or_verify(path,{'passed':True},True);control.save_or_verify(path,{'passed':True},False)
    assert path.stat().st_mtime_ns==before
    with pytest.raises(ValueError):control.save_or_verify(path,{'passed':False},True)
    with pytest.raises(FileNotFoundError):control.save_or_verify(local_out/'missing.json',{},False)

def test_shared_envelope_has_no_new_origin_or_resource_pool(local_out,monkeypatch):
    rootout=local_out;monkeypatch.setattr(c.parent,'OUT',rootout);monkeypatch.setattr(c,'OUT',rootout/'assessment')
    class Owner:
        start=17.;wall_start=31.;waited=9.;resident={};worker_accounting={}
        evidence={'workers':{rootout/'workers/source':1,c.OUT/'workers/operational_g000_control':2}}
        def begin(self,name):
            self.stage=name;self.deadline=100.;self.wall_deadline=114.
            c.parent.save(rootout/'stages'/f'{name}_start.json',{'cap_seconds':c.parent.STAGES[name],'monotonic':20.})
        def finish(self):c.parent.save(rootout/'stages'/f'{self.stage}_end.json',{'monotonic':22.})
    owner=Owner();view=control.SharedEnvelope(owner,{})
    view.begin('qualification');view.finish()
    assert owner.stage=='assessment_qualification' and view.start==17. and view.waited==9.
    assert len(view.evidence['workers'])==1 and len(owner.evidence['workers'])==2
    with pytest.raises(ValueError):view.begin('final')


def test_scientific_quota_and_checkpoint_routes_remain_distinct():
    inventory=control.lifecycle_inventory()
    for name,item in inventory.items():
        assert item['quota']==128 and item['startup_requests']==2 and item['ordinary_first_request_id']=='request-00000003'
        assert item['route']==name.rsplit('_',1)[1]
    assert transport.OBSERVER_MODULE=='tools.splendorust_policy_head_20261104.assessment.worker'


def test_actual_sidecar_graph_binding_rejects_other_checkpoint():
    receipt=c.read(OLD/'request-00000001.json');parent=c.read(OLD/'parent_profiles/request-00000001.json')
    child=c.read(OLD/'request-00000001_child_profile.json');payload=c.read(OLD/'startup_inputs/request-00000001.json')
    transport.research_child_identity(child,parent,receipt,payload,128,route='control',completed=True,startup=True,
        graph_identity_sha256=child['graph_identity_sha256'])
    with pytest.raises(ValueError,match='backend identity'):
        transport.research_child_identity(child,parent,receipt,payload,128,route='candidate',completed=True,startup=True,
            graph_identity_sha256='0'*64)


def test_public_saved_reader_uses_root_clock_and_forwards_publication_policy(local_out,monkeypatch):
    rootout=local_out;monkeypatch.setattr(c.parent,'OUT',rootout);monkeypatch.setattr(c,'OUT',rootout/'assessment')
    c.parent.save(rootout/'freeze.json',{'root':True});bound={'root_freeze':c.artifact(rootout/'freeze.json')}
    c.save(c.OUT/'freeze.json',bound)
    result={'exact_reproductions':96,'completed_games':406}
    c.save(c.OUT/'collection_completion.json',dict(status='assessment_collected_and_reproduced',games=406,replays=96,freeze=c.artifact(c.OUT/'freeze.json')))
    owner=SimpleNamespace(start=123.,wall_start=456.);seen=[]
    def saved(view,freeze,*,writing):
        assert view.start==123. and view.root is owner and freeze==bound
        seen.append(writing);return result
    monkeypatch.setattr(control,'_verify_saved',saved)
    assert control.verify_saved(owner,{},writing=True)==result
    assert control.verify_saved(owner,{},writing=False)==result
    assert seen==[True,False]


def forecast_fixture(monkeypatch):
    monkeypatch.setattr(c,'MAIN_GAMES',400);monkeypatch.setattr(c,'RAM_WAIT_SECONDS',10800.)
    monkeypatch.setattr(c,'STAGES',{**c.STAGES,'main':16*3600.,'replay':4*3600.,'final':12*3600.})
    monkeypatch.setattr(c,'PREFINAL_EVIDENCE_BYTES',40*c.GiB);monkeypatch.setattr(c,'MAX_EVIDENCE_BYTES',48*c.GiB)
    monkeypatch.setattr(c,'FINAL_RESERVE_BYTES',8*c.GiB);monkeypatch.setattr(c,'MAX_REPLAY_DECISIONS',96)
    costs=[dict(seconds=31.,nonwait_seconds=30.,ram_wait_seconds=1.,accounted_growth_bytes=c.MiB,
        completed_decisions=50,plies=50,raw_stream_growth={r:{'stdout.log':c.MiB,'stderr.log':0,'events.jsonl':4096} for r in c.VARIANTS}) for _ in range(6)]
    startups={r:dict(nonwait_seconds=10.,accounted_growth_bytes=c.MiB,raw_stream_bytes={'stdout.log':2*c.MiB,'stderr.log':0,'events.jsonl':8192}) for r in c.VARIANTS}
    closures={r:dict(seconds=1.) for r in c.VARIANTS}
    snapshot=dict(charged_ram_wait_seconds=0.,main_remaining_seconds=16*3600.,model_remaining_seconds=30*3600.,
        global_remaining_seconds=42*3600.,accounted_bytes=c.GiB,free_disk_bytes=45*c.GiB)
    return costs,startups,closures,snapshot

def test_forecast_carries_shared_wait_and_complete_final_reserve(monkeypatch):
    args=forecast_fixture(monkeypatch);value=control.forecast_projection(*args)
    assert value['passed'] and value['saved_final_seconds']==12*3600.
    assert value['remaining_ram_reserve_seconds']==10800.
    assert value['normal_main_worker_pairs']==40 and value['maximum_reproductions']==96
    assert value['main_seconds']==1.5*(400*30+40*22)+10800

@pytest.mark.parametrize('failure',['clock','bytes','main_raw','replay_raw'])
def test_forecast_rejects_real_scope_resource_overflow(monkeypatch,failure):
    costs,startups,closures,snapshot=forecast_fixture(monkeypatch)
    if failure=='clock':snapshot['model_remaining_seconds']=100.
    elif failure=='bytes':snapshot['accounted_bytes']=c.PREFINAL_EVIDENCE_BYTES-1
    elif failure=='main_raw':costs[0]['raw_stream_growth']['candidate']['stdout.log']=c.MAX_LOG_BYTES
    else:
        costs[0]['raw_stream_growth']['candidate']['stdout.log']=c.MAX_LOG_BYTES//100
        costs[0]['completed_decisions']=costs[0]['plies']=1
    assert not control.forecast_projection(costs,startups,closures,snapshot)['passed']

def test_new_pair_only_before_unplayed_game_after_guardloss(monkeypatch):
    specs=[]
    for i in range(12):specs.append(dict(stage='main',game_index=i,reason='forfeit' if i==4 else 'natural'))
    monkeypatch.setattr(control.design,'validate_spec',lambda x:None)
    plan=control.expected_worker_plan(specs)
    assert plan['main',4]['candidate']=='main_g000_candidate'
    assert plan['main',5]['candidate']=='main_g005_candidate'
    assert plan['main',9]['candidate']=='main_g005_candidate'
    assert plan['main',10]['candidate']=='main_g010_candidate'

