"""Saved real evidence and model-free assessment integration regressions."""
import ast,copy,inspect,json,os,subprocess,sys,tempfile,threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import pytest
from tools.splendorust_selfplay_pilot_20261102.assessment import common as c,control,adapters,checkpoint,qualification,scientific,transport,worker,graph

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
    code="import sys;from tools.splendorust_selfplay_pilot_20261102.assessment import control,worker,checkpoint;assert 'torch' not in sys.modules"
    result=subprocess.run([sys.executable,'-B','-c',code],cwd=c.ROOT,env={**os.environ,'PYTHONPATH':str(c.ROOT/'src')+os.pathsep+str(c.ROOT)},capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stderr

def test_exact_parent_reference_export():
    result=checkpoint.verify_parent_export()
    assert result['bytes']==19123540 and result['tensors']==81
    assert result['model_bin']['sha256']=='57f6e227f8ac0382b7fa67dba6b58ec6663d1f635c7b9f583937867cd6692deb'

def test_actual_learned_archive_export_roundtrip_and_no_overwrite(local_out):
    descriptor=c.artifact(CANDIDATE);item=checkpoint.export_candidate(descriptor)
    proof=checkpoint.verify_candidate_export(descriptor,item)
    assert proof['tensors']==81 and proof['live_python_inference_unchanged']
    with pytest.raises(FileExistsError):checkpoint.export_candidate(descriptor)
    target=c.ROOT/item['path'];raw=bytearray(target.read_bytes());raw[-1]^=1;target.write_bytes(raw)
    with pytest.raises(ValueError):checkpoint.verify_candidate_export(descriptor,c.artifact(target))

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

def test_candidate_profiles_bind_only_new_weights_same_search(local_out):
    descriptor=c.artifact(CANDIDATE);item=checkpoint.export_candidate(descriptor)
    for quota in (16,128):
        old=adapters.profile(quota,route='control');new=adapters.candidate_profile(quota,descriptor,item)
        for key in ('iterations','world_pool','cpuct','depth','dynamic_fpu','fpu_reduction','inference_batch_size','native_binary','internal_budget_seconds'):
            assert old[key]==new[key]
        assert new['checkpoint']==c.read(CANDIDATE)['checkpoint']['path']
        assert new['profile_id']!=old['profile_id']
        assert adapters.route_from_profile_id(new['profile_id'],quota)=='candidate'
        assert adapters.route_from_profile_id(old['profile_id'],quota)=='control'
        assert new['native_reference_model']==item

@pytest.mark.parametrize('quota',[True,0,64,256,512,128.0])
def test_only_matched128_and_reference16(quota):
    with pytest.raises(ValueError):adapters.profile_id(quota,route='candidate')

def test_parent_full_actual_saved67_suite_passes_new_science():
    clock=copy.deepcopy(c.read(OLD/'entrypoint_clock.json'))
    clock['result'].update(execution_schema=adapters.EXECUTION_SCHEMA,execution_route='control')
    freeze={'profiles':{'control':{str(q):c.artifact(adapters.profile_path(q,route='control')) for q in (16,128)}},
        'checkpoint_bindings':{'control':{'native_reference':c.artifact(checkpoint.MODEL_BIN)}}}
    plan=qualification.build_plan()
    result=qualification.verify_entrypoint(OLD,clock,plan,freeze)
    assert result['complete_decisions']==67 and result['native_numeric_parity']==46
    assert qualification.compare_control_parity(clock['result'],plan)['status']=='passed'

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

@pytest.mark.parametrize('scope,workers,games', [('qualification',4,6),('pilot',804,406)])
def test_scope_lifecycles_and_namespace_in_actual_child_import(scope,workers,games):
    code="from tools.splendorust_selfplay_pilot_20261102.assessment import control,common as c;print(len(control.lifecycle_inventory()),len(control.schedule()),c.OUT.name);assert set(c.ROUTE_QUOTAS.values())=={128}"
    env={**os.environ,'SELFPLAY_JOB_SCOPE':scope,'PYTHONPATH':str(c.ROOT/'src')+os.pathsep+str(c.ROOT)}
    result=subprocess.run([sys.executable,'-B','-c',code],cwd=c.ROOT,env=env,capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stderr
    assert result.stdout.strip()==f'{workers} {games} assessment'

def test_scientific_quota_and_checkpoint_routes_remain_distinct():
    inventory=control.lifecycle_inventory()
    for name,item in inventory.items():
        assert item['quota']==128 and item['startup_requests']==2 and item['ordinary_first_request_id']=='request-00000003'
        assert item['route']==name.rsplit('_',1)[1]
    assert transport.OBSERVER_MODULE=='tools.splendorust_selfplay_pilot_20261102.assessment.worker'


def test_actual_sidecar_graph_binding_rejects_other_checkpoint():
    receipt=c.read(OLD/'request-00000001.json');parent=c.read(OLD/'parent_profiles/request-00000001.json')
    child=c.read(OLD/'request-00000001_child_profile.json');payload=c.read(OLD/'startup_inputs/request-00000001.json')
    transport.research_child_identity(child,parent,receipt,payload,128,route='control',completed=True,startup=True,
        graph_identity_sha256=child['graph_identity_sha256'])
    with pytest.raises(ValueError,match='backend identity'):
        transport.research_child_identity(child,parent,receipt,payload,128,route='candidate',completed=True,startup=True,
            graph_identity_sha256='0'*64)

def test_same_owner_six_game_then_replay_flow(local_out,monkeypatch):
    rootout=local_out;monkeypatch.setattr(c.parent,'OUT',rootout);monkeypatch.setattr(c,'OUT',rootout/'assessment')
    assert c.MAIN_GAMES==0
    events=[]
    class Owner:
        start=10.;wall_start=20.;waited=0.;resident={};worker_accounting={};evidence={'workers':{}}
        def guard(self):pass
        def begin(self,name):
            events.append(('begin',name));self.stage=name;self.deadline=2000.;self.wall_deadline=3000.
            c.parent.save(rootout/'stages'/f'{name}_start.json',{'cap_seconds':c.parent.STAGES[name]})
        def finish(self):events.append(('finish',self.stage));c.parent.save(rootout/'stages'/f'{self.stage}_end.json',{})
    owner=Owner();rootlaunch={'pid':os.getpid(),'created':control.psutil.Process().create_time()}
    c.parent.save(rootout/'launch.json',rootlaunch)
    specs=control.schedule();freeze={}
    def binding(*args):c.save(c.OUT/'freeze.json',freeze);return freeze
    monkeypatch.setattr(control,'bind_checkpoint',binding)
    monkeypatch.setattr(control,'verify_runtime',lambda *args:None)
    monkeypatch.setattr(control,'load_inputs',lambda *args:(specs,{}))
    def qualify(name,workers,*args):
        workers[name]=SimpleNamespace(_closed=False)
        c.save(c.OUT/'qualification'/f'{name}.json',{'qualified':True})
        c.save(c.OUT/'startup_costs'/f'{name}.json',{'nonwait_seconds':10.})
        events.append(('qualify',name))
    monkeypatch.setattr(control,'qualify',qualify)
    monkeypatch.setattr(control,'advance_idle_worker',lambda child,envelope:events.append(('handoff',envelope.stage)))
    monkeypatch.setattr(control,'phase_admission',lambda *args,**kwargs:events.append(('admit',args[1])))
    monkeypatch.setattr(control,'prior_pair_closures',lambda *args:[])
    def play(spec,*args):events.append(('game',spec['game_index']));return {**spec,'reason':'natural','plies':2}
    monkeypatch.setattr(control,'play_game',play)
    monkeypatch.setattr(control.analysis,'summarize_operational',lambda *args,**kwargs:{'passed':True})
    def close(workers,*args):events.append(('close',len(workers)));workers.clear()
    monkeypatch.setattr(control,'close_workers',close)
    monkeypatch.setattr(control.design,'select_replay_sample',lambda *args,**kwargs:{'decisions':[{}]*12})
    monkeypatch.setattr(control,'replay_selected',lambda *args:events.append(('replays',12)))
    result=control.run(owner,{}, {})
    assert result['games']==6 and result['replays']==12
    assert [v for k,v in events if k=='begin']==['assessment_qualification','operational','reproduction']
    assert [v for k,v in events if k=='game']==list(range(6))
    assert len([1 for k,v in events if k=='qualify'])==4
    assert not any(k=='begin' and v=='final' for k,v in events)
    assert events.index(('close',2))<events.index(('begin','reproduction'))

def test_public_saved_reader_uses_root_clock_and_forwards_publication_policy(local_out,monkeypatch):
    rootout=local_out;monkeypatch.setattr(c.parent,'OUT',rootout);monkeypatch.setattr(c,'OUT',rootout/'assessment')
    c.parent.save(rootout/'freeze.json',{'root':True});bound={'root_freeze':c.artifact(rootout/'freeze.json')}
    c.save(c.OUT/'freeze.json',bound)
    result={'exact_reproductions':12,'completed_games':6}
    c.save(c.OUT/'collection_completion.json',dict(status='assessment_collected_and_reproduced',games=6,replays=12,freeze=c.artifact(c.OUT/'freeze.json')))
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
