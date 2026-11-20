"""Model-free tests; no learned models, games or performance benchmarks."""
from copy import deepcopy
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import ast
import struct
import threading
import time
import pytest
from . import queueing, workflow, identity, collector, preparation, data, common as c


def specs():
    return [dict(game_id=f'g{i:04d}',game_index=i,seed_decimal=str(10000+i),start_player=i%2,
        purpose='excluded_parallel_throughput_engineering_only') for i in range(16)]


@pytest.mark.parametrize('clients',[2,8])
def test_actual_cpu_service_never_aggregates_control_or_candidate(clients):
    from tools.splendorust_gpu_collector_20261105.backend import InferenceService, WIDTH
    batches=[];receipts=[];gate=threading.Barrier(clients);lock=threading.Lock()
    def forward(tokens,batch):
        with lock:batches.append((batch,len(tokens)))
        gate.wait(timeout=3)
        return [[-0.0]+[tokens[0]]*82]
    service=InferenceService(forward,'cpu',evidence=receipts.append,scheduler='original8')
    for index in range(clients):service.register(f'client{index}',index)
    def run(index):return service.infer(f'client{index}',[float(index)]*WIDTH,1,deadline=time.perf_counter()+60.)
    with ThreadPoolExecutor(max_workers=clients) as pool:results=list(pool.map(run,range(clients)))
    assert batches==[(1,WIDTH)]*clients
    assert all(item['physical_batch']==1 and len(item['requests'])==1 for item in receipts)
    for index,(rows,receipt) in enumerate(results):
        assert struct.pack('<83f',*rows[0])==struct.pack('<83f',-0.0,*([float(index)]*82))
        assert receipt['native_batch']==receipt['physical_batch']==1
    for index in range(clients):service.unregister(f'client{index}')
    service.close()


@pytest.mark.parametrize('processes,clients',[(1,8),(4,2)])
def test_same_queue_assigns_all_sixteen_once_across_independent_clients(tmp_path,processes,clients):
    queue=tmp_path/'queue.sqlite';stop=queueing.SharedStop(tmp_path/'stop.json')
    queueing.create_queue(queue,specs())
    def process(p):
        def take(i):return queueing.take_next(queue,stop,process_index=p,client_index=i,clients=clients)
        return queueing.run_clients(clients,take,lambda i,s,a:dict(status='completed',game_id=s['game_id']),stop)
    with ThreadPoolExecutor(max_workers=processes) as pool:results=list(pool.map(process,range(processes)))
    rows=queueing.assignments(queue)
    assert [x['queue_index'] for x in rows]==list(range(16))
    assert len({x['game_id'] for x in rows})==16
    assert all(x['slot'] in range(clients) and x['process_index'] in range(processes) for x in rows)
    assert sum(len(x['outcomes']) for x in results)==16
    assert all(x['status']=='passed' and x['drained'] for x in results)


def test_failure_stops_assignment_and_drains_without_retry(tmp_path):
    queue=tmp_path/'queue.sqlite';stop=queueing.SharedStop(tmp_path/'stop.json');queueing.create_queue(queue,specs())
    active=threading.Barrier(2);calls=[]
    def play(index,spec,assignment):
        calls.append(spec['game_id']);active.wait(timeout=3)
        if index==0:
            stop.set();return dict(status='deadline_failed')
        assert stop.path.parent==tmp_path
        return dict(status='completed')
    result=queueing.run_clients(2,lambda i:queueing.take_next(queue,stop,process_index=0,client_index=i,clients=2),play,stop)
    assert result['status']=='failed' and result['drained'] and len(calls)==2
    assert len(queueing.assignments(queue))==2
    assert queueing.take_next(queue,stop,process_index=0,client_index=0,clients=2) is None
    assert result['retries']==result['replacements']==0


def test_exception_stops_all_clients_and_preserves_error(tmp_path):
    stop=queueing.SharedStop(tmp_path/'stop.json')
    def take(i):raise ValueError('saved fixture failure')
    result=queueing.run_clients(2,take,lambda *a:None,stop)
    assert result['status']=='failed' and result['drained'] and stop.is_set()
    assert any('saved fixture failure' in x['error'] for x in result['errors'])


def test_existing_queue_or_wrong_size_never_replaced(tmp_path):
    path=tmp_path/'q.sqlite'
    with pytest.raises(ValueError):queueing.create_queue(path,specs()[:-1])
    queueing.create_queue(path,specs())
    with pytest.raises(ValueError):queueing.create_queue(path,specs())


@pytest.mark.parametrize('clients,index',[(4,0),(2,4),(8,1)])
def test_unplanned_configuration_rejected(tmp_path,clients,index):
    with pytest.raises(ValueError):queueing.take_next(tmp_path/'x',queueing.SharedStop(tmp_path/'stop'),process_index=index,client_index=0,clients=clients)


def test_overlap_is_real_and_order_fixed():
    events=[];y_started=threading.Event();x_checked=threading.Event();cells={'X':{'cell_id':'X'},'Y':{'cell_id':'Y'}}
    def play(arm,spec):
        key=(arm,spec['cell_id']);events.append(('play',*key))
        if key==('candidate','Y'):
            y_started.set();assert x_checked.wait(3)
        return {'status':'completed'}
    def check(arm,spec):
        key=(arm,spec['cell_id'])
        if key==('candidate','X'):
            assert y_started.wait(3);x_checked.set()
        events.append(('check',*key));return {'status':'passed'}
    result=workflow.run_comparison(cells,play,check,lambda *a:pytest.fail('Unexpected stop'),lambda *a:pytest.fail('Unexpected quarantine'))
    assert result['status']=='passed'
    assert [x for x in events if x[0]=='play']==[('play','control','X'),('play','candidate','X'),('play','candidate','Y'),('play','control','Y')]
    assert events.index(('check','candidate','X'))<events.index(('check','candidate','Y'))


@pytest.mark.parametrize('failure',['audit','identity','play'])
def test_any_overlap_failure_quarantines_y_and_drains(failure):
    y_started=threading.Event();stop=threading.Event();drained=threading.Event();quarantine=[];played=[]
    cells={'X':{'cell_id':'X'},'Y':{'cell_id':'Y'}}
    def play(arm,spec):
        played.append((arm,spec['cell_id']))
        if (arm,spec['cell_id'])==('candidate','Y'):
            y_started.set()
            if failure=='play':raise ValueError('Y failure')
            assert stop.wait(3);drained.set()
        return {'status':'completed'}
    def check(arm,spec):
        if (arm,spec['cell_id'])==('candidate','X'):
            assert y_started.wait(3)
            if failure in ('audit','identity'):raise ValueError(failure+' mismatch')
        return {'status':'passed'}
    with pytest.raises(ValueError):
        workflow.run_comparison(cells,play,check,lambda *a:stop.set(),lambda *a:quarantine.append(a))
    assert quarantine and ('control','Y') not in played
    assert stop.is_set() and (drained.is_set() or failure=='play')


def test_exact_identity_rejects_signed_zero_and_small_values():
    with pytest.raises(ValueError):identity.exact({'root_value':0.},{'root_value':-0.})
    with pytest.raises(ValueError):identity.exact([.5],[.50000001])
    with pytest.raises(ValueError):identity.exact({'visits':[1,2]},{'visits':[2,1]})
    identity.exact({'q':[.5,None],'visits':[1,2]},{'q':[.5,None],'visits':[1,2]})


def test_target_source_digest_authenticated_before_removing_only_provenance():
    row=dict(ply=0,actor=1,game_id='g0',clock={'elapsed':.1});other=deepcopy(row);other['clock']['elapsed']=.2
    example=dict(ply=0,actor=1,game_id='g0',row_sha256=c.digest(row),policy=[.5,.5],value=[1.,-1.])
    target=dict(examples=[example],teacher_simulations=512)
    peer=deepcopy(target);peer['examples'][0]['row_sha256']=c.digest(other)
    identity.exact(identity.target_projection(target,[row]),identity.target_projection(peer,[other]))
    peer['examples'][0]['row_sha256']='f'*64
    with pytest.raises(ValueError):identity.target_projection(peer,[other])
    peer=deepcopy(target);peer['examples'][0]['policy']=[.6,.4]
    with pytest.raises(ValueError):identity.exact(identity.target_projection(target,[row]),identity.target_projection(peer,[row]))


def test_only_orchestration_fields_excluded_from_move_identity():
    row={key:None for key in ('game_id','spec','route','ply','actor','before','observer','result','action','trace','after','training_capture')}
    row.update(clock={'elapsed':1},client_id='worker_0',slot=0)
    peer=deepcopy(row);peer.update(clock={'elapsed':2},client_id='worker_1',slot=1)
    identity.exact(identity.scientific_move(row),identity.scientific_move(peer))
    peer['training_capture']={'visits67':[1]*67}
    with pytest.raises(ValueError):identity.exact(identity.scientific_move(row),identity.scientific_move(peer))


def test_plan_is_model_free_exact_cpu_route_and_no_corpus_use():
    import sys
    from .. import schedule
    plan=preparation.build_plan(schedule.collection_specs(),schedule.qualification_specs())
    assert 'torch' not in sys.modules
    assert plan['packing']['physical_batch']==plan['packing']['native_batch']==1
    assert plan['control_clients']==8 and plan['candidate_clients']==2 and plan['candidate_processes']==4
    assert plan['internal_seconds']==60 and plan['outer_seconds']==75
    assert plan['teacher']==dict(simulations=512,worlds=3,cpuct=.4,depth=64,precision='float32')
    assert len(plan['engineering_games'])==2 and plan['qualification']['decisions_per_route']==116
    preparation.validate_plan(plan)


def test_loader_manifest_refuses_production_labels():
    rows=specs();good=data.build_manifest([{}]*16,rows,route='cpu',cell_id='X')
    assert good['purpose']=='excluded_engineering_qualification'
    rows[0]['purpose']='fresh_corrected_production_training_source'
    with pytest.raises(ValueError):data.build_manifest([{}]*16,rows,route='cpu',cell_id='X')


def test_private_reuse_does_not_mutate_public_accepted_modules():
    import tools.splendorust_gpu_collector_20261105.collector as old
    import tools.splendorust_gpu_collector_20261105.data as olddata
    assert collector._legacy is not old and data._legacy is not olddata
    assert collector._legacy.c is c and old.c is not c and olddata.c is not c
    assert collector._legacy.data is data


def test_worker_has_one_startup_guard_call_and_original_registry():
    from . import worker
    source=Path(worker.__file__).read_text();tree=ast.parse(source)
    calls=[node for node in ast.walk(tree) if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='verify_runtime_sources']
    assert len(calls)==1
    assert worker.active_registry_paths(Path('worker'),{'mode':'play'})==[Path('worker/runtime/active_decisions.json')]
    assert worker.active_registry_paths(Path('worker'),{'mode':'audit_cell'})==[]


def cohort_fixture():
    selected=specs();assigned=[];proofs=[]
    for i,game in enumerate(selected):
        assignment=dict(queue_index=i,game_id=game['game_id'],process_index=(i%8)//2,slot=i%2,
            assigned_perf_counter=1.+i,assigned_unix=101.+i)
        assigned.append(assignment)
        finished=dict(ended_perf_counter=1.5+i,ended_unix=101.5+i)
        proofs.append(dict(game_id=game['game_id'],outcome=dict(status='completed',game_id=game['game_id'],finished=finished),
            finished=finished,loaded=dict(plies=3,eligible_roots=2)))
    started=dict(cell_id='X',started_perf_counter=1.,started_unix=101.)
    cohort=dict(status='completed',cell_id='X',route='cpu',concurrency=8,games=16,plies=48,eligible_roots=32,
        full_cohort_completed=True,corpus_produced=False,engineering_targets_only=True,excluded_from_every_training_corpus=True,
        retries=0,replacements=0,training_updates=0,**{k:v for k,v in started.items() if k!='cell_id'},
        ended_perf_counter=25.,ended_unix=125.,wall_seconds=24.,games_per_hour=2400.,eligible_roots_per_hour=4800.,
        gameplay_end_perf_counter=16.5,gameplay_end_unix=116.5,gameplay_seconds=15.5,target_loader_seconds=8.5)
    queue=dict(assignments=assigned,outcomes={p['game_id']:p['outcome'] for p in proofs},errors=[],never_started=[])
    committed=[dict(position=i,spec=s,assignment=assigned[i]) for i,s in enumerate(selected)]
    return [dict(cell_id='X',specifications=selected),assigned,proofs,started,cohort,queue,committed,dict(unix_start=120.,unix_end=130.)]


def test_independent_cohort_reconstructs_all_clocks_and_counts():
    from .audit import check_cohort
    assert check_cohort(*cohort_fixture())==dict(plies=48,eligible_roots=32,wall_seconds=24.)


@pytest.mark.parametrize('corruption',['queued_spec','claimed_game','busy_client','outcome','plies','roots','clock','loader_time','rate','origin','assembly_exit'])
def test_independent_cohort_rejects_saved_corruption(corruption):
    from .audit import check_cohort
    args=[deepcopy(value) for value in cohort_fixture()]
    if corruption=='queued_spec':args[6][3]['spec']['seed_decimal']='999'
    elif corruption=='claimed_game':args[6][3]['assignment']=None
    elif corruption=='busy_client':args[2][0]['finished']['ended_perf_counter']=20.
    elif corruption=='outcome':args[5]['outcomes']['g0000']['status']='timeout'
    elif corruption=='plies':args[4]['plies']=47
    elif corruption=='roots':args[4]['eligible_roots']=31
    elif corruption=='clock':args[4]['gameplay_end_perf_counter']=16.4
    elif corruption=='loader_time':args[4]['target_loader_seconds']=8.4
    elif corruption=='rate':args[4]['games_per_hour']=2401.
    elif corruption=='origin':args[4]['started_unix']=102.
    else:args[7]['unix_end']=124.
    with pytest.raises(ValueError):check_cohort(*args)


def test_committed_queue_read_is_complete_and_read_only(tmp_path):
    path=tmp_path/'queue.sqlite';queueing.create_queue(path,specs());before=path.read_bytes()
    rows=queueing.inventory(path)
    assert len(rows)==16 and all(x['assignment'] is None for x in rows)
    assert path.read_bytes()==before
    with pytest.raises(Exception):queueing.inventory(tmp_path/'absent.sqlite')
    assert not (tmp_path/'absent.sqlite').exists()


def test_actual_worker_saved_role_calls_startup_guard_once(tmp_path,monkeypatch):
    from . import worker
    from .. import topology
    from tools import future_training_rules_20261021 as guard
    import psutil
    calls=[];sources={'test':'hash'}
    monkeypatch.setattr(c,'OUT',tmp_path)
    monkeypatch.setattr(guard,'verify_runtime_sources',lambda:calls.append('guard') or sources)
    monkeypatch.setattr(preparation,'validate_plan',lambda p:None)
    monkeypatch.setattr(topology,'verify_startup',lambda t,p:{'observed':True})
    class Process:
        mask=[0,2,4,6,8,12]
        def cpu_affinity(self,value=None):
            if value is not None:self.mask=value
            return self.mask
    monkeypatch.setattr(psutil,'Process',Process)
    monkeypatch.setattr(collector,'assemble_cell',lambda *a,**k:dict(status='completed',models=0))
    plan=dict(collection=dict(source_guard=sources,cells={'X':dict(cell_id='X')}),topology={},
        placement=dict(audit_cpus=[8],original_allowed=[0,2,4,6,8,12]))
    request=dict(mode='assemble',cpu=[8],cell='X',cell_folder=str(tmp_path/'cell'),started={},gameplay_end=0.)
    result=worker.run(tmp_path/'worker',plan,dict(identity={'pid':1,'created':2.},plan={},claim={}),lambda:None,request)
    assert calls==['guard'] and result==dict(status='completed',models=0)
    assert c.read(tmp_path/'worker/runtime_sources.json')['sources']==sources


@pytest.mark.parametrize('clients',[2,8])
def test_actual_session_reuses_game_path_and_closes_all_clients(tmp_path,monkeypatch,clients):
    from types import SimpleNamespace
    monkeypatch.setattr(c,'OUT',tmp_path)
    monkeypatch.setattr(collector._legacy,'q',SimpleNamespace(validate_runtime=lambda runtime:None))
    warmups=[];calls=[];closed=[]
    monkeypatch.setattr(collector,'warmup_runtime',lambda plan,runtime,folder:warmups.append(folder))
    class Adapter:
        def __init__(self,name,slot):self.client_id,self.slot=name,slot
        def close(self):closed.append(self.client_id)
    class Runtime:
        route='cpu'
        def make_adapter(self,name,slot,quota):
            assert quota==512
            return Adapter(name,slot)
        def abort(self):raise AssertionError('No abort expected')
    def original_game(plan,runtime,adapter,spec,folder,stopped,boundary):
        boundary();calls.append((adapter.client_id,adapter.slot,spec['game_id']))
        return dict(status='completed',game_id=spec['game_id'])
    monkeypatch.setattr(collector._legacy,'_play_game',original_game)
    cell=tmp_path/'cell';queueing.create_queue(cell/'queue.sqlite',specs())
    c.save(cell/'release.json',dict(services=1 if clients==8 else 4,clients=8))
    session=collector.Session({},Runtime(),tmp_path/'worker',process_index=0,clients=clients,
        cell_folder=cell,queue_path=cell/'queue.sqlite',stop_path=cell/'stop.json',boundary=lambda:None)
    result=session.run()
    assert result['status']=='passed' and len(calls)==16 and len(warmups)==1
    assert sorted(closed)==[f'worker_{i}' for i in range(clients)]
    assert c.read(tmp_path/'worker/ready.json')['clients']==clients
    assert len(list((cell/'assignments').glob('*.json')))==16
