"""Controller regressions: fixed sample, classified losses and sealed resource gates."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from datetime import datetime,timezone
import ast
import json
import subprocess
import sys
import pytest
from tools.splendorust_search512_20261101 import control as r,common as c,design,scientific


def basis():
    streams={'stdout.log':2_000_000,'stderr.log':0,'events.jsonl':100_000}
    costs=[dict(seconds=220.+i,nonwait_seconds=210.+i,ram_wait_seconds=10.,independent_seconds=1.,
        accounted_growth_bytes=12_000_000+i*100,plies=60,completed_decisions=60,
        maximum_receipt_seconds=5.,raw_stream_growth={v:dict(streams) for v in c.VARIANTS}) for i in range(6)]
    starts={v:dict(seconds=80.,nonwait_seconds=80.,accounted_growth_bytes=4_000_000,
        raw_stream_bytes=dict(streams)) for v in c.VARIANTS}
    closures={v:dict(seconds=4.) for v in c.VARIANTS}
    snap=dict(charged_ram_wait_seconds=60.,accounted_bytes=200_000_000,free_disk_bytes=42*c.GiB,
        main_remaining_seconds=c.STAGES['main']-10,model_remaining_seconds=c.COLLECTION_CUTOFF_SECONDS-1500,
        global_remaining_seconds=c.JOB_SECONDS-1500)
    return costs,starts,closures,snap


def test_forecast_separates_incremental_games_startups_ram_and_final():
    costs,starts,closed,snap=basis();value=r.forecast_projection(costs,starts,closed,snap)
    assert value['passed']
    assert value['main_nonwait_seconds']==1.5*(400*215+40*(160+8))
    assert value['remaining_ram_reserve_seconds']==c.RAM_WAIT_SECONDS-60
    assert value['main_seconds']==value['main_nonwait_seconds']+c.RAM_WAIT_SECONDS-60
    assert value['model_seconds']==value['main_seconds']+value['replay_seconds']
    assert value['total_seconds']==value['model_seconds']+value['saved_final_seconds']
    assert value['raw_stream_projections']['candidate']['stdout.log']==32_000_000
    assert value['replay_raw_stream_projections']['candidate']['stdout.log']==11_600_000


@pytest.mark.parametrize('fault,gate',[
 ('main','main_remaining'),('model','model_remaining'),('global','global_remaining'),
 ('disk','free_disk'),('evidence','prefinal_evidence'),('raw','raw_candidate_stdout.log'),
 ('replay_raw','replay_raw_candidate_stdout.log')])
def test_forecast_rejects_every_actual_envelope(fault,gate):
    costs,starts,closed,snap=basis()
    if fault in ('main','model','global'):snap[fault+'_remaining_seconds']=1.
    elif fault=='disk':snap['free_disk_bytes']=c.MINIMUM_FREE_DISK_BYTES
    elif fault=='evidence':snap['accounted_bytes']=c.PREFINAL_EVIDENCE_BYTES-1
    elif fault=='raw':costs[0]['raw_stream_growth']['candidate']['stdout.log']=30*c.MiB
    else:
        costs[0]['plies']=costs[0]['completed_decisions']=2
        costs[0]['raw_stream_growth']['candidate']['stdout.log']=3*c.MiB
    result=r.forecast_projection(costs,starts,closed,snap)
    assert not result['passed'] and not result['checks'][gate]


@pytest.mark.parametrize('field,value',[('nonwait_seconds',-1.),('accounted_growth_bytes',0),('completed_decisions',59),('ram_wait_seconds',1e8)])
def test_forecast_rejects_malformed_costs(field,value):
    args=basis();args[0][0][field]=value
    with pytest.raises(ValueError):r.forecast_projection(*args)


@pytest.mark.parametrize('forfeits',[set(),{3},{0,1,2,9,10,398,399},set(range(400))])
def test_full_fixed400_flow_never_replaces_or_retries_games(monkeypatch,forfeits):
    schedule=design.make_schedule('main');workers={};seen=[];starts=[];closes=[];active=[]
    def begin(spec,workers,*args):
        assert not workers
        index=spec['game_index'];starts.append(index);active.append(index)
        workers.update({route:f'main_g{index:03d}_{route}' for route in c.VARIANTS})
    def play(spec,workers,*args):
        assert len(workers)==2;seen.append(spec['game_index'])
        return dict(**spec,complete=True,reason='forfeit' if spec['game_index'] in forfeits else 'natural')
    def close(workers,*args):
        closes.append(tuple(workers.values()));workers.clear()
    monkeypatch.setattr(r,'begin_block',begin);monkeypatch.setattr(r,'play_game',play);monkeypatch.setattr(r,'close_workers',close)
    games=r.collect_main(schedule,workers,object(),{}, {})
    expected=sorted(set(range(0,400,10))|{i+1 for i in forfeits if i<399})
    assert seen==list(range(400)) and starts==expected and len(closes)==len(starts) and not workers
    plan=r.expected_worker_plan(games)
    for game in games:
        start=max(i for i in expected if i<=game['game_index'])
        assert plan[('main',game['game_index'])]=={v:f'main_g{start:03d}_{v}' for v in c.VARIANTS}
    assert len(starts)*2+4<=c.MAX_WORKER_LIFECYCLES


def test_infrastructure_failure_stops_before_next_game(monkeypatch):
    seen=[];workers={}
    monkeypatch.setattr(r,'begin_block',lambda spec,workers,*args:workers.update(control=1,candidate=2))
    def fail(spec,*args):seen.append(spec['game_index']);raise ValueError('malformed scientific evidence')
    monkeypatch.setattr(r,'play_game',fail)
    with pytest.raises(ValueError,match='malformed'):r.collect_main(design.make_schedule('main'),workers,None,{}, {})
    assert seen==[0]


def test_saved_rows_is_reiterable_lazy_and_checks_each_boundary(tmp_path,monkeypatch):
    monkeypatch.setattr(c,'ROOT',tmp_path);monkeypatch.setattr(c,'OUT',tmp_path/'out')
    games=[dict(stage='main',game_index=0,plies=3)]
    for i in range(3):
        path=r.game_folder(games[0])/'moves'/f'{i:06d}.json';path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps({'ply':i}))
    guarded=[];rows=r.SavedRows(games,lambda:guarded.append(1))
    assert not guarded
    iterator=iter(rows);assert next(iterator)=={'ply':0} and len(guarded)==1
    assert [x['ply'] for x in rows]==[0,1,2] and len(guarded)==4


@pytest.mark.parametrize('problem',['future','alive','errors','duplicate','modified'])
def test_previous_cleanup_must_be_exact_and_complete_before_new_start(tmp_path,monkeypatch,problem):
    monkeypatch.setattr(c,'ROOT',tmp_path);monkeypatch.setattr(c,'OUT',tmp_path/'out');monkeypatch.setattr(c,'WRITE_OBSERVER',None)
    descriptors=[]
    for route in c.VARIANTS:
        path=c.OUT/'workers'/route/'cleanup.json'
        c.save(path,dict(time=datetime.fromtimestamp(100,timezone.utc).isoformat(),alive=[],errors=[]))
        descriptors.append(c.artifact(path))
    wall=101.
    if problem=='future':wall=99.
    elif problem=='duplicate':descriptors[1]=descriptors[0]
    else:
        path=c.ROOT/descriptors[0]['path'];value=c.read(path)
        if problem=='alive':value['alive']=[123]
        elif problem=='errors':value['errors']=['unconfirmed owned exit']
        else:value['time']=datetime.fromtimestamp(99,timezone.utc).isoformat()
        path.write_text(c.canonical(value))
        if problem!='modified':descriptors[0]=c.artifact(path)
    with pytest.raises(ValueError):r.verify_closed_before(descriptors,wall)


def test_compact_closed_accounting_retains_slot_and_byte_charge(tmp_path,monkeypatch):
    monkeypatch.setattr(c,'ROOT',tmp_path);monkeypatch.setattr(c,'OUT',tmp_path/'out')
    folder=c.OUT/'workers/main_g000_control';record=c.OUT/'closed_workers/main_g000_control.json';record.parent.mkdir(parents=True);record.write_text('{}')
    env=r.Envelope(dict(monotonic_origin=0.,unix_origin=0.));env.evidence={'workers':{folder:1234}}
    env.worker_accounting[folder]=SimpleNamespace(_child=SimpleNamespace(process=SimpleNamespace(pid=123)))
    env.release_verified_worker(folder)
    assert folder in env.worker_accounting and env.worker_accounting[folder] is None and env.evidence['workers'][folder]==1234
    env.close_worker(folder)
    with pytest.raises(ValueError):env.close_worker(folder,SimpleNamespace(process=SimpleNamespace(pid=123)))


def test_failed_clock_classifier_is_not_a_generic_failure_as_loss():
    base=dict(status='worker_error',deadline_seconds=10.,detail='native internal deadline exceeded',
        receipt_elapsed_seconds=.5,received_before_deadline=True,accepted=False)
    with pytest.raises(ValueError):scientific.classify_failure(base)


def test_control_import_in_actual_child_has_no_model_runtime():
    script="import sys; from tools.splendorust_search512_20261101 import control,watchdog,independent_review; assert 'torch' not in sys.modules; print('model-free import passed')"
    result=subprocess.run([sys.executable,'-B','-c',script],capture_output=True,text=True,timeout=60)
    assert result.returncode==0,result.stderr
    assert 'model-free import passed' in result.stdout


def test_trusted_parent_request_and_receive_remain_inherited():
    from tools.splendorust_search512_20261101 import transport
    from tools.local_player_qualification_128.clock import PersistentWorker
    import inspect
    from tools.splendorust_boundary_timing_20261025.parent_profile import ProfiledStudyWorker as Accepted
    assert inspect.getsource(transport.PlacedProfiledStudyWorker._receive)==inspect.getsource(Accepted._receive)
    assert transport.PlacedProfiledStudyWorker._stdout_reader is PersistentWorker._stdout_reader


def test_full_saved_verifier_enforces_original_schedule_before_statistics():
    tree=ast.parse(Path(r.__file__).read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='verify_saved')
    source=ast.get_source_segment(Path(r.__file__).read_text(),fn)
    assert source.index('read_games(')<source.index('analysis.analyze_main(')
    assert source.index('transport.verify_closed_worker(')<source.index('analysis.analyze_main(')
    assert source.index('verify_replays(')<source.index('analysis.analyze_main(')
    assert 'boundary=envelope.guard' in source[source.index('analysis.summarize_timing('):]
