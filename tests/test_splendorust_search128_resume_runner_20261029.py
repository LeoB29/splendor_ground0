"""Model-free continuation, streaming and shared-boundary regression checks."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import threading
import weakref
import gc
import pytest
from tools.splendorust_search128_resume_20261029 import common as c,runner,transport,audit


@pytest.fixture
def output(tmp_path,monkeypatch):
    monkeypatch.setattr(c,'OUT',tmp_path/'new');monkeypatch.setattr(c,'ROOT',tmp_path)
    monkeypatch.setattr(c,'OLD_OUT',tmp_path/'old');monkeypatch.setattr(c,'WRITE_OBSERVER',None)
    (c.OLD_OUT/'workers').mkdir(parents=True)
    monkeypatch.setattr(c,'FIRST_RECOVERY_OUT',tmp_path/'first')
    monkeypatch.setattr(c,'CLOSURE_OUT',tmp_path/'closure')
    (c.FIRST_RECOVERY_OUT/'workers').mkdir(parents=True)
    return c.OUT


def test_shared_origin_prior_time_and_ram_carry(output,monkeypatch):
    claim=dict(monotonic_origin=c.ORIGINAL_MONOTONIC,unix_origin=c.ORIGINAL_UNIX)
    e=runner.Envelope(claim)
    assert e.waited==200 and e.end==c.ORIGINAL_MONOTONIC+86400
    with pytest.raises(ValueError,match='reset'):runner.Envelope(dict(claim,monotonic_origin=e.start+1))
    monkeypatch.setattr(runner.time,'monotonic',lambda:e.start+20000)
    monkeypatch.setattr(runner.time,'time',lambda:e.wall_start+20000)
    monkeypatch.setattr(e,'guard',lambda:None)
    e.begin('final')
    assert e.deadline==e.start+20000+14400-190
    assert c.read(c.OUT/'stages/final_start.json')['cap_seconds']==14210


def test_aggregate_bytes_include_old_and_new(output):
    (c.OLD_OUT/'saved.json').write_text(json.dumps({'old':'x'*1700}))
    c.save(c.OUT/'new.json',{'new':'y'*600})
    assert runner.bytes_used()==sum(p.stat().st_size for p in (c.OLD_OUT/'saved.json',c.OUT/'new.json'))


def test_save_or_verify_is_repeatable_without_overwrite(output):
    p=c.OUT/'prior_qualification_verification.json';value={'status':'passed','scientific_decisions':203}
    runner.save_or_verify(p,value);before=p.read_bytes();runner.save_or_verify(p,deepcopy(value))
    assert p.read_bytes()==before
    with pytest.raises(ValueError,match='differs'):runner.save_or_verify(p,dict(value,scientific_decisions=119))
    assert p.read_bytes()==before


def test_lazy_rows_reiterate_without_retaining_decoded_values(output,monkeypatch):
    class Row(dict):pass
    refs=[];reads=[]
    def read(path):
        gc.collect();assert sum(r() is not None for r in refs)<=1
        row=Row(index=int(Path(path).stem),payload='x'*100000)
        refs.append(weakref.ref(row));reads.append(path);return row
    monkeypatch.setattr(c,'read',read)
    rows=runner.SavedRows([dict(path=f'{i}.json',bytes=100001) for i in range(80)])
    assert len(rows)==80 and rows.maximum_bytes==100001
    assert sum(r['index'] for r in rows)==3160
    assert sum(r['index'] for r in rows)==3160 and len(reads)==160
    gc.collect();assert not any(r() is not None for r in refs)


def test_freeze_hash_is_cached_per_context(output,monkeypatch):
    calls=[];monkeypatch.setattr(c,'file_hash',lambda path:calls.append(path) or 'abc')
    context={}
    assert all(runner.recovery_freeze_sha(context)=='abc' for _ in range(500))
    assert len(calls)==1


def test_lazy_rows_guard_every_load_and_index_selected_only(output,monkeypatch):
    calls=[];guards=[]
    monkeypatch.setattr(c,'read',lambda p:calls.append(p) or {'read':p.name})
    rows=runner.SavedRows(boundary=lambda:guards.append(True))
    for i in range(12):
        row=dict(stage='main',pair_index=i//2,game_index=i,ply=0,player_role='challenger')
        rows.add(dict(path=f'{i}.json',bytes=80),row)
    assert rows.find(dict(stage='main',pair_index=4,game_index=8,ply=0,player_role='challenger'))=={'read':'8.json'}
    assert len(calls)==len(guards)==1
    list(rows);assert len(calls)==len(guards)==13


def segmented_fixture(monkeypatch,*,failure=False):
    def state(i):return {'i':i,'state':{'current_player':i%2,'terminal':i>=41,'winner':0 if i>=41 else None}}
    def transition(before,action):assert action=={'advance':1};return state(before['i']+1)
    monkeypatch.setattr(runner.ledger,'transition',transition)
    monkeypatch.setattr(runner.ledger,'check_observer',lambda *a:None)
    monkeypatch.setattr(runner,'verify_live_row',lambda *a:None)
    monkeypatch.setattr(audit,'validate_boundary_row',lambda *a:None)
    monkeypatch.setattr(audit,'classify_failure',lambda *a:'research_outer_hard_guard')
    spec={'stage':'main','pair_index':19,'game_index':38,'challenger_seat':0}
    def row(i,completed=True):return dict(**{k:spec[k] for k in ('stage','pair_index','game_index')},
        ply=i,before=state(i),after=state(i+1) if completed else None,action={'advance':1} if completed else None,
        seat=i%2,player_role='challenger' if i%2==0 else 'opponent',completed=completed,
        clock={'receipt_elapsed_seconds':.4,'accepted':completed,'status':'ok' if completed else 'timeout'})
    context={'bridge':{'bridge':{'ply':39},'before':state(39),'after':state(40),
        'source_row':{'observer':{},'clock':{'status':'resource_guard','accepted':False}},'result':{'action':{'advance':1}}}}
    events=[dict(kind='ordinary',value=row(i)) for i in range(39)]+[dict(kind='recovery_bridge'),dict(kind='ordinary',value=row(40,not failure))]
    ending=dict(**spec,complete=True,plies=41,final=state(40 if failure else 41),reason='forfeit' if failure else 'natural',winner=1 if failure else 0)
    if failure:ending.update(administrative_forfeit=True,failure_policy='research_hard_guard_9s_internal_10s_complete_response',
        failed_role='challenger',failed_seat=0,loser=0,failure='research_outer_hard_guard')
    return state(0),events,ending,spec,context


@pytest.mark.parametrize('failure',[False,True])
def test_segmented_transition_once_and_terminal_or_forfeit(monkeypatch,failure):
    initial,events,ending,spec,context=segmented_fixture(monkeypatch,failure=failure)
    raw=deepcopy(context['bridge']['source_row']['clock'])
    proof=runner.audit_segmented_game(initial,events,ending,spec,context,{})
    assert proof['transitions']==(40 if failure else 41) and proof['failed_moves']==int(failure)
    assert proof['recovered_transitions']==1 and proof['recovered_move_in_ordinary_timing'] is False
    assert sum(map(len,proof['latencies'].values()))==(39 if failure else 40)
    assert context['bridge']['source_row']['clock']==raw


@pytest.mark.parametrize('mutation',['duplicate','skip','relocate'])
def test_segmented_transition_rejects_duplicate_skip_or_relocation(monkeypatch,mutation):
    initial,events,ending,spec,context=segmented_fixture(monkeypatch)
    if mutation=='duplicate':events.insert(40,dict(kind='recovery_bridge'))
    if mutation=='skip':events.pop(39)
    if mutation=='relocate':events[38],events[39]=events[39],events[38]
    ending['plies']=len(events)
    with pytest.raises(ValueError):runner.audit_segmented_game(initial,events,ending,spec,context,{})


def test_actual_requalification_uses_same_retained_two_workers(output,monkeypatch):
    from tools.splendorust_search128_resume_20261029 import adapters,qualification
    made=[];called=[]
    def make(name,e,f,route):
        child=SimpleNamespace(output_dir=c.OUT/'workers'/name,route=route);made.append(child);return child
    def actual(child,e,f,**kwargs):
        called.append(child);return dict(status='ok',request_id='request-00000002',result={'counts':{'complete_decisions':67 if child.route=='research' else 52}})
    monkeypatch.setattr(runner,'make_worker',make);monkeypatch.setattr(adapters,'actual_entrypoint',actual)
    monkeypatch.setattr(qualification,'verify_entrypoint',lambda folder,receipt,*a,**k:{'status':'passed','raw_execution_decisions_bound':receipt['result']['counts']['complete_decisions']})
    workers={};value=runner.qualify_resumed_workers(SimpleNamespace(guard=lambda:None),{}, {},workers)
    assert len(made)==len(workers)==2 and made==called
    assert value['scientific_decisions']==119 and value['combined_scientific_decisions']==441
    assert sum(x['verification']['raw_execution_decisions_bound'] for x in value['receipts'])==119
    assert list(workers.values())==made


def test_stage_rebind_requires_idle_original_request_lock(output):
    child=SimpleNamespace(_closed=False,recovery_accounting=SimpleNamespace(state='idle'),_lock=threading.Lock(),
        _counter=2,absolute_deadline=123.,output_dir=c.OUT/'workers/main_g38_challenger')
    e=SimpleNamespace(effective_deadline=lambda:456.,stage='main',start=c.ORIGINAL_MONOTONIC,wall_start=c.ORIGINAL_UNIX)
    child._lock.acquire()
    with pytest.raises(ValueError,match='idle'):runner.rebind_retained_workers({'challenger':child},e)
    child._lock.release();runner.rebind_retained_workers({'challenger':child},e)
    assert child._counter==2 and child.absolute_deadline==456.
    assert c.read(c.OUT/'recovery_stage_transition.json')['ordinary_clocks_reset'] is False


def test_infrastructure_row_saved_without_forfeit_classifier(output,monkeypatch):
    from tools.splendorust_search128_resume_20261029 import schedule
    spec={'stage':'main','pair_index':19,'game_index':39,'challenger_seat':0,'seed_decimal':'1',
        'challenger_mode':'research','opponent_mode':'frozen'}
    state=SimpleNamespace(terminal=False,current_player=0)
    monkeypatch.setattr(schedule,'validate_spec',lambda s:None)
    monkeypatch.setattr(runner.rules,'CorrectedEnv',lambda **k:SimpleNamespace(initial_state=lambda:state))
    monkeypatch.setattr(runner.rules,'serialize_snapshot',lambda s:{'state':{'current_player':0}})
    monkeypatch.setattr(runner,'decode_training_snapshot',lambda s:state)
    row=dict(completed=False,clock={'status':'resource_guard','accepted':False},route='research',seat=0)
    monkeypatch.setattr(runner,'decision',lambda *a:deepcopy(row))
    monkeypatch.setattr(runner,'_require_profiles',lambda r:None);monkeypatch.setattr(runner,'verify_saved_decision',lambda *a:None)
    monkeypatch.setattr(runner,'deadline_failure',lambda *a:False)
    monkeypatch.setattr(runner,'captured_fault',lambda *a:pytest.fail('Infrastructure is not a player loss'))
    monkeypatch.setattr(runner,'close_and_audit',lambda *a:{'status':'passed'})
    with pytest.raises(ValueError,match='infrastructure'):
        runner.play_game(spec,SimpleNamespace(guard=lambda:None),{},workers={'challenger':object(),'opponent':object()})
    folder=runner.game_folder(spec)
    assert c.read(folder/'moves/0000.json')['clock']==row['clock']
    assert c.read(folder/'infrastructure_failure.json')['classified_as_player_loss'] is False
    assert not(folder/'result.json').exists() and not(folder/'captured_fault.json').exists()


def test_incomplete_finalization_never_calls_full_acceptance_or_models(output,monkeypatch):
    monkeypatch.setattr(runner,'verify_runtime',lambda *a:None)
    monkeypatch.setattr(runner,'verify_prior_preservation',lambda *a:None)
    monkeypatch.setattr(runner,'final_checks',lambda *a:pytest.fail('No full workload acceptance'))
    monkeypatch.setattr(runner,'make_worker',lambda *a:pytest.fail('No final learned requests'))
    result=runner.incomplete_checks(SimpleNamespace(guard=lambda:None),{},'primary resource fault')
    assert result['primary_failure_preserved']=='primary resource fault' and result['status']=='incomplete'
    assert result['full_workload_accepted'] is False and result['new_model_calls_for_finalization']==0


def test_worker_numerical_entrypoint_byte_identical_and_trusted_clock_inherited():
    from tools.splendorust_search128_strength_20261028 import worker as old
    from tools.splendorust_search128_resume_20261029 import worker as new
    assert Path(old.__file__).read_bytes()==Path(new.__file__).read_bytes()
    from tools.splendorust_search128_strength_20261028.transport import PlacedProfiledStudyWorker as prior
    for name in ('request','_receive','_stdout_reader','_stderr_reader'):
        assert getattr(transport.PlacedProfiledStudyWorker,name) is getattr(prior,name)
