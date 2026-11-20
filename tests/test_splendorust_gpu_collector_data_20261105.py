"""Saved evidence and synthetic scheduling tests only; never load a model or play."""
from copy import deepcopy
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import threading
import time
import pytest

from tools.splendorust_gpu_collector_20261105 import collector, data, qualification, common as c
from tools.splendorust_gpu_throughput_20261105 import scientific


def saved_row():
    # Explicit historical engineering fixture for decoder tests, not new corpus.
    return c.read(c.ROOT/'outputs/splendorust_selfplay_pilot_20261102_02/capture_fixtures/capture_on/008.json')


def test_fixed_seed_plan_and_exact_fixture_exception():
    plan=collector.build_plan()
    assert len(plan['specifications'])==len({x['seed_decimal'] for x in plan['specifications']})==32
    assert qualification.build_plan()['decisions_per_route']==116
    case=next(x for x in qualification.build_plan()['loader']['cases'] if x['id']=='noble_final_round')
    spec=qualification.fixture_spec(case,8)
    assert data.initial_snapshot(spec)==case['snapshot']
    for key,value in [('purpose','ordinary'),('game_id','g000'),('fixture_start',{'legacy':True})]:
        bad=deepcopy(spec);bad[key]=value
        with pytest.raises((ValueError,KeyError)):data.initial_snapshot(bad)


def test_actual_row_wrapper_requires_original_clock_and_preserves_visit_targets(monkeypatch):
    row=saved_row();before=row['before'];state=data.guard.decode_training_snapshot(before)
    row.update(game_id='synthetic_clock_plumbing',spec={'game_id':'synthetic_clock_plumbing'},actor=state.current_player,
        result={'action':row['action'],'diagnostics':row['trace']},
        clock=dict(internal_seconds=60.,outer_seconds=75.,execution_policy_id=c.EXECUTION_POLICY_ID,no_retry=True,
                   elapsed_seconds=.1,started_perf_counter=1.,status='ok',completed=True))
    seen=[]
    monkeypatch.setattr(scientific,'audit_decision',lambda *args,**kwargs:seen.append(kwargs))
    monkeypatch.setattr(scientific,'validate_result',lambda *args,**kwargs:None)
    decoded=data.validate_row(row,row['spec'],before,row['ply'])
    assert decoded['policy']==row['training_capture']['policy_target81'] and len(seen)==1
    bad=deepcopy(row);bad['clock']['outer_seconds']=90.
    with pytest.raises(ValueError):data.validate_row(bad,bad['spec'],before,bad['ply'])
    bad=deepcopy(row);bad['training_capture']['visits67'][0]+=1
    with pytest.raises(ValueError):data.validate_row(bad,bad['spec'],before,bad['ply'])


@pytest.mark.parametrize('winner,actor,expected',[(0,0,[1.,-1.]),(0,1,[-1.,1.]),(1,0,[-1.,1.]),
    (1,1,[1.,-1.]),(None,0,[0.,0.]),(None,1,[0.,0.])])
def test_actor_and_draw_mapping(monkeypatch,winner,actor,expected):
    monkeypatch.setattr(data.guard,'require_official_terminal_target',lambda snapshot,reason:winner)
    assert data.signed_outcomes({},'completed',actor)==expected


@pytest.mark.parametrize('reason',['cutoff','deadlock','timeout','completed'])
def test_unfinished_or_failure_never_gets_terminal_target(reason):
    with pytest.raises(ValueError):data.signed_outcomes(saved_row()['before'],reason,0)


def test_manifest_hash_and_legacy_identity_are_rejected_before_loader(monkeypatch,tmp_path):
    monkeypatch.setattr(c,'ROOT',tmp_path);monkeypatch.setattr(c,'OUT',tmp_path)
    spec=dict(game_id='g0',game_index=0,seed_decimal='123',start_player=0)
    item=c.save(tmp_path/'game.json',{'unused':'not a source game'})
    manifest=data.build_manifest([item],[spec],route='cpu',cell_id='test')
    data.validate_manifest(manifest,[spec])
    checks=data.qualify_manifest_rejections(manifest,[spec])
    assert len(checks)==6 and all(x['rejected'] for x in checks)
    for key in data.IDENTITY:
        bad=deepcopy(manifest);bad['metadata'][key]='historical'
        with pytest.raises(ValueError):data.validate_manifest(bad,[spec])
    bad=deepcopy(item);bad['path']='../outside.json'
    with pytest.raises(ValueError):data.read_artifact(bad)


def prepare_queue(tmp_path,monkeypatch,concurrency,negative=None):
    monkeypatch.setattr(c,'OUT',tmp_path);monkeypatch.setattr(c,'ROOT',tmp_path)
    specs=[dict(game_id=f'g{i:03d}',game_index=i,seed_decimal=str(i+100),start_player=i%2) for i in range(16)]
    gameplan={'specifications':specs}
    monkeypatch.setattr(collector,'build_plan',lambda:gameplan)
    monkeypatch.setattr(collector.q,'validate_runtime',lambda runtime:None)
    monkeypatch.setattr(collector,'warmup_runtime',lambda *args:None)
    monkeypatch.setattr(collector,'run_replays',lambda *args:{'status':'mock_exact','count':0})
    route='cpu' if concurrency==8 else 'gpu'
    runtime=SimpleNamespace(route=route,adapters=[],aborted=False)
    def make_adapter(client,slot,quota):
        adapter=SimpleNamespace(client_id=client,slot=slot,closed=False,active=False)
        adapter.close=lambda:setattr(adapter,'closed',True)
        runtime.adapters.append(adapter);return adapter
    runtime.make_adapter=make_adapter;runtime.abort=lambda:setattr(runtime,'aborted',True)
    first=threading.Barrier(concurrency);seen=[];lock=threading.Lock()
    def play(plan,runtime,adapter,spec,folder,stopped,boundary):
        assert len(runtime.adapters)==concurrency and not adapter.active
        adapter.active=True
        with lock:seen.append((spec['game_index'],adapter.client_id,adapter.slot))
        if spec['game_index']<concurrency:first.wait(timeout=4)
        if negative is not None and spec['game_index']==negative:
            stopped.set();adapter.active=False
            return dict(game_id=spec['game_id'],status='deadline_failed',completed_moves=0)
        time.sleep(.003*(1+spec['game_index']%3))
        c.save(folder/'target_game.json',{'synthetic':True,'spec':spec})
        adapter.active=False
        return dict(game_id=spec['game_id'],status='completed',completed_moves=1)
    monkeypatch.setattr(collector,'_play_game',play)
    def load(manifest,*,expected_specs,boundary,sink):
        for spec in expected_specs:sink(spec,[{'synthetic_test':True}])
        return dict(eligible_roots=16,games=[dict(plies=1) for _ in expected_specs])
    monkeypatch.setattr(data,'load_corpus',load)
    cell=dict(cell_id='test',route=route,concurrency=concurrency,game_indices=list(range(16)),specifications=specs)
    return dict(games=gameplan),cell,runtime,seen


@pytest.mark.parametrize('concurrency',[8,16])
def test_dynamic_queue_full_manifest_and_loader_are_inside_primary_clock(tmp_path,monkeypatch,concurrency):
    plan,spec,runtime,seen=prepare_queue(tmp_path,monkeypatch,concurrency)
    result=collector.run_cell(plan,spec,runtime,tmp_path/'cell')
    assert result['status']=='completed' and result['games']==16
    assert sorted(x[0] for x in seen)==list(range(16))
    assert len(runtime.adapters)==concurrency and all(x.closed for x in runtime.adapters)
    assert all(slot==int(client.split('_')[1])%8 for _,client,slot in seen)
    assert (tmp_path/'cell/loader_result.json').exists() and (tmp_path/'cell/target_manifest.json').exists()
    assert result['wall_seconds']>=result['gameplay_seconds'] and result['target_loader_seconds']>0
    assert result['model_replay']['status']=='mock_exact'


def test_queue_first_failure_preserved_without_replacement_or_corpus(tmp_path,monkeypatch):
    plan,spec,runtime,seen=prepare_queue(tmp_path,monkeypatch,8,negative=0)
    result=collector.run_cell(plan,spec,runtime,tmp_path/'cell')
    assert result['status']=='deadline_failed' and result['corpus_produced'] is False
    assert sorted(x[0] for x in seen)==list(range(8))
    assert not (tmp_path/'cell/corpus_manifest.json').exists()
    assert all(x.closed for x in runtime.adapters)
    outcomes=c.read(tmp_path/'cell/queue_outcomes.json')
    assert outcomes['outcomes']['g000']['status']=='deadline_failed'


def test_full_game_loader_rejects_nonofficial_reason_before_rows():
    spec={'game_id':'g0'}
    for reason in ('cutoff','timeout','deadlock'):
        document=dict(schema=data.GAME_SCHEMA,**data.IDENTITY,spec=spec,reason=reason)
        with pytest.raises(ValueError):data.load_game(document,spec)


@pytest.mark.parametrize('changed',[False,True])
def test_warmup_requires_accepted_root_bits_not_only_repetition(tmp_path,monkeypatch,changed):
    monkeypatch.setattr(c,'OUT',tmp_path);monkeypatch.setattr(c,'ROOT',tmp_path)
    source=c.save(tmp_path/'source.json',dict(result=dict(diagnostics=dict(phases=[dict(inference=[
        dict(tokens=[1.],output83=[0.]*83)])]))))
    plan=dict(cases=[dict(expected_root_tokens=[1.],expected_results={'gpu':source})])
    state={'calls':0,'registered':False}
    def infer(*args,**kwargs):
        state['calls']+=1
        return [[-0. if changed else 0.]*83],{'synthetic_test':True}
    service=SimpleNamespace(register=lambda *a:state.update(registered=True),
        unregister=lambda *a:state.update(registered=False),infer=infer)
    runtime=SimpleNamespace(route='gpu',service=service,operation=lambda *a,**kw:nullcontext(),abort=lambda:None)
    if changed:
        with pytest.raises(ValueError,match='bit-exact'):collector.warmup_runtime(plan,runtime,tmp_path/'warmup')
        assert state['calls']==1
    else:
        collector.warmup_runtime(plan,runtime,tmp_path/'warmup')
        assert state['calls']==5 and c.read(tmp_path/'warmup/warmup.json')['exact_accepted_root_bits']
    assert not state['registered']
