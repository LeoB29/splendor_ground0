"""Independent scope, failure-attribution and player-preservation checks."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from tools.splendorust_graph64_strength_20261025 import runner, audit, schedule
from tools.splendorust_search_scale_20261024 import audit as established


@pytest.mark.parametrize('mode,quota',[('candidate',32),('baseline',64)])
def test_only_qualified_graph64_and_original_eager32_can_reach_worker_admission(mode,quota):
    envelope=SimpleNamespace(admit=lambda:pytest.fail('Unqualified player reached admission'))
    with pytest.raises(ValueError,match='Only unchanged eager32'):
        runner.make_worker('forbidden',envelope,{},mode=mode,quota=quota)


def test_operational_and_main_schedule_keep_both_backends_and_swapped_worlds():
    specs=schedule.make_schedule()
    assert len(specs)==206
    for stage,count in [('operational',6),('main',200)]:
        rows=[row for row in specs if row['stage']==stage]
        assert len(rows)==count and len({row['seed_decimal'] for row in rows})==count//2
        for first,second in zip(rows[::2],rows[1::2]):
            assert first['seed_decimal']==second['seed_decimal']
            assert {first['challenger_seat'],second['challenger_seat']}=={0,1}
    assert all(row['challenger_mode']=='candidate' and row['challenger_quota']==64 and
               row['opponent_mode']=='baseline' and row['opponent_quota']==32 for row in specs)


def failed_clock(status,detail=None):
    return dict(status=status,accepted=False,request_id='request-00000001',deadline_seconds=1.,
        elapsed_seconds=1.1,receipt_elapsed_seconds=None,received_before_deadline=False,
        clock='perf_counter_ns',result=None,detail=detail,telemetry={})


def failed_game_fixture(monkeypatch,tmp_path,*,game_index,status,detail=None):
    monkeypatch.setattr(runner.c,'OUT',tmp_path)
    monkeypatch.setattr(runner.c,'ROOT',tmp_path)
    spec=schedule.stage_schedule('main')[game_index]
    calls=[]; closed=[]; opened=[]
    class Child:
        def __init__(self,name,mode,quota):
            self.runtime_mode,self.challenger_quota=mode,quota
            self.output_dir=tmp_path/'workers'/name
        def __enter__(self): return self
        def __exit__(self,*args): closed.append((self.runtime_mode,self.challenger_quota))
    def worker(name,envelope,freeze,*,mode,quota):
        runner.require_player(mode,quota); opened.append((mode,quota)); return Child(name,mode,quota)
    def decision(child,state,envelope,role):
        calls.append(role)
        return dict(arm='search',player_role=role,quota=child.challenger_quota,mode=child.runtime_mode,
            seat=state.current_player,observer=runner.rules.observation_payload(state),
            before=runner.rules.serialize_snapshot(state),after=None,action=None,trace=None,
            completed=False,clock=failed_clock(status,detail),worker_clock_file='controlled-test-clock')
    monkeypatch.setattr(runner,'make_worker',worker)
    monkeypatch.setattr(runner,'decision',decision)
    monkeypatch.setattr(runner,'verify_worker_logs',lambda *args,**kwargs:dict(status='passed'))
    monkeypatch.setattr(runner,'verify_graph_evidence',lambda *args,**kwargs:dict(status='passed'))
    original=audit.classify_failure
    def classify(receipt):
        # The actual failure must already be durable before adjudication.
        row=json.loads((runner.game_folder(spec)/'moves/0000.json').read_text())
        assert row['clock']==receipt and row['completed'] is False
        return original(receipt)
    monkeypatch.setattr(audit,'classify_failure',classify)
    envelope=SimpleNamespace(guard=lambda:None,effective_deadline=lambda:999999)
    return spec,envelope,calls,closed,opened


@pytest.mark.parametrize('game_index,losing_role',[(0,'challenger'),(1,'opponent')])
def test_real_timeout_is_preserved_then_scored_as_loss_for_exact_acting_player(monkeypatch,tmp_path,game_index,losing_role):
    spec,envelope,calls,closed,opened=failed_game_fixture(monkeypatch,tmp_path,
        game_index=game_index,status='timeout')
    ending,rows,_=runner.play_game(spec,envelope,{})
    assert calls==[losing_role] and opened==[('candidate',64),('baseline',32)]
    assert set(closed)=={('candidate',64),('baseline',32)}
    assert ending['reason']=='forfeit' and ending['failed_role']==losing_role
    assert ending['loser']==0 and ending['winner']==1 and ending['failure']=='timeout'
    assert rows[0]['after'] is None and ending['final']==rows[0]['before']
    assert json.loads((runner.game_folder(spec)/'result.json').read_text())==ending


@pytest.mark.parametrize('status,detail',[('resource_guard','RSS ceiling'),
    ('worker_error','STUDY_RESOURCE: memory'),('worker_error','unexpected graph failure')])
def test_infrastructure_failure_is_preserved_and_stops_without_game_result_or_retry(monkeypatch,tmp_path,status,detail):
    spec,envelope,calls,closed,opened=failed_game_fixture(monkeypatch,tmp_path,
        game_index=0,status=status,detail=detail)
    with pytest.raises(ValueError): runner.play_game(spec,envelope,{})
    assert calls==['challenger'] and len(closed)==2
    folder=runner.game_folder(spec)
    row=json.loads((folder/'moves/0000.json').read_text())
    assert row['clock']['status']==status and row['clock']['detail']==detail
    assert not (folder/'result.json').exists()


def test_failure_classifier_is_exact_established_rule_and_stage_clamp_is_never_loss():
    assert audit.classify_failure is established.classify_failure
    receipt=failed_clock('timeout'); receipt['deadline_seconds']=.5
    with pytest.raises(ValueError,match='stage-clamped'):
        audit.classify_failure(receipt)


def test_completed_replay_sample_preserves_mode_with_maximum24_per_role():
    rows=[]
    for role,mode,quota in [('challenger','candidate',64),('opponent','baseline',32)]:
        for seat in (0,1):
            for phase_base in (0,20,40):
                for category in range(4):
                    own,other=bool(category&1),bool(category&2)
                    players=[{'reserves':[]},{'reserves':[]}]
                    players[seat]['reserves']=[{'source':'deck'}] if own else []
                    players[1-seat]['reserves']=[{'source':'deck'}] if other else []
                    i=len(rows)
                    rows.append(dict(stage='main',completed=True,pair_index=i//2,game_index=i,
                        ply=phase_base+category,player_role=role,seat=seat,arm='search',mode=mode,quota=quota,
                        observer={'observation':{'current_player':seat,'players':players}}))
    sample=audit.select_replay_sample(rows)
    assert len(sample['decisions'])==48 and sample['empty_cells']==[]
    assert sum(row['mode']=='candidate' for row in sample['decisions'])==24
    assert sum(row['mode']=='baseline' for row in sample['decisions'])==24
    corrupted=deepcopy(rows); corrupted[-1]['mode']='candidate'
    with pytest.raises(ValueError,match='sample role/quota'):
        audit.select_replay_sample(corrupted)
