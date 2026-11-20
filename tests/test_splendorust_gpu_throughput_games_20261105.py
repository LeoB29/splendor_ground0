"""Model-free fixed-wave gameplay, durability, failure and replay regressions."""
from copy import deepcopy
from types import SimpleNamespace
import threading
import time

import pytest

from tools.splendorust_gpu_throughput_20261105 import common as c, games, qualification as q, scientific


def test_fresh_plan_is_disjoint_and_balances_starting_seats():
    plan=games.build_plan()
    old=c.read(c.ROOT/games.PRIOR_GAME_PLAN)["games"]
    assert len(plan["specifications"])==8 and len({row["seed_decimal"] for row in plan["specifications"]})==8
    assert not {row["seed_decimal"] for row in plan["specifications"]}&{row["seed_decimal"] for row in old["specifications"]}
    assert [row["start_player"] for row in plan["specifications"]]==[0,1]*4
    assert plan["internal_seconds"]==60 and plan["outer_seconds"]==75
    assert all(games.initial_snapshot(spec)==row["initial"] for spec,row in zip(plan["specifications"],plan["initial_states"]))


class FakeRuntime:
    route="cpu"
    def __init__(self):
        self.service=SimpleNamespace(lock=threading.RLock(),clients={},batch_count=0,completed_requests=0)
        self.adapters=[];self.events=[];self.aborts=0
    def make_adapter(self,client,slot,quota):
        assert quota==512 and client not in self.service.clients and slot not in self.service.clients.values()
        self.service.clients[client]=slot;self.events.append(("register",client,slot))
        value=SimpleNamespace(client_id=client,slot=slot,last_service_calls=[],closed=False)
        def close():
            if not value.closed:
                self.service.clients.pop(client);self.events.append(("close",client,slot));value.closed=True
        value.close=close;self.adapters.append(value)
        return value
    def abort(self):self.aborts+=1


def fixture(tmp_path,monkeypatch,concurrency,fail_index=None,fail_error=None,*,plies=1,fail_ply=0):
    monkeypatch.setattr(c,"OUT",tmp_path)
    specs=[dict(game_id=f"g{i:03d}",game_index=i) for i in range(8)]
    gameplan=dict(specifications=specs,throughput_includes="first release through durable natural results")
    plan=dict(games=gameplan)
    monkeypatch.setattr(games,"build_plan",lambda:deepcopy(gameplan))
    monkeypatch.setattr(games,"initial_snapshot",lambda spec:dict(index=spec["game_index"],ply=0))
    monkeypatch.setattr(games.guard,"require_official_terminal_target",lambda value,status:0 if value["ply"]==plies else pytest.fail("nonterminal"))
    monkeypatch.setattr(games.rules,"observation_payload",lambda state:dict(index=state.index,ply=state.ply) if hasattr(state,"index") else {})
    def decode(value):return SimpleNamespace(terminal=value["ply"]>=plies,current_player=0,index=value["index"],ply=value["ply"])
    monkeypatch.setattr(games.guard,"decode_training_snapshot",decode)
    monkeypatch.setattr(games,"actual_after",lambda before,action:dict(index=before["index"],ply=before["ply"]+1))
    monkeypatch.setattr(games,"validate_move",lambda row,spec,route,before,ply:row["after"])
    monkeypatch.setattr(scientific,"validate_result",lambda *args,**kwargs:None)
    monkeypatch.setattr(scientific,"audit_decision",lambda *args,**kwargs:None)
    monkeypatch.setattr(q,"validate_runtime",lambda runtime:None)
    runtime=FakeRuntime();lock=threading.Lock();seen=[];active=[0];peak=[0]
    rendezvous=threading.Barrier(concurrency)
    def ordinary(runtime,adapters,request,*,started,client_ids,evidence_callback):
        adapter=adapters["search"];client=client_ids["search"];index=request["observer"]["index"]
        if client.startswith("game_"):
            assert len([item for item in runtime.adapters if item.client_id.startswith("game_")])==8
            assert adapter.slot==index
            attempt=c.read(tmp_path/"games"/f"g{index:03d}"/"attempts"/f"{request['observer']['ply']:06d}.json")
            assert attempt["started_perf_counter"]==started
            with lock:
                seen.append(index);active[0]+=1;peak[0]=max(peak[0],active[0])
            rendezvous.wait(timeout=3)
            try:
                time.sleep(.001)
                if index==fail_index and request["observer"]["ply"]==fail_ply:raise fail_error
            finally:
                with lock:active[0]-=1
        else:
            assert client==f"replay_g{index:03d}" and adapter.slot==(index+1)%8
            assert len(runtime.service.clients)==1
        with runtime.service.lock:
            runtime.service.batch_count+=1;runtime.service.completed_requests+=1
        adapter.last_service_calls=[dict(marker=index)]
        result=dict(action=dict(action_type="TAKE",index=index),diagnostics=dict(inference_calls=1,phases=[dict(simulations=512)]))
        evidence_callback(dict(result=result,observer=request["observer"],action=result["action"],diagnostics=result["diagnostics"]))
        return result
    monkeypatch.setattr(q,"ordinary",ordinary)
    return plan,runtime,seen,peak


@pytest.mark.parametrize("concurrency",[1,2,4,8])
def test_fixed_waves_prepare_all_slots_then_complete_and_replay(tmp_path,monkeypatch,concurrency):
    plan,runtime,seen,peak=fixture(tmp_path,monkeypatch,concurrency)
    result=games.run_games(plan,runtime,tmp_path/"games",concurrency=concurrency)
    cohort=result["cohort"]
    assert result["status"]=="passed" and cohort["finished_games"]==8 and cohort["full_cohort_completed"]
    assert peak[0]==concurrency
    assert [set(seen[i:i+concurrency]) for i in range(0,8,concurrency)]==[set(group) for group in games.wave_groups(concurrency)]
    assert cohort["total_plies"]==cohort["inference_calls"]==8 and cohort["simulations"]==4096
    assert cohort["games_per_hour"]==8*3600/cohort["wall_seconds"]
    assert len(cohort["waves"])==8//concurrency
    ends=[c.read(tmp_path/"games"/f"g{i:03d}"/"finished.json")["ended_perf_counter"] for i in range(8)]
    assert cohort["ended_perf_counter"]==max(ends)
    assert result["audit"]["transitions"]==8 and result["replay"]["count"]==8
    assert runtime.service.clients=={} and all(adapter.closed for adapter in runtime.adapters)
    assert runtime.aborts==0
    assert games.audit_games(plan,tmp_path/"games",concurrency=concurrency)==result["audit"]
    assert games.verify_model_replays(plan,tmp_path/"games")["count"]==8


def test_real_decision_timeout_preserves_peer_and_stops_future_waves(tmp_path,monkeypatch):
    failure=TimeoutError("Full ordinary response exceeded fixed60s internal guard")
    plan,runtime,seen,_=fixture(tmp_path,monkeypatch,2,fail_index=1,fail_error=failure)
    result=games.run_games(plan,runtime,tmp_path/"games",concurrency=2)
    assert result["status"]=="deadline_failed" and set(seen)=={0,1}
    cohort=result["cohort"]
    assert cohort["started_game_ids"]==["g000","g001"] and cohort["completed_game_ids"]==["g000"]
    assert cohort["never_started_game_ids"]==[f"g{i:03d}" for i in range(2,8)]
    assert cohort["wall_seconds"] is cohort["games_per_hour"] is cohort["decisions_per_second"] is None
    assert result["replay"] is None and runtime.aborts==0
    assert (tmp_path/"games/g001/failure.json").exists()
    assert not (tmp_path/"games/g002/started.json").exists()
    assert not list((tmp_path/"games/model_replays").glob("*.json"))


def test_status_timeout_cannot_be_admitted_as_a_slow_game(tmp_path,monkeypatch):
    plan,runtime,seen,_=fixture(tmp_path,monkeypatch,2,fail_index=1,fail_error=TimeoutError("Status mutex exceeded250ms acquisition bound"))
    with pytest.raises(RuntimeError,match="integrity failure"):
        games.run_games(plan,runtime,tmp_path/"games",concurrency=2)
    assert set(seen)=={0,1}
    assert c.read(tmp_path/"games/cohort.json")["status"]=="integrity_failed"
    assert all(adapter.closed for adapter in runtime.adapters)


def test_late_failure_retains_moves_from_unfinished_games_without_a_rate(tmp_path,monkeypatch):
    failure=TimeoutError("Full ordinary response exceeded fixed60s internal guard")
    plan,runtime,seen,_=fixture(tmp_path,monkeypatch,2,fail_index=1,fail_error=failure,plies=2,fail_ply=1)
    result=games.run_games(plan,runtime,tmp_path/"games",concurrency=2)
    cohort=result["cohort"]
    assert result["status"]=="deadline_failed" and sorted(seen)==[0,0,1,1]
    assert cohort["finished_games"]==1 and cohort["total_plies"]==3 and cohort["completed_game_plies"]==2
    assert cohort["inference_calls"]==3 and cohort["simulations"]==3*512 and cohort["latency"]["count"]==3
    assert result["audit"]["transitions"]==3 and cohort["games_per_hour"] is None
    assert c.read(tmp_path/"games/g001/failure.json")["failed_ply"]==1


def test_between_decision_peer_is_not_fabricated_as_an_independent_timeout():
    failure=TimeoutError("Full ordinary response exceeded fixed60s internal guard")
    clock=dict(status="error",completed=False,internal_seconds=60.,outer_seconds=75.)
    rows={0:dict(game_id="g000",case_id="g000",status="deadline_failed",clock=clock),
          1:dict(game_id="g001",case_id="g001",status="cancelled_between_decisions",completed_moves=4)}
    assert games.resolve_wave_outcomes(rows,{0:failure})=="deadline_failed"
    assert rows[1]["status"]=="cancelled" and rows[1]["cancellation_kind"]=="between_decisions"
    assert rows[1]["cancellation_trigger_case_ids"]==["g000"] and "clock" not in rows[1]


def test_between_decision_stop_without_proven_failure_is_integrity_error():
    rows={0:dict(game_id="g000",case_id="g000",status="cancelled_between_decisions")}
    assert games.resolve_wave_outcomes(rows,{})=="integrity_failed"


def test_replay_selection_is_fixed_by_first_neural_ply_per_band():
    rows=[dict(ply=i,result=dict(diagnostics=dict(inference_calls=int(i in (3,5,22,42,50))))) for i in range(60)]
    assert games.replay_selection(rows)==[dict(ply=3,band=[0,19]),dict(ply=22,band=[20,39]),dict(ply=42,band=[40,1999])]


def test_long_successes_are_descriptive_not_rejected_by_old_latency_targets():
    clock=dict(status="ok",completed=True,internal_seconds=60.,outer_seconds=75.,execution_policy_id=c.EXECUTION_POLICY_ID,
               no_retry=True,started_perf_counter=100.,ended_perf_counter=120.,elapsed_seconds=20.)
    games.validate_clock(clock)
    summary=games.latency_summary([.5,5.,9.5,20.])
    assert summary["over_seconds"]=={"1":3,"9":2,"10":1}
    assert summary["latency_thresholds_are_descriptive"]


def test_saved_cohort_timing_tamper_is_rejected(tmp_path,monkeypatch):
    plan,runtime,_,_=fixture(tmp_path,monkeypatch,4)
    games.run_games(plan,runtime,tmp_path/"games",concurrency=4)
    path=tmp_path/"games/cohort.json";value=c.read(path);value["wall_seconds"]*=.5
    path.write_text(c.canonical(value),encoding="utf-8")
    with pytest.raises(ValueError,match="wall throughput"):
        games.audit_games(plan,tmp_path/"games",concurrency=4)


def test_saved_replay_changed_result_is_rejected(tmp_path,monkeypatch):
    plan,runtime,_,_=fixture(tmp_path,monkeypatch,4)
    games.run_games(plan,runtime,tmp_path/"games",concurrency=4)
    path=tmp_path/"games/model_replays/g000_000000.json";value=c.read(path)
    value["result"]["action"]["index"]=77;path.write_text(c.canonical(value),encoding="utf-8")
    with pytest.raises(ValueError,match="shifted-slot"):
        games.verify_model_replays(plan,tmp_path/"games")
