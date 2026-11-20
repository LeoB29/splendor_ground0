"""No learned/native calls: actual corrected decoding, guards and saved plans."""
from contextlib import contextmanager
from copy import deepcopy
import json
import subprocess
import sys
import threading
import time

import pytest

from tools.splendorust_gpu_live_20261105_02 import qualification as q
from tools.splendorust_gpu_live_20261105_02 import games as g


class Runtime:
    route = "cpu"
    runtime_sources = q.guard.SOURCE_HASHES
    def __init__(self):
        self.calls=[]; self.aborted=False
    @contextmanager
    def operation(self,client_id,*,started=None,kind="decision"):
        self.calls.append(dict(client_id=client_id,started=started,kind=kind))
        yield
    def abort(self):
        self.aborted=True


class ForcedPassAdapter:
    def __init__(self):
        self.profile=q.original.profile(512,route="candidate")
        self.request_deadline=None; self.calls=0
    def choose(self,observer,*,diagnostic_capture):
        self.calls+=1
        state=q.rules.state_from_observation(observer)
        action=q.rules.CorrectedEnv().legal_actions(state)[0]
        trace=dict(opponent_id="splendorust",profile_id=self.profile["profile_id"],
                   observer_seed=int(q.digest(observer)[:16],16),forced_pass=True,
                   inference_calls=0,inference_rows=0,phases=[])
        return action,trace


@pytest.fixture(scope="module")
def cases():
    return {x["id"]:x for x in q.corpus.build_fixtures()}


def observer(cases,name="forced_pass"):
    return q.rules.observation_payload(q.guard.decode_training_snapshot(cases[name]["snapshot"]))


def test_preparation_fixed35_roots_and_all_counterexamples():
    plan=q.build_plan()
    assert len(plan["decision_panel"])==35
    assert len(plan["structural_suite"]["counterexamples"])==15
    assert plan["required_full_suite_decisions_per_route"]==67
    assert plan["cross_backend_action_agreement_is_gate"] is False
    assert {x["path"] for x in plan["source_inputs"]}
    assert set(q.profile_bindings())=={"16","512"}


def test_actual_ordinary_forced_pass_without_model(cases):
    runtime,adapter=Runtime(),ForcedPassAdapter(); started=time.perf_counter()
    result=q.ordinary(runtime,{"search":adapter},dict(player="search",observer=observer(cases)),
                      started=started,client_ids={"search":"test"})
    assert result["action"]["action_type"]=="PASS"
    assert result["diagnostics"]["inference_calls"]==0
    assert runtime.calls==[dict(client_id="test",started=started,kind="ordinary")]
    assert adapter.request_deadline is None


@pytest.mark.parametrize("mutation",["extra_request","old_schema","missing_ruleset","wrong_catalogue","private_decks"])
def test_actual_ordinary_rejects_bad_inputs_before_player(cases,mutation):
    runtime,adapter=Runtime(),ForcedPassAdapter()
    request=dict(player="search",observer=observer(cases))
    if mutation=="extra_request":request["seed"]=1
    elif mutation=="old_schema":request["observer"]["schema"]="historical"
    elif mutation=="missing_ruleset":request["observer"].pop("ruleset_id")
    elif mutation=="wrong_catalogue":request["observer"]["catalogue_sha256"]="0"*64
    else:request["observer"]["observation"]["hidden_tier_decks"]={}
    with pytest.raises((ValueError,KeyError,AssertionError)):
        q.ordinary(runtime,{"search":adapter},request,started=time.perf_counter(),client_ids={"search":"test"})
    assert adapter.calls==0


def test_outer_guard_preserves_original_start_and_full_response():
    receipts=[]; runtime=Runtime()
    value,receipt=q.timed_call(lambda started:dict(started=started),abort=runtime.abort,on_receipt=receipts.append)
    assert receipt["status"]=="ok" and value["started"]==receipt["started_perf_counter"]
    assert receipt["internal_seconds"]==9. and receipt["outer_seconds"]==10.
    assert receipts==[receipt] and not runtime.aborted


def test_outer_timeout_aborts_without_waiting_for_blocked_thread():
    release=threading.Event(); runtime=Runtime(); receipts=[]
    began=time.perf_counter()
    try:
        with pytest.raises(TimeoutError):
            q.timed_call(lambda started:release.wait(1.),abort=runtime.abort,on_receipt=receipts.append,seconds=.02)
        assert runtime.aborted and time.perf_counter()-began<.5
        assert receipts[0]["status"]=="timeout" and receipts[0]["no_retry"]
    finally:release.set()


def test_direct_qualification_inherits_outer_start_and_internal_deadline():
    runtime=Runtime(); records=[]
    class Adapter:
        client_id="direct";request_deadline=None
        def direct_choose(self,*args,**kwargs):
            assert self.request_deadline is not None
            return self.request_deadline
    adapter=Adapter(); bound=q.BoundedAdapter(adapter,runtime,"search",records.append)
    deadline=bound.direct_choose({})
    assert runtime.calls[0]["kind"]=="qualification_direct_choose"
    assert deadline==runtime.calls[0]["started"]+9.
    assert adapter.request_deadline is None and records[0]["completed"]


def test_game_initials_fixed_fresh_and_balanced():
    plan=g.build_plan()
    assert len(plan["specifications"])==8
    assert len({x["seed_decimal"] for x in plan["specifications"]})==8
    assert [x["start_player"] for x in plan["specifications"]]==[0,1]*4
    assert plan["concurrency"]==8 and plan["model_replay_maximum_per_route"]==24
    assert all(g.initial_snapshot(spec)==row["initial"] for spec,row in zip(plan["specifications"],plan["initial_states"]))
    bad=deepcopy(plan["specifications"][0]);bad["seed_decimal"]="1"
    with pytest.raises(ValueError):g.initial_snapshot(bad)


@pytest.mark.parametrize("name",["face_T1_05","face_T1_07","face_T2_10","face_T2_12","face_T2_13","face_T2_14",
                                 "noble_nonpurchase","multiple_nobles","noble_final_round","private_reserves","gold_returns"])
def test_actual_transition_independently_matches_ledger(cases,name):
    before=cases[name]["snapshot"]
    state=q.guard.decode_training_snapshot(before)
    action=q.rules.serialize_action(q.rules.CorrectedEnv().legal_actions(state)[0])
    after=g.actual_after(before,action)
    assert after==q.ledger.transition(before,action)
    q.guard.check_training_transition(before,action,after)


def test_replay_selection_first_neural_each_fixed_band():
    rows=[dict(ply=i,result=dict(diagnostics=dict(inference_calls=int(i not in (0,20,40))))) for i in range(55)]
    assert g.replay_selection(rows)==[dict(ply=1,band=[0,19]),dict(ply=21,band=[20,39]),dict(ply=41,band=[40,1999])]


def test_percentiles_and_cross_route_nonoverlap():
    assert g.percentile([1.,2.,3.,4.],.5)==2.5
    assert g.percentile([1.,2.,3.,4.],.95)==pytest.approx(3.85)
    base=dict(game_ids=[f"g{i:03d}" for i in range(8)],finished_games=8,concurrency=8,
              wall_seconds=10.,games_per_hour=2880.,decisions_per_second=12.,ended_unix=20.,started_unix=10.)
    cpu=dict(wave=dict(base,route="cpu"));gpu=dict(wave=dict(base,route="gpu",started_unix=21.,ended_unix=31.))
    assert g.compare_game_rates(cpu,gpu)["cpu_over_gpu_wall_ratio"]==1.
    gpu["wave"]["started_unix"]=19.
    with pytest.raises(ValueError,match="never overlap"):g.compare_game_rates(cpu,gpu)


def test_model_free_import_and_full_preparation():
    code="from tools.splendorust_gpu_live_20261105_02 import qualification,games; qualification.build_plan();games.build_plan();import sys;assert 'torch' not in sys.modules;assert 'torch_directml' not in sys.modules"
    result=subprocess.run([sys.executable,"-B","-c",code],cwd=q.ROOT,capture_output=True,text=True,timeout=60)
    assert result.returncode==0,result.stderr


def test_forensic_chain_rejects_reordered_or_edited_prefix(tmp_path):
    path=tmp_path/"forensics.jsonl"
    writer=q.Forensics(path); writer.record(dict(event="started"));writer.record(dict(event="result"));writer.close()
    events,_=q.read_forensics(path)
    assert events==[dict(event="started"),dict(event="result")]
    lines=path.read_text().splitlines(); first=json.loads(lines[0]);first["event"]["event"]="changed"
    path.write_text(q.canonical(first)+"\n"+lines[1]+"\n")
    with pytest.raises(ValueError,match="forensic chain"):q.read_forensics(path)
