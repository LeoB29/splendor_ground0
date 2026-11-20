"""Model-free integration checks for selection, bounded execution and evidence."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from tools.splendorust_search_scale_20261024 import common as c, runner as r, launch


def summary(quota, passed=True):
    return dict(challenger_quota=quota,status="passed" if passed else "timing_not_qualified",
                timing_qualified=passed,initialization_seconds=1.,maximum_parent_validation_seconds=.01,
                qualification_log_verify_seconds=.01)


@pytest.mark.parametrize("a,b,wanted", [(True,True,64),(True,False,32),(False,True,64),(False,False,None)])
def test_selection_is_predeclared_and_uses_only_qualification(a,b,wanted):
    assert r.select_quota([summary(32,a),summary(64,b)]) == wanted


@pytest.mark.parametrize("rows", [[],[summary(32)],[summary(64),summary(32)],
    [summary(32),summary(32)],[summary(32.),summary(64.)],
    [summary(32),summary(64)|{"timing_qualified":1}],
    [summary(32),summary(64)|{"status":"infrastructure_failed"}],
    [summary(32),summary(64)|{"timing_qualified":False}],
    [summary(32),summary(64)|{"status":"timing_not_qualified"}],
    [summary(32),{}]])
def test_selection_rejects_missing_malformed_or_inconsistent_receipts(rows):
    with pytest.raises((ValueError,KeyError,TypeError)):
        r.select_quota(rows)


def receipt(status="timeout",detail=None,deadline=1.):
    return dict(status=status,detail=detail,request_id="request-00000001",clock="perf_counter_ns",
        deadline_seconds=deadline,elapsed_seconds=1.01,receipt_elapsed_seconds=None,
        received_before_deadline=False,accepted=False)


def test_clock_failure_is_candidate_failure_only_with_full_move_clock():
    assert r.timing_failure(receipt())
    assert r.timing_failure(receipt("worker_error","ONE_SECOND_INTERNAL_DEADLINE: exhausted"))
    assert not r.timing_failure(receipt("worker_error","PLAYER_ILLEGAL_ACTION: illegal"))
    with pytest.raises(ValueError,match="stage-clamped"):
        r.timing_failure(receipt(deadline=.5))
    for status in ("resource_guard","rss_limit","process_tree_limit","affinity_changed"):
        with pytest.raises(ValueError): r.timing_failure(receipt(status))


def test_entrypoint_batch_timeout_is_infrastructure_not_candidate_clock():
    assert not r.timing_failure(receipt(deadline=300.),entrypoint=True)
    assert r.timing_failure(receipt("worker_error","ONE_SECOND_INTERNAL_DEADLINE: exhausted",300.),entrypoint=True)
    for detail in ("resource limit", "source changed", "adapter_prerequisite", "catalogue changed"):
        assert not r.timing_failure(receipt("worker_error","ONE_SECOND_INTERNAL_DEADLINE: "+detail,300.),entrypoint=True)
    assert not r.timing_failure(receipt("worker_error","unexpected checker error",300.),entrypoint=True)


def test_fresh_paired_schedule_replaces_all_prior_worlds():
    from tools.splendorust_one_second_v3_20261024 import common as old
    rows=c.make_schedule()
    assert len(rows)==206 and len({x["seed_decimal"] for x in rows})==103
    assert {x["seed_decimal"] for x in rows}.isdisjoint({x["seed_decimal"] for x in old.make_schedule()})
    for stage,n in (("operational",3),("main",100)):
        selected=[x for x in rows if x["stage"]==stage]
        for pair in range(n):
            a,b=selected[2*pair:2*pair+2]
            assert a["seed_decimal"]==b["seed_decimal"]
            assert (a["search_seat"],b["search_seat"])==(pair%2,1-pair%2)
            assert (a["game_index"],b["game_index"])==(2*pair,2*pair+1)


@pytest.fixture
def workspace(monkeypatch,tmp_path):
    monkeypatch.setattr(c,"ROOT",tmp_path); monkeypatch.setattr(c,"OUT",tmp_path/"outputs/new")
    monkeypatch.setattr(c,"PRIOR",tmp_path/"outputs/prior")
    monkeypatch.setattr(c,"PROFILE",tmp_path/"outputs/reference.json")
    monkeypatch.setattr(c,"NATIVE_COPY",c.OUT/"native/challenger_bridge.exe")
    monkeypatch.setattr(c,"PROTOCOL",tmp_path/"docs/protocol.md")
    monkeypatch.setattr(c,"WRITE_OBSERVER",None)
    c.OUT.mkdir(parents=True); c.PRIOR.mkdir(parents=True)
    def write(path,value):
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value),encoding="utf-8")
    write(c.PROTOCOL,{"protocol":"test"})
    write(tmp_path/"docs/engineering/future_training_rules_20261021.md",{"rules":"test"})
    write(tmp_path/"tests/test_local_player_qualification_clock.py",{"test":"bound"})
    original=tmp_path/"native/reference.exe"; original.parent.mkdir(); original.write_bytes(b"identical-native")
    c.NATIVE_COPY.parent.mkdir(); c.NATIVE_COPY.write_bytes(original.read_bytes())
    profile={"iterations":16,"profile_id":c.profile_id(16),"native_binary":"native/reference.exe",
             "world_pool":3,"depth":64,"cpuct":.4,"artifacts":[c.artifact(original)]}
    write(c.PROFILE,profile); monkeypatch.setattr(c,"PROFILE_SHA",c.file_hash(c.PROFILE))
    for quota in (32,64):
        derived=deepcopy(profile); derived.update(iterations=quota,profile_id=c.profile_id(quota),
            native_binary=c.NATIVE_COPY.relative_to(c.ROOT).as_posix())
        derived["artifacts"][0]["path"]=derived["native_binary"]
        write(c.profile_path(quota),derived)
    write(c.PRIOR/"freeze.json",{"inputs":[],"torch_runtime":[],"compression_runtime":{"codec":"test"}})
    write(c.OUT/"prelaunch_tests.json",{"status":"passed","source_inventory":[]})
    write(c.OUT/"prelaunch_review.json",{"status":"passed","source_inventory":[]})
    for package in ("splendorust_one_second_v3_20261024","splendorust_search_scale_20261024"):
        p=tmp_path/"tools"/package/"compression.py"; p.parent.mkdir(parents=True); p.write_text("same codec")
    return write


def prepare_mocks(monkeypatch):
    from tools.splendorust_search_scale_20261024 import qualification as q, compression
    from tools.matched_strength_20261020 import statistics
    monkeypatch.setattr(r,"verify_runtime_sources",lambda:{"guard":"passed"})
    monkeypatch.setattr(r,"preserve_prior_attempts",lambda:{"attempts":[]})
    monkeypatch.setattr(r,"seed_inventory",lambda:{"status":"passed","collisions":[]})
    monkeypatch.setattr(r,"verify_runtime",lambda *a:None)
    monkeypatch.setattr(r,"processor_topology",lambda:{"selected_logical_processor":0})
    monkeypatch.setattr(r.shutil,"disk_usage",lambda p:SimpleNamespace(free=10*c.GiB))
    monkeypatch.setattr(q,"build_plan",lambda q:{"quota":q})
    monkeypatch.setattr(q,"model_free_checks",lambda q:{"status":"passed","quota":q})
    monkeypatch.setattr(compression,"runtime_contract",lambda:{"codec":"test"})
    monkeypatch.setattr(statistics,"runtime_contract",lambda:{"numpy":"test"})


def test_prepare_binds_exact_profiles_and_unchanged_limits(workspace,monkeypatch):
    prepare_mocks(monkeypatch); r.prepare()
    freeze=c.read(c.OUT/"freeze.json")
    assert freeze["seconds_per_move"]==1 and freeze["internal_seconds"]==.9 and freeze["margin_seconds"]==.7
    assert freeze["job_seconds"]==21600 and freeze["collection_cutoff_seconds"]==19800
    assert freeze["stage_seconds"]=={"qualification":3600,"operational":1800,"main":14400,"final":1800}
    assert freeze["automatic_retries"]==0 and freeze["larger_data_deferred"]
    assert freeze["calibration_quotas"]==[32,64] and freeze["max_evidence_bytes"]==512*c.MiB
    assert {a["path"] for a in freeze["native_artifacts"]}=={"native/reference.exe","outputs/new/native/challenger_bridge.exe"}
    assert c.artifact(c.profile_path(64)) in freeze["inputs"]


@pytest.mark.parametrize("field,value",[("world_pool",4),("depth",128),("cpuct",.5),("iterations",128)])
def test_prepare_rejects_any_additional_search_change(workspace,monkeypatch,field,value):
    prepare_mocks(monkeypatch)
    profile=c.profile_for(64); profile[field]=value; workspace(c.profile_path(64),profile)
    with pytest.raises(ValueError,match="more than declared"):
        r.prepare()
    assert not (c.OUT/"freeze.json").exists()


class Envelope:
    def __init__(self): self.start=0.; self.wall_start=1000.; self.guards=0; self.folders=[]
    def guard(self): self.guards+=1
    def admit(self): pass
    def register_worker(self,p): self.folders.append(p)
    def close_worker(self,p): pass
    def timeout(self,maximum): return maximum
    def effective_deadline(self): return 10000.


def test_worker_receives_frozen_quota_and_metadata_validation(workspace,monkeypatch):
    import sys
    from tools.splendorust_search_scale_20261024 import adapters
    monkeypatch.delitem(sys.modules,"torch",raising=False)
    calls=[]; validations=[]
    class Worker:
        def __init__(self,*args,**kwargs):
            calls.append((args,kwargs)); self.initialization={"result":{"opaque":"identity"}}
            self.process=SimpleNamespace(pid=42); self._failure=None
        def _sample(self): return {"process_tree":[{"pid":42}]}
        def close(self): self.closed=True
    monkeypatch.setattr(c,"StudyWorker",Worker)
    monkeypatch.setattr(adapters,"validate_ready_metadata",lambda *a,**k:validations.append((a,k)))
    envelope=Envelope(); freeze={"native_artifacts":["bound"],"topology":{"selected_logical_processor":4}}
    child=r.make_worker("example",envelope,freeze,64)
    args,kw=calls[0]
    assert args[0]=="tools.splendorust_search_scale_20261024.worker"
    assert kw["env"]["STUDY_SEARCH_SIMULATIONS"]=="64" and kw["env"]["CUDA_VISIBLE_DEVICES"]==""
    assert kw["affinity"]==[4] and kw["max_rss_bytes"]==2*c.GiB and kw["max_log_bytes"]==16*c.MiB
    assert validations[0][1]["expected_quota"]==64 and validations[0][1]["worker_pid"]==42
    assert child.challenger_quota==64 and envelope.folders


def test_comparison_freeze_seals_all_calibration_evidence(workspace,monkeypatch):
    from tools.splendorust_search_scale_20261024 import statistics
    workspace(c.OUT/"freeze.json",{"source":"frozen"}); workspace(c.OUT/"schedule.json",c.make_schedule())
    def qualified(envelope,freeze,quota):
        workspace(c.OUT/"workers"/f"qualification_{quota}"/"trace.json",{"quota":quota})
        workspace(c.OUT/f"qualification_{quota}"/"000.json",{"quota":quota})
        workspace(c.OUT/f"entrypoint_checks_{quota}.json",{"status":"passed"})
        workspace(c.OUT/f"qualification_summary_{quota}.json",summary(quota))
        workspace(c.OUT/f"qualification_log_integrity_{quota}.json",{"status":"passed"})
        return summary(quota)
    monkeypatch.setattr(r,"qualify_one",qualified); monkeypatch.setattr(r,"verify_runtime",lambda *a:None)
    monkeypatch.setattr(statistics,"analyze_games",lambda *a,**k:{"synthetic":True})
    monkeypatch.setattr(statistics,"verify_analysis",lambda *a,**k:None)
    quota,overhead=r.qualify(Envelope(),{})
    assert quota==64 and overhead>30
    freeze=c.read(c.OUT/"comparison_freeze.json")
    assert freeze["challenger_simulations"]==64 and freeze["reference_simulations"]==16
    assert freeze["source_freeze"]==c.artifact(c.OUT/"freeze.json")
    assert not freeze["automatic_fallback"] and not freeze["clocks_reset"]
    bound={x["path"] for x in freeze["calibration_evidence"]}
    assert {"outputs/new/workers/qualification_32/trace.json","outputs/new/workers/qualification_64/trace.json",
            "outputs/new/timing_selection.json"}.issubset(bound)
    workspace(c.OUT/"workers/qualification_32/trace.json",{"changed":True})
    with pytest.raises(ValueError,match="Changed frozen input"):
        c.verify(freeze["calibration_evidence"])


def test_final_replay_uses_frozen_quota(workspace,monkeypatch):
    from tools.splendorust_search_scale_20261024 import audit
    spec={"stage":"main","pair_index":0,"game_index":0,"search_seat":0,"seed_decimal":"123"}
    monkeypatch.setattr(c,"make_schedule",lambda:[spec])
    snapshot=r.rules.serialize_snapshot(r.rules.CorrectedEnv(seed=123).initial_state())
    frozen={"challenger_simulations":64,"source_freeze":{"a":1},"challenger_profile":{"b":2},
            "reference_profile":{"c":3},"schedule":{"d":4},"calibration_evidence":[{"e":5}]}
    workspace(c.OUT/"comparison_freeze.json",frozen)
    selected=spec|{"challenger_simulations":64}; folder=r.game_folder(selected)
    workspace(folder/"initial.json",selected|{"initial":snapshot})
    workspace(folder/"result.json",selected|{"plies":0})
    verified=[]; monkeypatch.setattr(c,"verify",lambda items:verified.append(items))
    class ReachedReplay(Exception): pass
    def replay(initial,rows,game,**kw):
        assert initial==snapshot and rows==[] and kw["expected_identity"]=={
            "stage":"main","pair_index":0,"game_index":0,"challenger_simulations":64}
        raise ReachedReplay
    monkeypatch.setattr(audit,"audit_game",replay)
    envelope=Envelope(); envelope.comparison_freeze=c.artifact(c.OUT/"comparison_freeze.json")
    with pytest.raises(ReachedReplay): r.final_checks(envelope,{},[selected])
    assert verified[0]==[envelope.comparison_freeze]
    assert verified[1]==[frozen[k] for k in ("source_freeze","challenger_profile","reference_profile","schedule")]+frozen["calibration_evidence"]


@pytest.mark.parametrize("failure",[None,"collection","storage","final","margin","forfeit"])
def test_operational_gate_keeps_fixed_resource_reserves(workspace,monkeypatch,failure):
    clock=[0.]; used=[100*c.MiB]; calls=[]
    monkeypatch.setattr(r.time,"perf_counter",lambda:clock[0])
    monkeypatch.setattr(r.time,"monotonic",lambda:100.)
    monkeypatch.setattr(r.time,"time",lambda:1100.)
    monkeypatch.setattr(r,"bytes_used",lambda:used[0])
    def play(spec,*args):
        calls.append(spec); clock[0]+=100 if failure=="collection" else 1
        used[0]+=(2 if failure=="storage" else .01)*c.MiB
        return ({"reason":"forfeit" if failure=="forfeit" else "natural"},
            [{"clock":{"receipt_elapsed_seconds":.71 if failure=="margin" else .6}}],
            10. if failure=="final" else .1)
    monkeypatch.setattr(r,"play_game",play)
    schedule=[x|{"challenger_simulations":64} for x in c.make_schedule()]
    if failure:
        with pytest.raises(ValueError): r.operational(Envelope(),{},schedule,100.)
    else:
        r.operational(Envelope(),{},schedule,100.)
        gate=c.read(c.OUT/"operational_gate.json")
        assert gate["passed"] and len(calls)==6 and gate["arithmetic_factor"]==pytest.approx(45)
        assert gate["outcomes_used"] is False and gate["remaining_collection_seconds"]==14400
        assert gate["projected_prefinal_bytes"]<448*c.MiB and gate["projected_final_seconds"]<1800


def test_watchdog_rejects_reused_pid_and_never_extends_deadline(monkeypatch):
    class Child:
        pid=11; killed=False; waits=[]
        def create_time(self): return 2.
        def kill(self): self.killed=True
        def wait(self,timeout): self.waits.append(timeout)
    child=Child(); monkeypatch.setattr(launch.psutil,"Process",lambda pid:child)
    handle=SimpleNamespace(pid=11,poll=lambda:0,returncode=0,wait=lambda timeout:0)
    result=launch.cleanup_owned(handle,None,(),monotonic_origin=launch.time.monotonic()-c.JOB_SECONDS-1,
        unix_origin=launch.time.time(),root_identity={"pid":11,"created":1.})
    assert result["errors"] and not child.killed and result["effective_allowance_seconds"]==0


@pytest.mark.parametrize("quota,fail_main,wanted",[(None,False,"no_qualifying_challenger"),
    (32,False,"completed_verified"),(64,False,"completed_verified"),(64,True,"incomplete")])
def test_driver_propagates_selected_quota_through_all_stages(workspace,monkeypatch,quota,fail_main,wanted):
    workspace(c.OUT/"freeze.json",{"source":"frozen"})
    sha=c.file_hash(c.OUT/"freeze.json")
    workspace(c.OUT/"watchdog_claim.json",{"freeze_sha256":sha,"monotonic_origin":0.,"unix_origin":1000.})
    workspace(c.OUT/"schedule.json",c.make_schedule())
    stages=[]; games=[]; seen={"operational":0,"final":0}
    class DriverEnvelope(Envelope):
        def __init__(self,claim): super().__init__(); self.stage=None
        def enable_evidence_accounting(self): pass
        def begin(self,stage): self.stage=stage; stages.append(stage)
        def finish(self): pass
    monkeypatch.setattr(r,"Envelope",DriverEnvelope)
    monkeypatch.setattr(r.psutil,"Process",lambda:SimpleNamespace(create_time=lambda:1.))
    monkeypatch.setattr(r,"verify_runtime",lambda *a:None)
    monkeypatch.setattr(r,"verify_runtime_sources",lambda:{"guard":"passed"})
    def qualify(envelope,freeze):
        if quota is not None:
            workspace(c.OUT/"comparison_freeze.json",{"challenger_simulations":quota})
            envelope.comparison_freeze=c.artifact(c.OUT/"comparison_freeze.json")
        return quota,None if quota is None else 100.
    monkeypatch.setattr(r,"qualify",qualify)
    def operational(envelope,freeze,schedule,overhead):
        seen["operational"]+=1
        assert len(schedule)==206 and all(s["challenger_simulations"]==quota for s in schedule)
    def play(spec,*args):
        assert spec["stage"]=="main" and spec["challenger_simulations"]==quota
        games.append(spec)
        if fail_main and len(games)==3: raise ValueError("preserved injected infrastructure failure")
    def final(envelope,freeze,schedule):
        seen["final"]+=1
        assert len(games)==200 and all(s["challenger_simulations"]==quota for s in schedule)
        return {"status":"verified"}
    monkeypatch.setattr(r,"operational",operational); monkeypatch.setattr(r,"play_game",play)
    monkeypatch.setattr(r,"final_checks",final)
    exit_code=r.run(sha)
    result=c.read(c.OUT/"completion.json")
    assert result["status"]==wanted and result["automatic_retries"]==0 and result["larger_data_deferred"]
    assert exit_code==(1 if fail_main else 0)
    if quota is None:
        assert stages==["qualification"] and not games and seen=={"operational":0,"final":0}
    elif fail_main:
        assert len(games)==3 and seen=={"operational":1,"final":0}
        assert "injected infrastructure" in c.read(c.OUT/"failure.json")["error"]
    else:
        assert stages==["qualification","operational","main","final"] and len(games)==200
        assert [g["game_index"] for g in games]==list(range(200))
        assert seen=={"operational":1,"final":1}


@pytest.mark.parametrize("detail,timing",[("ONE_SECOND_INTERNAL_DEADLINE: exhausted",True),
    ("STUDY_RESOURCE: RAM unavailable",False),("SEARCH_SCALE_ADAPTER: source changed",False)])
def test_entrypoint_failure_is_preserved_and_only_timing_continues(workspace,monkeypatch,detail,timing):
    from tools.splendorust_search_scale_20261024 import log_audit
    workspace(c.OUT/"qualification_plan_64.json",{"timed_requests":[]})
    failure=receipt("worker_error",detail,300.)
    class Worker:
        output_dir=c.OUT/"workers/qualification_64"
        initialization={"elapsed_seconds":1.}
        def __enter__(self): return self
        def __exit__(self,*a): self.closed=True
        def request(self,*a,**kw): return failure
    child=Worker(); monkeypatch.setattr(r,"make_worker",lambda *a:child)
    monkeypatch.setattr(log_audit,"verify_worker_logs",lambda *a,**kw:{"status":"passed"})
    if timing:
        result=r.qualify_one(Envelope(),{},64)
        assert result["status"]=="timing_not_qualified" and result["requests"]==0
        assert result["failure"]==failure and not result["timing_qualified"]
    else:
        with pytest.raises(ValueError,match="entrypoint checks failed"):
            r.qualify_one(Envelope(),{},64)
        assert not (c.OUT/"qualification_summary_64.json").exists()
    assert child.closed and c.read(c.OUT/"entrypoint_checks_64.json")==failure
