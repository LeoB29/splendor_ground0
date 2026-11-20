"""Model-free changed-clock entrypoint regressions; no native/model launch."""
from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest

from tools.splendorust_gpu_throughput_20261105 import adapters as a, backend as b
from tools.splendorust_gpu_throughput_20261105 import common as c, qualification as q, scientific as s

PRIOR = c.ROOT / "outputs/splendorust_gpu_concurrency_20261105_01"


def accepted_case():
    folder = PRIOR / "cells/cpu_c1/worker/cell/g000"
    return q.read(folder / "request.json"), q.read(folder / "result.json")


def new_result():
    request, result = accepted_case()
    result["wrapper_id"] = s.wrapper_id(512)
    return request, result


def test_explicit_identity_projection_changes_only_wrapper_and_rejects_other_domains():
    request, old = accepted_case()
    raw = deepcopy(old); raw["wrapper_id"] = s.wrapper_id(512)
    assert s.normalize_result(raw) == old
    assert raw["wrapper_id"] != old["wrapper_id"]
    assert s.validate_result(raw, request["observer"], "search", expected_quota=512)
    assert s.validate_legacy_result(old, request["observer"], "search", expected_quota=512)
    for bad in (old, dict(raw, wrapper_id="invented"), dict(raw, profile_sha256="0"*64),
                dict(raw, execution_policy_id=c.EXECUTION_POLICY_ID)):
        with pytest.raises(ValueError):
            s.normalize_result(bad)
    with pytest.raises(ValueError):
        s.validate_legacy_result(raw, request["observer"], "search", expected_quota=512)


@pytest.mark.parametrize("finish,allowed", [(112., True), (159.9, True), (160., False)])
def test_service_accepts_late_old9s_result_but_preserves60s_expiry(monkeypatch, finish, allowed):
    clock = [100.]
    monkeypatch.setattr(b, "time", SimpleNamespace(perf_counter=lambda: clock[0]))
    def forward(tokens, batch):
        clock[0] = finish
        return [[.25]*83 for _ in range(batch)]
    service = b.InferenceService(forward, "cpu"); service.register("one", 0)
    try:
        if allowed:
            outputs, receipt = service.infer("one", [0.]*b.WIDTH, deadline=160.)
            assert outputs == [[.25]*83]
            assert receipt["deadline_perf"] == 160.
        else:
            with pytest.raises(TimeoutError, match="Completed inference"):
                service.infer("one", [0.]*b.WIDTH, deadline=160.)
            assert service.failure is not None
    finally:
        service.close()


def bare_runtime():
    runtime = b.Runtime.__new__(b.Runtime)
    runtime.lock = threading.RLock(); runtime.active = {}; runtime.operation_counter = 0
    runtime._publish_active = lambda: None
    return runtime


@pytest.mark.parametrize("expiry", ["perf", "unix", "publication"])
def test_nested75s_original_origin_and_postpublication_expiry(monkeypatch, expiry):
    clock = [100., 1000.]
    monkeypatch.setattr(b, "time", SimpleNamespace(perf_counter=lambda: clock[0], time=lambda: clock[1]))
    runtime = bare_runtime()
    with pytest.raises(TimeoutError, match="dual-clock75s"):
        with runtime.operation("one", started=90., kind="ordinary") as outer:
            assert outer["deadline_perf"] == 165.
            assert outer["deadline_unix"] == 1065.
            assert outer["execution_policy_id"] == c.EXECUTION_POLICY_ID
            with runtime.operation("one", kind="native_exchange") as inner:
                assert inner is outer
            if expiry == "perf": clock[0] = 165.
            elif expiry == "unix": clock[1] = 1065.
            else:
                runtime._publish_active = lambda: clock.__setitem__(0, 165.)
    assert not runtime.active


def test_native_orchestration_uses_original_active_deadline_without_profile_mutation(monkeypatch):
    request, expected = accepted_case()
    proxy = a.ProxySearchOpponent.__new__(a.ProxySearchOpponent)
    proxy.profile = q.original.profile(512, route="candidate")
    before = deepcopy(proxy.profile); proxy.active_deadline = 160.
    monkeypatch.setattr(a, "time", SimpleNamespace(perf_counter=lambda: 115.))
    phases = iter(expected["diagnostics"]["phases"])
    def exchange(native_request):
        assert proxy.deadline == proxy.active_deadline == 160.
        assert native_request["iterations"] == 512
        row = deepcopy(next(phases))
        proxy.inference_calls += row["neural_callback_count"]
        proxy.inference_rows += row["neural_callback_rows"]
        return dict(kind="result", **row)
    proxy.exchange = exchange
    action, trace = proxy._choose_with_original_deadline(request["observer"])
    assert q.rules.serialize_action(action) == expected["action"]
    assert trace == expected["diagnostics"]
    assert proxy.profile == before and proxy.profile["internal_budget_seconds"] == 9.
    assert proxy.deadline is None


def test_complete_ordinary_accepts15s_and_retains_new_raw_metadata(monkeypatch):
    request, expected = new_result()
    clock = [100.]; origins = []; raw = []
    monkeypatch.setattr(q, "time", SimpleNamespace(perf_counter=lambda: clock[0]))
    @contextmanager
    def operation(client, **kwargs):
        origins.append(kwargs["started"]); yield
    adapter = SimpleNamespace(profile=q.original.profile(512, route="candidate"), request_deadline=None)
    def choose(observer, **kwargs):
        assert adapter.request_deadline == 160.
        clock[0] = 115.
        return q.rules.deserialize_action(expected["action"]), deepcopy(expected["diagnostics"])
    adapter.choose = choose
    result = q.ordinary(SimpleNamespace(operation=operation), {"search":adapter}, request,
                        started=100., client_ids={"search":"one"}, evidence_callback=raw.append)
    assert result == expected and raw[0]["result"] == expected
    assert origins == [100.] and adapter.request_deadline is None


def test_timed_call_accepts_old_late_response_and_receipts_effective_policy():
    clock = [100.]; receipts = []
    def operation(started):
        assert started == 100.
        clock[0] = 115.
        return "complete"
    result, receipt = q.timed_call(operation, abort=lambda: pytest.fail("no abort"),
                                   on_receipt=receipts.append, clock=lambda: clock[0])
    assert result == "complete" and receipt["elapsed_seconds"] == 15.
    assert receipt["internal_seconds"] == 60. and receipt["outer_seconds"] == 75.
    assert receipt["execution_policy_id"] == c.EXECUTION_POLICY_ID and receipts == [receipt]


def test_accepted_math_packing_source_equivalence_and67_only_plan():
    plan = q.build_plan()
    assert plan["source_equivalence"]["status"] == "source_equivalence_verified"
    assert plan["required_full_suite_decisions_per_route"] == 67
    assert plan["fixed_root_decisions_per_route"] == 0 and plan["decision_panel"] == []
    assert plan["execution_policy_id"] == c.EXECUTION_POLICY_ID
    assert len(plan["reused_evidence"]) == 5


def test_saved67_auditor_binds_raw_new_response_to_only_wrapper_projection(tmp_path):
    """Historical receipts are an explicitly synthetic test fixture, not new evidence."""
    prior = PRIOR / "qualification/worker/qualification_cpu"
    suite = q.read(prior / "entrypoint_suite.json")
    old_events, _ = q.read_forensics(prior / "entrypoint_forensics.jsonl")
    events = []
    for original_event in old_events:
        event = deepcopy(original_event)
        if event.get("operation") in ("outer_clock", "outer_ordinary_clock"):
            event.update(internal_seconds=60., outer_seconds=75., execution_policy_id=c.EXECUTION_POLICY_ID)
        if event.get("event") == "complete_result_before_worker_validation":
            event["result"]["wrapper_id"] = s.wrapper_id(event["result"]["simulations_quota"])
            events.append(event)
            events.append(dict(event="explicit_scientific_wrapper_projection",execution_policy_id=c.EXECUTION_POLICY_ID,
                               changed_fields=["wrapper_id"],raw_result_sha256=q.digest(event["result"]),
                               projection_sha256=q.digest(s.normalize_result(event["result"]))))
        else:
            events.append(event)
    plan = {"qualification": q.build_plan()}
    summary = dict(status="actual_entrypoint_qualified", route="cpu",suite_sha256=q.digest(suite),
                   execution_policy_id=c.EXECUTION_POLICY_ID,fixed_root_decisions=0,
                   fixed_roots_sha256=q.digest([]),elapsed_root_seconds=[],
                   reused_evidence=plan["qualification"]["reused_evidence"],
                   source_equivalence=plan["qualification"]["source_equivalence"])
    (tmp_path/"qualification.json").write_text(q.canonical(summary),encoding="utf-8")
    (tmp_path/"entrypoint_suite.json").write_text(q.canonical(suite),encoding="utf-8")
    def write_events():
        previous = None
        with (tmp_path/"entrypoint_forensics.jsonl").open("w",encoding="utf-8") as stream:
            for index,event in enumerate(events):
                row=dict(sequence=index,previous_sha256=previous,event=event)
                stream.write(q.canonical(row)+"\n"); previous=q.digest(row)
    write_events()
    audit = q.verify_qualification(plan,tmp_path)
    assert audit["actual_decisions"] == 67 and audit["raw_new_results"] == 37
    next(event for event in events if event.get("event")=="explicit_scientific_wrapper_projection")["changed_fields"] = ["wrapper_id","action"]
    write_events()
    with pytest.raises(ValueError,match="wrapper-only projection"):
        q.verify_qualification(plan,tmp_path)
