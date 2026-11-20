"""Model-free lifecycle tests: once-only guard, bounded routes and no admission bypass."""
import json
import sys
import time
from types import SimpleNamespace

import pytest

from tools.splendorust_gpu_collector_20261105 import worker as w, common as c, backend as b


@pytest.fixture
def scope(tmp_path, monkeypatch):
    out = tmp_path/"new_study"; out.mkdir()
    monkeypatch.setattr(c, "OUT", out); monkeypatch.setattr(c, "PLAN", out/"plan.json")
    return out


def setup_worker(scope, monkeypatch, *, fail_cpu=False, expired=False):
    from tools import future_training_rules_20261021 as guard
    from tools.splendorust_gpu_collector_20261105 import qualification, collector
    import psutil
    calls = []; sources = {"source": "authenticated"}
    c.save(c.PLAN, {"test_frozen_plan": True})
    previous = scope/"accepted_identity.json"
    c.save(previous, dict(compilation_fingerprint="accepted-graph"))
    plan = dict(worker_limits=dict(qualification_seconds=3600, games_seconds=14400),
        model_seconds=c.MODEL_SECONDS, total_seconds=c.TOTAL_SECONDS, worker_affinity=list(range(19)),
        source_guard=sources, runtime=dict(source_guard=sources, ruleset_id="corrected", catalogue_sha256="catalogue",
                                         turn_contract="contract"),
        backend_source_equivalence={"proof": True}, native_capture={"capture": True},
        compiled_identity=c.artifact(previous), compilation_fingerprint="accepted-graph",
        qualification_routes=["cpu", "gpu"], inputs=[])
    c.save(scope/"claim.json", dict(plan_sha256=c.sha(c.PLAN), model_seconds=plan["model_seconds"],
        total_seconds=plan["total_seconds"], monotonic_origin=time.monotonic()-(c.MODEL_SECONDS+1 if expired else 0),
        unix_origin=time.time()))
    monkeypatch.setattr(w, "validate_contract", lambda *args: None)
    monkeypatch.setattr(w, "verify_admission", lambda *args: None)
    def source_guard(): calls.append("source_guard"); return sources
    monkeypatch.setattr(guard, "verify_runtime_sources", source_guard)
    class Process:
        def cpu_affinity(self, value=None): return list(range(19))
        def create_time(self): return 1.
    monkeypatch.setattr(psutil, "Process", Process)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(set_num_threads=lambda n: None, set_num_interop_threads=lambda n: None))
    monkeypatch.setattr(b, "source_equivalence", lambda: {"proof": True})
    monkeypatch.setattr(b, "capture_provenance", lambda: {"capture": True})
    runtimes = []
    def build(runtime_plan, route, folder, **kwargs):
        if runtimes: assert runtimes[-1].closed
        assert kwargs["runtime_sources"] == sources
        assert kwargs["implementation"] == ("eager" if route == "cpu" else "compiled")
        assert kwargs["scheduler"] == ("original8" if route == "cpu" else "row_fifo32")
        assert calls.count("source_guard") == 1
        calls.append("build_"+route); folder.mkdir()
        identity = dict(compilation_fingerprint="accepted-graph" if route == "gpu" else None)
        c.save(folder/"runtime_identity.json", identity)
        class Runtime:
            closed = False; active = {}
            service = SimpleNamespace(pending={}, clients={}, completed_requests=0, batch_count=0)
            def close(self): self.closed = True; calls.append("close_"+route)
            def abort(self): calls.append("abort_"+route)
        runtime = Runtime(); runtime.folder = folder; runtime.route = route; runtime.identity = identity
        runtimes.append(runtime); return runtime
    monkeypatch.setattr(b, "build_runtime", build)
    def qualify(plan, route, runtime, folder, boundary):
        assert not folder.exists(); boundary(); calls.append("qualify_"+route)
        if fail_cpu: raise ValueError("preserved exact failure")
        return dict(status="qualified", complete_decisions=116)
    monkeypatch.setattr(qualification, "run_qualification", qualify)
    monkeypatch.setattr(collector, "run_cell", lambda *args: pytest.fail("No games during qualification"))
    return plan, calls


def test_serial_qualification_guard_once_closes_each_route(scope, monkeypatch):
    plan, calls = setup_worker(scope, monkeypatch)
    assert w.run(plan, scope/"worker", "qualification", claim_sha=c.sha(scope/"claim.json")) == 0
    assert calls == ["source_guard", "build_cpu", "qualify_cpu", "close_cpu", "build_gpu", "qualify_gpu", "close_gpu"]
    result = c.read(scope/"worker/worker_result.json")
    assert result["status"] == "passed" and result["result"]["complete_decisions"] == 232
    assert c.read(scope/"worker/cleanup.json")["success"] is True
    assert c.read(scope/"worker/gpu_runtime/compilation_admission.json")["checked_before_requests"]


def test_first_qualification_failure_is_saved_and_never_retried(scope, monkeypatch):
    plan, calls = setup_worker(scope, monkeypatch, fail_cpu=True)
    assert w.run(plan, scope/"worker", "qualification", claim_sha=c.sha(scope/"claim.json")) == 1
    assert calls == ["source_guard", "build_cpu", "qualify_cpu", "abort_cpu", "close_cpu"]
    assert "preserved exact failure" in c.read(scope/"worker/worker_failure.json")["error"]
    assert not (scope/"worker/worker_result.json").exists()


def test_expired_original_global_clock_blocks_model_start(scope, monkeypatch):
    plan, calls = setup_worker(scope, monkeypatch, expired=True)
    assert w.run(plan, scope/"worker", "qualification", claim_sha=c.sha(scope/"claim.json")) == 1
    assert calls == ["source_guard"]
    assert "original global model cutoff" in c.read(scope/"worker/worker_failure.json")["error"]


def test_runtime_initialization_preserves_exception_and_actual_interval(scope):
    error = RuntimeError("original preparation failure")
    def operation(): raise error
    with pytest.raises(RuntimeError) as caught:
        w.initialize_runtime(operation, scope/"gpu_runtime", "gpu", "compiled")
    assert caught.value is error
    saved = c.read(scope/"initialization/gpu_runtime.json")
    assert saved["status"] == "failed" and saved["elapsed_seconds"] >= 0
    assert saved["started_perf_counter"] <= saved["ended_perf_counter"]
    assert saved["inside_collection_timing"] is False


def test_qualification_cannot_supply_premature_admission():
    with pytest.raises(ValueError, match="does not self-admit"):
        w.verify_admission({}, "qualification", "0"*64)


def test_gate_cannot_be_redirected_or_forged(scope, monkeypatch):
    from tools.splendorust_gpu_collector_20261105 import audit
    c.save(c.PLAN, {"frozen": True})
    source = scope/"qualification/saved_audit.json"
    c.save(source, dict(status="test_passed_proof", exact=True))
    expected = dict(status="collector_admitted", passed=True)
    monkeypatch.setattr(audit, "qualification_gate", lambda proof: expected if proof.get("exact") else dict(passed=False))
    path = scope/"qualification_gate.json"
    c.save(path, dict(expected, plan_sha256=c.sha(c.PLAN), source_audits=[c.artifact(source)]))
    result = w.verify_admission({}, "games", c.sha(path))
    assert result["verified_before_model_creation"] is True
    with pytest.raises(ValueError, match="controller-bound"):
        w.verify_admission({}, "games", "0"*64)
    value = c.read(path); value["source_audits"] = [c.artifact(c.PLAN)]
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="independent collector audit"):
        w.verify_admission({}, "games", c.sha(path))


def test_source_guard_call_is_single_and_precedes_model_import():
    import ast
    tree = ast.parse(__import__("inspect").getsource(w.run))
    checks = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
              and isinstance(node.func, ast.Name) and node.func.id == "verify_runtime_sources"]
    imports = [node for node in ast.walk(tree) if isinstance(node, ast.Import) and any(x.name == "torch" for x in node.names)]
    assert len(checks) == len(imports) == 1 and checks[0].lineno < imports[0].lineno
