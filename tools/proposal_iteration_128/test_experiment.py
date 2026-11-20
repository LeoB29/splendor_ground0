"""Conditional authorization, preserved provenance, and whole-run resource gates."""
from contextlib import contextmanager

import pytest

from tools.proposal_iteration_128 import experiment as p


def cost(**changes):
    kwargs = dict(game_worker_wall=1000., audit_worker_wall=100., stress_extra_wall=40.,
                  stress_roots=10, size_bytes=10*1024**2, free_bytes=100*1024**3,
                  elapsed_seconds=0., remaining_stages=1, probe_seconds=150.)
    kwargs.update(changes)
    return p.cost_projection(**kwargs)


def test_stage_a_reserves_both_stages_storage_and_time():
    one = cost()
    free = one["projected_main_bytes"]+p.CONFIG["disk_reserve_bytes"]+20*1024**2
    assert cost(free_bytes=free)["reasonable"]
    assert not cost(free_bytes=free, remaining_stages=2)["reasonable"]
    elapsed = p.CONFIG["maximum_runtime_hours"]*3600-1.5*one["forecast_seconds"]
    assert cost(elapsed_seconds=elapsed)["reasonable"]
    assert not cost(elapsed_seconds=elapsed, remaining_stages=2)["reasonable"]
    assert cost(remaining_stages=2)["remaining_forecast_hours"] > 2*one["forecast_hours"]


@pytest.mark.parametrize("changes", [dict(game_worker_wall=0), dict(stress_extra_wall=0),
                                     dict(stress_roots=0), dict(remaining_stages=3),
                                     dict(audit_worker_wall=float("nan"))])
def test_invalid_probe_cost_cannot_authorize_main(changes):
    with pytest.raises(ValueError):
        cost(**changes)


@pytest.mark.parametrize("unchanged,pass_a,pass_b,expected,label", [
    (True, True, True, [], "no_selected_model_update"),
    (False, False, True, ["probe_a", "stage_a"], "practical_progress_not_established"),
    (False, True, False, ["probe_a", "stage_a", "probe_b", "stage_b"], "progress_new_corpus_advantage_not_established"),
    (False, True, True, ["probe_a", "stage_a", "probe_b", "stage_b"], "positive_controlled_iteration"),
])
def test_complete_pipeline_gate_order(tmp_path, monkeypatch, unchanged, pass_a, pass_b, expected, label):
    monkeypatch.setattr(p, "OUTPUT", tmp_path)
    monkeypatch.setattr(p.windows, "assert_windowless_worker", lambda *a: {})
    monkeypatch.setattr(p, "validate_freeze", lambda: "bound")

    @contextmanager
    def awake():
        yield {}
    monkeypatch.setattr(p.parent, "inhibit_system_sleep", awake)
    monkeypatch.setattr(p, "prepare_learning", lambda binding: unchanged)
    built, finished = [], []
    monkeypatch.setattr(p, "build_stage", lambda stage, binding: built.append(stage))
    monkeypatch.setattr(p, "operational_decision", lambda stage, binding: True)
    monkeypatch.setattr(p, "finish_stage", lambda stage, binding: {"stage_pass": pass_a if stage == "stage_a" else pass_b})
    monkeypatch.setattr(p, "finish", lambda binding, classification, stages: finished.append((classification, stages)))
    p.run()
    assert built == expected
    assert finished == [(label, [s for s in expected if s.startswith("stage_")])]
    assert not (tmp_path / "failure.json").exists()
    with pytest.raises(ValueError, match="One-shot"):
        p.run()


def test_operational_hold_does_not_start_arena_or_publish_strength(tmp_path, monkeypatch):
    monkeypatch.setattr(p, "OUTPUT", tmp_path)
    monkeypatch.setattr(p.windows, "assert_windowless_worker", lambda *a: {})
    monkeypatch.setattr(p, "validate_freeze", lambda: "bound")
    @contextmanager
    def awake():
        yield {}
    monkeypatch.setattr(p.parent, "inhibit_system_sleep", awake)
    monkeypatch.setattr(p, "prepare_learning", lambda binding: False)
    built = []
    monkeypatch.setattr(p, "build_stage", lambda stage, binding: built.append(stage))
    monkeypatch.setattr(p, "operational_decision", lambda stage, binding: False)
    monkeypatch.setattr(p, "finish", lambda *a: pytest.fail("Hold must not finalize strength"))
    p.run()
    assert built == ["probe_a"]


def stage_a_receipts(base, passed=True):
    result = {"manifest": "bound", "stage": "stage_a", "stage_pass": passed,
              "share": .6 if passed else .5, "ci95": [.55, .65] if passed else [.45, .55],
              "all_prescribed_audits_pass": True}
    p.old.save_new(base / "stage_a_result.json", result)
    p.old.save_new(base / "stage_a_complete.json", {"manifest": "bound"})
    p.old.save_new(base / "stage_a_final_audit.json", {
        "manifest": "bound", "passed": True, "full_inner_replay": False,
        "result": p.artifact(base / "stage_a_result.json"),
        "complete": p.artifact(base / "stage_a_complete.json"),
        "independent_statistics": {"share": result["share"], "stage_pass": passed},
    })


def test_second_stage_requires_audit_bound_unmodified_success(tmp_path, monkeypatch):
    monkeypatch.setattr(p, "OUTPUT", tmp_path)
    closed = []
    monkeypatch.setattr(p, "close_stage", lambda stage, binding: closed.append(stage))
    stage_a_receipts(tmp_path)
    assert p.stage_b_authorized("bound")
    assert closed == ["stage_a"]
    result = p.old.read_json(tmp_path / "stage_a_result.json")
    result["share"] = .99
    p.old.atomic_write(tmp_path / "stage_a_result.json", result)
    with pytest.raises(ValueError, match="Changed artifact"):
        p.stage_b_authorized("bound")
    assert closed == ["stage_a"]


def test_failed_a_cannot_authorize_b_even_with_valid_hashes(tmp_path, monkeypatch):
    monkeypatch.setattr(p, "OUTPUT", tmp_path)
    monkeypatch.setattr(p, "close_stage", lambda *a: pytest.fail("Must fail before closure"))
    stage_a_receipts(tmp_path, False)
    with pytest.raises(ValueError, match="verified practical progress"):
        p.stage_b_authorized("bound")


def test_no_silent_automatic_retry_or_extra_generation():
    assert p.CONFIG["conditional_stage_b"]
    assert not any(p.CONFIG[k] for k in ("automatic_retry", "automatic_successor", "promotion", "scheduled_monitoring"))
    assert set(p.CONFIG["games"]) == {"probe_a", "stage_a", "probe_b", "stage_b", "test"}
