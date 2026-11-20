"""Fixed decisions, resource ceilings and autonomous stage boundaries."""
from contextlib import contextmanager
import copy

import pytest

from tools.proposal_breadth_learning_128 import common as c, experiment as p, verify


def cost(**changes):
    args = dict(collection_selection=1300., collection_assessment=700., collection_structural=100.,
                audit_selection=1000., audit_assessment=500., audit_structural=200.,
                probe_selection_count=10000, probe_assessment_count=5000,
                main_selection_replays=60000, main_assessment_replays=30000,
                size_bytes=20*1024**2, free_bytes=100*1024**3, elapsed_seconds=1800.)
    args.update(changes)
    return p.cost_projection(**args)


def test_forecast_includes_both_sparse_blocks_structure_and_entire_execution():
    ordinary = cost()
    assert ordinary["reasonable"] and ordinary["audit_seconds_before_allowance"] > 0
    assert cost(collection_assessment=1400.)["remaining_seconds"] > ordinary["remaining_seconds"]
    assert cost(audit_structural=400.)["remaining_seconds"] > ordinary["remaining_seconds"]
    assert not cost(elapsed_seconds=8*3600-10)["reasonable"]
    assert not cost(free_bytes=ordinary["projected_bytes"]+c.CONFIG["disk_reserve_bytes"]-1)["reasonable"]


@pytest.mark.parametrize("changes", [dict(collection_selection=0.), dict(audit_assessment=0.),
                                     dict(probe_assessment_count=0), dict(audit_structural=-1.),
                                     dict(elapsed_seconds=float("nan"))])
def test_bad_probe_never_authorizes_main(changes):
    with pytest.raises(ValueError):
        cost(**changes)


def fake_run(tmp_path, monkeypatch, gate=True):
    monkeypatch.setattr(p, "OUTPUT", tmp_path)
    monkeypatch.setattr(p.windows, "assert_windowless_worker", lambda *a: {})
    monkeypatch.setattr(p, "validate_freeze", lambda: "binding")
    monkeypatch.setattr(p, "budget_boundary", lambda *a: None)
    @contextmanager
    def awake():
        yield {"fixture": True}
    monkeypatch.setattr(p.parent.parent, "inhibit_system_sleep", awake)
    calls = []
    monkeypatch.setattr(p, "build_learning", lambda *a: calls.append("learning"))
    monkeypatch.setattr(p, "finish_if_identical", lambda *a: False)
    monkeypatch.setattr(p, "build_sources", lambda *a: calls.append("sources"))
    monkeypatch.setattr(p, "build_stage", lambda stage, *a: calls.append(stage))
    monkeypatch.setattr(p, "operational_decision", lambda *a: gate)
    monkeypatch.setattr(p, "finish", lambda *a: calls.append("finish"))
    return calls


@pytest.mark.parametrize("gate,expected", [(True, ["learning", "sources", "probe", "main", "finish"]),
                                         (False, ["learning", "sources", "probe"])])
def test_pipeline_respects_cost_gate_and_has_no_automatic_arena(tmp_path, monkeypatch, gate, expected):
    calls = fake_run(tmp_path, monkeypatch, gate)
    p.run()
    assert calls == expected and not (tmp_path/"failure.json").exists()
    with pytest.raises(ValueError, match="One-shot"):
        p.run()


def test_failure_preserves_partial_and_stops_before_main(tmp_path, monkeypatch):
    calls = fake_run(tmp_path, monkeypatch)
    def broken(*a):
        calls.append("source_failure")
        p.old.save_new(tmp_path/"partial.json", {"retained": True})
        raise RuntimeError("retained source mismatch")
    monkeypatch.setattr(p, "build_sources", broken)
    with pytest.raises(RuntimeError):
        p.run()
    assert calls == ["learning", "source_failure"]
    assert (tmp_path/"partial.json").exists()
    assert "retained source mismatch" in p.old.read_json(tmp_path/"failure.json")["traceback"]
    assert not (tmp_path/"final_status.json").exists()


def test_resource_or_source_shortage_records_hold_without_retry(tmp_path, monkeypatch):
    calls = fake_run(tmp_path, monkeypatch)
    def short(*a):
        raise c.BudgetExceeded("Fixed source pool short; no extension")
    monkeypatch.setattr(p, "build_sources", short)
    p.run()
    assert calls == ["learning"]
    assert "source pool short" in p.old.read_json(tmp_path/"operational_hold.json")["reason"]
    assert not (tmp_path/"failure.json").exists()


def test_budget_is_enforced_in_worker_boundaries(tmp_path, monkeypatch):
    c.old.save_new(tmp_path/"launch.json", {"epoch_seconds": 1})
    monkeypatch.setattr(c.time, "time", lambda: 28802.)
    with pytest.raises(c.BudgetExceeded, match="Eight-hour"):
        c.budget_boundary(tmp_path)


def test_prospective_identity_closure_skips_all_evaluation(tmp_path, monkeypatch):
    calls = fake_run(tmp_path, monkeypatch)
    monkeypatch.setattr(p, "finish_if_identical", lambda *a: calls.append("identical") or True)
    p.run()
    assert calls == ["learning", "identical"]


def rows():
    result = []
    for fit in c.FITS:
        for seat in (0, 1):
            for phase in (0, 1, 2):
                for k in range(20):
                    i = len(result)
                    result.append({"index": i, "pair": i, "game_index": i, "fit": fit, "seat": seat, "phase": phase,
                                   "arm_scores": {"anchor": .5, "broad": .5+(k%9-2)/256, "narrow": .5+(k%4-1)/256}})
    return result


def test_separate_statistics_match_real_verifier_and_reject_invalid_units():
    values = rows()
    actual, independent = verify.statistics(values), p.independent_statistics(values)
    for name in p.CONTRASTS:
        assert independent["contrasts"][name]["mean"] == actual["contrasts"][name]["mean_score_difference"]
        assert independent["contrasts"][name]["ci95"] == pytest.approx(actual["contrasts"][name]["ci95"], abs=1e-12)
    for key in ("anchor_practical_gain", "anchor_superiority", "narrow_superiority", "gate_pass", "classification"):
        assert independent[key] == actual[key]
    bad = copy.deepcopy(values)
    bad[0]["game_index"] = bad[1]["game_index"]
    with pytest.raises(ValueError, match="Repeated"):
        p.independent_statistics(bad)
    bad = copy.deepcopy(values)
    bad[0]["arm_scores"]["broad"] = .501
    with pytest.raises(ValueError, match="Score"):
        p.independent_statistics(bad)


def test_control_identity_and_zero_progress_never_pass():
    values = rows()
    for r in values:
        r["arm_scores"] = dict.fromkeys(("anchor", "broad", "narrow"), .5)
    result = p.independent_statistics(values)
    assert result["classification"] == "practical_progress_not_established"
    assert not any(result[k] for k in ("anchor_practical_gain", "anchor_superiority", "narrow_superiority", "gate_pass"))


@pytest.mark.parametrize("broad,narrow,expected", [
    (127/256, 120/256, False),  # Beating a damaged control is insufficient.
    (132/256, 133/256, False),  # Progress must also beat matched training.
    (129/256, 128/256, False),  # Positive but below the practical threshold.
    (130/256, 129/256, True),
])
def test_progress_requires_both_absolute_utility_and_attribution(broad, narrow, expected):
    values = rows()
    for row in values:
        row["arm_scores"] = {"anchor": .5, "broad": broad, "narrow": narrow}
    result = p.independent_statistics(values)
    assert result["gate_pass"] is expected


def test_status_write_failure_cannot_abort_scientific_work(tmp_path, monkeypatch):
    monkeypatch.setattr(p, "OUTPUT", tmp_path)
    monkeypatch.setattr(p, "REPORTER", None)
    def deny_replace(*args):
        raise PermissionError("Fixture advisory progress file held open")
    monkeypatch.setattr(p.os, "replace", deny_replace)
    result = p.event("fixture_status", completed=1, total=2)
    assert result["stage"] == "fixture_status"
    assert any(error["operation"] == "progress_replace" for error in p.REPORTER.last_errors)
    assert len(list((tmp_path/"status_events").glob("*.json"))) == 1
