"""Resource limits and autonomous sequencing, without production computation."""
from contextlib import contextmanager

import pytest

from tools.proposal_diagnostic_128 import experiment as p
from tools.proposal_diagnostic_128 import common as c


def cost(**changes):
    args = dict(collection_wall=2000., audit_wall=1200., replay_wall=1100.,
                probe_continuations=15000, main_replay_ceiling=80000,
                size_bytes=10*1024**2, free_bytes=100*1024**3, elapsed_seconds=900.)
    args.update(changes)
    return p.cost_projection(**args)


def test_cost_reserves_audit_finalization_and_whole_run():
    ordinary = cost()
    assert ordinary["reasonable"]
    assert ordinary["audit_seconds_before_allowance"] > 0
    assert not cost(elapsed_seconds=8*3600 - 10)["reasonable"]
    assert not cost(free_bytes=ordinary["projected_bytes"] + p.CONFIG["disk_reserve_bytes"] - 1)["reasonable"]


@pytest.mark.parametrize("changes", [dict(collection_wall=0.), dict(replay_wall=0.),
                                     dict(audit_wall=1.), dict(probe_continuations=0),
                                     dict(size_bytes=float("nan")), dict(elapsed_seconds=-1)])
def test_bad_probe_cannot_authorize_main(changes):
    with pytest.raises(ValueError):
        cost(**changes)


def install_fake_run(tmp_path, monkeypatch, gate):
    monkeypatch.setattr(p, "OUTPUT", tmp_path)
    monkeypatch.setattr(p.windows, "assert_windowless_worker", lambda *args: {})
    monkeypatch.setattr(p, "validate_freeze", lambda: "binding")
    monkeypatch.setattr(p, "budget_boundary", lambda *args: None)
    @contextmanager
    def awake():
        yield {"mock": True}
    monkeypatch.setattr(p.source.parent, "inhibit_system_sleep", awake)
    p.old.save_new(tmp_path / "selection.json", {"main": [], "probe": []})
    calls = []
    monkeypatch.setattr(p.data, "cached_analysis", lambda *args: calls.append("cache"))
    monkeypatch.setattr(p, "build_stage", lambda stage, binding: calls.append(stage))
    monkeypatch.setattr(p, "operational_decision", lambda *args: gate)
    monkeypatch.setattr(p, "finish", lambda *args: calls.append("finish"))
    return calls


@pytest.mark.parametrize("gate,expected", [(True, ["cache", "probe", "main", "finish"]),
                                         (False, ["cache", "probe"])])
def test_pipeline_has_only_operational_gate(tmp_path, monkeypatch, gate, expected):
    calls = install_fake_run(tmp_path, monkeypatch, gate)
    p.run()
    assert calls == expected
    assert not (tmp_path / "failure.json").exists()
    with pytest.raises(ValueError, match="One-shot"):
        p.run()


def test_failure_preserves_partial_without_main_or_success(tmp_path, monkeypatch):
    calls = install_fake_run(tmp_path, monkeypatch, True)
    def broken(stage, binding):
        calls.append(stage)
        raise RuntimeError("retained fixture discrepancy")
    monkeypatch.setattr(p, "build_stage", broken)
    with pytest.raises(RuntimeError):
        p.run()
    assert calls == ["cache", "probe"]
    assert "retained fixture discrepancy" in p.old.read_json(tmp_path / "failure.json")["traceback"]
    assert not (tmp_path / "final_status.json").exists()


def test_budget_limit_preserves_hold_not_completion(tmp_path, monkeypatch):
    calls = install_fake_run(tmp_path, monkeypatch, True)
    def expired(*args):
        raise p.BudgetExceeded("test eight-hour boundary")
    monkeypatch.setattr(p, "budget_boundary", expired)
    p.run()
    assert calls == ["cache"]
    assert "eight-hour" in p.old.read_json(tmp_path / "operational_hold.json")["reason"]
    assert not (tmp_path / "failure.json").exists()


def test_common_runtime_checks_are_enforced(tmp_path, monkeypatch):
    c.old.save_new(tmp_path / "launch.json", {"epoch_seconds": 1})
    monkeypatch.setattr(c.time, "time", lambda: 1 + 8*3600 + 1)
    with pytest.raises(c.BudgetExceeded, match="Eight-hour"):
        c.budget_boundary(tmp_path)


def test_fixed_panel_and_score_scale():
    rows = []
    for i in range(201):
        rows.append({"index": i, "pair": i, "fit": p.FITS[i//67], "arm_scores": {
            "anchor": .5, "treatment": .625, "replay": .375, "random": .5,
            "first_three": .25, "union": .75}})
    stats = p.independent_statistics(rows)
    assert stats["treatment_minus_anchor"] == {"mean": .125, "ci95": [.125, .125]}
    assert stats["treatment_minus_replay"]["mean"] == .25
    with pytest.raises(ValueError, match="Complete paired"):
        p.independent_statistics(rows[:-1])
    rows[0]["arm_scores"]["treatment"] = .500001
    with pytest.raises(ValueError, match="Score grid"):
        p.independent_statistics(rows)


def test_no_training_promotion_extension_or_monitor():
    assert p.CONFIG["maximum_runtime_hours"] == 8
    assert p.CONFIG["states"] == {"probe": 12, "main": 201}
    assert not any(p.CONFIG[k] for k in ("training", "automatic_retry", "promotion",
                                        "automatic_successor", "scheduled_monitoring"))
