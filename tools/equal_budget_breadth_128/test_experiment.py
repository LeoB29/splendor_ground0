"""Primary-only inference, operational ceilings, durable pipeline and coverage."""
from contextlib import contextmanager
import copy
from types import SimpleNamespace

import pytest

from tools.equal_budget_breadth_128 import common as c, experiment as p, verify


def rows():
    result = []
    for fit in c.FITS:
        for seat in (0, 1):
            for phase in (0, 1, 2):
                for k in range(20):
                    i = len(result)
                    result.append({"index": i, "pair": i, "game_index": i, "fit": fit, "seat": seat, "phase": phase,
                        "arm_scores": {"anchor": .5 + (k % 3 - 1) / 256,
                                       "uniform": .5 + (k % 9 - 2) / 256,
                                       "teacher": .5 + (k % 4 - 1) / 256}})
    return result


def test_independent_statistics_agree_with_verifier_on_nonconstant_panel():
    values = rows()
    expected, actual = verify.statistics(values), p.independent_statistics(values)
    for name in p.CONTRASTS:
        assert actual["contrasts"][name]["mean"] == expected["contrasts"][name]["mean_score_difference"]
        assert actual["contrasts"][name]["ci95"] == pytest.approx(expected["contrasts"][name]["ci95"], abs=1e-12)
    for key in ("anchor_practical_gain", "anchor_superiority", "gate_pass", "classification"):
        assert actual[key] == expected[key]
    assert p.independent_statistics(list(reversed(values))) == actual


@pytest.mark.parametrize("uniform,teacher,expected", [
    (0, 4, (False, False, False)), (1, -4, (False, True, False)),
    (2, 2, (True, True, True)), (2, 4, (True, True, True)), (2, -4, (True, True, True)),
])
def test_teacher_comparison_neither_rescues_nor_vetoes_primary(uniform, teacher, expected):
    values = rows()
    for row in values:
        row["arm_scores"] = {"anchor": .5, "uniform": .5 + uniform / 256, "teacher": .5 + teacher / 256}
    result = p.independent_statistics(values)
    assert tuple(result[k] for k in ("anchor_practical_gain", "anchor_superiority", "gate_pass")) == expected
    assert result["classification"] == ("positive_conditional_equal_budget_breadth" if expected[-1] else "practical_progress_not_established")


@pytest.mark.parametrize("extra,passes", [(100, False), (101, True)])
def test_exact_aggregate_half_percentage_point_boundary(extra, passes):
    values = rows()
    for row in values:
        row["arm_scores"] = {"anchor": .5, "uniform": .5 + (1 + int(row["index"] < extra)) / 256, "teacher": .9}
        row["arm_scores"]["teacher"] = 230 / 256
    result = p.independent_statistics(values)
    assert result["contrasts"]["uniform_minus_anchor"]["mean"] == (360 + extra) / (360 * 256)
    assert result["anchor_practical_gain"] is passes and result["anchor_superiority"]


@pytest.mark.parametrize("damage", ["duplicate_game", "missing_row", "changed_cell", "wrong_units", "extra_arm"])
def test_invalid_statistical_population_rejected(damage):
    values = rows()
    if damage == "duplicate_game": values[0]["game_index"] = values[1]["game_index"]
    elif damage == "missing_row": values.pop()
    elif damage == "changed_cell": values[0]["phase"] = 1
    elif damage == "wrong_units": values[0]["arm_scores"]["uniform"] = .501
    else: values[0]["arm_scores"]["other"] = .5
    with pytest.raises(ValueError): p.independent_statistics(values)


def cost_arguments():
    return {"collection": {"selection": 184.32, "assessment": 207.36},
            "replay": {"selection": 147.456, "assessment": 165.888},
            "probe_counts": {"selection": 18432, "assessment": 6912},
            "main_counts": {"selection": 368640, "assessment": 138240},
            "main_replays": {"selection": 69120, "assessment": 25920},
            "collection_structural": 100., "audit_structural": 200.,
            "size_bytes": 20 * 1024**2, "free_bytes": 100 * 1024**3, "elapsed_seconds": 1800.}


def test_forecast_covers_both_blocks_structure_and_total_budget():
    args = cost_arguments()
    value = p.cost_projection(**args)
    collection = (368640 * .01 + 138240 * .03 + 20 * 100) / 8
    replay = (69120 * .008 + 25920 * .024 + 20 * 200) / 8
    assert value["reasonable"]
    assert value["collection_seconds_before_allowance"] == pytest.approx(collection)
    assert value["audit_seconds_before_allowance"] == pytest.approx(replay)
    assert value["remaining_seconds"] == pytest.approx(1.35 * (collection + replay) + 300)
    assert not p.cost_projection(**dict(args, elapsed_seconds=8 * 3600 - 1))["reasonable"]
    assert not p.cost_projection(**dict(args, free_bytes=value["projected_bytes"] + c.CONFIG["disk_reserve_bytes"] - 1))["reasonable"]
    for block in p.BLOCKS:
        changed = copy.deepcopy(args)
        changed["replay"][block] *= 2
        assert p.cost_projection(**changed)["remaining_seconds"] > value["remaining_seconds"]


@pytest.mark.parametrize("damage", ["negative", "nan", "missing", "unmeasured"])
def test_invalid_or_unmeasured_probe_cannot_authorize_main(damage):
    args = cost_arguments()
    if damage == "negative": args["audit_structural"] = -1
    elif damage == "nan": args["collection"]["selection"] = float("nan")
    elif damage == "missing": args["replay"].pop("assessment")
    else:
        args["collection"]["assessment"] = 0
        assert not p.cost_projection(**args)["reasonable"]
        return
    with pytest.raises(ValueError): p.cost_projection(**args)


def coverage_fixture(support=5, stress=False):
    lengths, supports = p.geometry(support)
    spec = {"index": 0, "canonical_support": support}
    ordinary = {b: verify.ordinary_indices("test", 0, b, lengths[b]) for b in p.BLOCKS}
    replay = {b: list(range(lengths[b])) if stress else ordinary[b] for b in p.BLOCKS}
    sample = {"stress": stress, "ordinary": ordinary, "replay": replay}
    item = {"path": "fixture/sample.json", "sha256": "sample"}
    root = {"path": "fixture/root.json", "sha256": "root", "index": 0,
            "continuations": sum(lengths[b] * supports[b] for b in p.BLOCKS)}
    receipt = {"manifest": "fixture", "passed": True, "spec": spec,
               "root": {k: root[k] for k in ("path", "sha256")}, "sample_artifact": item,
               "stress": stress, "ordinary_world_indices": ordinary, "replay_world_indices": replay,
               "full_inner_replay": stress}
    for b in p.BLOCKS:
        receipt[f"support_{b}"] = supports[b]
        for kind, count in (("recorded", lengths[b]), ("replayed", len(replay[b]))):
            receipt[f"{kind}_{b}_worlds"] = count
            receipt[f"{kind}_{b}_continuations"] = count * supports[b]
    for kind in ("recorded", "replayed"):
        for unit in ("worlds", "continuations"):
            receipt[f"{kind}_{unit}"] = sum(receipt[f"{kind}_{b}_{unit}"] for b in p.BLOCKS)
    return receipt, spec, root, sample, item, "fixture"


@pytest.mark.parametrize("support", [1, 2, 3, 4, 5, 6, 7, 8, 40])
def test_geometry_and_receipts_cover_all_supports(support):
    lengths, counts = p.geometry(support)
    assert lengths == {"selection": 128, "assessment": 128}
    assert counts == {"selection": min(support, 8), "assessment": min(support, 3)}
    for stress in (False, True): p.validate_coverage(*coverage_fixture(support, stress))


@pytest.mark.parametrize("damage", ["link", "sample", "worlds", "work", "support", "total", "root", "disclosure"])
def test_corrupt_coverage_receipt_rejected(damage):
    args = copy.deepcopy(coverage_fixture())
    receipt = args[0]
    if damage == "link": receipt["root"]["sha256"] = "wrong"
    elif damage == "sample": receipt["replay_world_indices"] = {**receipt["replay_world_indices"], "selection": [0]}
    elif damage == "worlds": receipt["recorded_selection_worlds"] -= 1
    elif damage == "work": receipt["replayed_selection_continuations"] += 1
    elif damage == "support": receipt["support_assessment"] = 4
    elif damage == "total": receipt["recorded_continuations"] += 1
    elif damage == "root": args[2]["continuations"] += 1
    else: receipt["full_inner_replay"] = True
    with pytest.raises(ValueError): p.validate_coverage(*args)


def fake_run(tmp_path, monkeypatch, gate=True):
    monkeypatch.setattr(p, "OUTPUT", tmp_path)
    monkeypatch.setattr(p, "REPORTER", None)
    monkeypatch.setattr(p.windows, "assert_windowless_worker", lambda *a: {})
    monkeypatch.setattr(p, "validate_freeze", lambda: "fixture")
    monkeypatch.setattr(p, "budget_boundary", lambda *a: None)
    @contextmanager
    def awake(): yield {"fixture": True}
    monkeypatch.setattr(p.parent.parent, "inhibit_system_sleep", awake)
    calls = []
    monkeypatch.setattr(p.models, "prepare", lambda *a: calls.append("models"))
    monkeypatch.setattr(p, "build_sources", lambda *a: calls.append("sources"))
    monkeypatch.setattr(p, "build_stage", lambda stage, *a: calls.append(stage))
    monkeypatch.setattr(p, "operational_decision", lambda *a: gate)
    monkeypatch.setattr(p, "finish", lambda *a: calls.append("finish"))
    return calls


@pytest.mark.parametrize("gate", [True, False])
def test_pipeline_operational_gate_and_no_successor(tmp_path, monkeypatch, gate):
    calls = fake_run(tmp_path, monkeypatch, gate)
    p.run()
    assert calls == (["models", "sources", "probe", "main", "finish"] if gate else ["models", "sources", "probe"])
    with pytest.raises(ValueError, match="One-shot"): p.run()


def test_budget_stops_preserve_partial_evidence(tmp_path, monkeypatch):
    calls = fake_run(tmp_path, monkeypatch)
    def shortage(*a):
        p.old.save_new(tmp_path / "retained_partial.json", {"preserved": True})
        raise c.BudgetExceeded("Fixed source pool short; no extension")
    monkeypatch.setattr(p, "build_sources", shortage)
    p.run()
    assert calls == ["models"] and (tmp_path / "retained_partial.json").exists()
    assert "source pool short" in p.old.read_json(tmp_path / "operational_hold.json")["reason"]
    assert not (tmp_path / "final_status.json").exists()


def test_budget_boundary_enforces_deadline_and_disk(tmp_path, monkeypatch):
    c.old.save_new(tmp_path / "launch.json", {"epoch_seconds": 1.})
    monkeypatch.setattr(c.time, "time", lambda: 8 * 3600 + 2.)
    with pytest.raises(c.BudgetExceeded, match="Eight-hour"): c.budget_boundary(tmp_path)
    monkeypatch.setattr(c.time, "time", lambda: 2.)
    monkeypatch.setattr(c.shutil, "disk_usage", lambda *a: SimpleNamespace(free=c.CONFIG["disk_reserve_bytes"] - 1))
    with pytest.raises(c.BudgetExceeded, match="disk reserve"): c.budget_boundary(tmp_path)


def test_status_contention_is_advisory_scientific_write_is_strict(tmp_path, monkeypatch):
    from tools.proposal_ranking_128_recovery_01 import reporting
    reporter = p.StatusReporter(tmp_path)
    monkeypatch.setattr(p, "REPORTER", reporter)
    def locked(*a, **kw): raise PermissionError("fixture Windows lock")
    monkeypatch.setattr(reporting.os, "replace", locked)
    monkeypatch.setattr(reporting.time, "sleep", lambda *a: None)
    assert p.event("fixture", completed=1)["stage"] == "fixture"
    assert reporter.last_errors and list(tmp_path.glob("status_events/*.json"))
    with pytest.raises(PermissionError): p.old.save_new(tmp_path / "scientific.json", {"strict": True})
