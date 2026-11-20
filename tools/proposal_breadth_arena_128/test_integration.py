"""Small real stage integration and synthetic full-size result reconciliation."""
import copy

import pytest

from tools.proposal_breadth_arena_128 import common as c, experiment as p, arena, verify


def test_actual_six_game_stage_seals_and_reconciles_prescribed_audits(tmp_path, monkeypatch):
    from tools.proposal_breadth_arena_128.test_arena import BINDING, fixture_registry, private_state
    registry = fixture_registry()
    c.old.save_new(tmp_path / "models_lock.json", registry)
    arena.initialize_models(tmp_path)
    snapshot = c.old.serialize_state_snapshot(private_state())
    monkeypatch.setattr(p, "OUTPUT", tmp_path)
    monkeypatch.setattr(p.models, "validate", lambda *args: registry)
    for name in ("freeze.json", "schedule.json"):
        c.old.save_new(tmp_path / name, {"synthetic_fixture": True})
    calls = []
    def pool(stage, kind, arguments, binding, upstream):
        assert stage == "test" and binding == BINDING
        calls.append(kind)
        launch_path = tmp_path / f"{stage}_{kind}_launch.json"
        c.old.save_new(launch_path, {"manifest": binding, "stage": stage, "kind": kind,
                                    "dependencies": [c.artifact(path) for path in upstream]})
        if kind == "games":
            c.old.save_new(tmp_path / f"{stage}_launch.json", {"manifest": binding, "stage": stage})
            files = [p.collect_job(stage, spec, binding, tmp_path, snapshot) for spec in arguments]
        else:
            files = [p.audit_job(stage, item, binding, tmp_path, snapshot) for item in arguments]
        c.old.save_new(tmp_path / f"{stage}_{kind}_lock.json", {"manifest": binding, "files": files,
                       "launch": c.artifact(launch_path), "elapsed_seconds": 1.})
        return files
    monkeypatch.setattr(p, "run_pool", pool)
    p.build_stage("test", BINDING)
    complete = c.old.read_json(tmp_path / "test_complete.json")
    assert calls == ["games", "audits"] and complete["games"] == 6
    assert complete["all_prescribed_audits_pass"] and not complete["full_inner_replay"]
    coverage = complete["coverage"]
    assert coverage["games"] == 6 and coverage["roots"] == 12
    assert coverage["recorded_continuations"] == 6*128*(8+4)
    assert coverage["recorded_worlds"] == 12*128
    assert coverage["replayed_continuations"] == coverage["recorded_continuations"]
    assert coverage["stress_roots"] == 12
    sample = c.verified(complete["sample"])
    assert len(sample["stress_cells"]) == 171
    assert sample["coverage"]["missing_stress_cells"]
    original_validate, original_root_path = verify.validate_sample, arena.root_path
    monkeypatch.setattr(verify, "validate_sample", lambda stage, binding, base=tmp_path: original_validate(stage,binding,base))
    monkeypatch.setattr(arena, "root_path", lambda stage,index,turn,base=tmp_path: original_root_path(stage,index,turn,base))
    games, audits, receipts, closed_coverage = p.close_stage("test", BINDING)
    assert closed_coverage == coverage and len(games["files"]) == len(audits["files"]) == 6
    roots = [root for receipt in receipts for root in receipt["roots"]]
    assert {root["planner"] for root in roots} == {"candidate", "reference"}
    assert {root["fit"] for root in roots} == set(c.FITS)
    assert all(root["stress"] and len(root["ordinary_world_indices"]) == 8
               and len(root["replay_world_indices"]) == 128 and not root["full_fixture_replay"] for root in roots)
    # Rewriting a delivered receipt's accounting cannot be accepted merely
    # because its corresponding game and sample links remain correct.
    damaged = copy.deepcopy(receipts)
    damaged[0]["roots"][0]["replayed_branches"] -= 1
    with pytest.raises(ValueError, match="Delivered continuation mismatch"):
        p.checked_audit_coverage(damaged, sample, complete["sample"]["sha256"])


def final_fixture(tmp_path, monkeypatch, *, damaged_receipt=False):
    """Exercise finalization, not gameplay, with402 explicit synthetic games.

    Whole-stage simulation/provenance closure is covered by the actual fixture
    below; only that expensive closure and ancestor checks are substituted here.
    Both independently implemented final statistics are used unchanged.
    """
    monkeypatch.setattr(p, "OUTPUT", tmp_path)
    monkeypatch.setattr(p, "REPORTER", None)
    monkeypatch.setattr(p, "validate_freeze", lambda: "binding")
    monkeypatch.setattr(p.models, "validate", lambda *args: None)
    scores = [1., 1., .5, 0., 1., .5, 0.]
    files, audits, rows = [], [], []
    for spec in arena.specs("main"):
        index, seat = spec["index"], spec["candidate_seat"]
        score = scores[(index*5+index//9) % len(scores)]
        winner = None if score == .5 else seat if score else 1-seat
        row = {"index": index, "pair": spec["pair"], "fit": spec["fit"], "seat": seat, "score": score}
        rows.append(copy.deepcopy(row))
        path = tmp_path / "main/games" / f"game_{index:04d}.json"
        c.old.save_new(path, {"manifest": "binding", "spec": spec, "reason": "completed",
                              "winner": winner, "synthetic_fixture": True})
        files.append(c.artifact(path, index=index))
        if damaged_receipt and index == 0:
            row["score"] = .5 if score != .5 else 1.
        row.update(margin=1 if winner == seat else -1 if winner is not None else 0,
                   turns=2, reason="completed", timing={"wall": 1., "cpu": .5})
        roots = [{"planner": role, "continuations": 128*support,
                  "timing": {"wall": support*.1, "cpu": support*.05}}
                 for role, support in (("candidate",8),("reference",4))]
        path = tmp_path / "main/audits" / f"game_{index:04d}.json"
        c.old.save_new(path, {"manifest": "binding", "game_artifact": files[-1],
                              "passed": True, "full_inner_replay": False,
                              "game": row, "roots": roots, "synthetic_fixture": True})
        audits.append(c.artifact(path, index=index))
    coverage = {"games": 402, "roots": 804, "recorded_worlds": 804*128,
                "recorded_continuations": 402*128*12, "replayed_worlds": 804*8,
                "replayed_continuations": 402*8*12, "stress_roots": 0}
    def closed(stage, binding):
        assert stage in ("probe", "main") and binding == "binding"
        return ({"files": files, "elapsed_seconds": 31.},
                {"files": audits, "elapsed_seconds": 7.},
                [c.verified(item) for item in audits], coverage)
    monkeypatch.setattr(p, "close_stage", closed)
    for name in ("models_lock.json", "main/audit_sample.json", "probe_complete.json", "main_complete.json"):
        c.old.save_new(tmp_path / name, {"synthetic_fixture": True})
    return rows


def test_finish_reconciles_raw_game_wdl_with_independent_receipt_statistics(tmp_path, monkeypatch):
    rows = final_fixture(tmp_path, monkeypatch)
    assert {row["score"] for row in rows} == {0., .5, 1.}
    pair_totals = [int(2*rows[i]["score"]+2*rows[i+1]["score"]) for i in range(0,402,2)]
    assert len(set(pair_totals)) > 2 and any(value % 2 for value in pair_totals)
    p.finish("binding")
    status = c.old.read_json(tmp_path / "final_status.json")
    result, audit = c.verified(status["result"]), c.verified(status["audit"])
    assert status["complete"] and status["all_prescribed_audits_pass"] and audit["passed"]
    assert not status["automatic_successor"] and audit["raw_game_endpoints_agree"]
    assert result["games"] == 402 and result["pairs"] == 201
    assert result["share"] == sum(row["score"] for row in rows)/402
    for key in ("wins", "draws", "losses", "share", "ci95", "classification", "stage_pass"):
        assert result[key] == audit["independent_statistics"][key]
    for fit in c.FITS:
        assert result["fits"][str(fit)] == {"games": 134,
            "share": sum(row["score"] for row in rows if row["fit"] == fit)/134}
    assert result["continuations_by_role"] == {"candidate": 402*128*8, "reference": 402*128*4}
    assert all(value["count"] == 402 for value in result["decision_seconds"].values())
    assert status["classification"] == result["classification"]
    c.verify_hash(status["report"])


def test_rehashed_receipt_score_disagreement_prevents_result(tmp_path, monkeypatch):
    final_fixture(tmp_path, monkeypatch, damaged_receipt=True)
    with pytest.raises(ValueError, match="Saved game/receipt endpoint disagreement"):
        p.finish("binding")
    assert not (tmp_path / "result.json").exists()
    assert not (tmp_path / "final_status.json").exists()


def test_finalization_budget_expiry_retains_evidence_without_false_completion(tmp_path, monkeypatch):
    final_fixture(tmp_path, monkeypatch)
    def boundary(*args):
        if (tmp_path / "report.md").exists():
            raise c.BudgetExceeded("Finalization exceeded original deadline")
    monkeypatch.setattr(p, "budget_boundary", boundary)
    with pytest.raises(c.BudgetExceeded, match="original deadline"):
        p.finish("binding")
    assert (tmp_path / "result.json").exists() and (tmp_path / "final_audit.json").exists()
    assert (tmp_path / "report.md").exists() and not (tmp_path / "final_status.json").exists()
