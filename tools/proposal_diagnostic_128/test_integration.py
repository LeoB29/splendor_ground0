"""Real receipt-summary/finalization interfaces with cheap synthetic evidence.

No network, model inference or continuation simulation. Only the raw model/world
record decoder and expensive frozen inventory/stage-lock checks are substituted;
the two independently implemented statistics paths and receipt summary run.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from tools.proposal_diagnostic_128 import common as c
from tools.proposal_diagnostic_128 import experiment as p
from tools.proposal_diagnostic_128 import verify as v

BINDING = "synthetic-integration-binding"


def synthetic_row(index, spec):
    anchor = 89 + (17*index) % 71
    numerators = {"anchor": anchor,
                  "treatment": anchor + (11*index) % 21 - 5,
                  "replay": anchor + (7*index) % 19 - 11,
                  "random": anchor + (13*index) % 31 - 17,
                  "first_three": anchor - index % 10,
                  "union": anchor + (3*index) % 19 - 2}
    proposed = {"anchor": 3, "treatment": 3 if index % 4 == 0 else 4,
                "replay": 5, "random": 6}
    choices = {**proposed, "first_three": 0, "union": 4}
    if index % 17 == 0:
        numerators = dict.fromkeys(v.ARMS, anchor)
        proposed = dict.fromkeys(v.PROPOSERS, None)
        choices = dict.fromkeys(v.ARMS, 0)
    scores = {arm: value/256 for arm, value in numerators.items()}
    fixed = {arm: None if proposed[arm] is None else (83+(19*index+offset*7) % 97)/256
             for offset, arm in enumerate(v.PROPOSERS)}
    return {**{key: spec[key] for key in ("index", "pair", "fit", "role", "phase")},
            "arm_scores": scores, "fixed_proposal_scores": fixed,
            "choices": choices, "proposed_indices": proposed}


def install_evidence(tmp_path, monkeypatch):
    monkeypatch.setattr(p, "OUTPUT", tmp_path)
    monkeypatch.setattr(p, "validate_freeze", lambda: BINDING)
    monkeypatch.setattr(p.engine, "statistics_row", lambda record: record["fixture_row"])
    monkeypatch.setattr(c.shutil, "disk_usage", lambda path: SimpleNamespace(free=100*1024**3))
    # All real budget checks pass unless a test later advances the clock.
    started = p.time.time()
    p.old.save_new(tmp_path / "launch.json", {"epoch_seconds": started, "manifest": BINDING})
    selection, locks, rows = {}, {}, {}
    for stage, count in (("probe", 12), ("main", 201)):
        specs = [{"index": i, "pair": i if stage == "main" else i//2,
                  "fit": p.FITS[i % 3] if stage == "main" else p.FITS[(i//2) % 3],
                  "role": "anchor" if i % 2 else "treatment", "phase": (i//2) % 3,
                  "privacy": {"own": i % 2, "opponent": (i//2) % 2},
                  "canonical_support": 3 if i % 17 == 0 else 20}
                 for i in range(count)]
        selection[stage] = specs
        plan = v.sample_plan(specs, stage, BINDING)
        plan_path = tmp_path / stage / "audit_sample.json"
        p.old.save_new(plan_path, plan)
        plan_item = p.artifact(plan_path)
        roots, audits, rows[stage] = [], [], []
        for spec, sample in zip(specs, plan["selections"], strict=True):
            index = spec["index"]
            row = synthetic_row(index, spec)
            rows[stage].append(row)
            support = 3 if index % 17 == 0 else 7
            seal_path = tmp_path / stage / "selections" / f"state_{index:04d}.json"
            p.old.save_new(seal_path, {"manifest": BINDING, "index": index, "fixture": "sealed selection"})
            seal = p.artifact(seal_path)
            root_path = tmp_path / stage / "roots" / f"state_{index:04d}.json"
            record = {"manifest": BINDING, "stage": stage, "spec": spec,
                      "selection_artifact": seal, "fixture_row": row,
                      "timing": {"wall": 10.+index % 5, "cpu": 9.}}
            p.old.save_new(root_path, record)
            # Match the collector's actual extra metadata. The audit root link
            # deliberately contains only path/hash, as the real verifier does.
            roots.append(p.artifact(root_path, index=index, continuations=256*support))
            replayed = sum(len(indices) for indices in sample["replay"].values())
            receipt = {"manifest": BINDING, "stage": stage, "spec": spec, "passed": True,
                       "root": p.artifact(root_path), "selection_artifact": seal,
                       "sample_artifact": plan_item, "row": row, "privacy": spec["privacy"],
                       "support": support, "stress": sample["stress"],
                       "ordinary_world_indices": sample["ordinary"], "replay_world_indices": sample["replay"],
                       "recorded_worlds": 256, "recorded_continuations": 256*support,
                       "replayed_worlds": replayed, "replayed_continuations": replayed*support,
                       "moves": replayed*support*3, "reasons": {"completed": replayed*support},
                       "full_inner_replay": replayed == 256,
                       "timing": {"wall": 6.+index % 3, "cpu": 5., "replay_wall": 4.,
                                  "structural_wall": 2.+index % 3, "ordinary_replay_wall": 1.,
                                  "full_extra_replay_wall": 3.}}
            audit_path = tmp_path / stage / "audits" / f"state_{index:04d}.json"
            p.old.save_new(audit_path, receipt)
            audits.append(p.artifact(audit_path, index=index))
        locks[stage, "collection"] = {"manifest": BINDING, "files": roots}
        locks[stage, "audits"] = {"manifest": BINDING, "files": audits}
        p.old.save_new(tmp_path / f"{stage}_complete.json", {"manifest": BINDING, "states": count, "fixture": True})
    p.old.save_new(tmp_path / "selection.json", selection)
    p.old.save_new(tmp_path / "cached_analysis.json", {"manifest": BINDING, "independent_assessment": False})

    def stage_lock(stage, kind, binding):
        assert binding == BINDING
        return locks[stage, kind]

    monkeypatch.setattr(p, "stage_lock", stage_lock)
    return selection, locks, rows, started


def test_finish_consumes_real_summary_schema_and_nonconstant_integer_bootstrap(tmp_path, monkeypatch):
    selection, locks, rows, _ = install_evidence(tmp_path, monkeypatch)
    assert len({r["arm_scores"]["treatment"]-r["arm_scores"]["anchor"] for r in rows["main"]}) > 10
    assert any(int(256*r["arm_scores"]["treatment"]) % 2 for r in rows["main"])
    p.finish(BINDING)
    result = p.old.read_json(tmp_path / "result.json")
    audit = p.old.read_json(tmp_path / "final_audit.json")
    final = p.old.read_json(tmp_path / "final_status.json")
    assert final["complete"] and final["all_prescribed_audits_pass"]
    assert final["classification"] == result["classification"] == result["statistics"]["classification"]
    assert result["statistics"]["states"] == result["statistics"]["source_pairs"] == 201
    assert result["statistics"]["bootstrap_samples"] == 50000
    assert result["statistics"]["primary"] == "treatment_minus_anchor"
    for name in p.CONTRASTS:
        left = audit["independent_statistics"][name]
        right = result["statistics"]["contrasts"][name]
        assert left["mean"] == right["mean_score_difference"]
        assert np.allclose(left["ci95"], right["ci95"], rtol=0., atol=1e-12)
    receipts = [p.verified(item) for item in locks["main", "audits"]["files"]]
    for key in ("recorded_worlds", "recorded_continuations", "replayed_worlds", "replayed_continuations"):
        assert result["coverage"][key] == sum(r[key] for r in receipts)
    assert result["coverage"]["recorded_worlds"] == 201*256
    assert result["fixed_proposal_scores_descriptive"]["anchor"]["eligible_states"] < 201
    assert result["changes_descriptive"]["treatment"]["fourth_changed_vs_anchor"] > 0
    assert p.verified(final["result"]) == result
    assert p.verified(final["audit"]) == audit
    assert p.verify_hash(final["report"]).read_text(encoding="utf-8").find("not full-game win rates") >= 0


def test_budget_expiry_during_finalization_preserves_partial_without_final_status(tmp_path, monkeypatch):
    _, locks, _, started = install_evidence(tmp_path, monkeypatch)
    # Use the actual budget implementation. Advance the wall clock only after
    # statistical results and the report were written, before completion seal.
    monkeypatch.setattr(c.time, "time", lambda: started + (8*3600+1 if (tmp_path / "report.md").exists() else 1))
    with pytest.raises(p.BudgetExceeded, match="Eight-hour"):
        p.finish(BINDING)
    assert (tmp_path / "result.json").exists()
    assert (tmp_path / "final_audit.json").exists()
    assert (tmp_path / "report.md").exists()
    assert not (tmp_path / "final_status.json").exists()
    assert all(p.verify_hash(item).exists() for item in locks["main", "audits"]["files"])


def test_operational_gate_uses_actual_top_level_receipt_cost_fields(tmp_path, monkeypatch):
    _, locks, _, started = install_evidence(tmp_path, monkeypatch)
    monkeypatch.setattr(p.time, "time", lambda: started+100)
    roots = [p.verified(item) for item in locks["probe", "collection"]["files"]]
    receipts = [p.verified(item) for item in locks["probe", "audits"]["files"]]
    assert all("coverage" not in receipt and "recorded_continuations" in receipt for receipt in receipts)
    plan = p.old.read_json(tmp_path / "main/audit_sample.json")
    ceiling = sum(sum(len(values) for values in sample["replay"].values())*7 for sample in plan["selections"])
    expected = p.cost_projection(
        sum(r["timing"]["wall"] for r in roots), sum(r["timing"]["wall"] for r in receipts),
        sum(r["timing"]["replay_wall"] for r in receipts), sum(r["recorded_continuations"] for r in receipts),
        ceiling, sum(path.stat().st_size for path in (tmp_path / "probe").rglob("*") if path.is_file()),
        100*1024**3, 100.)
    assert p.operational_decision(BINDING)
    observed = p.old.read_json(tmp_path / "operational_decision.json")
    assert all(observed[key] == value for key, value in expected.items())
    assert observed["reasonable"] and observed["audit_seconds_before_allowance"] > 0
    assert not (tmp_path / "operational_hold.json").exists()
