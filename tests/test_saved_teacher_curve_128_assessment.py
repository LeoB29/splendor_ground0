"""Bounded saved-curve firewall, endpoint, metrics and independent audit tests."""
import copy
import json
from pathlib import Path

import pytest
import torch

from tools.saved_teacher_curve_128 import assessment as a
from tools.focused_teacher_fit_128.test_assessment import fixture_panel as prior_fixture
from splendor_ai.alphazero.strong_policy import save_model


def rewrite(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return a.c.artifact(path)


def curve_fixture(tmp_path, monkeypatch):
    predecessor = tmp_path/"predecessor"
    predecessor.mkdir()
    rows, inputs, descriptors = prior_fixture(predecessor, monkeypatch)
    registry = a.old.read_json(predecessor/"models_lock.json")
    model = a.prior.StrongPolicyInference(a.c.verify_hash(registry["models"]["focused_161001"])).model
    for parameter in model.parameters():
        parameter.data.zero_()
    model.eval()
    parent = {"binding": a.c.PARENT_BINDING, "models": {}, "training": {}, "fixture": True}
    for update in a.c.CHECKPOINTS:
        parent["models"][str(update)] = {}
        for fit in a.c.FITS:
            path = tmp_path/f"checkpoint_{update}_{fit}.pt"
            save_model(path, model, update, {"fit": fit, "manifest": a.c.PARENT_BINDING})
            parent["models"][str(update)][str(fit)] = a.c.artifact(path)
    final_predictions, final_rows = [], []
    for row, item, descriptor in zip(rows, inputs, descriptors):
        # Synthetic predecessor anchor is the same zero model as checkpoint zero.
        selection_path = a.c.verify_hash(item["selection_artifact"])
        selection = a.c.verified(item["selection_artifact"])
        anchor = selection["identity"]["proposals"]["anchor"]
        anchor["learned_logits"] = [0.]*len(row["features"])
        anchor["offered_group_scores"] = [0.]*len(row["members"])
        selection["selection_lock"]["identity_digest"] = a.old.digest(selection["identity"])
        selection_ref = rewrite(selection_path, selection)
        item["selection_artifact"] = descriptor["selection"] = selection_ref
        root_path = a.c.verify_hash(descriptor["root"])
        root = a.c.verified(descriptor["root"])
        root["selection_artifact"] = selection_ref
        root["selection_lock"] = selection["selection_lock"]
        root_ref = rewrite(root_path, root)
        audit_path = a.c.verify_hash(descriptor["audit"])
        audit = a.c.verified(descriptor["audit"])
        audit["root"] = root_ref
        descriptor["root"], descriptor["audit"] = root_ref, rewrite(audit_path, audit)
        _, totals, missing = a.prior._old_input(item)
        proposal = a.prior._predict(row, model)
        choices = a.prior._choices(totals, missing, proposal["offered_group_scores"], item["old_choices"])
        final_predictions.append({"proposal": proposal, "choices": choices})
        final_rows.append(a.prior._row(item["spec"], choices, root["assessment"]["worlds"]))
    path = tmp_path/"previous_predictions.json"
    a.old.save_new(path, {"manifest": a.c.PARENT_BINDING, "predictions": final_predictions})
    parent["predictions"] = a.c.artifact(path)
    path = tmp_path/"previous_assessment.json"
    a.old.save_new(path, {"manifest": a.c.PARENT_BINDING, "rows": final_rows})
    parent["assessment"] = a.c.artifact(path)
    monkeypatch.setattr(a.source, "panel", lambda ignored: (rows, inputs, descriptors))
    # Full statistical implementation is checked independently on 336-row fixtures.
    def stats(curves, boundary=None):
        if boundary is not None:
            boundary()
        return {"fixture_checkpoints": sorted(map(int, curves)), "states": len(curves["0"])}
    monkeypatch.setattr(a.statistics, "summarize", stats)
    monkeypatch.setattr(a.statistics, "independent_summarize", stats)
    return parent, rows, inputs, descriptors


def test_all_thirteen_checkpoints_locked_before_first_outcome_and_independent_audit(tmp_path, monkeypatch):
    parent, rows, inputs, descriptors = curve_fixture(tmp_path, monkeypatch)
    reader, accesses, progress = a.c.verified, [], []
    outcome_paths = {value["root"]["path"] for value in descriptors} | {parent["assessment"]["path"]}
    def guarded(ref):
        if ref["path"] in outcome_paths:
            lock = a.old.read_json(tmp_path/"predictions_lock.json")
            assert lock["complete"] and lock["checkpoints"] == list(a.c.CHECKPOINTS)
            assert all(len(lock["predictions"][str(u)]) == 2 for u in a.c.CHECKPOINTS)
            accesses.append(ref["path"])
        return reader(ref)
    monkeypatch.setattr(a.c, "verified", guarded)
    ref = a.assess("fixture", parent, tmp_path, lambda stage, **values: progress.append((stage, values)))
    result = reader(ref)
    assert len(accesses) == 3 and len(result["curves"]) == 13
    assert sum(stage == "checkpoint_predictions" for stage, value in progress) == 13
    assert result["endpoint_zero_reproduced"] and result["endpoint_final_reproduced"]
    assert result["new_simulations"] == 0 and not result["new_training"]
    for update in a.c.CHECKPOINTS:
        metrics = result["old_label_metrics"][str(update)]["groups"]
        assert metrics["train"]["metrics"]["multiple_outsiders"]["accuracy_denominator"] == 1
        assert metrics["validation"]["metrics"]["all_rows"]["rows"] == 1
        assert metrics["validation"]["metrics"]["all_rows"]["accuracy_denominator"] == 0
        assert metrics["validation"]["metrics"]["all_rows"]["mean_cross_entropy"] == 0
    for name in ("_label_record", "_label_metrics", "_initial_endpoint", "_open_outcomes"):
        monkeypatch.setattr(a, name, lambda *args, **kwargs: pytest.fail("Audit reused collection arithmetic"))
    for name in ("_predict", "_old_input", "_choices", "_row"):
        monkeypatch.setattr(a.prior, name, lambda *args, **kwargs: pytest.fail("Audit reused collection inference/arithmetic"))
    checked = reader(a.audit("fixture", parent, ref, tmp_path))
    assert checked["passed"] and checked["states_per_checkpoint"] == 2
    assert checked["old_label_metrics_independently_reproduced"] and checked["unknown_outcomes"] == 0
    assert checked["new_training"] is False and checked["new_trajectory_replays"] == 0
    with pytest.raises(ValueError, match="Preserve"):
        a.assess("fixture", parent, tmp_path)


@pytest.mark.parametrize("damage", ["missing_checkpoint", "missing_fit", "wrong_row_fit", "wrong_generation", "wrong_model_fit", "wrong_binding"])
def test_checkpoint_and_source_identity_rejected_before_outcomes(tmp_path, monkeypatch, damage):
    parent, rows, inputs, descriptors = curve_fixture(tmp_path, monkeypatch)
    if damage == "missing_checkpoint":
        del parent["models"]["512"]
    elif damage == "missing_fit":
        del parent["models"]["512"]["161003"]
    elif damage == "wrong_row_fit":
        rows[0]["source_fit"] = 161003
    else:
        item = parent["models"]["0"]["161001"]
        path = a.c.verify_hash(item)
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if damage == "wrong_generation":
            payload["generation"] = 1
        elif damage == "wrong_model_fit":
            payload["metadata"]["fit"] = 161002
        else:
            payload["metadata"]["manifest"] = "different"
        torch.save(payload, path)
        parent["models"]["0"]["161001"] = a.c.artifact(path)
    monkeypatch.setattr(a, "_open_outcomes", lambda *args: pytest.fail("Opened outcomes before validating inputs"))
    with pytest.raises(ValueError):
        a.assess("fixture", parent, tmp_path)
    assert not (tmp_path/"predictions_lock.json").exists()


def test_missing_and_incomplete_global_seals_block_outcome_access(tmp_path, monkeypatch):
    parent, rows, inputs, descriptors = curve_fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="Seal"):
        a._open_outcomes(descriptors[0], inputs[0], {}, tmp_path)
    path = tmp_path/"predictions_lock.json"
    a.old.save_new(path, {"complete": False})
    with pytest.raises(ValueError, match="Incomplete"):
        a._open_outcomes(descriptors[0], inputs[0], a.c.artifact(path), tmp_path)


@pytest.mark.parametrize("damage", ["initializer_logits", "final_choices", "final_scores"])
def test_both_endpoints_must_reproduce_saved_evidence(tmp_path, monkeypatch, damage):
    parent, rows, inputs, descriptors = curve_fixture(tmp_path, monkeypatch)
    if damage == "initializer_logits":
        path = a.c.verify_hash(inputs[0]["selection_artifact"])
        seal = a.c.verified(inputs[0]["selection_artifact"])
        seal["identity"]["proposals"]["anchor"]["learned_logits"][0] += .1
        seal["selection_lock"]["identity_digest"] = a.old.digest(seal["identity"])
        inputs[0]["selection_artifact"] = rewrite(path, seal)
    else:
        name = "predictions" if damage == "final_choices" else "assessment"
        path = a.c.verify_hash(parent[name])
        saved = a.c.verified(parent[name])
        if damage == "final_choices":
            saved["predictions"][0]["choices"]["focused_outsider"] = 4
        else:
            saved["rows"][0]["delta_bounds"]["focused_supported_minus_anchor"] = [1/256, 1/256]
        parent[name] = rewrite(path, saved)
    with pytest.raises(ValueError):
        a.assess("fixture", parent, tmp_path)


def test_metric_ties_empty_singleton_and_extreme_logits_are_independent():
    from tools.focused_teacher_fit_128.test_assessment import old_seal
    collected, checked = [], []
    for index, (n, sums, values) in enumerate(((3, [0, 0, 0], [0, 0, 0]),
            (4, [0, 0, 0, 128], [0, 0, 0, -1000]),
            (6, [0, 0, 0, 128, 128, 0], [0, 0, 0, -1000, 1000, -1000]))):
        seal, selected = old_seal(n, [x//128 for x in sums])
        row = {"index": index, "id": str(index), "source_fit": 161001, "split": "train"}
        proposal = {"offered_group_scores": values}
        choice = a.prior._choices(sums, [0]*n, values, selected["choices"])
        collected.append(a._label_record(row, sums, proposal, choice))
        checked.append(a._independent_label_record(row, seal, proposal, choice))
    a.prior.compare(collected, checked, rel_tol=0., abs_tol=1e-12)
    left, right = a._label_metrics(collected), a._independent_label_metrics(checked)
    a.prior.compare(left, right, rel_tol=0., abs_tol=1e-12)
    m = left["groups"]["train"]["metrics"]["all_rows"]
    assert m["rows"] == 3 and m["accuracy_denominator"] == 2
    assert m["first_winner_count"] == 1 and m["tied_maximum_count"] == 2
    assert m["mean_cross_entropy"] == pytest.approx(2000/3)
    assert m["old_regret_units_sum"] == 0


def test_raw_shared_outcomes_keep_unknown_bounds_and_fixed_choices():
    from tools.focused_teacher_fit_128.test_assessment import old_seal
    from tools.teacher_signal_capture_128.test_engine import worlds, outcome
    from tools.teacher_signal_capture_128.test_verify import specs
    seal, selected = old_seal(5, [0, 0, 0, 0, 1])
    choice = a.prior._choices(selected["sums_wdl"], selected["unknown_counts"], [0, 0, 0, 1, 2], selected["choices"])
    fresh = worlds([1, 0, 0, 0, 0])
    fresh[0]["outcomes"][4] = outcome(None)
    row = a.prior._row(specs()[0], choice, fresh)
    assert row == a.prior._independent_row(specs()[0], choice, fresh)
    assert row["choices"]["focused_outsider"] == 4
    assert row["delta_bounds"]["focused_outsider_minus_anchor_outsider"] == [-1/256, 1/256]


def test_actual_saved_initial_checkpoint_reproduces_old_only_anchor_input(monkeypatch):
    """One old row and checkpoint, never the production 13-checkpoint curve."""
    root = a.old.read_json(a.c.PARENT/"fits/fit_161001/result.json")
    cp = next(value for value in root["checkpoints"] if value["update"] == 0)
    parent = {"models": {"0": {"161001": cp["checkpoint"]}}}
    model = a._checkpoint_model(parent, 0, 161001)
    parent_model = a.prior.StrongPolicyInference(a.c.verify_hash(root["initializer"])).model
    schedule = a.old.read_json(a.prior.c.SOURCE/"schedule.json")["main"]
    spec = next(item for item in schedule if item["fit"] == 161001 and item["split"] == "train")
    old_root = a.c.verified(spec["source_root"])
    exports = torch.load(a.c.verify_hash(spec["source_export"]), map_location="cpu", weights_only=True)
    row = next(value for value in exports if value["id"] == spec["source_rowid"])
    item = {"snapshot": old_root["snapshot"], "candidates": old_root["candidates"]}
    monkeypatch.setattr(a.old, "trajectory", lambda *args: pytest.fail("No simulations in integration"))
    expected = a.prior._predict(row, parent_model)
    actual = a.prior._independent_proposal(row, item, model)
    a.prior.compare(actual, expected)
    state = a.old.deserialize_state_snapshot(item["snapshot"])
    alternate = a.prior.input_audit.hidden_alternative(state)
    a.prior.compare(actual, a.prior._independent_proposal(row, item, model, alternate))


def test_report_consumes_complete_actual_statistics_schema_and_old_metrics(tmp_path, monkeypatch):
    from test_saved_teacher_curve_128_statistics import synthetic_curves
    from tools.saved_teacher_curve_128 import reporting
    monkeypatch.setitem(a.c.CONFIG, "bootstrap_samples", 79)
    monkeypatch.setitem(a.c.CONFIG, "bootstrap_chunk", 17)
    curves = synthetic_curves()
    stats = a.statistics.summarize(curves)
    labels, parent = {}, {"training": {str(fit): {} for fit in a.c.FITS}}
    for update in a.c.CHECKPOINTS:
        records = []
        for row in curves[str(update)]:
            support = row["support"]-min(3, row["support"])
            records.append({"index": row["index"], "id": str(row["index"]), "fit": row["fit"], "split": row["split"],
                            "outsider_support": support, "choice": 3 if support else None,
                            "target": 3 if support else None, "loss": .5 if support > 1 else 0.,
                            "first_winner": True if support else None, "any_old_max": True if support else None,
                            "old_regret_units": 0})
        labels[str(update)] = a._label_metrics(records)
        for fit in a.c.FITS:
            path = tmp_path/f"training_{fit}_{update}.json"
            a.old.save_new(path, {"metrics": {"all_rows": {"first_winner_accuracy": .4+.6*update/1784}},
                                  "regret_capture_vs_initializer": update/1784})
            parent["training"][str(fit)][str(update)] = a.c.artifact(path)
    refs = reporting.write_report("synthetic-fixture", {"statistics": stats, "old_label_metrics": labels}, parent, tmp_path)
    report = a.c.verify_hash(refs["report"]).read_text(encoding="utf-8")
    for update in a.c.CHECKPOINTS:
        assert report.count(f"| {update} |") == 2
    assert "reused development" in report.lower() and "No model or checkpoint is selected" in report
    assert a.c.verify_hash(refs["figure"]).read_bytes().startswith(b"\x89PNG")
    assert a.c.verify_hash(refs["figure_pdf"]).read_bytes().startswith(b"%PDF")
    renderer = a.old.read_json(tmp_path/"render_complete.json")
    assert renderer["torch_imported"] is False and renderer["checkpoints"] == list(a.c.CHECKPOINTS)
    assert (tmp_path/"render_stderr.txt").read_text(encoding="utf-8") == ""
    with pytest.raises(ValueError, match="Preserve"):
        reporting.write_report("synthetic-fixture", {"statistics": stats}, parent, tmp_path)
