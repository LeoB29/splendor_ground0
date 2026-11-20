"""Saved-checkpoint/source authentication, with no policy inference or fitting."""
import copy

import pytest
import torch

from tools.higher_teacher_curve_128 import common as c, source as s


def checkpoint(update=0, fit=161001):
    initial = {"model_kind": "fixture-model", "model_config": {"width": 2},
               "model_state_dict": {"policy.weight": torch.tensor([[1., 2.]]),
                                    "value_head.weight": torch.tensor([3.])}}
    payload = copy.deepcopy(initial)
    payload.update(generation=update, metadata={"experiment": "higher_teacher_rate_128", "fit": fit,
                   "manifest": c.PARENT_BINDING, "checkpoint_selection": "fixed_final",
                   "training_split": "train", "assessment_used": False})
    if update:
        payload["model_state_dict"]["policy.weight"] += 0.5
    return payload, initial


@pytest.mark.parametrize("update", c.CHECKPOINTS)
def test_all_saved_checkpoint_generations_allow_finite_policy_change(update):
    payload, initial = checkpoint(update)
    assert s.checkpoint_identity(payload, initial, 161001, update)


@pytest.mark.parametrize("change", ["kind", "config", "generation", "metadata", "keys", "shape", "dtype", "nan", "initial", "value"])
def test_tensor_identity_and_zero_model_reject_changes(change):
    payload, initial = checkpoint(0 if change == "initial" else 128)
    update = payload["generation"]
    if change == "kind": payload["model_kind"] = "another-model"
    if change == "config": payload["model_config"]["width"] = 3
    if change == "generation": payload["generation"] = 999
    if change == "metadata": payload["metadata"]["assessment_used"] = True
    if change == "keys": payload["model_state_dict"]["extra"] = torch.tensor([1.])
    if change == "shape": payload["model_state_dict"]["policy.weight"] = torch.ones(3)
    if change == "dtype": payload["model_state_dict"]["policy.weight"] = payload["model_state_dict"]["policy.weight"].double()
    if change == "nan": payload["model_state_dict"]["policy.weight"][0, 0] = float("nan")
    if change == "initial": payload["model_state_dict"]["policy.weight"][0, 0] += 1
    if change == "value": payload["model_state_dict"]["value_head.weight"][0] += 1
    with pytest.raises(ValueError): s.checkpoint_identity(payload, initial, 161001, update)


def fit_evidence():
    fit = c.FITS[0]
    config = {"train_seeds": {str(fit): 123}, "checkpoints": list(c.CHECKPOINTS)}
    points = [{"update": update, "checkpoint": {"path": f"model_{update}.pt", "sha256": f"model-{update}"},
               "training": {"path": f"training_{update}.json", "sha256": f"report-{update}"}}
              for update in c.CHECKPOINTS]
    ref = {"path": "fit.json", "sha256": "fixture-fit"}
    exposures = {f"row-{i}": 13 if i < 512 else 12 for i in range(9461)}
    result = {"manifest": c.PARENT_BINDING, "fit": fit, "config": config, "seed": 123,
              "final_always_selected": True, "updates": 1784, "selected_update": 1784,
              "checkpoints": points, "selected_model": points[-1]["checkpoint"],
              "selected_training": points[-1]["training"], "batch_sequence_sha256": "stream",
              "row_exposures": sum(exposures.values()), "per_row_exposures": exposures,
              "train_rows": 9461, "epochs_entered": 13, "assessment_read": False, "validation_read": False,
              "optimization_stability": {"all_postupdate_model_tensors_finite": True,
                  "all_preclip_norms_finite": True, "gradient_clip_limit": 1., "preclip_gradient_norms": [.5]*1784},
              "matched_low_rate_control": {"high_learning_rate": 1e-4, "low_learning_rate": 1e-5,
                  "learning_rate_multiplier": 10, "old_fit_replayed": False,
                  "only_learning_rate_changes_fitting_recipe": True, "same_fixed_final_update": True,
                  "same_original_initializer_and_training_rows": True, "same_seed_batch_stream_and_per_row_exposures": True}}
    receipt = {key: copy.deepcopy(result[key]) for key in (
        "manifest", "fit", "final_always_selected", "selected_update", "selected_model", "selected_training",
        "batch_sequence_sha256", "row_exposures", "per_row_exposures", "assessment_read", "validation_read",
        "optimization_stability", "matched_low_rate_control")}
    receipt.update(result=ref, passed=True, complete_fit_bitwise_identical=True,
                   all_training_metrics_independently_reproduced=True, value_head_frozen=True,
                   optimization_stability_independently_reproduced=True, matched_low_rate_control_verified=True)
    return result, receipt, ref, fit, config


def test_complete_fit_receipt_accepts_every_saved_checkpoint():
    assert s.fitting_receipt(*fit_evidence())


@pytest.mark.parametrize("change", ["missing_checkpoint", "duplicate_checkpoint", "wrong_final", "missing_audit",
                                    "validation_use", "assessment_use", "seed", "exposure", "fit", "config",
                                    "stability_audit", "stability_value", "matched_audit", "matched_rate", "wrong_population"])
def test_fit_receipt_rejects_partial_changed_or_contaminated_trajectories(change):
    result, receipt, ref, fit, config = fit_evidence()
    if change == "missing_checkpoint": result["checkpoints"].pop(3)
    if change == "duplicate_checkpoint": result["checkpoints"][3] = result["checkpoints"][2]
    if change == "wrong_final": result["selected_model"] = result["checkpoints"][0]["checkpoint"]
    if change == "missing_audit": receipt["all_training_metrics_independently_reproduced"] = False
    if change == "validation_use": result["validation_read"] = True
    if change == "assessment_use": receipt["assessment_read"] = True
    if change == "seed": result["seed"] += 1
    if change == "exposure": receipt["per_row_exposures"]["row-0"] += 1
    if change == "fit": receipt["fit"] = c.FITS[1]
    if change == "config": result["config"] = {"other": True}
    if change == "stability_audit": receipt["optimization_stability_independently_reproduced"] = False
    if change == "stability_value": result["optimization_stability"]["preclip_gradient_norms"][0] = float("nan")
    if change == "matched_audit": receipt["matched_low_rate_control_verified"] = False
    if change == "matched_rate": result["matched_low_rate_control"]["low_learning_rate"] = 1e-4
    if change == "wrong_population": result["train_rows"] = 168
    with pytest.raises(ValueError): s.fitting_receipt(result, receipt, ref, fit, config)


@pytest.fixture(scope="module")
def parent():
    # This is an identity-only inspection. It reads saved tensors and metadata,
    # and must never create a policy prediction or reproduce an optimizer step.
    from unittest.mock import patch
    from splendor_ai.training.model import LegalActionAttentionScorerModel
    def forbidden(*args, **kwargs):
        raise AssertionError("Checkpoint/source authentication invoked inference or new play")
    with patch.object(LegalActionAttentionScorerModel, "forward", forbidden), \
         patch.object(torch.optim.AdamW, "step", forbidden), \
         patch.object(c.old, "trajectory", forbidden):
        return s.closed_parent()


def test_real_parent_authenticates_all_models_metrics_and_closure(parent):
    assert parent["binding"] == c.PARENT_BINDING
    assert parent["result"]["sha256"] == s.RESULT_SHA and parent["audit"]["sha256"] == s.AUDIT_SHA
    assert list(parent["models"]) == [str(update) for update in c.CHECKPOINTS]
    assert set(parent["training"]) == {str(fit) for fit in c.FITS}
    assert all(set(models) == {str(fit) for fit in c.FITS} for models in parent["models"].values())
    assert all(list(values) == [str(update) for update in c.CHECKPOINTS] for values in parent["training"].values())
    assert parent["inventory"] == {ref["path"]: ref["sha256"] for ref in parent["evidence"]}
    assert parent["checkpoint_zero_equals_initializer"] and parent["final_checkpoint_equals_prior_endpoint"]
    assert parent["historical_fit_replays"] == parent["historical_simulation_replays"] == 0
    assert parent["new_inference"] is False
    assert len(parent["schedule"]) == len(parent["evaluation_sources"]) == 336
    assert len(parent["controls"]) == 9
    assert parent["training_scopes"]["0"]["rows"] == parent["training_scopes"]["1784"]["rows"] == 9461
    assert all(parent["training_scopes"][str(update)]["rows"] == 168 for update in c.CHECKPOINTS[1:-1])
    assert parent["validation_old_targets"]["path"] == "outputs/higher_teacher_rate_128/validation_old_targets.json"
    assert parent["validation_file"]["path"] == "outputs/full_teacher_fit_128/validation.pt"
    # The retained teacher context is directly consumed by downstream assessment.
    context = c.old.read_json(c.ROOT/"outputs/teacher_signal_capture_128/final_status.json")
    assert parent["inventory"][context["result"]["path"]] == s.inherited_source.RESULT_SHA


@pytest.mark.parametrize("change", ["swapped_model", "swapped_metric", "binding", "dropped_evidence", "reordered_panel", "final_reference",
                                    "wrong_scope", "control", "validation_reference"])
def test_descriptor_changes_rejected_even_with_real_valid_file_refs(parent, monkeypatch, change):
    monkeypatch.setattr(s, "closed_parent", lambda: parent)
    altered = copy.deepcopy(parent)
    if change == "swapped_model": altered["models"]["1"]["161001"] = parent["models"]["2"]["161001"]
    if change == "swapped_metric": altered["training"]["161001"]["1"] = parent["training"]["161001"]["2"]
    if change == "binding": altered["binding"] = "different-parent"
    if change == "dropped_evidence": altered["evidence"].pop()
    if change == "reordered_panel": altered["schedule"].reverse()
    if change == "final_reference": altered["assessment"] = altered["result"]
    if change == "wrong_scope": altered["training_scopes"]["0"]["rows"] = 168
    if change == "control": altered["controls"]["lowrate_161001"] = altered["controls"]["specialist_161001"]
    if change == "validation_reference": altered["validation_file"] = altered["models"]["0"]["161001"]
    with pytest.raises(ValueError): s.validate(altered)
    assert s.validate(parent) == parent


def test_real_panel_reads_old_inputs_only_and_keeps_all_source_pairs(parent, monkeypatch):
    read = c.old.read_json
    def guarded(path):
        name = str(path).replace("\\", "/")
        assert not name.endswith("assessment_result.json"), "Panel opened scored final outcomes"
        assert "/main/roots/" not in name, "Panel opened independent raw assessment worlds"
        return read(path)
    monkeypatch.setattr(c.old, "read_json", guarded)
    rows, inputs, descriptors = s.panel(parent)
    assert len(rows) == len(inputs) == len(descriptors) == 336
    assert [row["index"] for row in rows] == list(range(336))
    for split in ("train", "validation"):
        selected = [row for row in rows if row["split"] == split]
        assert len(selected) == 168 and len({row["pair"] for row in selected}) == 42
    assert [item["spec"] for item in inputs] == parent["schedule"]


def test_panel_rejects_changed_descriptor_before_parent_input_access(parent, monkeypatch):
    altered = copy.deepcopy(parent)
    altered["schedule"] = altered["schedule"][:-1]
    def forbidden(*args, **kwargs):
        raise AssertionError("Changed panel reached inherited input loader")
    monkeypatch.setattr(s.inherited_assessment, "_panel", forbidden)
    with pytest.raises(ValueError, match="panel identity"):
        s.panel(altered)


def test_real_full_validation_old_targets_and_private_control_refs(parent, monkeypatch):
    read = c.old.read_json
    def guarded(path):
        name = str(path).replace("\\", "/")
        assert not name.endswith("assessment_result.json") and "/main/roots/" not in name
        return read(path)
    monkeypatch.setattr(c.old, "read_json", guarded)
    rows = s.validation_rows(parent)
    assert len(rows) == len({row["id"] for row in rows}) == 2489
    assert {row["split"] for row in rows} == {"validation"}
    assert len({row["pair"] for row in rows}) == 42
    assert all(set(row) == s.inherited_source.ROW_FIELDS for row in rows)
    controls = s.control_models(parent)
    assert set(controls) == {f"{family}_{fit}" for family in ("anchor", "specialist", "lowrate") for fit in c.FITS}


@pytest.mark.parametrize("change", ["scope", "count", "id", "full_slice", "probe_slice", "validation_use"])
def test_saved_training_scope_cannot_silently_mix_full_corpus_and_probe(parent, change):
    report = copy.deepcopy(c.verified(parent["training"]["161001"]["0"]))
    fitted = c.verified(parent["fit_results"]["161001"])
    train_ids = set(fitted["per_row_exposures"])
    probe_ids = {spec["source_rowid"] for spec in parent["schedule"] if spec["split"] == "train"}
    assert s.training_report_scope(report, 0, train_ids, probe_ids)["rows"] == 9461
    if change == "scope": report["diagnostic_scope"] = "original_168_train_probe"
    if change == "count": report["metrics"]["all_rows"]["rows"] = 168
    if change == "id": report["rows"][0]["id"] = "another-row"
    if change == "full_slice": report["metrics_by_slice"]["full_train"]["metrics"]["all_rows"]["rows"] = 168
    if change == "probe_slice": report["metrics_by_slice"]["original_168_train"]["metrics"]["all_rows"]["rows"] = 9461
    if change == "validation_use": report["validation_read"] = True
    with pytest.raises(ValueError): s.training_report_scope(report, 0, train_ids, probe_ids)


def test_control_and_validation_accessors_reject_swapped_actual_artifacts(parent, monkeypatch):
    altered = copy.deepcopy(parent)
    altered["validation_file"] = parent["models"]["0"]["161001"]
    monkeypatch.setattr(s.inherited_source, "load_rows", lambda *args: pytest.fail("Changed reference reached tensor loader"))
    with pytest.raises(ValueError, match="validation tensor identity"):
        s.validation_rows(altered)
    altered = copy.deepcopy(parent)
    altered["controls"]["lowrate_161001"] = altered["controls"]["specialist_161001"]
    with pytest.raises(ValueError, match="controls changed"):
        s.control_models(altered)
