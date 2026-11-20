"""Saved-checkpoint/source authentication, with no policy inference or fitting."""
import copy

import pytest
import torch

from tools.saved_teacher_curve_128 import common as c, source as s


def checkpoint(update=0, fit=161001):
    initial = {"model_kind": "fixture-model", "model_config": {"width": 2},
               "model_state_dict": {"policy.weight": torch.tensor([[1., 2.]]),
                                    "value_head.weight": torch.tensor([3.])}}
    payload = copy.deepcopy(initial)
    payload.update(generation=update, metadata={"experiment": "focused_teacher_fit_128", "fit": fit,
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
    exposures = {f"row-{i}": 500 for i in range(168)}
    result = {"manifest": c.PARENT_BINDING, "fit": fit, "config": config, "seed": 123,
              "final_always_selected": True, "updates": 1784, "selected_update": 1784,
              "checkpoints": points, "selected_model": points[-1]["checkpoint"],
              "selected_training": points[-1]["training"], "batch_sequence_sha256": "stream",
              "row_exposures": sum(exposures.values()), "per_row_exposures": exposures,
              "train_rows": 168, "assessment_read": False, "validation_read": False}
    receipt = {key: copy.deepcopy(result[key]) for key in (
        "manifest", "fit", "final_always_selected", "selected_update", "selected_model", "selected_training",
        "batch_sequence_sha256", "row_exposures", "per_row_exposures", "assessment_read", "validation_read")}
    receipt.update(result=ref, passed=True, complete_fit_bitwise_identical=True,
                   all_training_metrics_independently_reproduced=True, value_head_frozen=True)
    return result, receipt, ref, fit, config


def test_complete_fit_receipt_accepts_every_saved_checkpoint():
    assert s.fitting_receipt(*fit_evidence())


@pytest.mark.parametrize("change", ["missing_checkpoint", "duplicate_checkpoint", "wrong_final", "missing_audit",
                                    "validation_use", "assessment_use", "seed", "exposure", "fit", "config"])
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


@pytest.mark.parametrize("change", ["swapped_model", "swapped_metric", "binding", "dropped_evidence", "reordered_panel", "final_reference"])
def test_descriptor_changes_rejected_even_with_real_valid_file_refs(parent, monkeypatch, change):
    monkeypatch.setattr(s, "closed_parent", lambda: parent)
    altered = copy.deepcopy(parent)
    if change == "swapped_model": altered["models"]["1"]["161001"] = parent["models"]["2"]["161001"]
    if change == "swapped_metric": altered["training"]["161001"]["1"] = parent["training"]["161001"]["2"]
    if change == "binding": altered["binding"] = "different-parent"
    if change == "dropped_evidence": altered["evidence"].pop()
    if change == "reordered_panel": altered["schedule"].reverse()
    if change == "final_reference": altered["assessment"] = altered["result"]
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
