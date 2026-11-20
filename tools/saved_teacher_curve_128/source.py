"""Read-only closure and checkpoint ancestry for the saved learning curve.

Every optimizer trajectory and old simulation has already been independently
checked. This module authenticates those receipts and saved tensors; it never
calls a model, optimizer, simulator, or new evaluation routine.
"""
from pathlib import Path

import torch

from tools.focused_teacher_fit_128 import assessment as inherited_assessment
from tools.focused_teacher_fit_128 import source as inherited_source
from . import common as c

RESULT_SHA = "771eca943dde7759ca406ffe6a03e55bfb73b32b0bf3ac611f7a1c00d38023af"
AUDIT_SHA = "074648f8056f4e690802b993102f86dd789ea59c1c706c2b192a76929dfff146"


def _plain(ref):
    return {key: ref[key] for key in ("path", "sha256")}


def _same(left, right):
    return _plain(left) == _plain(right)


def checkpoint_identity(payload, initializer, fit, update):
    """Inspect saved tensors only; zero must be the unchanged original model."""
    metadata = {"experiment": "focused_teacher_fit_128", "fit": fit,
                "manifest": c.PARENT_BINDING, "checkpoint_selection": "fixed_final",
                "training_split": "train", "assessment_used": False}
    c.require(fit in c.FITS and update in c.CHECKPOINTS
              and payload["model_kind"] == initializer["model_kind"]
              and payload["model_config"] == initializer["model_config"]
              and payload["generation"] == update and payload["metadata"] == metadata,
              "Saved checkpoint metadata differs from its audited trajectory")
    actual, initial = payload["model_state_dict"], initializer["model_state_dict"]
    c.require(actual.keys() == initial.keys(), "Checkpoint tensor keys changed")
    for name, tensor in actual.items():
        original = initial[name]
        c.require(torch.is_tensor(tensor) and torch.is_tensor(original)
                  and tensor.dtype == original.dtype and tensor.shape == original.shape
                  and torch.isfinite(tensor).all().item(), "Checkpoint tensor schema or values changed")
        if update == 0 or name.startswith("value_head."):
            c.require(torch.equal(tensor, original), "Initializer or frozen value-head identity changed")
    return True


def fitting_receipt(result, receipt, result_ref, fit, config):
    """Check complete optimizer/checkpoint accounting without reproducing it again."""
    c.require(result["manifest"] == receipt["manifest"] == c.PARENT_BINDING
              and result["fit"] == receipt["fit"] == fit and _same(receipt["result"], result_ref)
              and result["config"] == config, "Fit/audit identity differs")
    for flag in ("passed", "complete_fit_bitwise_identical",
                 "all_training_metrics_independently_reproduced", "value_head_frozen", "final_always_selected"):
        c.require(receipt[flag] is True, "Incomplete inherited optimizer reproduction")
    c.require(result["final_always_selected"] is True
              and result["updates"] == result["selected_update"] == receipt["selected_update"] == c.CHECKPOINTS[-1]
              and [row["update"] for row in result["checkpoints"]] == list(c.CHECKPOINTS)
              and result["selected_model"] == result["checkpoints"][-1]["checkpoint"] == receipt["selected_model"]
              and result["selected_training"] == result["checkpoints"][-1]["training"] == receipt["selected_training"],
              "Missing checkpoint or changed fixed-final endpoint")
    for key in ("batch_sequence_sha256", "row_exposures", "per_row_exposures"):
        c.require(result[key] == receipt[key], "Inherited fit exposure accounting changed")
    c.require(result["train_rows"] == 168 and len(result["per_row_exposures"]) == 168
              and sum(result["per_row_exposures"].values()) == result["row_exposures"]
              and all(type(n) is int and n > 0 for n in result["per_row_exposures"].values())
              and result["seed"] == config["train_seeds"][str(fit)]
              and config["checkpoints"] == list(c.CHECKPOINTS), "Changed training population or stream")
    for value in (result, receipt):
        c.require(value["assessment_read"] is False and value["validation_read"] is False,
                  "Inherited training used forbidden data")
    return True


def closed_parent():
    """Authenticate the completed parent and all 39 checkpoints without inference."""
    refs = {}

    def add(ref, read=True):
        item = _plain(ref)
        c.require(item["path"] not in refs or refs[item["path"]] == item,
                  "Conflicting ancestor evidence")
        refs[item["path"]] = item
        return c.verified(item) if read else c.verify_hash(item)

    freeze_ref = c.artifact(c.PARENT/"freeze.json")
    freeze = add(freeze_ref)
    manifest = add(freeze["manifest"])
    c.require(freeze["owner_authorized"] is True
              and freeze["binding"] == c.PARENT_BINDING == c.old.digest(manifest),
              "Focused parent freeze changed")
    # The final result authenticates this exact manifest/data closure. Rehashing
    # the entire dependency inventory is sufficient here; no fit, trajectory,
    # prior test suite, or old feature reconstruction is rerun.
    for path, digest in manifest["identity"]["dependencies"].items():
        add({"path": path, "sha256": digest}, False)
    checks = add(manifest["checks"])
    c.require(checks["exit_code"] == 0 and checks["identity"] == manifest["identity"],
              "Focused parent preflight no longer matches its freeze")
    for ref in checks["logs"]:
        add(ref, False)
    for ref in manifest["source_parent"]["evidence"]:
        add(ref, False)
    status_ref = c.artifact(c.PARENT/"final_status.json")
    status = add(status_ref)
    c.require(status["manifest"] == c.PARENT_BINDING and status["complete"] is True
              and status["all_prescribed_audits_pass"] is True
              and status["result"]["sha256"] == RESULT_SHA and status["audit"]["sha256"] == AUDIT_SHA,
              "Focused parent is not the exact completed audited study")
    result, audit = add(status["result"]), add(status["audit"])
    add(status["report"], False)
    c.require(result["manifest"] == audit["manifest"] == c.PARENT_BINDING
              and result["all_prescribed_audits_pass"] is True and audit["passed"] is True
              and audit["result"] == status["result"] and audit["training_reproduced"] is True
              and audit["historical_simulations_replayed"] is False and result["new_simulations"] == 0,
              "Focused final closure differs")
    data = add(result["data"])
    c.require(data["manifest"] == c.PARENT_BINDING and data["parent"] == manifest["source_parent"]
              and data["schedule"] == data["parent"]["schedule"]
              and data["assessment_in_payload"] is False and data["historical_replay"] is False
              and data["rows"] == 336 and data["train_rows"] == data["validation_rows"] == 168
              and data["independent_lawful_inputs_checked"] is True
              and data["source_trace_membership_checked"] is True,
              "Focused old-only input closure differs")
    inherited_source.validate_schedule(data["schedule"])
    c.require(set(data["files"]) == {"train", "validation"}
              and data["files"]["train"]["path"] != data["files"]["validation"]["path"],
              "Training and validation payloads were merged")
    for ref in data["files"].values():
        add(ref, False)
    registry = add(result["models"])
    fit_lock = add(c.artifact(c.PARENT/"fits_lock.json"))
    audit_lock = add(c.artifact(c.PARENT/"fit_audits_lock.json"))
    c.require(registry["manifest"] == fit_lock["manifest"] == audit_lock["manifest"] == c.PARENT_BINDING
              and registry["fixed_final"] is True and registry["all_fit_audits_pass"] is True
              and registry["fits"] == fit_lock["files"]
              and registry["audits"] == audit_lock["files"] == audit["fits"]
              and [ref["fit"] for ref in fit_lock["files"]] == list(c.FITS)
              and [ref["fit"] for ref in audit_lock["files"]] == list(c.FITS),
              "Incomplete focused fit registry")
    models = {str(update): {} for update in c.CHECKPOINTS}
    training, initializers, fit_results, fit_audits = {}, {}, {}, {}
    training_identities = set()
    for fit, fr, ar in zip(c.FITS, registry["fits"], registry["audits"], strict=True):
        fitted, checked = add(fr), add(ar)
        fitting_receipt(fitted, checked, fr, fit, manifest["identity"]["config"])
        c.require(result["training"][str(fit)] == fitted, "Final result links another training trajectory")
        original = data["parent"]["models"][f"anchor_{fit}"]
        c.require(fitted["initializer"] == original, "Focused fit changed original initializer")
        add(original, False)
        initializers[str(fit)] = _plain(original)
        initial = torch.load(c.verify_hash(original), map_location="cpu", weights_only=True)
        training[str(fit)] = {}
        training_identities.add(fitted["training_rows_sha256"])
        fit_results[str(fit)], fit_audits[str(fit)] = _plain(fr), _plain(ar)
        for point in fitted["checkpoints"]:
            update = point["update"]
            payload = torch.load(add(point["checkpoint"], False), map_location="cpu", weights_only=True)
            checkpoint_identity(payload, initial, fit, update)
            report = add(point["training"])
            c.require(report["assessment_read"] is False and report["validation_read"] is False
                      and report["guide_is_stopping_rule"] is False
                      and report["metrics"]["all_rows"]["rows"] == len(report["rows"]) == 168,
                      "Saved training metric belongs to another population")
            models[str(update)][str(fit)] = _plain(point["checkpoint"])
            training[str(fit)][str(update)] = _plain(point["training"])
        final = registry["models"][f"focused_{fit}"]
        c.require(final["source_fit"] == fit and final["selected_update"] == c.CHECKPOINTS[-1]
                  and _same(final, models[str(c.CHECKPOINTS[-1])][str(fit)]),
                  "Last curve checkpoint differs from assessed fixed-final model")
    c.require(len(training_identities) == 1, "The three fits used different training sets")
    assessment_audit = add(result["assessment_audit"])
    # Outcome-bearing final assessment is only authenticated here. Its contents
    # are consumed by the new assessor after every curve prediction is sealed.
    add(result["assessment"], False)
    c.require(audit["assessment"] == result["assessment_audit"]
              and assessment_audit["manifest"] == c.PARENT_BINDING and assessment_audit["passed"] is True
              and assessment_audit["result"] == result["assessment"]
              and assessment_audit["models"] == result["models"]
              and assessment_audit["states"] == 336 and assessment_audit["unknown_outcomes"] == 0
              and assessment_audit["new_trajectory_replays"] == 0,
              "Focused assessment closure changed")
    for flag in ("all_new_arithmetic_pass", "lawful_inputs_and_private_invariance_pass",
                 "new_focused_inference_independently_reproduced", "old_choices_independently_reconstructed",
                 "inherited_assessment_audits"):
        c.require(assessment_audit[flag] is True, "Incomplete independent focused assessment")
    predictions_ref = assessment_audit["predictions"]
    predictions = add(predictions_ref)
    c.require(predictions["manifest"] == c.PARENT_BINDING and predictions["complete"] is True
              and predictions["states"] == len(predictions["predictions"]) == 336
              and predictions["models"] == result["models"]
              and predictions["all_predictions_precede_assessment_open"] is True
              and predictions["selection_used_assessment"] is False,
              "Focused prediction lock changed")
    descriptors = data["parent"]["evaluation_sources"]
    for index, (prediction, descriptor, spec) in enumerate(zip(predictions["predictions"], descriptors,
                                                            data["schedule"], strict=True)):
        c.require(prediction["index"] == descriptor["index"] == spec["index"] == index
                  and prediction["spec"] == descriptor["spec"] == spec
                  and prediction["fit"] == spec["fit"] and prediction["row_id"] == spec["source_rowid"]
                  and _same(prediction["model"], models[str(c.CHECKPOINTS[-1])][str(spec["fit"])])
                  and prediction["source_selection"] == descriptor["selection"],
                  "Fixed-final prediction changed source or model")
        for key in ("root", "audit", "selection"):
            add(descriptor[key], False)
    return {"binding": c.PARENT_BINDING, "parent_binding": c.PARENT_BINDING,
            "freeze": freeze_ref, "manifest": freeze["manifest"], "completion": status_ref,
            "result": status["result"], "audit": status["audit"],
            "assessment": result["assessment"], "assessment_audit": result["assessment_audit"],
            "predictions": predictions_ref, "data": result["data"], "final_models": result["models"],
            "models": models, "training": training, "initializers": initializers,
            "fit_results": fit_results, "fit_audits": fit_audits,
            "schedule": data["schedule"], "evaluation_sources": descriptors,
            "checkpoint_zero_equals_initializer": True, "final_checkpoint_equals_prior_endpoint": True,
            "historical_fit_replays": 0, "historical_simulation_replays": 0,
            "new_inference": False, "evidence": [refs[key] for key in sorted(refs)],
            "inventory": {key: refs[key]["sha256"] for key in sorted(refs)}}


def validate(parent):
    """Reauthenticate the saved descriptor, including its complete label-to-file map."""
    c.require(parent["binding"] == parent["parent_binding"] == c.PARENT_BINDING,
              "Wrong saved-curve parent binding")
    c.require(parent == closed_parent(), "Saved-curve parent evidence or checkpoint mapping changed")
    return parent


def panel(parent):
    """Return the unchanged old-input panel, without opening assessment matrices."""
    # A caller validates the complete descriptor at its freeze boundary. Here
    # retain the narrower old-input path so prediction locking cannot read scores.
    c.require(parent["binding"] == c.PARENT_BINDING, "Wrong panel parent")
    data = c.verified(parent["data"])
    c.require(data["manifest"] == c.PARENT_BINDING and data["schedule"] == parent["schedule"]
              and data["parent"]["evaluation_sources"] == parent["evaluation_sources"],
              "Saved panel identity changed")
    for ref in data["files"].values():
        c.verify_hash(ref)
    rows, inputs, descriptors = inherited_assessment._panel(c.PARENT_BINDING, c.PARENT)
    c.require(descriptors == parent["evaluation_sources"]
              and [item["spec"] for item in inputs] == parent["schedule"], "Saved panel changed")
    return rows, inputs, descriptors
