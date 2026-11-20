"""Authenticate exact completed scaling TRAIN bytes and original161 initializers; no new labels."""
from pathlib import Path
import hashlib
import os
import uuid

import torch

from tools.independent_game_scaling_128 import learning as inherited
from tools.independent_game_scaling_128 import training_source as inherited_source
from . import common as c

COUNTS = {"old": 9461, "added": 11520}
_CLOSED = None


def _plain_ref(ref):
    return {key: ref[key] for key in ("path", "sha256")}


def validate_adamw(payload, names, config, update):
    """Structural validation, not reconstruction of historical AdamW moments."""
    optimizer = payload["optimizer_state_dict"]
    ids = list(range(len(names)))
    expected = {"lr":config["lr"],"betas":(.9,.999),"eps":1e-8,"weight_decay":config["weight_decay"],
                "amsgrad":False,"foreach":None,"maximize":False,"capturable":False,
                "differentiable":False,"fused":None,"params":ids}
    c.require(optimizer["param_groups"] == [expected] and set(optimizer["state"]) == (set(ids) if update else set()),
              "Saved AdamW hyperparameters, IDs or state coverage differ")
    c.require(names and len(set(names)) == len(names)
              and all(not name.startswith(("value_head.","base.value_head.")) for name in names), "Invalid AdamW parameter order")
    for index,name in zip(ids if update else [],names if update else [],strict=True):
        state,parameter = optimizer["state"][index],payload["model_state_dict"][name]
        c.require(set(state) == {"step","exp_avg","exp_avg_sq"}
                  and state["step"].shape == torch.Size([]) and float(state["step"]) == update,
                  "Saved AdamW step differs")
        for key in ("exp_avg","exp_avg_sq"):
            value = state[key]
            c.require(value.shape == parameter.shape and value.dtype == parameter.dtype == torch.float32
                      and bool(torch.isfinite(value).all()), "Saved AdamW moment shape/dtype/value differs")
        c.require(bool((state["exp_avg_sq"] >= 0).all()), "Negative AdamW second moment")
    c.require(all(bool(torch.isfinite(value).all()) for value in payload["model_state_dict"].values()),
              "Nonfinite inherited model")


def validate_continuation(record, fit_id, *, expected_update=7136):
    """Check model/optimizer identity, order and audited original trajectory."""
    result, audit = c.verified(record["result"]), c.verified(record["audit"])
    c.require(result["fit"] == audit["fit"] == fit_id and result["arm"] == audit["arm"] == "large"
              and result["complete"] and audit["result"] == record["result"]
              and all(audit[key] is True for key in inherited.AUDIT_FLAGS)
              and result["manifest"] == audit["manifest"]
              and result["selected_update"] == audit["selected_update"] == expected_update
              and result["selected_model"] == audit["selected_model"] == record["model"]
              and result["selected_training"] == audit["selected_training"] == record["final_training"]
              and result["initializer"] == record["initializer"]
              and result["checkpoints"][0]["training"] == record["initial_training"],
              "Continuation does not belong to the audited large fit")
    c.require(not any(v[key] for v in (result, audit) for key in ("assessment_read", "validation_read")),
              "Held-out information in inherited fitting")
    resume = c.verified(record["resume"])
    c.require(resume["checkpoint"] == record["optimizer"] and resume["update"] == expected_update
              and resume["identity_sha256"] == c.old.digest(result["identity"]), "Inherited optimizer pointer changed")
    payload = torch.load(c.verify_hash(record["optimizer"]), map_location="cpu", weights_only=True)
    model = torch.load(c.verify_hash(record["model"]), map_location="cpu", weights_only=True)
    c.require(payload["identity"] == result["identity"] and payload["update"] == expected_update
              and payload["preclip_gradient_norms"] == result["optimization_stability"]["preclip_gradient_norms"]
              and len(payload["preclip_gradient_norms"]) == expected_update
              and inherited._equal_torch(payload["model_state_dict"], model["model_state_dict"]),
              "Inherited optimizer/model trajectory differs")
    groups = payload["optimizer_state_dict"]["param_groups"]
    c.require(len(groups) == 1 and groups[0]["lr"] == result["config"]["lr"]
              and groups[0]["weight_decay"] == result["config"]["weight_decay"]
              and len(groups[0]["params"]) == len(result["trainable_parameter_names"])
              and result["identity"]["trainable_parameter_names"] == result["trainable_parameter_names"],
              "Inherited AdamW recipe or parameter order differs")
    c.require(model["metadata"] == inherited._metadata(fit_id,"large",result["manifest"],expected_update)
              and model["generation"] == expected_update, "Inherited checkpoint identity differs")
    validate_adamw(payload,result["trainable_parameter_names"],result["config"],expected_update)
    for key in ("initializer", "initial_training", "final_training"):
        c.verify_hash(record[key])
    return result


def closed_parent():
    global _CLOSED
    if _CLOSED is not None:
        return _CLOSED
    parent = c.PARENT
    freeze_ref, status_ref = c.artifact(parent/"freeze.json"), c.artifact(parent/"final_status.json")
    freeze, status = c.verified(freeze_ref), c.verified(status_ref)
    manifest = c.verified(freeze["manifest"])
    c.require(freeze["binding"] == c.old.digest(manifest) == c.PARENT_BINDING
              and status["manifest"] == c.PARENT_BINDING and status["complete"] is True
              and status["all_prescribed_audits_pass"] is True, "Scaling study is not complete/authenticated")
    result, final_audit = c.verified(status["result"]), c.verified(status["audit"])
    c.require(final_audit["passed"] is True and final_audit["result"] == status["result"]
              and result["manifest"] == final_audit["manifest"] == c.PARENT_BINDING
              and result["learning"] == final_audit["learning"], "Scaling closure differs")
    completion = c.verified(result["learning"])
    models = inherited.validate_models(c.PARENT_BINDING, parent)
    data = inherited_source.validate(c.PARENT_BINDING, parent, authenticate_parent=False)
    c.require(c.verified(completion["models"]) == models and c.verified(completion["data"]) == data,
              "Scaling data/model completion references differ")
    c.require(data["counts"]["old"] == 9461 and data["counts"]["added"] == 11520
              and data["old128_only"] is True and not data["assessment_in_payload"]
              and data["partition"]["large"] == list(range(11520)), "Large TRAIN corpus changed")
    continuations = {}
    evidence = [freeze_ref, freeze["manifest"], status_ref, status["result"], status["audit"],
                result["learning"], completion["models"], completion["data"], data["audit"],
                data["files"]["old"], data["files"]["added"]]
    for fit in c.FITS:
        index = c.FITS.index(fit)*3+2
        fit_ref, audit_ref = models["fits"][index], models["audits"][index]
        fitted = c.verified(fit_ref)
        folder = parent/"fits"/f"fit_{fit}"/"large"
        resume_ref = c.artifact(folder/"resume.json")
        resume = c.verified(resume_ref)
        record = {"result":fit_ref,"audit":audit_ref,"resume":resume_ref,"optimizer":resume["checkpoint"],
                  "model":fitted["selected_model"],"initializer":fitted["initializer"],
                  "initial_training":fitted["checkpoints"][0]["training"],"final_training":fitted["selected_training"]}
        validate_continuation(record,fit)
        c.require(record["model"] == models["models"][f"large_{fit}"]
                  and record["initializer"] == models["models"][f"anchor_{fit}"]
                  and fitted["config"] == inherited.CONFIG, "Unrecognized scaling continuation")
        continuations[str(fit)] = record
        evidence.extend(record.values())
    unique = {str(c.verify_hash(ref).resolve()):_plain_ref(ref) for ref in evidence}
    _CLOSED = {"binding":c.PARENT_BINDING,"freeze":freeze_ref,"status":status_ref,
        "data":completion["data"],"partition":data["partition"],"continuations":continuations,
        "models":{f"{arm}_{fit}":models["models"][f"{'large' if arm == 'control' else 'anchor'}_{fit}"]
                  for arm in ("anchor","control") for fit in c.FITS},
        "source_parent":manifest["source_parent"],"source_schedule":manifest["schedule"],
        "evidence":list(unique.values()),"historical_simulations_replayed":0,"historical_fits_replayed":0,
        "historical_rng_state_present":False,"deterministic_training_without_rng_draws":True}
    _CLOSED["historical_optimizer_moments_independently_replayed"] = False
    _CLOSED["historical_optimizer_provenance"] = "completed_writer_checkpoint_authenticated_now_and_structurally_validated"
    return _CLOSED


def prepare(binding, base=c.OUTPUT, parent=None):
    base = Path(base)
    parent = closed_parent() if parent is None else parent
    if base.resolve().is_relative_to(c.OUTPUT.resolve()):
        c.require(parent == closed_parent(), "Production parent override forbidden")
    data = c.verified(parent["data"])
    value = {"manifest":binding,"parent":parent,"files":{key:data["files"][key] for key in COUNTS},
        "row_digests":{key:data["row_digests"][key] for key in COUNTS},"counts":dict(COUNTS),
        "partition":data["partition"],"old128_only":True,"tensor_bytes_reused":True,
        "assessment_in_payload":False,"validation_read":False,"new_labels":0,
        "historical_simulations_replayed":0,"historical_fits_replayed":0}
    path = base/"data_lock.json"
    inherited._new_json_or_equal(path,value)
    return c.artifact(path)


def _lock(binding, base=c.OUTPUT):
    base = Path(base)
    lock = c.old.read_json(base/"data_lock.json")
    parent = lock["parent"]
    original = c.verified(parent["data"])
    c.require(lock["manifest"] == binding and parent["binding"] == c.PARENT_BINDING
              and lock["counts"] == COUNTS and lock["partition"] == original["partition"]
              and all(lock[key] == {pool:original[key][pool] for pool in COUNTS} for key in ("files","row_digests"))
              and lock["old128_only"] is lock["tensor_bytes_reused"] is True
              and lock["assessment_in_payload"] is lock["validation_read"] is False
              and lock["new_labels"] == lock["historical_simulations_replayed"] == lock["historical_fits_replayed"] == 0,
              "Structured TRAIN seal changed")
    return lock


def validate(binding, base=c.OUTPUT, authenticate_parent=True):
    lock = _lock(binding,base)
    if authenticate_parent:
        c.require(lock["parent"] == closed_parent(), "Closed parent changed")
    for ref in lock["files"].values():
        c.verify_hash(ref)
    return lock


def load_pools(binding, base=c.OUTPUT):
    lock = _lock(binding,base)
    pools = {}
    for pool in COUNTS:
        rows = torch.load(c.verify_hash(lock["files"][pool]),map_location="cpu",weights_only=True)
        c.require(len(rows) == COUNTS[pool] and inherited_source.row_digest(rows) == lock["row_digests"][pool],
                  "Inherited TRAIN tensors changed")
        inherited._validate_rows(rows)
        pools[pool] = rows
    return pools


def partitions(binding, base=c.OUTPUT):
    return _lock(binding,base)["partition"]


def continuation(fit_id, binding, base=c.OUTPUT):
    c.require(fit_id in c.FITS, "Unknown continuation fit")
    record = _lock(binding,base)["parent"]["continuations"][str(fit_id)]
    validate_continuation(record,fit_id)
    return record


def initializer(fit_id, base=c.OUTPUT):
    binding = c.old.read_json(Path(base)/"data_lock.json")["manifest"]
    c.require(fit_id in c.FITS,"Unknown initializer fit")
    ref = _lock(binding,base)["parent"]["models"][f"anchor_{fit_id}"]
    c.verify_hash(ref)
    return ref


def control_models(binding, base=c.OUTPUT):
    return {key:value for key,value in _lock(binding,base)["parent"]["models"].items() if key.startswith("anchor_")}


def _feature_inputs(binding, base, rows):
    from . import model
    fixture = rows is not None
    if fixture:
        c.require(not base.resolve().is_relative_to(c.OUTPUT.resolve()), "Feature fixture cannot target production")
        pools = rows
        c.require(set(pools) == {"old", "added"} and all(pools.values()), "Both TRAIN pools required")
        inputs = {"files": None, "row_digests": {key: inherited_source.row_digest(value) for key, value in pools.items()},
                  "data_lock": None}
    else:
        lock = _lock(binding, base)
        pools = load_pools(binding, base)
        inputs = {"files": {key: lock["files"][key] for key in ("old", "added")},
                  "row_digests": {key: lock["row_digests"][key] for key in ("old", "added")},
                  "data_lock": c.artifact(base/"data_lock.json")}
    c.require(all(row["split"] == "train" for pool in pools.values() for row in pool), "Held-out features forbidden")
    return pools, {"manifest": binding, "inputs": inputs, "fixture": fixture,
        "counts": {key: len(value) for key, value in pools.items()}, "feature_implementation": c.artifact(Path(model.__file__)),
        "model_config": model.MODEL_CONFIG}


def audit_features(binding, base=None, *, rows=None):
    """Check every new TRAIN action/card input, without network inference.

    The independent scalar path decodes take, return and payment blocks, while
    production uses the recorded net flow. Reused lawful input audits establish
    their observation/privacy ancestry; neither path reads assessment outcomes.
    """
    from . import model
    from tools.proposal_ranking_128_recovery_01.reporting import StatusReporter
    base = Path(c.OUTPUT if base is None else base)
    c.budget_boundary(base)
    torch.set_num_threads(c.CONFIG["torch_threads"])
    pools, identity = _feature_inputs(binding, base, rows)
    path = base/"training_feature_audit.json"
    c.require(not path.exists(), "Feature audit already complete; authenticate its receipt")
    totals, maximum_error = {}, 0.
    reporter = StatusReporter(base/"training_feature_status", {"manifest": binding})
    with torch.no_grad():
        for kind, pool in pools.items():
            production_hash, independent_hash = hashlib.sha256(), hashlib.sha256()
            actions = known = 0
            for index, row in enumerate(pool):
                c.budget_boundary(base)
                observation, features = row["observation"][None], row["features"][None]
                c.require(torch.isfinite(observation).all().item() and torch.isfinite(features).all().item(), "Nonfinite feature input")
                for arm in model.ARMS:
                    actual, mask = model.pair_features(observation, features, arm)
                    expected, independent_mask = model.independent_pair_features(observation, features, arm)
                    c.require(torch.equal(mask, independent_mask) and actual.shape == expected.shape
                              and torch.isfinite(actual).all().item() and torch.isfinite(expected).all().item(),
                              "Independent relation feature shape/mask differs")
                    error = float((actual-expected).abs().max())
                    c.require(error <= 1e-7, "Independent relation arithmetic differs")
                    maximum_error = max(maximum_error, error)
                    for digest, tensor in ((production_hash, actual), (independent_hash, expected)):
                        digest.update(c.old.digest({"id": row["id"], "arm": arm, "shape": list(tensor.shape)}).encode())
                        digest.update(tensor.contiguous().numpy().tobytes())
                        digest.update(mask.contiguous().numpy().tobytes())
                actions += len(row["features"])
                known += int(mask.sum())
                if (index+1) % 128 == 0 or index+1 == len(pool):
                    reporter.event("training_relation_features_audit", pool=kind, rows=index+1, total_rows=len(pool))
            totals[kind] = {"rows": len(pool), "legal_action_rows": actions, "known_card_rows": known,
                "production_features_sha256": production_hash.hexdigest(), "independent_features_sha256": independent_hash.hexdigest()}
    value = {**identity, "passed": True, "arms": list(model.ARMS), "totals": totals,
        "maximum_absolute_error": maximum_error, "absolute_tolerance": 1e-7,
        "all_training_rows_and_legal_actions_checked": True, "independent_payment_return_take_arithmetic": True,
        "assessment_read": False, "validation_read": False, "new_simulations": 0, "network_inference": 0}
    temporary = base/("training_feature_audit."+uuid.uuid4().hex+".tmp")
    c.old.save_new(temporary, value)
    os.replace(temporary, path)
    return c.artifact(path)


def validate_features(binding, base=None, *, rows=None):
    from . import model
    base = Path(c.OUTPUT if base is None else base)
    value = c.old.read_json(base/"training_feature_audit.json")
    if rows is None:
        lock = _lock(binding, base)
        expected_inputs = {"files": {key: lock["files"][key] for key in ("old", "added")},
            "row_digests": {key: lock["row_digests"][key] for key in ("old", "added")},
            "data_lock": c.artifact(base/"data_lock.json")}
        expected = {"manifest": binding, "inputs": expected_inputs, "fixture": False,
            "counts": {key: COUNTS[key] for key in ("old", "added")},
            "feature_implementation": c.artifact(Path(model.__file__)), "model_config": model.MODEL_CONFIG}
    else:
        _, expected = _feature_inputs(binding, base, rows)
    c.require(all(value[key] == item for key, item in expected.items()), "Training feature audit identity changed")
    c.require(value["passed"] is value["all_training_rows_and_legal_actions_checked"]
              is value["independent_payment_return_take_arithmetic"] is True
              and value["arms"] == list(model.ARMS) and value["absolute_tolerance"] == 1e-7
              and 0 <= value["maximum_absolute_error"] <= value["absolute_tolerance"]
              and value["assessment_read"] is value["validation_read"] is False
              and value["new_simulations"] == value["network_inference"] == 0
              and set(value["totals"]) == {"old", "added"}, "Incomplete training feature audit")
    for key in ("old", "added"):
        c.require(value["totals"][key]["rows"] == expected["counts"][key]
                  and value["totals"][key]["legal_action_rows"] >= expected["counts"][key]
                  and all(len(value["totals"][key][name]) == 64 for name in
                      ("production_features_sha256", "independent_features_sha256")), "Feature audit coverage changed")
    return value
