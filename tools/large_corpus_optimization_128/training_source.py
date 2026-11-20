"""Authenticate completed scaling evidence without replaying historical work."""
from pathlib import Path

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
    c.require(optimizer["param_groups"] == [expected] and set(optimizer["state"]) == set(ids),
              "Saved AdamW hyperparameters, IDs or state coverage differ")
    c.require(names and len(set(names)) == len(names)
              and all(not name.startswith("value_head.") for name in names), "Invalid AdamW parameter order")
    for index,name in zip(ids,names,strict=True):
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
              "Continuation TRAIN seal changed")
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
    return continuation(fit_id,binding,base)["initializer"]


def control_models(binding, base=c.OUTPUT):
    return _lock(binding,base)["parent"]["models"]
