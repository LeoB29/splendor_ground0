"""Authenticated immutable data and matched BROAD controls; no old replay.

The completed coverage experiment already audited every added input, membership,
target and game-disjointness claim. Its pinned completion and frozen inventory
authenticate those receipts. Reusing the exact tensor bytes avoids another raw
trajectory conversion, while fitting opens only the two TRAIN tensors.
"""
from pathlib import Path
import hashlib
import os
import uuid

import torch

from tools.independent_game_coverage_128_recovery_01 import training_source as coverage_source
from tools.independent_game_coverage_128_recovery_01 import learning as coverage_learning
from tools.soft_teacher_targets_128 import source as inherited
from . import common as c

RESULT_SHA = "52d2af32ff64c73ef606ec74e38a5e0c789e87d33180dae357ae850dbebfaade"
AUDIT_SHA = "ca49aba2e55704ac76d2145557e03bec99b2de408a8af8d205dac3b2d8ee879c"
DATA_SHA = "7013b84d738a87c1fa9b0586ee79e758f1db130f0b3cbdb9dbe4406dc8e49812"
COUNTS = {"old": 9461, "added": 720, "validation": 2489}
DATA_FIELDS = ("files", "row_digests", "counts", "added_identities", "partition", "retired_evaluation_panels")
PANEL_FIELDS = ("schedule", "panel_file", "panel_digest", "original_train_probe_ids")
row_digest, validate_row = inherited.row_digest, inherited.validate_row
_plain, _check_panel_identity = inherited._plain, inherited._check_panel_identity


def _data_receipt(data, original, audit):
    """Check inherited conversion and retirement claims, without opening outcomes."""
    parent = data["parent"]
    c.require(data["manifest"] == c.PARENT_BINDING and data["counts"] == COUNTS
              and data["assessment_in_payload"] is data["historical_replay"] is False
              and data["tensor_bytes_reused"] is True
              and data["inherited_data"] == parent["reused_data"]
              and data["inherited_data"]["sha256"] == inherited.LOWRATE_DATA_SHA
              and original["manifest"] == inherited.LOWRATE_BINDING,
              "Completed training data identity changed")
    for new, old in (("old", "train"), ("validation", "validation")):
        c.require(data["files"][new] == original["files"][old]
                  and data["row_digests"][new] == original["row_digests"][old],
                  "Original TRAIN/VAL tensor split changed")
    c.require(data["added_identities"] == [coverage_source.row_identity(d) for d in parent["added_sources"]]
              and data["partition"] == coverage_source.partition(data["added_identities"])
              and data["retired_evaluation_panels"] == parent["retired_evaluation_panels"],
              "Recorded game partition or retirement changed")
    proof, retired = parent["training_disjointness"], data["retired_evaluation_panels"]
    old_seeds = proof["old_split_environment_seeds"]
    c.require(proof["historical_games"] == 1204 and proof["added_distinct_games"] == 720
              and proof["all_inherited_train_validation_and_anchor_ancestry_checked"] is True
              and len(set(proof["added_environment_seeds"])) == 720
              and len(set(old_seeds["train"])) == 159 and len(set(old_seeds["validation"])) == 42
              and set(old_seeds["train"]).isdisjoint(old_seeds["validation"])
              and set(proof["added_environment_seeds"]).isdisjoint(proof["historical_environment_seeds"])
              and retired["permanent"] is True and retired["rows"] == 720
              and retired["ids"] == [r["id"] for r in data["added_identities"]],
              "Independent-game/split or permanent retirement proof changed")
    c.require(audit["manifest"] == c.PARENT_BINDING and audit["passed"] is True
              and audit["added_rows"] == 720 and audit["assessment_in_payload"] is False
              and audit["historical_trajectory_replays"] == audit["new_simulations"] == 0
              and all(audit[k] == data[k] for k in ("files", "row_digests", "partition", "retired_evaluation_panels"))
              and audit["training_disjointness"] == proof
              and all(audit[k] is True for k in ("independent_public_private_invariant_encoding",
                  "all_canonical_memberships_checked", "every128_world_target_independently_summed",
                  "actual_source_turn_membership_checked", "inherited_tensor_bytes_unchanged")),
              "Completed conversion audit changed")
    c.require(original["assessment_in_payload"] is original["historical_replay"] is False
              and original["new_simulations"] == original["historical_fit_replays"] == 0
              and all(original[k] is True for k in ("independent_lawful_inputs_audit_inherited",
                  "source_trace_membership_checked", "new_old_target_arithmetic_checked")),
              "Original full-corpus conversion audit changed")
    inherited.validate_schedule(original["schedule"])
    return True


def fit_receipt(trained, checked, fit_ref, fit, model, anchor, partition):
    c.require(trained["manifest"] == checked["manifest"] == c.PARENT_BINDING
              and trained["fit"] == checked["fit"] == fit
              and trained["complete"] is True and trained["config"] == coverage_learning.CONFIG
              and checked["result"] == fit_ref and trained["initializer"] == anchor
              and trained["narrow_indices"] == partition["narrow_indices"][str(fit)]
              and trained["paired_schedule"] == checked["paired_schedule"]
              and trained["validation_read"] is checked["validation_read"] is False
              and trained["assessment_read"] is checked["assessment_read"] is False,
              "Matched BROAD fitting receipt changed")
    for flag in ("passed", "complete_fit_bitwise_identical", "exact_between_arm_old_and_cell_streams",
                 "soft_target_arithmetic_independently_reproduced"):
        c.require(checked[flag] is True, "Incomplete matched fitting audit")
    arm, audit = trained["arms"]["broad"], checked["arms"]["broad"]
    for flag in ("complete_fit_bitwise_identical", "all_training_metrics_independently_reproduced",
                 "optimization_stability_independently_reproduced", "soft_target_arithmetic_independently_reproduced",
                 "value_head_frozen", "final_always_selected"):
        c.require(audit[flag] is True, "Incomplete BROAD reproduction audit")
    c.require(arm["selected_model"] == audit["selected_model"] == model
              and model["source_fit"] == fit and model["arm"] == "broad"
              and arm["initializer"] == anchor and arm["selected_update"] == audit["selected_update"] == 1784
              and arm["updates"] == audit["updates"] == 1784
              and arm["old_rows"] == 9461 and arm["added_rows_exposed"] == 720
              and arm["row_exposures"] == audit["row_exposures"] == 114044
              and arm["old_exposures"] == arm["added_exposures"] == 57022
              and arm["validation_read"] is arm["assessment_read"] is False
              and arm["seed"] == coverage_learning.CONFIG["train_seeds"][str(fit)]
              and [p["update"] for p in arm["checkpoints"]] == [0, 1784]
              and arm["checkpoints"][-1]["checkpoint"] == model
              and arm["checkpoints"][-1]["training"] == arm["selected_training"] == audit["selected_training"],
              "BROADS's fixed endpoint, population or initializer changed")
    for key in ("batch_sequence_sha256", "batch_lengths", "old_batch_stream_sha256", "added_cell_batch_stream_sha256",
                "per_row_exposures", "per_added_cell_exposures", "per_added_game_exposures", "per_old_game_exposures",
                "optimization_stability"):
        c.require(arm[key] == audit[key], "BROADS's exact exposure/stability audit changed")
    return True


def closed_parent():
    """Hash the sealed closure once; inherit independent audits, never re-run fits."""
    refs = {}
    def add(ref, read=True):
        plain = _plain(ref)
        c.require(plain["path"] not in refs or refs[plain["path"]] == plain,
                  "Conflicting immutable source identities")
        if plain["path"] not in refs:
            c.verify_hash(plain)
            refs[plain["path"]] = plain
        return c.old.read_json(c.ROOT/plain["path"]) if read else c.ROOT/plain["path"]

    freeze = add(c.artifact(c.PARENT/"freeze.json"))
    manifest = add(freeze["manifest"])
    c.require(freeze["owner_authorized"] is True and freeze["binding"] == c.PARENT_BINDING == c.old.digest(manifest)
              and manifest["identity"]["config"] == c.coverage.CONFIG, "Completed coverage freeze changed")
    prior = manifest["source_parent"]
    c.require(prior["inventory"] == {r["path"]: r["sha256"] for r in prior["evidence"]},
              "Inherited authenticated inventory changed")
    for ref in prior["evidence"]:
        add(ref, False)
    for path, sha in manifest["identity"]["dependencies"].items():
        add({"path": path, "sha256": sha}, False)
    checks = add(manifest["checks"])
    c.require(checks["exit_code"] == 0 and checks["identity"] == manifest["identity"], "Coverage preflight changed")
    for ref in checks["logs"]:
        add(ref, False)
    status = add(c.artifact(c.PARENT/"final_status.json"))
    c.require(status["manifest"] == c.PARENT_BINDING and status["complete"] is True
              and status["all_prescribed_audits_pass"] is True
              and status["result"]["sha256"] == RESULT_SHA and status["audit"]["sha256"] == AUDIT_SHA,
              "Coverage is not the fixed completed audited experiment")
    result, audit = add(status["result"]), add(status["audit"])
    c.require(result["manifest"] == audit["manifest"] == c.PARENT_BINDING
              and audit["passed"] is result["all_prescribed_audits_pass"] is True
              and audit["result"] == status["result"] and result["learning"] == audit["learning"],
              "Coverage final audit linkage changed")
    complete = add(result["learning"])
    data = add(complete["data"])
    c.require(complete["manifest"] == c.PARENT_BINDING and complete["data"]["sha256"] == DATA_SHA
              and data["parent"] == prior, "Coverage learning/data ancestry changed")
    original, conversion = add(data["inherited_data"]), add(data["audit"])
    _data_receipt(data, original, conversion)
    for ref in (*data["files"].values(), original["panel_file"]):
        add(ref, False)
    registry = add(complete["models"])
    fits, audits = add(complete["fits"]), add(complete["fit_audits"])
    c.require(registry["manifest"] == fits["manifest"] == audits["manifest"] == c.PARENT_BINDING
              and registry["complete"] is registry["all_new_fits_independently_reproduced"] is True
              and registry["assessment_used_for_selection"] is False
              and registry["fits"] == fits["files"] and registry["audits"] == audits["files"]
              and [r["fit"] for r in registry["fits"]] == list(c.FITS)
              == [r["fit"] for r in registry["audits"]], "Completed model registry changed")
    models = {f"{arm}_{fit}": registry["models"][f"{arm}_{fit}"]
              for arm in ("anchor", "broad") for fit in c.FITS}
    fit_refs = {}
    for fit, fr, ar in zip(c.FITS, registry["fits"], registry["audits"], strict=True):
        trained, checked = add(fr), add(ar)
        anchor, model = models[f"anchor_{fit}"], models[f"broad_{fit}"]
        c.require(anchor == prior["models"][f"anchor_{fit}"]
                  and anchor["sha256"] == inherited.inherited_models.EXPECTED[f"anchor_{fit}"]
                  and anchor["source_fit"] == fit, "Original initializer changed")
        fit_receipt(trained, checked, fr, fit, model, anchor, data["partition"])
        for ref in (anchor, model):
            add(ref, False)
        for point in trained["arms"]["broad"]["checkpoints"]:
            add(point["checkpoint"], False)
            add(point["training"], False)
        fit_refs[str(fit)] = {"result": fr, "audit": ar}
    panel_parent = original["parent"]
    c.require(panel_parent["source_binding"] == c.SOURCE_BINDING
              and panel_parent["schedule"] == original["schedule"], "Old336 source identity changed")
    for descriptor in panel_parent["evaluation_sources"]:
        for key in ("selection", "root", "audit"):
            add(descriptor[key], False)
    return {"binding": c.PARENT_BINDING, "result": status["result"], "audit": status["audit"],
            "learning": result["learning"], "coverage_data": complete["data"], "original_data": data["inherited_data"],
            "conversion_audit": data["audit"], "models": models, "broad_fits": fit_refs,
            "model_config": prior["model_config"], "corpus": prior["corpus"],
            "source_binding": c.SOURCE_BINDING, "schedule": original["schedule"],
            "evaluation_sources": panel_parent["evaluation_sources"],
            "training_disjointness": prior["training_disjointness"],
            "retired_evaluation_panels": data["retired_evaluation_panels"],
            "evidence": [refs[p] for p in sorted(refs)], "inventory": {p: refs[p]["sha256"] for p in sorted(refs)},
            "dependencies": {p: refs[p]["sha256"] for p in sorted(refs) if not p.startswith("outputs/")},
            "historical_fit_replays": 0, "historical_trajectory_replays": 0, "new_simulations": 0,
            "conversion_audits_inherited": True, "assessment_in_training_payload": False}


def prepare(binding, base=None, parent=None):
    base = Path(c.OUTPUT if base is None else base)
    parent = closed_parent() if parent is None else parent
    c.require(not (base/"data_lock.json").exists(), "One-shot data preparation")
    data, original = c.verified(parent["coverage_data"]), c.verified(parent["original_data"])
    _data_receipt(data, original, c.verified(parent["conversion_audit"]))
    value = {"manifest": binding, "parent": parent, **{k: data[k] for k in DATA_FIELDS},
             **{k: original[k] for k in PANEL_FIELDS}, "tensor_bytes_reused": True,
             "conversion_audits_inherited": True, "assessment_in_payload": False,
             "new_simulations": 0, "historical_fit_replays": 0, "historical_trajectory_replays": 0}
    c.old.save_new(base/"data_lock.json", value)
    return c.artifact(base/"data_lock.json")


def _lock(binding, base):
    base = Path(base)
    value = c.old.read_json(base/"data_lock.json")
    c.require(value["manifest"] == binding and value["counts"] == COUNTS
              and value["tensor_bytes_reused"] is value["conversion_audits_inherited"] is True
              and value["assessment_in_payload"] is False
              and value["new_simulations"] == value["historical_fit_replays"] == value["historical_trajectory_replays"] == 0,
              "Reused data seal changed")
    if (base/"freeze.json").exists():
        freeze = c.old.read_json(base/"freeze.json")
        manifest = c.verified(freeze["manifest"])
        c.require(freeze["binding"] == binding == c.old.digest(manifest)
                  and manifest["source_parent"] == value["parent"], "Data differs from frozen source parent")
    else:
        c.require(not base.resolve().is_relative_to(c.OUTPUT.resolve()), "Production data lacks freeze")
    parent = value["parent"]
    c.require(parent["binding"] == c.PARENT_BINDING and parent["coverage_data"]["sha256"] == DATA_SHA
              and parent["original_data"]["sha256"] == inherited.LOWRATE_DATA_SHA,
              "Completed immutable data reference changed")
    data, original = c.verified(parent["coverage_data"]), c.verified(parent["original_data"])
    c.require(all(value[k] == data[k] for k in DATA_FIELDS)
              and all(value[k] == original[k] for k in PANEL_FIELDS)
              and parent["conversion_audit"] == data["audit"]
              and parent["corpus"] == data["parent"]["corpus"]
              and parent["model_config"] == data["parent"]["model_config"]
              and parent["training_disjointness"] == data["parent"]["training_disjointness"]
              and parent["retired_evaluation_panels"] == data["retired_evaluation_panels"]
              and parent["evaluation_sources"] == original["parent"]["evaluation_sources"]
              and parent["schedule"] == original["schedule"], "Immutable tensors, partition or source metadata changed")
    return value


def load_rows(binding, split, base=None):
    base = c.OUTPUT if base is None else base
    split = "old" if split == "train" else split
    c.require(split in COUNTS, "Unknown tensor split")
    lock = _lock(binding, base)
    rows = torch.load(c.verify_hash(lock["files"][split]), map_location="cpu", weights_only=True)
    c.require(len(rows) == COUNTS[split] and row_digest(rows) == lock["row_digests"][split], "Tensor contents changed")
    if split == "added":
        for i, (row, identity) in enumerate(zip(rows, lock["added_identities"], strict=True)):
            coverage_source.validate_added_row(row)
            c.require(row["index"] == 11950+i and all(row[k] == v for k, v in identity.items()), "Added row order changed")
    else:
        expected_split = "train" if split == "old" else "validation"
        corpus = lock["parent"]["corpus"]
        expected = [(f"breadth128:{shard['spec']['index']}:{r['turn']}", shard["spec"]["fit"], f"breadth128:{shard['spec']['pair']}")
                    for shard in corpus["shards"] if corpus["splits"][str(shard["spec"]["pair"])] == expected_split
                    for r in shard["roots"]]
        c.require(len(expected) == len(rows), "Original row population changed")
        for row, identity in zip(rows, expected, strict=True):
            validate_row(row)
            c.require(row["split"] == expected_split and (row["id"], row["source_fit"], row["pair"]) == identity,
                      "Original split/row identity changed")
    return rows


def load_pools(binding, base=None):
    return {key: load_rows(binding, key, base) for key in ("old", "added")}


def load_panel_rows(binding, base=None):
    lock = _lock(binding, c.OUTPUT if base is None else base)
    rows = torch.load(c.verify_hash(lock["panel_file"]), map_location="cpu", weights_only=True)
    c.require(len(rows) == 336 and row_digest(rows) == lock["panel_digest"], "Original336 tensors changed")
    for row, spec in zip(rows, lock["schedule"], strict=True):
        validate_row(row)
        _check_panel_identity(row, spec)
    return rows


def narrow_indices(fit, binding, base=None):
    c.require(fit in c.FITS, "Unknown partition fit")
    return _lock(binding, c.OUTPUT if base is None else base)["partition"]["narrow_indices"][str(fit)]


def control_models(binding, base=None):
    return _lock(binding, c.OUTPUT if base is None else base)["parent"]["models"]


def initializer(fit, base=None):
    base = c.OUTPUT if base is None else base
    c.require(fit in c.FITS, "Unknown initializer fit")
    binding = c.old.read_json(Path(base)/"data_lock.json")["manifest"]
    ref = control_models(binding, base)[f"anchor_{fit}"]
    c.require(ref["source_fit"] == fit and ref["sha256"] == inherited.inherited_models.EXPECTED[f"anchor_{fit}"],
              "Original initializer changed")
    c.verify_hash(ref)
    return ref


def broad_fit(fit, binding, base=None):
    c.require(fit in c.FITS, "Unknown matched BROAD fit")
    return _lock(binding, c.OUTPUT if base is None else base)["parent"]["broad_fits"][str(fit)]["result"]


def broad_audit(fit, binding, base=None):
    c.require(fit in c.FITS, "Unknown matched BROAD audit")
    return _lock(binding, c.OUTPUT if base is None else base)["parent"]["broad_fits"][str(fit)]["audit"]


def broad_model(fit, binding, base=None):
    c.require(fit in c.FITS, "Unknown matched BROAD model")
    return control_models(binding, base)[f"broad_{fit}"]


def evaluation_sources(binding, base=None):
    return _lock(binding, c.OUTPUT if base is None else base)["parent"]["evaluation_sources"]


def old_inputs(binding, base=None):
    """Expose pre-assessment selections only; never open independent outcomes."""
    result = []
    for descriptor in evaluation_sources(binding, base):
        saved = c.verified(descriptor["selection"])
        identity, locked = saved["identity"], saved["selection_lock"]
        c.require(saved["manifest"] == c.SOURCE_BINDING and saved["stage"] == "main"
                  and saved["index"] == descriptor["index"] and identity["spec"] == descriptor["spec"]
                  and locked["identity_digest"] == c.old.digest(identity)
                  and locked["old_selection_sha256"] == c.old.digest(saved["old_selection"]), "Original selection changed")
        statistics = locked["selection_statistics"]
        result.append({"index": descriptor["index"], "spec": descriptor["spec"], "snapshot": identity["snapshot"],
                       "candidates": identity["candidates"], "old_sums_wdl": statistics["sums_wdl"],
                       "old_unknown_counts": statistics["unknown_counts"], "old_choices": locked["choices"],
                       "selection_artifact": descriptor["selection"]})
    c.require([row["index"] for row in result] == list(range(336)), "Original336 selection population changed")
    return result


def validate(binding, base=None, authenticate_parent=True):
    lock = _lock(binding, c.OUTPUT if base is None else base)
    if authenticate_parent:
        c.require(lock["parent"] == closed_parent(), "Closed source ancestry changed")
    # Parent evidence is either authenticated here or by the calling freeze
    # validator. Repeating every old audit/encoding here adds no new evidence.
    for ref in (*lock["files"].values(), lock["panel_file"]):
        c.verify_hash(ref)
    return lock


def _feature_inputs(binding, base, rows):
    from . import model
    fixture = rows is not None
    if fixture:
        c.require(not base.resolve().is_relative_to(c.OUTPUT.resolve()), "Feature fixture cannot target production")
        pools = rows
        c.require(set(pools) == {"old", "added"} and all(pools.values()), "Both TRAIN pools required")
        inputs = {"files": None, "row_digests": {key: row_digest(value) for key, value in pools.items()},
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
