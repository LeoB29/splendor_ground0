"""Closed original128 supervision and independently encoded new TRAIN positions.

Historical tensors and audits are inherited without replay. New tensors contain
only lawful public inputs and128-world labels; no assessment payload is opened.
"""
from collections import Counter
from pathlib import Path
import os
import uuid

import torch

from scripts.analysis import verify_fast_selector_transfer_157 as input_audit
from splendor_ai.alphazero.strong_policy import encode_position
from splendor_ai.training.model import LegalActionAttentionScorerConfig
from tools.teacher_precision_512 import training_source as parent_source
from tools.independent_game_coverage_128_recovery_01 import training_source as coverage
from . import common as c

COUNTS = {"old": 9461, "added": 11520, "validation": 2489}
CORE_COUNTS = {"old": 9461, "added": 720, "validation": 2489}
PANEL = "independent_game_scaling_128"
row_digest, validate_row = parent_source.row_digest, parent_source.validate_row
CELLS = tuple((f, s, p) for f in c.FITS for s in (0, 1) for p in range(3))
_SOURCE_CACHE = {}


def closed_parent():
    """Reuse the already validated closed R1 ancestry, never precision512 labels."""
    result = parent_source.closed_parent()
    c.require(result["binding"] == c.PARENT_BINDING and result["coverage_data"]["sha256"] == parent_source.DATA_SHA,
              "Unrecognized original128 TRAIN source")
    return result


def prepare(binding, base=c.OUTPUT, parent=None):
    base = Path(base)
    parent = closed_parent() if parent is None else parent
    data = c.verified(parent["coverage_data"])
    original = c.verified(parent["original_data"])
    parent_source._data_receipt(data, original, c.verified(parent["conversion_audit"]))
    value = {"manifest": binding, "parent": parent,
             "files": data["files"], "row_digests": data["row_digests"], "counts": CORE_COUNTS,
             "added_identities": data["added_identities"], "retired_evaluation_panels": data["retired_evaluation_panels"],
             "historical_trajectory_replays": 0, "historical_fit_replays": 0, "new_simulations": 0,
             "old128_targets_only": True, "conversion_audits_inherited": True,
             "assessment_in_payload": False, "tensor_bytes_reused": True}
    c.old.save_new(base/"inherited_source_lock.json", value)
    return c.artifact(base/"inherited_source_lock.json")


def inherited_lock(binding, base=c.OUTPUT):
    path = Path(base)/"inherited_source_lock.json"
    # This immutable seal is checked once per worker. Freeze/stage dependency
    # authentication remains responsible for its bytes at execution boundaries.
    key = (str(path.resolve()), binding)
    if key not in _SOURCE_CACHE:
        value = c.old.read_json(path)
        parent = value["parent"]
        c.require(value["manifest"] == binding and value["counts"] == CORE_COUNTS
                  and parent["binding"] == c.PARENT_BINDING
                  and parent["coverage_data"]["sha256"] == parent_source.DATA_SHA
                  and value["old128_targets_only"] is value["conversion_audits_inherited"] is value["tensor_bytes_reused"] is True
                  and value["assessment_in_payload"] is False
                  and value["historical_trajectory_replays"] == value["historical_fit_replays"] == value["new_simulations"] == 0,
                  "Inherited original128 source contract changed")
        data = c.verified(parent["coverage_data"])
        c.require(all(value[k] == data[k] for k in ("files", "row_digests", "counts", "added_identities", "retired_evaluation_panels")),
                  "Inherited TRAIN bytes/identities changed")
        _SOURCE_CACHE[key] = value
    return _SOURCE_CACHE[key]


def control_models(binding, base=c.OUTPUT):
    models = inherited_lock(binding, base)["parent"]["models"]
    return {f"anchor_{fit}": models[f"anchor_{fit}"] for fit in c.FITS} | {
        f"control_{fit}": models[f"broad_{fit}"] for fit in c.FITS}


def initializer(fit, base=c.OUTPUT):
    c.require(fit in c.FITS, "Unknown original initializer")
    binding = c.old.read_json(Path(base)/"inherited_source_lock.json")["manifest"]
    ref = control_models(binding, base)[f"anchor_{fit}"]
    c.require(ref["source_fit"] == fit and ref["sha256"] == parent_source.inherited.inherited_models.EXPECTED[f"anchor_{fit}"],
              "Original initializer differs")
    c.verify_hash(ref)
    return ref


def broad_fit(fit, binding, base=c.OUTPUT):
    return inherited_lock(binding, base)["parent"]["broad_fits"][str(fit)]["result"]


def broad_audit(fit, binding, base=c.OUTPUT):
    return inherited_lock(binding, base)["parent"]["broad_fits"][str(fit)]["audit"]


def new_identity(spec):
    return {"id": f"{PANEL}:{spec['game_index']}:{spec['turn']}", "pair": f"{PANEL}:{spec['game_index']}",
            "split": "train", "source_fit": spec["fit"], "seat": spec["seat"], "phase": spec["phase"],
            "privacy": spec["privacy"], "source_panel": PANEL, "source_game": spec["game_index"],
            "environment_seed": c.CONFIG["source_seed"]["train"]+spec["game_index"],
            "cell": [PANEL, spec["fit"], spec["seat"], spec["phase"]]}


def target_from_sums(sums):
    n = len(sums)
    c.require(1 <= n <= 8 and all(type(v) is int and -128 <= v <= 128 for v in sums), "Invalid new128 target")
    outside = max(range(3, n), key=lambda j: sums[j]) if n > 3 else -1
    return {"old_sums_wdl": list(sums), "teacher": max(range(n), key=lambda j: sums[j]),
            "narrow_support": min(4, n), "narrow_teacher": max(range(min(4, n)), key=lambda j: sums[j]),
            "outsider_start": min(3, n), "outsider_support": max(0, n-3),
            "outsider_teacher": outside, "outsider_label": outside-3 if outside >= 0 else -1}


def validate_added_row(row):
    if row["index"] < 12670:
        return coverage.validate_added_row(row)
    c.require(set(row) == coverage.inherited.ROW_FIELDS | coverage.EXTRA_FIELDS
              and 12670 <= row["index"] < 23470, "Unexpected scaling tensor payload")
    core = {key: row[key] for key in coverage.inherited.ROW_FIELDS}
    core["index"] = row["index"]-11950
    validate_row(core)
    c.require(row["split"] == "train" and row["source_panel"] == PANEL
              and row["cell"] == [PANEL, row["source_fit"], row["seat"], row["phase"]]
              and row["pair"] == f"{PANEL}:{row['source_game']}"
              and row["id"].startswith(row["pair"]+":")
              and row["environment_seed"] == c.CONFIG["source_seed"]["train"]+row["source_game"],
              "New TRAIN identity/split changed")
    c.require(all(row[k] == v for k, v in target_from_sums(row["old_sums_wdl"]).items()), "128 target arithmetic changed")
    return True


def encode_new(spec, root, config, state):
    """Independent lawful encoding/membership checks for every new fitting row."""
    encoded, keys, legal = encode_position(state, config)
    obs, features, groups, other_keys, other_legal = input_audit.input_invariance(state, config)
    c.require(torch.equal(encoded["observation"], obs) and torch.equal(encoded["features"], features)
              and encoded["groups"].tolist() == groups and keys == other_keys and legal == other_legal
              and state.current_player == spec["seat"] and state.turn_index == spec["turn"]
              and min(state.turn_index//20, 2) == spec["phase"]
              and c.baseline.privacy(state) == spec["privacy"]
              and c.old.digest(c.old.serialize_state_snapshot(state)) == spec["snapshot_sha256"],
              "New tensor public/private invariance or source turn failed")
    offered = root["identity"]["union"]
    members = []
    for key, serial in zip(offered["keys"], offered["actions"], strict=True):
        action = c.old.decode_action(serial)
        c.require(action in legal and repr(c.old.canonical_successor_action_key(state, action)) == key,
                  "Teacher action does not identify its legal canonical group")
        ids = [j for j, a in enumerate(legal) if repr(c.old.canonical_successor_action_key(state, a)) == key]
        c.require(ids, "Empty teacher canonical membership")
        members.append(torch.tensor(ids, dtype=torch.long))
    row = {**encoded, **new_identity(spec), **target_from_sums(root["sums_wdl"]),
           "index": 12670+spec["index"], "members": members}
    validate_added_row(row)
    return row


def tensor_path(index, base=c.OUTPUT):
    return Path(base)/"labels"/"tensors"/f"state_{index:05d}.pt"


def _write_tensor(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        saved = torch.load(path, map_location="cpu", weights_only=True)
        c.require(row_digest(saved) == row_digest(rows), "Existing exported tensor differs; preserve evidence")
    else:
        temporary = path.with_name(path.name+"."+uuid.uuid4().hex+".tmp")
        torch.save(rows, temporary)
        os.replace(temporary, path)
    return c.artifact(path)


def prepare_targets(binding, labels_lock_ref, labels_audits_lock_ref, base=c.OUTPUT):
    from . import label_source
    base = Path(base)
    inherited = inherited_lock(binding, base)
    plan = label_source.load_plan(binding, base)
    collection, audits = c.verified(labels_lock_ref), c.verified(labels_audits_lock_ref)
    c.require(collection["manifest"] == audits["manifest"] == binding
              and collection["complete"] is audits["complete"] is True
              and collection["all_prescribed_audits_pass"] is audits["all_prescribed_audits_pass"] is True
              and len(collection["files"]) == len(audits["files"]) == 10800, "Complete10800 label seals required")
    core = torch.load(c.verify_hash(inherited["files"]["added"]), map_location="cpu", weights_only=True)
    c.require(len(core) == 720 and row_digest(core) == inherited["row_digests"]["added"], "Inherited720 tensor changed")
    rows, tensor_refs = list(core), []
    for i, (spec, root_ref, audit_ref) in enumerate(zip(plan["specs"], collection["files"], audits["files"], strict=True)):
        c.budget_boundary(base)
        root, audit = c.verified(root_ref), c.verified(audit_ref)
        c.require(root_ref["index"] == audit_ref["index"] == root["index"] == audit["index"] == i
                  and root["spec"] == audit["spec"] == spec and audit["root"] == {k:root_ref[k] for k in ("path", "sha256")}
                  and root["manifest"] == audit["manifest"] == binding and audit["passed"] is True
                  and audit["every128_world_target_independently_summed"] is True
                  and audit["independent_public_private_invariant_encoding"] is True
                  and audit["all_canonical_memberships_checked"] is True
                  and root["unknown_outcomes"] == audit["unknown_outcomes"] == 0
                  and root["assessment_read"] is audit["assessment_read"] is False, "Uncertified fitting target")
        path = c.verify_hash(audit["tensor"])
        c.require(path.resolve() == tensor_path(i, base).resolve(), "Wrong per-root tensor artifact")
        one = torch.load(path, map_location="cpu", weights_only=True)
        c.require(len(one) == 1 and row_digest(one) == audit["tensor_row_digest"], "Audited per-root tensor changed")
        row = one[0]
        validate_added_row(row)
        c.require(row["index"] == 12670+i and all(row[k] == v for k,v in new_identity(spec).items())
                  and row["old_sums_wdl"] == root["sums_wdl"], "Exported tensor/root identity differs")
        rows.append(row); tensor_refs.append(audit["tensor"])
    c.require(row_digest(rows[:720]) == inherited["row_digests"]["added"]
              and len({r["pair"] for r in rows}) == len({r["environment_seed"] for r in rows}) == 11520,
              "Nested original core changed or added source games overlap")
    files = {"old": inherited["files"]["old"], "validation": inherited["files"]["validation"],
             "added": _write_tensor(base/"added.pt", rows)}
    digests = {"old": inherited["row_digests"]["old"], "validation": inherited["row_digests"]["validation"],
               "added": row_digest(rows)}
    identities = inherited["added_identities"]+[new_identity(s) for s in plan["specs"]]
    allocation = label_source.partitions(plan["specs"])
    audit = {"manifest": binding, "passed": True, "rows": 11520, "inherited_rows": 720, "new_rows": 10800,
             "every_new_tensor_independently_encoded": True, "every_new_target_audited": True,
             "old128_only": True, "inherited_core_unchanged": True, "old_train_unchanged": True,
             "assessment_in_payload": False, "validation_read": False, "historical_trajectory_replays": 0,
             "files": files, "row_digests": digests, "partition": allocation,
             "labels": labels_lock_ref, "label_audits": labels_audits_lock_ref,
             "tensor_files": tensor_refs, "plan": c.artifact(base/"label_plan.json")}
    audit_path = base/"training_source_audit.json"
    if audit_path.exists(): c.require(c.old.read_json(audit_path) == audit, "Prior conversion audit differs")
    else: c.old.save_new(audit_path, audit)
    result = {"manifest": binding, "inherited": c.artifact(base/"inherited_source_lock.json"),
              "parent": inherited["parent"], "files": files, "row_digests": digests, "counts": COUNTS,
              "added_identities": identities, "partition": allocation, "audit": c.artifact(audit_path),
              "label_plan": c.artifact(base/"label_plan.json"), "labels": labels_lock_ref, "label_audits": labels_audits_lock_ref,
              "old128_only": True, "assessment_in_payload": False, "validation_read": False,
              "historical_replay": False, "tensor_bytes_reused": True}
    if (base/"data_lock.json").exists(): c.require(c.old.read_json(base/"data_lock.json") == result, "Prior data seal differs")
    else: c.old.save_new(base/"data_lock.json", result)
    return c.artifact(base/"data_lock.json")


def _lock(binding, base=c.OUTPUT):
    from . import label_source
    base = Path(base)
    value = c.old.read_json(base/"data_lock.json")
    inherited = inherited_lock(binding, base)
    plan = label_source.load_plan(binding, base)
    audit = c.verified(value["audit"])
    c.require(value["manifest"] == audit["manifest"] == binding and value["counts"] == COUNTS
              and value["parent"] == inherited["parent"] and value["inherited"] == c.artifact(base/"inherited_source_lock.json")
              and value["old128_only"] is value["tensor_bytes_reused"] is True
              and value["assessment_in_payload"] is value["validation_read"] is value["historical_replay"] is False
              and value["partition"] == label_source.partitions(plan["specs"])
              and value["added_identities"] == inherited["added_identities"]+[new_identity(s) for s in plan["specs"]],
              "Final scaling TRAIN seal changed")
    c.require(audit["passed"] is audit["every_new_tensor_independently_encoded"] is audit["every_new_target_audited"] is True
              and audit["old128_only"] is audit["inherited_core_unchanged"] is audit["old_train_unchanged"] is True
              and audit["assessment_in_payload"] is audit["validation_read"] is False
              and audit["historical_trajectory_replays"] == 0
              and all(audit[k] == value[k] for k in ("files", "row_digests", "partition", "labels", "label_audits"))
              and all(value["files"][k] == inherited["files"][k] and value["row_digests"][k] == inherited["row_digests"][k]
                      for k in ("old", "validation")), "Final tensor conversion audit changed")
    return value


def load_rows(binding, split, base=c.OUTPUT):
    c.require(split in COUNTS, "Unknown TRAIN tensor split")
    lock = _lock(binding, base)
    rows = torch.load(c.verify_hash(lock["files"][split]), map_location="cpu", weights_only=True)
    c.require(len(rows) == COUNTS[split] and row_digest(rows) == lock["row_digests"][split], "Fitting tensor changed")
    if split == "added":
        for i,(row,identity) in enumerate(zip(rows,lock["added_identities"],strict=True)):
            validate_added_row(row)
            c.require(row["index"] == 11950+i and all(row[k] == v for k,v in identity.items()), "Fitting row order changed")
    else:
        for row in rows:
            validate_row(row)
            c.require(row["split"] == ("train" if split == "old" else "validation"), "Validation crossed training firewall")
    return rows


def load_pools(binding, base=c.OUTPUT):
    return {key: load_rows(binding, key, base) for key in ("old", "added")}


def partitions(binding, base=c.OUTPUT):
    return _lock(binding, base)["partition"]


def arm_indices(arm, binding, base=c.OUTPUT):
    c.require(arm in c.LEARNING_ARMS, "Unknown scaling arm")
    return partitions(binding, base)[arm]


def validate(binding, base=c.OUTPUT, authenticate_parent=True):
    value = _lock(binding, base) if (Path(base)/"data_lock.json").exists() else inherited_lock(binding, base)
    if authenticate_parent: c.require(value["parent"] == closed_parent(), "Closed inherited source differs")
    for ref in value["files"].values(): c.verify_hash(ref)
    return value
