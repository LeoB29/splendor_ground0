"""Authenticated, old-target-only inputs for one bounded learning diagnosis.

The completed independent assessment is sealed here by identity. Its outcome
arrays are never copied into either training tensor file. Historical fits and
trajectory audits are inherited; no old simulation or optimizer is replayed.
"""
from collections import Counter, defaultdict
from pathlib import Path

import torch

from scripts.analysis import verify_fast_selector_transfer_157 as independent
from splendor_ai.training.model import LegalActionAttentionScorerConfig
from tools.teacher_signal_capture_128 import models as inherited_models
from tools.teacher_signal_capture_128 import source as inherited_source
from . import common as c

RESULT_SHA = "13c7f7f717d77bfd86d51691bb2140c15480cb170d312a47d720abe5921e19ba"
AUDIT_SHA = "9f455eef9e7a46d81eb15764977ad5f0f22d6f64def60da8d70f72cb6efad67f"
ORIGINAL_FIELDS = {"observation", "features", "groups", "group_count", "members",
                   "id", "pair", "split", "seat", "phase", "privacy", "teacher",
                   "narrow_teacher", "narrow_support"}
NEW_FIELDS = {"index", "source_fit", "old_sums_wdl", "outsider_start",
              "outsider_support", "outsider_teacher", "outsider_label"}
ROW_FIELDS = ORIGINAL_FIELDS | NEW_FIELDS


def _plain(ref):
    return {key: ref[key] for key in ("path", "sha256")}


def _equal(a, b):
    return c.old.digest(a) == c.old.digest(b)


def validate_schedule(schedule):
    """Independently check the retained 84 whole pairs and fixed partition."""
    c.require(len(schedule) == 336 and [s["index"] for s in schedule] == list(range(336)),
              "Focused panel must retain every main root in original order")
    c.require(len({s["source_rowid"] for s in schedule}) == 336, "Repeated source row")
    pairs = defaultdict(list)
    for spec in schedule:
        c.require(spec["fit"] in c.FITS and spec["split"] in ("train", "validation")
                  and spec["source_arm"] == "candidate" and spec["seat"] in (0, 1)
                  and spec["game_index"] == 2*spec["pair"]+spec["seat"]
                  and spec["phase"] == min(spec["turn"]//20, 2)
                  and spec["source_rowid"] == f"breadth128:{spec['game_index']}:{spec['turn']}",
                  "Invalid retained source identity")
        pairs[spec["pair"]].append(spec)
    c.require(len(pairs) == 84, "Incomplete whole-pair panel")
    for pair, rows in pairs.items():
        first = rows[0]
        fields = ("fit", "split", "pair_rank", "omitted_phase", "environment_seed", "search_stream")
        c.require(len(rows) == 4 and len({tuple(row[k] for k in fields) for row in rows}) == 1,
                  "Source pair crossed split or lost a root")
        rank = first["pair_rank"]
        omitted = (rank+list(c.FITS).index(first["fit"])) % 3
        c.require(type(rank) is int and 0 <= rank < 14 and first["omitted_phase"] == omitted
                  and {(r["seat"], r["phase"]) for r in rows}
                  == {(seat, phase) for seat in (0, 1) for phase in range(3) if phase != omitted},
                  "Changed phase-balanced whole pair")
    for split in ("train", "validation"):
        rows = [s for s in schedule if s["split"] == split]
        c.require(len(rows) == 168 and len({r["pair"] for r in rows}) == 42
                  and Counter(r["fit"] for r in rows) == {fit: 56 for fit in c.FITS}
                  and Counter(r["phase"] for r in rows) == {0: 56, 1: 56, 2: 56},
                  "Changed training/validation allocation")
    return True


def closed_parent():
    """Read completion/identity receipts, hashing assessment files without using them."""
    refs = {}

    def add(ref, read=True):
        item = _plain(ref)
        c.require(item["path"] not in refs or refs[item["path"]] == item,
                  "Conflicting source identities")
        refs[item["path"]] = item
        return c.verified(item) if read else c.verify_hash(item)

    freeze_ref = c.artifact(c.SOURCE/"freeze.json")
    freeze = add(freeze_ref)
    manifest = add(freeze["manifest"])
    c.require(freeze["binding"] == c.SOURCE_BINDING == c.old.digest(manifest),
              "Signal-capture source binding changed")
    for path, digest in manifest["identity"]["dependencies"].items():
        add({"path": path, "sha256": digest}, False)
    checks = add(manifest["checks"])
    c.require(checks["exit_code"] == 0 and checks["identity"] == manifest["identity"],
              "Signal-capture checks do not match its freeze")
    for ref in checks["logs"]:
        add(ref, False)
    status_ref = c.artifact(c.SOURCE/"final_status.json")
    status = add(status_ref)
    c.require(status["manifest"] == c.SOURCE_BINDING and status["complete"] is True
              and status["all_prescribed_audits_pass"] is True
              and status["result"]["sha256"] == RESULT_SHA
              and status["audit"]["sha256"] == AUDIT_SHA, "Signal capture is not the exact completed study")
    result, audit = add(status["result"]), add(status["audit"])
    add(status["report"], False)
    c.require(result["manifest"] == audit["manifest"] == c.SOURCE_BINDING
              and result["all_prescribed_audits_pass"] is True and audit["passed"] is True
              and audit["result"] == status["result"], "Source completion linkage changed")
    source = add(manifest["source"])
    schedule = add(manifest["schedule"])
    c.require(schedule == source["schedule"] and source["selection_uses_outcomes"] is False
              and source["historical_trajectory_replays"] == 0, "Source schedule seal changed")
    inherited_source.validate_schedule(schedule)
    validate_schedule(schedule["main"])
    c.require(source["inventory"] == {r["path"]: r["sha256"] for r in source["source_evidence"]},
              "Source evidence inventory changed")
    for ref in source["source_evidence"]:
        add(ref, False)
    model_parent = inherited_models.closed_parent()
    c.require(model_parent == manifest["model_parent"], "Original model ancestry changed")
    registry = add(c.artifact(c.SOURCE/"models_lock.json"))
    c.require(registry["manifest"] == c.SOURCE_BINDING
              and all(registry[key] == value for key, value in model_parent.items())
              and registry["ancestor_learning_replayed"] is False
              and registry["specialist_reselection"] is False, "Original models are not sealed")
    for ref in model_parent["evidence"]:
        add(ref, False)
    c.require(source["training_data"] == model_parent["training_data"], "Training ancestry differs")
    evaluation = []
    for stage in ("probe", "main"):
        complete = add(audit["stages"][stage])
        c.require(complete["manifest"] == c.SOURCE_BINDING
                  and complete["all_prescribed_audits_pass"] is True, "Unfinished inherited stage")
        collection, receipts = add(complete["collection"]), add(complete["audits"])
        specs = schedule[stage]
        c.require(collection["manifest"] == receipts["manifest"] == c.SOURCE_BINDING
                  and [r["index"] for r in collection["files"]]
                  == [r["index"] for r in receipts["files"]] == list(range(len(specs))),
                  "Incomplete inherited root/audit coverage")
        for spec, root_ref, audit_ref in zip(specs, collection["files"], receipts["files"], strict=True):
            # Authentication reads bytes, never chooses a row/action from fresh scores.
            add(root_ref, False)
            receipt = add(audit_ref)
            c.require(receipt["manifest"] == c.SOURCE_BINDING and receipt["passed"] is True
                      and receipt["stage"] == stage and receipt["spec"] == spec
                      and receipt["root"] == _plain(root_ref)
                      and receipt["unresolved_assessment_continuations"] == 0,
                      "Inherited source root audit changed")
            add(receipt["sample_artifact"], False)
            selection = add(receipt["selection_artifact"])
            c.require(selection["manifest"] == c.SOURCE_BINDING and selection["stage"] == stage
                      and selection["index"] == spec["index"] and selection["identity"]["spec"] == spec
                      and selection["selection_lock"]["identity_digest"] == c.old.digest(selection["identity"])
                      and selection["selection_lock"]["old_selection_sha256"] == c.old.digest(selection["old_selection"]),
                      "Old-input selection seal changed")
            if stage == "main":
                evaluation.append({"index": spec["index"], "spec": spec,
                                   "root": _plain(root_ref), "audit": _plain(audit_ref),
                                   "selection": _plain(receipt["selection_artifact"])})
    first = model_parent["models"][f"anchor_{c.FITS[0]}"]
    config = torch.load(c.verify_hash(first), map_location="cpu", weights_only=True)["model_config"]
    return {"source_binding": c.SOURCE_BINDING, "schedule": schedule["main"],
            "models": model_parent["models"], "model_config": config,
            "evaluation_sources": evaluation, "completion": status_ref,
            "result": status["result"], "audit": status["audit"],
            "evidence": [refs[k] for k in sorted(refs)],
            "inventory": {k: refs[k]["sha256"] for k in sorted(refs)},
            "new_simulations": 0, "historical_fit_replays": 0,
            "assessment_used_for_selection": False,
            "inherited_coverage": result["coverage"]}


def old_targets(root):
    """Recompute exact original sums and first-maximum outsider without new play."""
    n = len(root["candidates"]["keys"])
    c.require(1 <= n <= 8 and len(root["candidates"]["actions"]) == n
              and len(set(root["candidates"]["keys"])) == n
              and len(root["worlds"]) == 128, "Invalid original teacher support")
    sums = [0]*n
    for world in root["worlds"]:
        c.require(len(world["outcomes"]) == n, "Original outcome support changed")
        for i, outcome in enumerate(world["outcomes"]):
            value = outcome["wdl"]
            c.require(type(value) in (int, float) and value in (-1, 0, 1)
                      and outcome["reason"] in c.old.VALID_NATURAL_TERMINATIONS,
                      "Original teacher target is unresolved or invalid")
            sums[i] += int(value)
    teacher = max(range(n), key=sums.__getitem__)
    c.require(root["choice"] == teacher and root["means"] == [v/128 for v in sums],
              "Original teacher arithmetic changed")
    outsider = max(range(3, n), key=sums.__getitem__) if n > 3 else -1
    return {"old_sums_wdl": sums, "outsider_start": min(3, n),
            "outsider_support": max(0, n-3), "outsider_teacher": outsider,
            "outsider_label": outsider-3 if outsider >= 0 else -1}


def reconstruct_row(spec, config):
    """Verify actual source membership and independently reconstruct lawful tensors."""
    root = inherited_source.load_root(spec)  # No predecessor budget or trajectory replay.
    original = inherited_source._export_row(spec)
    c.require(set(original) == ORIGINAL_FIELDS, "Unexpected original export fields")
    state = c.old.deserialize_state_snapshot(root["snapshot"])
    obs, features, groups, keys, legal = independent.input_invariance(state, config)
    members = [[i for i, action in enumerate(legal)
                if repr(c.old.canonical_successor_action_key(state, action)) == key]
               for key in root["candidates"]["keys"]]
    c.require(torch.equal(original["observation"], obs) and torch.equal(original["features"], features)
              and original["groups"].tolist() == groups and original["group_count"] == len(keys)
              and [m.tolist() for m in original["members"]] == members and all(members),
              "Export tensors differ from independent lawful inputs")
    target = old_targets(root)
    c.require(original["teacher"] == root["choice"]
              and original["narrow_teacher"] == max(range(min(4, len(members))), key=target["old_sums_wdl"].__getitem__),
              "Original broad/narrow labels changed")
    row = {**original, "index": spec["index"], "source_fit": spec["fit"], **target}
    validate_row(row)
    return row


def validate_row(row):
    """Strict payload allowlist prevents assessment data entering a fitting row."""
    c.require(set(row) == ROW_FIELDS, "Unexpected fitting payload fields")
    n = len(row["members"])
    sums = row["old_sums_wdl"]
    c.require(type(row["index"]) is int and 0 <= row["index"] < 336
              and row["source_fit"] in c.FITS and row["split"] in ("train", "validation")
              and 1 <= n <= 8 and len(sums) == n
              and all(type(x) is int and -128 <= x <= 128 for x in sums), "Invalid fitting target metadata")
    best = max(range(3, n), key=sums.__getitem__) if n > 3 else -1
    c.require(row["outsider_start"] == min(3, n) and row["outsider_support"] == max(0, n-3)
              and row["outsider_teacher"] == best
              and row["outsider_label"] == (best-3 if best >= 0 else -1), "Changed outsider target or tie order")
    c.require(row["teacher"] == max(range(n), key=sums.__getitem__)
              and row["narrow_support"] == min(4, n)
              and row["narrow_teacher"] == max(range(min(4, n)), key=sums.__getitem__), "Changed inherited teacher target")
    c.require(row["observation"].ndim == 1 and row["features"].ndim == 2
              and row["observation"].dtype == row["features"].dtype == torch.float32
              and torch.isfinite(row["observation"]).all().item() and torch.isfinite(row["features"]).all().item()
              and row["groups"].dtype == torch.long and row["groups"].ndim == 1
              and len(row["groups"]) == len(row["features"]), "Invalid lawful input tensors")
    c.require(type(row["group_count"]) is int and row["group_count"] >= n
              and set(row["groups"].tolist()) == set(range(row["group_count"])), "Invalid full legal canonical groups")
    seen = set()
    for member in row["members"]:
        c.require(member.dtype == torch.long and member.ndim == 1 and len(member) > 0
                  and len(set(member.tolist())) == len(member)
                  and all(0 <= i < len(row["features"]) for i in member.tolist()), "Invalid offered membership")
        group = int(row["groups"][member[0]])
        c.require(group not in seen and torch.equal(member, torch.where(row["groups"] == group)[0]),
                  "Offered group aliases differ")
        seen.add(group)
    return True


def row_digest(rows):
    def plain(value):
        if torch.is_tensor(value):
            return {"dtype": str(value.dtype), "shape": list(value.shape), "values": value.tolist()}
        if isinstance(value, dict):
            return {key: plain(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [plain(item) for item in value]
        return value
    return c.old.digest(plain(rows))


def _write_tensor(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        torch.save(rows, stream)
    return c.artifact(path)


def prepare(binding, base=c.OUTPUT):
    base = Path(base)
    parent = closed_parent()
    config = LegalActionAttentionScorerConfig(**parent["model_config"])
    rows = [reconstruct_row(spec, config) for spec in parent["schedule"]]
    files, digests, counts = {}, {}, {}
    for split in ("train", "validation"):
        selected = [row for row in rows if row["split"] == split]
        c.require(len(selected) == 168, "Changed partition during preparation")
        files[split] = _write_tensor(base/f"{split}.pt", selected)
        digests[split] = row_digest(selected)
        counts[split] = {"rows": len(selected), "pairs": len({r["pair"] for r in selected}),
                         "no_outsider": sum(r["outsider_support"] == 0 for r in selected),
                         "singleton_outsider": sum(r["outsider_support"] == 1 for r in selected)}
    value = {"manifest": binding, "parent": parent, "schedule": parent["schedule"],
             "files": files, "row_digests": digests, "counts": counts,
             "rows": 336, "train_rows": 168, "validation_rows": 168,
             "assessment_in_payload": False, "historical_replay": False,
             "independent_lawful_inputs_checked": True, "source_trace_membership_checked": True,
             "new_simulations": 0, "historical_fit_replays": 0}
    c.old.save_new(base/"data_lock.json", value)
    return c.artifact(base/"data_lock.json")


def _lock(binding, base):
    base = Path(base)
    value = c.old.read_json(base/"data_lock.json")
    c.require(value["manifest"] == binding and value["assessment_in_payload"] is False
              and value["historical_replay"] is False and value["new_simulations"] == 0
              and value["rows"] == 336 and value["train_rows"] == value["validation_rows"] == 168,
              "Invalid focused data seal")
    freeze = base/"freeze.json"
    if freeze.exists():
        sealed = c.old.read_json(freeze)
        manifest = c.verified(sealed["manifest"])
        c.require(sealed["binding"] == binding == c.old.digest(manifest)
                  and manifest["source_parent"] == value["parent"], "Focused source differs from experiment freeze")
    else:
        c.require(not base.resolve().is_relative_to(c.OUTPUT.resolve()), "Production source lacks freeze")
    c.require(value["schedule"] == value["parent"]["schedule"], "Changed source schedule")
    return value


def load_rows(binding, split, base=c.OUTPUT):
    """Read only the requested old-target tensor partition, never another split."""
    c.require(split in ("train", "validation"), "Unknown fitting split")
    lock = _lock(binding, base)
    rows = torch.load(c.verify_hash(lock["files"][split]), map_location="cpu", weights_only=True)
    specs = [spec for spec in lock["schedule"] if spec["split"] == split]
    c.require(len(rows) == len(specs) == 168 and row_digest(rows) == lock["row_digests"][split],
              "Focused tensor population changed")
    for row, spec in zip(rows, specs, strict=True):
        validate_row(row)
        c.require(row["split"] == split and row["index"] == spec["index"]
                  and row["id"] == spec["source_rowid"] and row["source_fit"] == spec["fit"]
                  and row["pair"] == f"breadth128:{spec['pair']}"
                  and row["seat"] == spec["seat"] and row["phase"] == spec["phase"]
                  and row["privacy"] == spec["privacy"], "Focused row changed identity or split")
    return rows


def initializer(fit, base=c.OUTPUT):
    c.require(fit in c.FITS, "Unknown initializer fit")
    lock = c.old.read_json(Path(base)/"data_lock.json")
    ref = lock["parent"]["models"][f"anchor_{fit}"]
    c.require(ref["sha256"] == inherited_models.EXPECTED[f"anchor_{fit}"]
              and ref["source_fit"] == fit, "Original initializer differs")
    c.verify_hash(ref)
    return ref


def evaluation_sources(binding, base=c.OUTPUT):
    return _lock(binding, base)["parent"]["evaluation_sources"]


def old_inputs(binding, base=c.OUTPUT):
    """Expose only the original pre-assessment record for locking new predictions."""
    result = []
    for descriptor in evaluation_sources(binding, base):
        saved = c.verified(descriptor["selection"])
        identity, locked = saved["identity"], saved["selection_lock"]
        c.require(saved["manifest"] == c.SOURCE_BINDING and saved["stage"] == "main"
                  and saved["index"] == descriptor["index"]
                  and identity["spec"] == descriptor["spec"]
                  and locked["identity_digest"] == c.old.digest(identity)
                  and locked["old_selection_sha256"] == c.old.digest(saved["old_selection"]),
                  "Changed pre-assessment inputs")
        statistics = locked["selection_statistics"]
        result.append({"index": descriptor["index"], "spec": descriptor["spec"],
                       "snapshot": identity["snapshot"], "candidates": identity["candidates"],
                       "old_sums_wdl": statistics["sums_wdl"],
                       "old_unknown_counts": statistics["unknown_counts"],
                       "old_choices": locked["choices"], "selection_artifact": descriptor["selection"]})
    c.require([item["index"] for item in result] == list(range(336)), "Incomplete old input panel")
    return result


def validate(binding, base=c.OUTPUT):
    """Reauthenticate ancestry and all new exports, without replaying simulations."""
    lock = _lock(binding, base)
    parent = closed_parent()
    c.require(lock["parent"] == parent and lock["independent_lawful_inputs_checked"] is True
              and lock["source_trace_membership_checked"] is True, "Data source ancestry changed")
    config = LegalActionAttentionScorerConfig(**parent["model_config"])
    for split in ("train", "validation"):
        saved = load_rows(binding, split, base)
        expected = [reconstruct_row(spec, config) for spec in parent["schedule"] if spec["split"] == split]
        c.require(row_digest(saved) == row_digest(expected), "Independent focused export reconstruction failed")
    return lock
