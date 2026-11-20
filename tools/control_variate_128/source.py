"""Authenticated saved terminal targets, public143 evaluator and split firewall.

Preparation reads source/audit metadata and hashes complete old files. It never
opens a saved assessment root, replays a trajectory, or loads protected143 test
predictions. The completed independent audits certify those immutable bytes.
"""
from collections import Counter, defaultdict
import math
from pathlib import Path

from tools.teacher_signal_capture_128 import source as prior_source
from . import common as c

RESULT_SHA = "13c7f7f717d77bfd86d51691bb2140c15480cb170d312a47d720abe5921e19ba"
AUDIT_SHA = "9f455eef9e7a46d81eb15764977ad5f0f22d6f64def60da8d70f72cb6efad67f"
VALUE_BINDING = "72b0878756dee34e1be59151e76fc157366287b5a2ebc240d255c6a29804ed02"
VALUE = c.ROOT / "outputs/value_generalization_scale_143"
VALUE_HASHES = {
    "manifest.json": "050fb3a771df9817665ebd744fe3a476ddf2e6dec8f4d7ae8852569275b1c4f6",
    "simple.json": "a5c87ab3fb0f3f22a0e0e25a5adbfcad1bb3422c7f40cdc060d33a0d3a9b720c",
    "selection_lock.json": "484a9ad799ee046274319f355cb63ee0be25b93d38657df584cd42fb3cad2986",
    "verification.json": "7984744ffcf8cb656afbeaee435b46bcc2fb2f278c1661e26c569307bd71134a",
    "result.json": "2bf923c2a15dfa62a19aa1cf1ec8f2f1756f46e0ed6c4da4773586d0d47a5098",
    "final_status.json": "72a5e073f201b29ec75400d8162a368fd54c86743edb5ae1b0127f87d0123de8",
    "serialization_recovery.json": "8badf267fd9c1cd04d5931f015e6318f191ef885a92cfe975fa716e880b681fd",
}
EXTRA_FIELDS = ("support", "reference", "cv_stress", "cv_audit_indices", "cv_timing_indices")


def _base(base):
    return Path(c.OUTPUT if base is None else base)


def _plain(ref):
    return {k: ref[k] for k in ("path", "sha256")}


def _original(spec):
    return {k: v for k, v in spec.items() if k not in EXTRA_FIELDS}


def _rank(tag, *parts):
    return c.old.digest([tag, *parts])


def _world_indices(spec, timing=False):
    tag = c.CONFIG["timing_tag"] if timing else c.CONFIG["audit_world_tag"]
    seed = c.CONFIG["timing_seed"] if timing else c.CONFIG["audit_seed"]
    return sorted(sorted(range(128), key=lambda i: (_rank(tag, seed, spec["split"], spec["index"], i), i))[:8])


def plans(specs):
    """Identity-only fixed timing cells and full-prefix stress selections."""
    c.require([s["index"] for s in specs] == list(range(336)), "Panel index/order changed")
    cells, stress = {}, set()
    for split in ("train", "validation"):
        group = [s for s in specs if s["split"] == split]
        c.require(len(group) == 168, "Panel split count changed")
        selected, rows = set(), []
        def add(name, eligible):
            choice = min(eligible, key=lambda s: (_rank(c.CONFIG["audit_stress_tag"], c.CONFIG["audit_seed"], split, s["index"]), s["index"]))["index"] if eligible else None
            rows.append({"cell": name, "eligible_count": len(eligible), "selected_index": choice})
            if choice is not None:
                selected.add(choice)
        for fit in c.FITS:
            fitrows = [s for s in group if s["fit"] == fit]
            for seat in (0, 1):
                for phase in range(3):
                    add(f"fit_seat_phase|{fit}|{seat}|{phase}", [s for s in fitrows if (s["seat"], s["phase"]) == (seat, phase)])
            for privacy in range(4):
                add(f"fit_privacy|{fit}|{privacy}", [s for s in fitrows if int(bool(s["privacy"]["own"]))+2*int(bool(s["privacy"]["opponent"])) == privacy])
            for support in (1, 2, 3):
                add(f"fit_short_support|{fit}|{support}", [s for s in fitrows if s["support"] == support])
        c.require(len(rows) == 39 and len(selected) <= c.CONFIG["maximum_stress_roots_per_split"], "Stress plan exceeds scope")
        stress.update(selected)
        cells[split] = rows
    timing = []
    for fit in c.FITS:
        for seat in (0, 1):
            for phase in range(3):
                eligible = [s for s in specs if (s["split"], s["fit"], s["seat"], s["phase"]) == ("train", fit, seat, phase)]
                c.require(bool(eligible), "Missing TRAIN timing cell")
                timing.append(min(eligible, key=lambda s: (_rank(c.CONFIG["timing_tag"], c.CONFIG["timing_seed"], s["index"]), s["index"]))["index"])
    augmented = [{**s, "cv_stress": s["index"] in stress,
                  "cv_audit_indices": list(range(128)) if s["index"] in stress else _world_indices(s),
                  "cv_timing_indices": _world_indices(s, True)} for s in specs]
    return {"schedule": augmented, "timing_indices": sorted(timing), "stress_cells": cells,
            "stress_indices": sorted(stress), "selection_uses_outcomes": False}


def _value_parent(add):
    refs = {name: {"path": (VALUE/name).relative_to(c.ROOT).as_posix(), "sha256": sha} for name, sha in VALUE_HASHES.items()}
    records = {name: add(ref, name != "result.json") for name, ref in refs.items()}
    manifest, simple, lock = (records[n] for n in ("manifest.json", "simple.json", "selection_lock.json"))
    status, audit, recovery = (records[n] for n in ("final_status.json", "verification.json", "serialization_recovery.json"))
    c.require(c.old.digest(manifest) == VALUE_BINDING and all(records[n]["manifest"] == VALUE_BINDING for n in records if n not in ("manifest.json", "result.json")), "Frozen143 binding changed")
    c.require(status["status"] == "complete" and audit["status"] == recovery["status"] == "passed"
              and status["result_sha256"] == audit["result_sha256"] == recovery["result_sha256"] == VALUE_HASHES["result.json"]
              and status["verification_sha256"] == VALUE_HASHES["verification.json"]
              and status["serialization_recovery_sha256"] == VALUE_HASHES["serialization_recovery.json"]
              and audit["all_features_latents_labels_reconstructed"] is audit["selection_rule_verified"] is True
              and recovery["model_selection_unchanged"] is True and recovery["training_or_new_games"] is False,
              "Frozen143 audit/serialization closure changed")
    c.require(lock["files"]["simple.json"] == VALUE_HASHES["simple.json"]
              and simple["model"] == "public_score_resource_ridge_15_v1" and simple["ridge"] == .001
              and simple["selected_index"] == 0 and len(simple["coefficients"]) == 15
              and all(type(v) in (int, float) and math.isfinite(v) for v in simple["coefficients"])
              and simple["training_ids"] == list(range(4096)) and simple["validation_ids"] == list(range(4096, 4608)), "Frozen public evaluator changed")
    for path, sha in manifest["dependencies"].items():
        add({"path": path, "sha256": sha}, False)
    return {"binding": VALUE_BINDING, "files": refs, "evaluator": refs["simple.json"],
            "feature_order": ["score", "five_bonuses", "five_ordinary_tokens", "gold", "purchased", "reserved", "nobles"],
            "normalizers": [15]+[7]*10+[5, 20, 3, 3], "protected_test_predictions_read": False}


def closed_parent():
    refs = {}
    def add(ref, read=True):
        item = _plain(ref)
        c.require(item["path"] not in refs or refs[item["path"]] == item, "Conflicting authenticated source")
        if item["path"] not in refs:
            c.verify_hash(item)
            refs[item["path"]] = item
        return c.old.read_json(c.ROOT/item["path"]) if read else c.ROOT/item["path"]
    freeze = add(c.artifact(c.SOURCE/"freeze.json"))
    manifest = add(freeze["manifest"])
    c.require(freeze["owner_authorized"] is True and freeze["binding"] == c.SOURCE_BINDING == c.old.digest(manifest), "Saved panel freeze changed")
    for path, sha in manifest["identity"]["dependencies"].items():
        add({"path": path, "sha256": sha}, False)
    checks = add(manifest["checks"])
    c.require(checks["exit_code"] == 0 and checks["identity"] == manifest["identity"], "Panel preflight changed")
    for ref in checks["logs"]:
        add(ref, False)
    status = add(c.artifact(c.SOURCE/"final_status.json"))
    c.require(status["manifest"] == c.SOURCE_BINDING and status["complete"] is status["all_prescribed_audits_pass"] is True
              and status["result"]["sha256"] == RESULT_SHA and status["audit"]["sha256"] == AUDIT_SHA, "Panel is not the fixed completed source")
    add(status["result"], False)
    final = add(status["audit"])
    c.require(final["manifest"] == c.SOURCE_BINDING and final["passed"] is True and final["result"] == status["result"], "Panel final audit changed")
    selected, scheduled = add(manifest["source"]), add(manifest["schedule"])
    c.require(selected["schedule"] == scheduled and selected["selection_uses_outcomes"] is False
              and selected["all_validation_pairs_included"] is selected["probe_pairs_disjoint"] is True
              and selected["inventory"] == {r["path"]: r["sha256"] for r in selected["source_evidence"]}, "Panel source/split ancestry changed")
    prior_source.validate_schedule(scheduled)
    for ref in selected["source_evidence"]:
        add(ref, False)
    complete = add(final["stages"]["main"])
    c.require(complete["manifest"] == c.SOURCE_BINDING and complete["all_prescribed_audits_pass"] is True
              and complete["coverage"]["states"] == 336 and complete["coverage"]["unresolved_assessment_continuations"] == 0,
              "Saved assessment is incomplete or unresolved")
    roots, audits = add(complete["collection"]), add(complete["audits"])
    c.require(roots["manifest"] == audits["manifest"] == c.SOURCE_BINDING
              and [r["index"] for r in roots["files"]] == [r["index"] for r in audits["files"]] == list(range(336)), "Saved root/audit population changed")
    descriptors, specs = [], []
    for spec, rr, ar in zip(scheduled["main"], roots["files"], audits["files"], strict=True):
        checked = add(ar)
        support = checked["support_assessment"]
        c.require(checked["manifest"] == c.SOURCE_BINDING and checked["passed"] is True
                  and checked["spec"] == spec and checked["stage"] == "main" and checked["root"] == _plain(rr)
                  and 1 <= support <= 8 and checked["recorded_worlds"] == 128
                  and checked["recorded_assessment_continuations"] == checked["terminal_continuations"] == rr["continuations"] == 128*support
                  and checked["unresolved_assessment_continuations"] == 0, "Saved all-candidate terminal audit changed")
        reference = checked["row"]["choices"]["teacher"]
        c.require(type(reference) is int and 0 <= reference < support, "Invalid old fixed teacher reference")
        for ref in (rr, checked["selection_artifact"], checked["sample_artifact"]):
            add(ref, False)
        descriptors.append({"index": spec["index"], "spec": spec, "root": rr, "audit": ar,
                            "selection": checked["selection_artifact"], "support": support, "reference": reference})
        specs.append({**spec, "support": support, "reference": reference})
    value = _value_parent(add)
    return {"binding": c.SOURCE_BINDING, "result": status["result"], "audit": status["audit"],
            "source": manifest["source"], "original_schedule": manifest["schedule"], "complete": final["stages"]["main"],
            "descriptors": descriptors, **plans(specs), "value": value, "evaluator": value["evaluator"],
            "evidence": [refs[p] for p in sorted(refs)], "inventory": {p: refs[p]["sha256"] for p in sorted(refs)},
            "dependencies": {p: refs[p]["sha256"] for p in sorted(refs) if not p.startswith("outputs/")},
            "assessment_roots_read": False, "old_trajectory_replays": 0, "old_training_replays": 0,
            "new_simulations": 0, "independent_terminal_audits_inherited": True}


def _parent_receipt(parent):
    c.require(parent["binding"] == c.SOURCE_BINDING and parent["result"]["sha256"] == RESULT_SHA
              and parent["audit"]["sha256"] == AUDIT_SHA and parent["evaluator"] == parent["value"]["evaluator"]
              and parent["evaluator"]["sha256"] == VALUE_HASHES["simple.json"]
              and parent["inventory"] == {r["path"]: r["sha256"] for r in parent["evidence"]}
              and parent["assessment_roots_read"] is False and parent["independent_terminal_audits_inherited"] is True
              and parent["new_simulations"] == parent["old_trajectory_replays"] == parent["old_training_replays"] == 0,
              "Completed source receipt changed")
    specs = [{**d["spec"], "support": d["support"], "reference": d["reference"]} for d in parent["descriptors"]]
    expected = plans(specs)
    c.require(all(parent[k] == v for k, v in expected.items()), "Identity-only source/audit/timing plan changed")
    groups = defaultdict(list)
    for spec in specs:
        groups[spec["pair"]].append(spec)
    c.require(len(groups) == 84 and Counter((r[0]["split"], r[0]["fit"]) for r in groups.values()) == Counter({(s, f): 14 for s in ("train", "validation") for f in c.FITS}), "Source split/pair balance changed")
    for pair, rows in groups.items():
        c.require(len(rows) == 4 and len({(s["fit"], s["split"]) for s in rows}) == 1
                  and {s["game_index"] for s in rows} == {2*pair, 2*pair+1}
                  and len({(s["seat"], s["phase"]) for s in rows}) == 4, "Whole source pair split changed")
    return True


def prepare(binding, base=None, parent=None):
    base = _base(base)
    parent = closed_parent() if parent is None else parent
    _parent_receipt(parent)
    value = {"manifest": binding, "parent": parent, "schedule": parent["schedule"],
             "timing_indices": parent["timing_indices"], "stress_cells": parent["stress_cells"],
             "counts": dict(c.CONFIG["states"]), "assessment_in_payload": False,
             "validation_read": False, "historical_replay": False}
    c.old.save_new(base/"data_lock.json", value)
    return c.artifact(base/"data_lock.json")


def _lock(binding, base):
    base = _base(base)
    value = c.old.read_json(base/"data_lock.json")
    c.require(value["manifest"] == binding and value["counts"] == c.CONFIG["states"]
              and value["assessment_in_payload"] is value["validation_read"] is value["historical_replay"] is False,
              "Data lock changed")
    if (base/"freeze.json").exists():
        freeze = c.old.read_json(base/"freeze.json")
        manifest = c.verified(freeze["manifest"])
        c.require(freeze["binding"] == binding == c.old.digest(manifest) and manifest["source_parent"] == value["parent"], "Data parent differs from new freeze")
    else:
        c.require(not base.resolve().is_relative_to(c.OUTPUT.resolve()), "Production source requires freeze")
    c.require(value["schedule"] == value["parent"]["schedule"] and value["timing_indices"] == value["parent"]["timing_indices"]
              and value["stress_cells"] == value["parent"]["stress_cells"], "Data schedule differs from authenticated parent")
    return value


def validate(binding, base=None, authenticate_parent=True):
    value = _lock(binding, base)
    _parent_receipt(value["parent"])
    if authenticate_parent:
        c.require(value["parent"] == closed_parent(), "Completed source identity changed")
    return value


def schedule(binding, base=None, split="train"):
    c.require(split in ("train", "validation"), "Unknown source split")
    return [s for s in _lock(binding, base)["schedule"] if s["split"] == split]


def timing_specs(binding, base=None):
    lock = _lock(binding, base)
    return [s for s in lock["schedule"] if s["index"] in lock["timing_indices"]]


def timing_world_indices(spec):
    c.require(spec["split"] == "train" and spec["cv_timing_indices"] == _world_indices(spec, True), "Timing must use fixed TRAIN worlds")
    return list(spec["cv_timing_indices"])


def audit_indices(spec, binding=None, base=None):
    if binding is not None:
        c.require(_lock(binding, base)["schedule"][spec["index"]] == spec, "Audit spec differs from source seal")
    expected = list(range(128)) if spec["cv_stress"] else _world_indices(spec)
    c.require(spec["cv_audit_indices"] == expected, "Prefix replay plan changed")
    return expected


def evaluator(base=None):
    base = _base(base)
    binding = c.old.read_json(base/"data_lock.json")["manifest"]
    ref = _lock(binding, base)["parent"]["evaluator"]
    c.require(ref["sha256"] == VALUE_HASHES["simple.json"], "Frozen143 evaluator changed")
    c.verify_hash(ref)
    return ref


def selection_barrier(binding, base):
    base = _base(base)
    c.require((base/"selection_lock.json").exists() and (base/"selection_audit.json").exists(), "Validation outcomes require audited TRAIN selection")
    selected, audit = (c.old.read_json(base/n) for n in ("selection_lock.json", "selection_audit.json"))
    c.require(selected["manifest"] == audit["manifest"] == binding and selected["status"] == "selected"
              and selected["selected_before_validation"] is True and selected["validation_read"] is False
              and audit["passed"] is True and _plain(audit["selection"]) == _plain(c.artifact(base/"selection_lock.json"))
              and c.verified(audit["selection"]) == selected, "Validation selection barrier changed")
    rule = selected["config"]
    c.require(rule["horizon"] in c.HORIZONS and type(rule["n"]) is int and 16 <= rule["n"] <= 128
              and type(rule["M"]) is int and rule["n"] < rule["M"] <= 512
              and type(rule["beta"]) in (int, float) and math.isfinite(rule["beta"]), "Invalid fixed validation rule")
    return selected


def _check_root(root, selected, descriptor):
    spec, support = descriptor["spec"], descriptor["support"]
    c.require(root["manifest"] == selected["manifest"] == c.SOURCE_BINDING and root["stage"] == selected["stage"] == "main"
              and root["spec"] == spec and root["selection_artifact"] == descriptor["selection"]
              and root["selection_lock"] == selected["selection_lock"]
              and root["selection_lock"]["created_at_ns"] < root["assessment_started_at_ns"], "Saved pre-assessment selection identity changed")
    identity = selected["identity"]
    c.require(c.old.digest(identity) == selected["selection_lock"]["identity_digest"]
              and all(root[k] == identity[k] for k in ("snapshot", "candidates", "old_teacher_choice", "spec", "privacy"))
              and c.old.digest(root["snapshot"]) == spec["snapshot_sha256"]
              and root["snapshot"]["current_player"] == spec["seat"] and not root["snapshot"]["terminal"], "Saved state/action identity changed")
    c.require(root["old_teacher_choice"] == descriptor["reference"] == selected["selection_lock"]["choices"]["teacher"]
              and len(root["candidates"]["actions"]) == len(root["candidates"]["keys"]) == support
              and len(set(root["candidates"]["keys"])) == support, "Fixed teacher/support changed")
    block = root["assessment"]
    seeds = [2210000000+1000000+spec["index"]*512+i for i in range(128)]
    c.require(block["candidate_indices"] == list(range(support)) and block["world_seeds"] == seeds
              and len(block["worlds"]) == 128 and set(seeds).isdisjoint(root["old_selection"]["world_seeds"]), "Independent target world block changed")
    for seed, world in zip(seeds, block["worlds"], strict=True):
        c.require(world["seed"] == seed and len(world["state_sha256"]) == 64 and len(world["outcomes"]) == support, "Saved world/candidate coupling changed")
        for outcome in world["outcomes"]:
            c.require(outcome["reason"] in c.old.VALID_NATURAL_TERMINATIONS and type(outcome["wdl"]) in (int, float)
                      and outcome["wdl"] in (-1, 0, 1) and outcome["winner"] in (None, 0, 1)
                      and outcome["wdl"] == c.old.outcome_value(outcome["winner"], spec["seat"]), "Unknown or inconsistent saved terminal target")
    return True


def load_root(spec, binding, base=None, *, stage="train", allow_validation=False):
    base = _base(base)
    c.require(stage in ("train", "validation") and (stage == "validation" or spec["split"] == "train"), "Source split/stage mismatch")
    c.require(spec["split"] == stage, "No cross-split source substitution")
    if stage == "validation":
        selection_barrier(binding, base)  # allow_validation never bypasses this seal.
    lock = _lock(binding, base)
    index = spec["index"]
    c.require(type(index) is int and 0 <= index < 336 and lock["schedule"][index] == spec, "Root differs from sealed schedule")
    descriptor = lock["parent"]["descriptors"][index]
    root, selected = c.verified(descriptor["root"]), c.verified(descriptor["selection"])
    _check_root(root, selected, descriptor)
    return root, descriptor["root"]
