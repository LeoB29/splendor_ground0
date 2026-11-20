"""Read-only authentication of the stopped precision run and reusable labels.

No trajectory, policy evaluation, fit, replay or resampling is performed here.
Absence/process checks apply at recovery preparation and launch only. Ordinary
validation deliberately permits subsequent scientific outputs in the old base.
"""
from __future__ import annotations

from datetime import datetime
import json
import math
import os
from pathlib import Path
import re
import subprocess

from tools.teacher_precision_512 import common as c

ORIGINAL = c.OUTPUT
BINDING = "d2321f90c0d92ec0834a3aa1b4e9ace9834a452428d233c2e41b3f1f0c257b0c"
ABORT_ERROR = "Eight-hour execution ceiling reached; preserve evidence"
HOLD_REASON = "Cooperative abort; preserve evidence and diagnose before recovery"
LABEL_VERIFICATION = {
    "label_roots": 720, "label_audits": 720, "new_blocks": 2160, "block_completion_receipts": 2160,
    "teacher_worlds_per_root": 512, "all512_sums_independently_checked": True,
    "target_tensor_checked": True, "old128_authenticated": True, "unknown_outcomes": 0,
    "crossfit_score_percentage_points": 3454*100/(720*1024),
}
PINNED_FILES = {
    "freeze.json": "bf738d315c4c083b3ed6841408936e4e1a03140326db3b718f5811da8b072f1b",
    "manifest.json": "4502703eb412d9246823801e3bce8af3020defce1ed849499ee5ac2dcf032ec6",
    "data_lock.json": "751cb1cee454fcf2433bdc7ce1feebf7f70cbd1db7ff1517e6fe37225144f82c",
    "label_plan.json": "57a894664cf1ec43084c01bc651c63bca27080e340b036a9e233c29ef358d3fa",
    "launch.json": "9ff2c25471b306eeaf4654c7fc7b2da45e550e94f4dae543ed52ddd6c3edf3b5",
    "abort.json": "fa33d4b0e1acb31a79abcf9ef4403784d7c23e935310cb961201131623011c9b",
    "operational_hold.json": "b0bb0fb58f32c8693503f7d2bda182a733428aa74c7c797abfbfad932e8e8c00",
    "learning_fits_launch.json": "fb0340a00fb21ddbaa17ead983c1bba76df53b4a8bfb5afb03fdd9d1abaa4f17",
    "labels_complete.json": "b172ed4ac4d68192a33a4f20997f3915949dcea39fc2e2423ed66a655876418e",
    "labels_lock.json": "fc87d7236a871fcd5e07bbc16231a5691e4262bea52e882565ca05abd07b2b5a",
    "labels_audits_lock.json": "0f45184199bbf6bcabeff8f37767fa64d3aed8cdde501115ca39dd2999813231",
    "target_lock.json": "db62b0146fb7df2c25db3b7a84983ed0e48befd17a7a09fbd19ecfbbe4ed68da",
    "runtime/process.json": "b6f5c743ce1a38f872472c65ea562f4cbffed0a46c453b587297dc8df7563e6c",
    "runtime/acknowledged.json": "c87ff61c51486bac76796cf2b33a8cede56a5decd156f72a81148e1a12ce44ec",
}
FORBIDDEN = (
    "failure.json", "active_run.json", "learning_fits_lock.json", "learning_audits_launch.json",
    "learning_audits_lock.json", "learning_fit_audits_launch.json", "learning_fit_audits_lock.json",
    "learning_complete.json", "models_lock.json", "model_lock.json",
    "models.json", "source", "probe", "main", "selection.json", "selection_seal.json",
    "result.json", "final_audit.json", "final_status.json",
)


def _plain(ref):
    return {k: ref[k] for k in ("path", "sha256")}


def _ref(path, sha=None):
    path = Path(path).resolve()
    try:
        name = path.relative_to(c.ROOT.resolve()).as_posix()
    except ValueError:
        name = path.as_posix()
    return {"path": name, "sha256": c.old.file_hash(path) if sha is None else sha}


def _epoch(value):
    parsed = datetime.fromisoformat(value)
    c.require(parsed.tzinfo is not None, "Timestamp lacks time zone")
    return parsed.timestamp()


def prior_active_seconds(launch, abort, hold, fit_launch, completed):
    """Charge original launch through its durable hold, excluding the pause."""
    start = launch["epoch_seconds"]
    c.require(type(start) in (int, float) and math.isfinite(start)
              and abs(start-_epoch(launch["time"])) < .00001, "Invalid original launch clock")
    c.require(abort["error"] == ABORT_ERROR and hold["reason"] == HOLD_REASON,
              "Unexpected interruption; inspect before recovery")
    c.require(launch["manifest"] == hold["manifest"] == fit_launch["manifest"]
              == completed["manifest"] == BINDING and launch["owner_authorized"] is True,
              "Original operational receipts do not share binding")
    moments = [start, _epoch(completed["time"]), _epoch(fit_launch["time"]),
               _epoch(abort["time"]), _epoch(hold["time"])]
    c.require(moments == sorted(moments) and 8*3600 < moments[-1]-start < 48*3600,
              "Original active time or stage order changed")
    return moments[-1]-start


class _Closure:
    """Deduplicate artifact hashes; never load a referenced assessment payload."""
    def __init__(self):
        self.refs = {}
        self.checked = set()

    def add(self, ref):
        plain = _plain(ref)
        c.require(isinstance(plain["path"], str) and isinstance(plain["sha256"], str)
                  and re.fullmatch(r"[0-9a-f]{64}", plain["sha256"]), "Malformed artifact reference")
        prior = self.refs.setdefault(plain["path"], plain)
        c.require(prior == plain, "Conflicting hashes for an inherited artifact")
        return plain

    def walk(self, value):
        if isinstance(value, dict):
            if {"path", "sha256"} <= value.keys():
                self.add(value)
            for item in value.values():
                self.walk(item)
        elif isinstance(value, list):
            for item in value:
                self.walk(item)

    def path(self, ref):
        plain = self.add(ref)
        if plain["path"] not in self.checked:
            path = c.verify_hash(plain)
            self.checked.add(plain["path"])
            return path
        return c.ROOT/plain["path"]

    def read(self, ref):
        return c.old.read_json(self.path(ref))

    def finish(self):
        for ref in self.refs.values():
            self.path(ref)
        return {p: self.refs[p]["sha256"] for p in sorted(self.refs)}


def _metadata(base, closure):
    refs = {name: _ref(base/name, sha) for name, sha in PINNED_FILES.items()}
    saved = {name: closure.read(ref) for name, ref in refs.items()}
    freeze, manifest = saved["freeze.json"], saved["manifest.json"]
    c.require(freeze["binding"] == c.old.digest(manifest) == BINDING
              and freeze["owner_authorized"] is manifest["owner_authorized"] is True
              and freeze["manifest"] == refs["manifest.json"]
              and freeze["data"] == refs["data_lock.json"]
              and freeze["label_plan"] == refs["label_plan.json"], "Frozen parent ancestry changed")
    c.require(manifest["identity"]["config"] == c.CONFIG
              and manifest["identity"]["fits"] == list(c.FITS), "Frozen scientific configuration changed")
    checks = closure.read(manifest["checks"])
    c.require(checks["exit_code"] == 0 and checks["identity"] == manifest["identity"],
              "Original official checks changed")
    for value in (saved, checks):
        closure.walk(value)
    for name, sha in manifest["identity"]["dependencies"].items():
        closure.add({"path": name, "sha256": sha})
    parent = manifest["source_parent"]
    c.require(parent == saved["data_lock.json"]["parent"]
              and parent["inventory"] == {r["path"]: r["sha256"] for r in parent["evidence"]},
              "Original source closure changed")
    launch, process, ack = (saved[k] for k in ("launch.json", "runtime/process.json", "runtime/acknowledged.json"))
    c.require(process["manifest"] == BINDING and process["pid"] == ack["pid"] == launch["pid"]
              and ack["launch"] == refs["launch.json"], "Original process ancestry changed")
    prior = prior_active_seconds(launch, saved["abort.json"], saved["operational_hold.json"],
                                 saved["learning_fits_launch.json"], saved["labels_complete.json"])
    fit = saved["learning_fits_launch.json"]
    c.require(fit["expected_indices"] == [0, 1, 2] and fit["stage"] == "learning" and fit["kind"] == "fits"
              and fit["dependencies"] == [refs[n] for n in
                  ("freeze.json", "data_lock.json", "target_lock.json", "labels_complete.json")],
              "Original fit-launch dependency contract changed")
    return refs, saved, prior


def _check_root(root, audit, spec, root_ref, plan_ref, full):
    c.require(root["manifest"] == audit["manifest"] == BINDING
              and root["spec"] == audit["spec"] == spec
              and root["index"] == audit["index"] == spec["index"]
              and root["row_id"] == audit["row_id"] == spec["row_id"]
              and root["complete"] is True and root["worlds"] == 512 and root["new_worlds"] == 384
              and root["unknown_outcomes"] == audit["unknown_outcomes"] == 0
              and root["assessment_read"] is audit["assessment_read"] is False
              and len(root["blocks"]) == 3, "Incomplete or changed inherited label root")
    c.require(audit["root"] == _plain(root_ref) and audit["plan"] == plan_ref
              and audit["passed"] is audit["old128_authenticated"] is True
              and audit["all512_sums_independently_checked"] is True
              and audit["old_trajectory_replays"] == 0 and audit["recorded_worlds"] == 384
              and audit["full_inner_replay"] is full and audit["replayed_worlds"] == (384 if full else 24)
              and audit["replayed_continuations"] == audit["replayed_worlds"]*spec["support"]
              and root["recorded_continuations"] == audit["recorded_continuations"] == 384*spec["support"]
              and root["inherited_continuations"] == 128*spec["support"], "Label audit certificate changed")
    for key, source_key in (("old_selection", "selection"), ("source_root", "root"), ("source_audit", "audit")):
        c.require(root[key] == spec["descriptor"][source_key], "Label inherited128 identity changed")


def _label_closure(base, closure, refs, saved):
    from tools.teacher_precision_512 import labels, label_source
    plan, roots, audits, target, complete = (saved[k] for k in
        ("label_plan.json", "labels_lock.json", "labels_audits_lock.json", "target_lock.json", "labels_complete.json"))
    c.require(len(plan["specs"]) == 720 and [s["index"] for s in plan["specs"]] == list(range(720))
              and [s["row_index"] for s in plan["specs"]] == list(range(11950, 12670))
              and plan["data"] == refs["data_lock.json"] and plan["assessment_read"] is False
              and plan["old128_authenticated"] is True and plan["sample"] == label_source.sample_plan(plan["specs"]),
              "Label TRAIN population or prescribed audit sample changed")
    for seal in (roots, audits):
        c.require(seal["manifest"] == BINDING and seal["complete"] is seal["all_prescribed_audits_pass"] is True
                  and seal["states"] == 720 and seal["worlds_per_root"] == 512 and seal["new_worlds_per_root"] == 384
                  and seal["assessment_read"] is False and seal["plan"] == refs["label_plan.json"]
                  and len(seal["files"]) == 720 and [r["index"] for r in seal["files"]] == list(range(720)),
                  "Incomplete inherited720 label seal")
    c.require(complete["manifest"] == BINDING and complete["labels"] == refs["labels_lock.json"]
              and complete["audits"] == refs["labels_audits_lock.json"] and complete["rows"] == 720
              and complete["new_worlds_per_root"] == 384 and complete["old_worlds_replayed"] is False
              and complete["all_prescribed_audits_pass"] is True, "Label completion ancestry changed")
    data = saved["data_lock.json"]
    c.require(target["manifest"] == BINDING and target["data"] == refs["data_lock.json"]
              and target["label_roots"] == refs["labels_lock.json"] and target["label_audits"] == refs["labels_audits_lock.json"]
              and target["rows"] == 720 and target["target_worlds"] == 512 and target["old_target_worlds"] == 128
              and target["old_data"] == data["files"]["old"] and target["original_added"] == data["files"]["added"]
              and target["assessment_read"] is target["validation_read"] is False and target["historical_replays"] == 0
              and all(target[k] is True for k in ("passed", "only_added_targets_change", "all512_sums_independently_checked",
                                                 "all_label_audits_passed", "original_inputs_unchanged")),
              "Target seal ancestry changed")
    diagnostic = closure.read(complete["crossfit_diagnostic"])
    c.require(diagnostic["manifest"] == BINDING and diagnostic["collection"] == refs["labels_lock.json"]
              and diagnostic["audits"] == refs["labels_audits_lock.json"] and diagnostic["states"] == 720
              and diagnostic["descriptive_only"] is diagnostic["independent_columnar_arithmetic_checked"] is True
              and diagnostic["assessment_read"] is diagnostic["model_gate"] is diagnostic["resampling"] is False
              and diagnostic["new_simulations"] == diagnostic["historical_trajectory_replays"] == 0
              and len(diagnostic["rows"]) == 720, "Descriptive diagnostic receipt changed")
    totals, reconstructed_diagnostic = [], []
    for spec, root_ref, audit_ref in zip(plan["specs"], roots["files"], audits["files"], strict=True):
        i, support = spec["index"], spec["support"]
        c.require((c.ROOT/root_ref["path"]).resolve() == labels.root_path(i, base).resolve()
                  and (c.ROOT/audit_ref["path"]).resolve() == labels.audit_path(i, base).resolve(),
                  "Wrong label root/audit path")
        root, audit = closure.read(root_ref), closure.read(audit_ref)
        closure.walk(root); closure.walk(audit)
        full = i in plan["sample"]["full_replay_indices"]
        _check_root(root, audit, spec, root_ref, refs["label_plan.json"], full)
        old_selection = closure.read(root["old_selection"])["selection"]
        columns = [labels.independent_sums(old_selection["worlds"], support)]
        used = set(old_selection["world_seeds"])
        c.require(len(used) == 128 and root["old_sums_wdl"] == spec["old_sums_wdl"] == columns[0],
                  "Inherited128 column arithmetic changed")
        for b, block_ref in enumerate(root["blocks"], 1):
            receipt_ref = block_ref["completion"]
            c.require(block_ref["block"] == b
                      and (c.ROOT/block_ref["path"]).resolve() == labels.block_path(i, b, base).resolve()
                      and (c.ROOT/receipt_ref["path"]).resolve() == labels.block_receipt_path(i, b, base).resolve(),
                      "Block completion path changed")
            receipt, block = closure.read(receipt_ref), closure.read(block_ref)
            c.require(receipt == {"manifest": BINDING, "index": i, "row_id": spec["row_id"],
                        "spec_sha256": c.old.digest(spec), "block": b, "artifact": _plain(block_ref)},
                      "Unsealed or changed completed128 block")
            c.require(block["manifest"] == BINDING and block["index"] == i and block["row_id"] == spec["row_id"]
                      and block["spec_sha256"] == c.old.digest(spec) and block["block"] == b
                      and block["complete"] is True and block["assessment_read"] is False and block["unknown_cells"] == []
                      and block["world_tag"] == c.CONFIG["label_world_tag"]
                      and block["candidate_indices"] == list(range(support))
                      and block["world_seeds"] == labels.world_seeds(i, b)
                      and [w["seed"] for w in block["worlds"]] == block["world_seeds"], "Label block contract changed")
            c.require(used.isdisjoint(block["world_seeds"]), "Label world namespace overlap")
            used.update(block["world_seeds"])
            column = labels.independent_sums(block["worlds"], support)
            c.require(column == block["sums_wdl"], "Label block aggregate differs from saved outcomes")
            columns.append(column)
            expected_replay = list(range(128)) if full else labels.ordinary_indices(i, b)
            c.require(audit["replay_world_indices"][str(b)] == expected_replay, "Prescribed replay sample changed")
        total = [sum(col[j] for col in columns) for j in range(support)]
        c.require(root["sums_wdl512"] == total and len(used) == 512, "Complete512 label arithmetic changed")
        totals.append(total)
        reconstructed_diagnostic.append({"index": i, "row_id": spec["row_id"], **labels.crossfit_row(columns)})
    c.require(diagnostic["rows"] == reconstructed_diagnostic, "Crossfit rows differ from saved block arithmetic")
    numerator = sum(r["sum_heldout_wdl_differences"] for r in reconstructed_diagnostic)
    c.require(diagnostic["mean_score_gain"] == numerator/(720*1024)
              and diagnostic["score_percentage_points"] == 100*numerator/(720*1024)
              and diagnostic["choice_disagreements"] == sum(r["choice_disagreements"] for r in reconstructed_diagnostic),
              "Crossfit reduction changed")
    # Tensor inspection checks labels/inputs only; no model or inference is loaded.
    import torch
    from tools.teacher_precision_512 import training_source
    rows = torch.load(closure.path(target["file"]), map_location="cpu", weights_only=True)
    original = torch.load(closure.path(target["original_added"]), map_location="cpu", weights_only=True)
    c.require(len(rows) == len(original) == 720 and training_source.row_digest(rows) == target["row_digest"],
              "Precision target tensor changed")
    stripped = []
    for i, (row, total) in enumerate(zip(rows, totals, strict=True)):
        c.require(row["index"] == 11950+i and row["id"] == plan["specs"][i]["row_id"]
                  and row["teacher_worlds"] == 512 and row["teacher_sums_wdl"] == total
                  and row["teacher_outsider"] == (max(range(3, len(total)), key=lambda j: total[j]) if len(total)>3 else -1),
                  "Tensor512 target does not match complete saved label")
        stripped.append({k: v for k, v in row.items() if k not in training_source.TARGET_FIELDS})
    c.require(training_source.row_digest(stripped) == training_source.row_digest(original),
              "Precision tensor altered inherited inputs or128 targets")
    result = {"label_roots": 720, "label_audits": 720, "new_blocks": 2160, "block_completion_receipts": 2160,
              "teacher_worlds_per_root": 512, "all512_sums_independently_checked": True,
              "target_tensor_checked": True, "old128_authenticated": True, "unknown_outcomes": 0,
              "crossfit_score_percentage_points": diagnostic["score_percentage_points"]}
    c.require(result == LABEL_VERIFICATION, "Stopped label verification contract changed")
    return result


def _known_processes(base, closure, saved):
    process = saved["runtime/process.json"]
    known = [{"pid": process["pid"], "executable": process["executable"], "role": "supervisor"}]
    refs = []
    for path in sorted((base/"workers").glob("*/*_health.json")):
        ref = _ref(path)
        health = closure.read(ref)
        c.require(health["manifest"] == BINDING and health["parent_pid"] == process["pid"]
                  and health["console_window"] == 0 and health["console_process_count"] == 0,
                  "Original worker health identity changed")
        known.append({"pid": health["pid"], "executable": health["executable"],
                      "parent_pid": health["parent_pid"], "role": health["stage"]+"_"+health["kind"]})
        refs.append(ref)
    c.require(len(known) == 36 and len({k["pid"] for k in known}) == 36
              and saved["abort.json"]["pid"] in {k["pid"] for k in known}, "Original process inventory incomplete")
    return known, refs


def _process_snapshot(pids):
    """One hidden read-only OS query; PID reuse is classified separately."""
    c.require(os.name == "nt", "Production stopped-process check requires Windows")
    ids = ",".join(str(int(p)) for p in sorted(set(pids)))
    script = ("$wanted=@("+ids+"); @(Get-CimInstance Win32_Process | Where-Object { $_.ProcessId -in $wanted } | "
              "ForEach-Object { @{pid=[int]$_.ProcessId; parent_pid=[int]$_.ParentProcessId; "
              "executable=$_.ExecutablePath; created=if ($_.CreationDate) {$_.CreationDate.ToUniversalTime().ToString('o')} else {$null}} }) "
              "| ConvertTo-Json -Compress")
    output = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
                            capture_output=True, text=True, check=True, timeout=30,
                            creationflags=subprocess.CREATE_NO_WINDOW).stdout.strip()
    value = json.loads(output) if output else []
    return value if isinstance(value, list) else [value]


def classify_processes(known, observed, earliest, held):
    """Reject live/ambiguous original identities, tolerate demonstrably reused PIDs."""
    by_pid = {p["pid"]: p for p in observed}
    c.require(len(by_pid) == len(observed), "Duplicate OS process observation")
    reports = []
    for expected in known:
        current = by_pid.get(expected["pid"])
        status = "absent"
        if current is not None:
            c.require(current.get("created"), "Cannot distinguish original process from reused PID")
            created = _epoch(current["created"])
            if created > held or created < earliest-2:
                status = "reused_pid_outside_original_run"
            else:
                c.require(current.get("executable"), "Cannot authenticate live original PID executable")
                same = os.path.normcase(current["executable"]) == os.path.normcase(expected["executable"])
                c.require(not same, "Original supervisor or worker remains live; recovery must hold")
                status = "reused_pid_other_executable"
        reports.append({"pid": expected["pid"], "status": status})
    return reports


def _assert_no_scientific_outputs(base):
    c.require(not any((base/name).exists() for name in FORBIDDEN),
              "Stopped parent advanced or failed; inspect before recovery")
    expected = {base/"fits"/f"fit_{fit}"/"precision" for fit in c.FITS}
    c.require(all(p.is_dir() for p in expected), "Original empty fit directories missing")
    allowed = expected | {p.parent for p in expected}
    c.require(all(p.is_dir() and p in allowed for p in (base/"fits").rglob("*")),
              "Original fit contains checkpoint, result or unexpected artifact")
    for pattern in ("source_*", "probe_*", "main_*", "*models*", "*result*", "*checkpoint*"):
        c.require(not any(base.glob(pattern)), "Unexpected fresh source/model/evaluation artifact")


def assert_stopped(evidence):
    """Prelaunch-only absence/process guard. Never call after recovery starts."""
    base = c.ROOT/evidence["base"]
    _assert_no_scientific_outputs(base)
    launch = c.verified(evidence["receipts"]["launch.json"])
    hold = c.verified(evidence["receipts"]["operational_hold.json"])
    process = c.verified(evidence["receipts"]["runtime/process.json"])
    c.require(launch["pid"] in {p["pid"] for p in evidence["known_processes"]}, "Missing original supervisor identity")
    return classify_processes(evidence["known_processes"], _process_snapshot(p["pid"] for p in evidence["known_processes"]),
                              _epoch(process["time"]), _epoch(hold["time"]))


def stopped_parent(base=ORIGINAL):
    """Authenticate all reusable stopped evidence without modifying any file."""
    base = Path(base)
    _assert_no_scientific_outputs(base)
    closure = _Closure()
    refs, saved, prior = _metadata(base, closure)
    verified_labels = _label_closure(base, closure, refs, saved)
    known, health_refs = _known_processes(base, closure, saved)
    inventory = closure.finish()
    evidence = {"binding": BINDING, "original_binding": BINDING,
                "base": _ref(base/"freeze.json")["path"].rsplit("/", 1)[0],
                "receipts": refs, "prior_active_seconds": prior, "stop_error": ABORT_ERROR,
                "stop_reason": HOLD_REASON, "known_processes": known, "worker_health": health_refs,
                "label_verification": verified_labels, "inventory": inventory,
                "inventory_sha256": c.old.digest(inventory),
                "evidence": [{"path": p, "sha256": sha} for p, sha in inventory.items()],
                "new_simulations": 0, "historical_replays": 0, "inference_calls": 0,
                "paused_time_excluded": True, "mutable_progress_pinned": False}
    evidence["stopped_process_observations"] = assert_stopped(evidence)
    return evidence


def validate_inventory(evidence, deep=False):
    """Validate immutable recovery inputs, permitting legitimate new outputs."""
    c.require(evidence["binding"] == evidence["original_binding"] == BINDING and evidence["stop_error"] == ABORT_ERROR
              and evidence["stop_reason"] == HOLD_REASON and evidence["paused_time_excluded"] is True
              and evidence["mutable_progress_pinned"] is False
              and evidence["new_simulations"] == evidence["historical_replays"] == evidence["inference_calls"] == 0
              and evidence["label_verification"] == LABEL_VERIFICATION,
              "Recovery parent contract changed")
    inventory = evidence["inventory"]
    c.require(c.old.digest(inventory) == evidence["inventory_sha256"]
              and evidence["evidence"] == [{"path": p, "sha256": inventory[p]} for p in sorted(inventory)],
              "Recovery evidence inventory changed")
    base = c.ROOT/evidence["base"]
    closure = _Closure()
    refs, saved, prior = _metadata(base, closure)
    c.require(refs == evidence["receipts"] and prior == evidence["prior_active_seconds"],
              "Recovery prior active time/receipts changed")
    for ref in evidence["worker_health"]:
        closure.add(ref)
    c.require(all(inventory.get(p) == ref["sha256"] for p, ref in closure.refs.items()),
              "Recovery inventory dropped original ancestry")
    if deep:
        # The preparation pass already reconstructed every saved sum and tensor.
        # Identical bytes preserve that proof; repeat hashes, not its arithmetic.
        # Use pinned original health receipts, not a mutable directory listing.
        for ref in evidence["evidence"]:
            closure.add(ref)
        c.require(closure.finish() == inventory, "Immutable recovery closure changed")
    return BINDING
