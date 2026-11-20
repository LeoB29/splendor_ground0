"""Authenticate interrupted recovery01 without replaying or selecting outcomes.

These checks establish that forty complete TRAIN files are safe to import.
They do not replace the original prescribed independent prefix audits: all 168
TRAIN roots, imported and newly collected, still undergo those audits before
calibration, and no validation outcome is read here.
"""
from __future__ import annotations

from datetime import datetime
import hashlib
import json
import math
import re

from tools.control_variate_128 import common as c, source, verify
from tools.control_variate_128_recovery_01 import recovery as r1

PRIOR = c.ROOT / "outputs/control_variate_128_recovery_01"
BINDING = "c24d8611a9936a05e90d8a93f4f56d4c942d5b62505992cb99538666a5ae8f34"
TRAIN_INDICES = tuple(range(38)) + (40, 41)
TRAIN_INVENTORY_SHA256 = "e867b463c397b42f7d59f308665fe4a8873ec2fb248da10bce3deaa2e988ad72"
INTERRUPTION_ALLOWANCE_SECONDS = 60.
PINNED_FILES = {
    "manifest.json": "3a019ac76d5e008f596c4451f5181c14c882fbe8f6b80c0be874b33485bb2e99",
    "freeze.json": "77a7458454bf1fbc783f71dc8cc92fbea772eeb72a52063ab10fbd2d4f036121",
    "data_lock.json": "2998d272a8ec3937c1b8ec8ac24083c06ff5aca95ff70c46db71be5686a26e74",
    "timing_lock.json": "071c26d3f5daf9fb2441939b4d48308e7f406903d21b61e3076423065f2dee85",
    "timing_audits_lock.json": "c4b7a39bf969bbce277bef42dce02ced7ecda8dbbc5fcfe843fdd24e35ab2466",
    "forecast.json": "e8f5de138db3161c123b3edbadc86249587cbf3687b49e2c6a8137ea608da86a",
    "launch.json": "390e61b7f5e895a6aecd28dabcf7e345c91e41b7985c2f2df475b5368a096b5e",
    "progress.json": "b2f28760231f87c5f09017539d13a3d246133eefa61e33991e5b33e86aa83654",
    "train_launch.json": "2fa56c442c01c8c28c026fc27efc44c8a777a41eb7d11e877eb4de060c699281",
    "timing_reuse.json": "1763962a7a088b0546c1f3cc6ec7bf49e6669d7464227205d7f9ab9f263d2e6b",
    "runtime/process.json": "4f99e3f94e740e3c9180eef8edf4f91595d1eee11eba23f6de4903052d32393a",
    "runtime/acknowledged.json": "16442fc63ddac6cea9da448d5446715badcab3e5e94f5b83365b5c05f81f31b1",
    "runtime/stdout.txt": "14118aaa0f2441089ec1abb3fc1cae13d49cd7a5cfd5de2d57e11c1930f0a0ea",
    "runtime/stderr.txt": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
}
FORBIDDEN = (
    "train/audits", "train_lock.json", "train_audits_launch.json", "train_audits_lock.json",
    "validation", "validation_launch.json", "validation_lock.json", "validation_audits_lock.json",
    "selection_lock.json", "selection_audit.json", "assessment_result.json", "assessment_audit.json",
    "result.json", "final_audit.json", "final_status.json", "failure.json", "abort.json", "operational_hold.json",
)
IDENTITY_KEYS = ("index", "pair", "fit", "split", "game_index", "phase", "seat")
RECORD_KEYS = {
    "complete", "evaluator", "fit", "forced_root_counts_as_ply", "game_index", "horizons", "index",
    "manifest", "new_terminal_simulations", "pair", "phase", "reference", "seat", "source", "spec",
    "split", "stage", "statistics", "support", "timing", "worlds", "y", "z",
}


def prior_active_seconds(launch, progress):
    """Charge a conservative post-progress minute; exclude the interrupted pause.

The exact process exit instant was not recorded. This explicit accounting proxy
ends sixty seconds after the last durable progress event, later than the service
update interruption reported in the status investigation. It is not an exact
exit-time assertion and does not treat two paused days as computation.
    """
    began = launch["epoch_seconds"]
    previous = launch["previous_elapsed_seconds"]
    c.require(type(began) in (int, float) and math.isfinite(began)
              and type(previous) in (int, float) and math.isfinite(previous) and previous >= 0,
              "Invalid inherited active time")
    launch_time, last_time = (datetime.fromisoformat(value["time"]) for value in (launch, progress))
    c.require(launch_time.tzinfo is not None and last_time.tzinfo is not None,
              "Inherited timestamps must carry time zones")
    c.require(abs(launch_time.timestamp()-began) < .00001 and last_time.timestamp() >= began,
              "Inherited launch/progress time order changed")
    return previous + last_time.timestamp() - began + INTERRUPTION_ALLOWANCE_SECONDS


def check_record(record, spec, saved, source_ref, evaluator_ref, coefficients):
    """Check complete structure, saved-world coupling and public score projection.

This performs arithmetic and comparisons only. No policy evaluation, simulated
move, new terminal game, validation root or calibration statistic is evaluated.
    """
    c.require(set(record) == RECORD_KEYS and record["manifest"] == BINDING
              and record["stage"] == record["split"] == spec["split"] == "train"
              and record["spec"] == spec and record["source"] == source_ref
              and record["evaluator"] == evaluator_ref and record["complete"] is True
              and record["horizons"] == list(c.HORIZONS)
              and record["forced_root_counts_as_ply"] is True and record["new_terminal_simulations"] == 0,
              "Incomplete or changed inherited TRAIN root identity")
    c.require(all(record[key] == saved["spec"][key] == spec[key] for key in IDENTITY_KEYS),
              "Inherited statistical root identity changed")
    support = len(saved["candidates"]["actions"])
    c.require(1 <= support <= 8 and record["support"] == spec["support"] == support
              and record["reference"] == spec["reference"] == saved["old_teacher_choice"],
              "Inherited candidate/reference mapping changed")
    c.require(set(record["timing"]) == {"cpu", "wall"}
              and all(type(v) in (int, float) and math.isfinite(v) and v >= 0
                      for v in record["timing"].values()), "Invalid inherited root timing")
    worlds = saved["assessment"]["worlds"]
    c.require(len(worlds) == len(record["worlds"]) == 128, "Inherited world population incomplete")
    z = {str(h): [] for h in c.HORIZONS}
    for wi, (expected, actual) in enumerate(zip(worlds, record["worlds"], strict=True)):
        c.require(set(actual) == {"index", "seed", "state_sha256", "candidates"}
                  and actual["index"] == wi and actual["seed"] == expected["seed"]
                  and actual["state_sha256"] == expected["state_sha256"]
                  and len(actual["candidates"]) == support, "Inherited world/candidate coupling changed")
        rows = {str(h): [] for h in c.HORIZONS}
        for ci, candidate in enumerate(actual["candidates"]):
            c.require(set(candidate) == set(z), "Incomplete inherited horizon set")
            for h in c.HORIZONS:
                point = candidate[str(h)]
                c.require(all(type(point[k]) is str and re.fullmatch(r"[0-9a-f]{64}", point[k])
                              for k in ("state_sha256", "trace_sha256")), "Invalid inherited endpoint digest")
                verify.verify_endpoint(point, spec["seat"], h, coefficients, expected["outcomes"][ci])
                rows[str(h)].append(point["z"])
        for h in z:
            z[h].append(rows[h])
    y = [[(1+outcome["wdl"])/2 for outcome in world["outcomes"]] for world in worlds]
    c.require(record["y"] == y and record["z"] == z, "Inherited outcome/endpoint projection changed")
    expected_statistics = {**{key: spec[key] for key in IDENTITY_KEYS}, "support": support,
                           "reference": spec["reference"], "y": y, "z": z}
    c.require(record["statistics"] == expected_statistics, "Inherited statistical projection changed")
    return True


def authenticate():
    """Authenticate the fixed interrupted predecessor and its forty saved roots."""
    c.require(r1.OUTPUT == PRIOR, "Recovery01 output binding was redirected")
    c.require(not any((PRIOR/name).exists() for name in FORBIDDEN),
              "Interrupted parent advanced or failed; inspect before continuation")
    references = {name: {"path": (PRIOR/name).relative_to(c.ROOT).as_posix(), "sha256": sha}
                  for name, sha in PINNED_FILES.items()}
    for ref in references.values():
        c.verify_hash(ref)
    c.require(r1.validate_freeze() == BINDING, "Interrupted recovery01 freeze changed")
    freeze, manifest, launch, progress, forecast = (c.verified(references[name+".json"])
                                                  for name in ("freeze", "manifest", "launch", "progress", "forecast"))
    c.require(freeze["binding"] == c.old.digest(manifest) == BINDING
              and freeze["manifest"] == references["manifest.json"], "Interrupted manifest linkage changed")
    c.require(launch["manifest"] == progress["manifest"] == forecast["manifest"] == BINDING
              and launch["owner_authorized"] is True and launch["maximum_active_seconds"] == 10800.
              and progress["stage"] == "train" and progress["completed"] == 40 and progress["total"] == 168
              and progress["status_pid"] == launch["pid"]
              and forecast["outcome_blind"] is forecast["within_runtime_ceiling"] is forecast["disk_reserve_pass"] is True
              and forecast["cost_estimates_unchanged"] is forecast["idle_pause_excluded"] is True,
              "Interrupted progress/launch/forecast contract changed")
    process, acknowledged = (c.verified(references["runtime/"+name+".json"])
                             for name in ("process", "acknowledged"))
    c.require(process["manifest"] == BINDING and process["pid"] == acknowledged["pid"] == launch["pid"]
              and acknowledged["launch"] == references["launch.json"], "Interrupted process receipt changed")
    schedule = source.schedule(BINDING, PRIOR, "train")
    train_launch = c.verified(references["train_launch.json"])
    c.require(train_launch["manifest"] == BINDING and train_launch["args_sha256"] == c.old.digest(schedule),
              "Interrupted collection arguments changed")
    timing = c.verified(references["timing_lock.json"])
    c.require(forecast["timing"] == timing["files"], "Interrupted forecast/timing link changed")
    files = sorted((PRIOR/"train/roots").iterdir())
    expected_names = [f"state_{i:04d}.json" for i in TRAIN_INDICES]
    c.require([p.name for p in files] == expected_names and all(p.is_file() for p in files),
              "Interrupted root inventory changed")
    inventory = {p.name: c.old.file_hash(p) for p in files}
    inventory_digest = hashlib.sha256(json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    c.require(inventory_digest == TRAIN_INVENTORY_SHA256, "Interrupted saved root bytes changed")
    specs = {spec["index"]: spec for spec in schedule}
    evaluator_ref = source.evaluator(PRIOR)
    coefficients = c.verified(evaluator_ref)["coefficients"]
    roots = []
    for index, path in zip(TRAIN_INDICES, files, strict=True):
        c.require(index in specs, "Saved root is not TRAIN")
        spec = specs[index]
        saved, source_ref = source.load_root(spec, BINDING, PRIOR, stage="train")
        ref = {"path": path.relative_to(c.ROOT).as_posix(), "sha256": inventory[path.name], "index": index}
        record = c.verified(ref)
        check_record(record, spec, saved, source_ref, evaluator_ref, coefficients)
        roots.append(ref)
    return {**{name: references[name+".json"] for name in
               ("manifest", "freeze", "data_lock", "launch", "progress", "forecast", "timing_lock", "timing_audits_lock",
                "train_launch", "timing_reuse")},
            "runtime_evidence": {name: ref for name, ref in references.items() if name.startswith("runtime/")},
            "binding": BINDING, "checks": manifest["checks"], "train_roots": roots,
            "train_inventory_sha256": TRAIN_INVENTORY_SHA256,
            "previous_elapsed_seconds": prior_active_seconds(launch, progress),
            "interruption_accounting": {
                "last_durable_progress_time": progress["time"], "additional_seconds": INTERRUPTION_ALLOWANCE_SECONDS,
                "method": "prior active seconds plus recovery01 launch-to-last-progress plus sixty seconds",
                "exact_exit_time_known": False, "idle_pause_excluded": True},
            "timing_roots": 18, "saved_train_roots": 40, "missing_train_roots": 128,
            "saved_root_structure_and_projection_verified": True,
            "all_168_independent_train_audits_still_required": True,
            "validation_evaluated": False, "new_simulations": 0, "historical_trajectory_replays": 0}
