"""Independent immutable registration; never modifies the source study."""
from __future__ import annotations

import ctypes
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/alternative_ranker_128"
STUDY_ROOT = ROOT / "outputs/structured_proposer_128"
STUDY_BINDING = "57cd1786992c123e2e8fdedf6e4cf567196588efc8bd1ab9d2bbb24b7908f454"
STUDY_MANIFEST_HASH = "2e65863f2c315ca0969fcb5a505f09b990431df6d78e45dfc1622db72b3b1f83"
PROTOCOL = ROOT / "docs/engineering/alternative_ranker_128.md"
EVALUATION_CONFIG = {
    "recipes": ["ridge", "hgb"], "roots": 720, "cells": 18,
    "roots_per_cell": 40, "worlds": 128,
    "bootstrap_samples": 50000, "bootstrap_seed": 2713000000,
    "ci_quantiles": [.0125, .9875], "familywise_alpha": .05,
    "endpoint": "fixed_choice_teacher_outsider_score_minus_anchor",
    "short_support": "retain_zero_gain", "unknowns": "conservative_bounds",
    "automatic_evaluation": False, "automatic_arena": False,
}
RESOURCE_CONFIG = {
    "threads": 1, "extra_workers": 1, "priority": "below_normal",
    "maximum_seconds": 7200, "maximum_own_bytes": 2 * 1024**3,
    "minimum_available_bytes": 768 * 1024**2,
    "minimum_commit_headroom_bytes": 2 * 1024**3,
    "disk_reserve_bytes": 5 * 1024**3,
    "ongoing_minimum_available_bytes": 128 * 1024**2,
    "ongoing_minimum_commit_headroom_bytes": 1024**3,
    "maximum_process_rss_bytes": 1536 * 1024**2,
    "guard_interval_seconds": 2,
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def artifact(path):
    path = Path(path).resolve()
    require(path.is_relative_to(ROOT), "Artifact outside workspace")
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": file_hash(path)}


def verify_hash(ref):
    path = (ROOT / ref["path"]).resolve()
    require(path.is_relative_to(ROOT), "Reference escapes workspace")
    require(file_hash(path) == ref["sha256"], "Artifact identity changed: " + ref["path"])
    return path


def read_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def verified(ref):
    return read_json(verify_hash(ref))


def new_json(path, value):
    path = Path(path).resolve()
    require(path.is_relative_to(OUTPUT) or
            (path.is_relative_to(ROOT / "outputs") and
             path.relative_to(ROOT / "outputs").parts[0].startswith("alternative_ranker_128_")),
            "Write outside this add-on's output namespace")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def own_inventory():
    files = list((ROOT / "tools/alternative_ranker_128").glob("*.py"))
    files += [PROTOCOL, ROOT / "run_alternative_ranker_128.py"]
    return {p.relative_to(ROOT).as_posix(): file_hash(p) for p in sorted(files)}


def packages():
    return {name: importlib.metadata.version(name)
            for name in ("numpy", "scipy", "scikit-learn", "torch", "threadpoolctl")}


def source_contract():
    freeze = read_json(STUDY_ROOT / "freeze.json")
    require(freeze["binding"] == STUDY_BINDING and
            freeze["manifest"]["sha256"] == STUDY_MANIFEST_HASH,
            "Unexpected source study")
    manifest = verified(freeze["manifest"])
    data = verified(freeze["data"])
    require(data["counts"] == {"old": 9461, "added": 11520}
            and data["assessment_in_payload"] is False,
            "TRAIN identity differs")
    dependencies = manifest["identity"]["dependencies"]
    for path, expected in dependencies.items():
        require(file_hash(ROOT / path) == expected, "Frozen source changed: " + path)
    for ref in data["files"].values():
        verify_hash(ref)
    return {"freeze": artifact(STUDY_ROOT / "freeze.json"),
            "manifest": freeze["manifest"], "data": freeze["data"],
            "train_files": data["files"], "counts": data["counts"],
            "dependencies": dependencies}


def blind_presence():
    # Names/existence only. Never parse a fresh outcome, source or fitting metric.
    paths = ["probe", "main", "selection.json", "selection_seal.json",
             "probe_complete.json", "main_complete.json", "final_status.json"]
    return {name: (STUDY_ROOT / name).exists() for name in paths}


def register(checks_path):
    from .features import FEATURE_CONFIG
    from .learning import CONFIG
    require(not (OUTPUT / "freeze.json").exists(), "Already frozen; preserve it")
    presence = blind_presence()
    require(not any(presence.values()), "Fresh panel already materialized; do not claim pre-result freeze")
    checks_ref = artifact(checks_path)
    checks = verified(checks_ref)
    require(checks["passed"] is True and checks["source_inventory"] == own_inventory(),
            "Preflight missing or source changed since checks")
    manifest = {"schema": "alternative_ranker_registration_v1", "time": now(),
        "owner_authorized": True, "study_binding": STUDY_BINDING,
        "source": source_contract(), "own_sources": own_inventory(),
        "features": FEATURE_CONFIG, "learning": CONFIG, "evaluation": EVALUATION_CONFIG,
        "resources": RESOURCE_CONFIG, "packages": packages(), "python": sys.version,
        "platform": platform.platform(), "checks": checks_ref,
        "pre_result_presence": presence, "fresh_outcomes_accessed": False,
        "freeze_kind": "complete_deterministic_algorithm_before_weights",
        "no_current_study_changes": True}
    require(not any(blind_presence().values()), "Fresh stage began during registration")
    binding = digest(manifest)
    new_json(OUTPUT / "manifest.json", manifest)
    result = {"binding": binding, "manifest": artifact(OUTPUT / "manifest.json"),
              "time": now(), "owner_authorized": True, "fresh_outcomes_accessed": False}
    new_json(OUTPUT / "freeze.json", result)
    return result


def validate_freeze():
    freeze = read_json(OUTPUT / "freeze.json")
    manifest = verified(freeze["manifest"])
    require(digest(manifest) == freeze["binding"], "Registration changed")
    require(manifest["own_sources"] == own_inventory(), "Registered add-on source changed")
    require(manifest["packages"] == packages(), "Registered library versions changed")
    require(manifest["python"] == sys.version, "Registered Python version changed")
    for ref in (manifest["source"]["freeze"], manifest["source"]["manifest"],
                manifest["source"]["data"]):
        verify_hash(ref)
    for path, sha in manifest["source"]["dependencies"].items():
        require(file_hash(ROOT / path) == sha, "Registered shared dependency changed: " + path)
    return freeze


def resource_snapshot():
    import psutil
    memory = psutil.virtual_memory()
    result = {"available_bytes": memory.available, "disk_free_bytes": shutil.disk_usage(ROOT).free}
    if os.name == "nt":
        class PerformanceInfo(ctypes.Structure):
            _fields_ = [("cb", ctypes.c_ulong)] + [(name, ctypes.c_size_t) for name in
                ("CommitTotal", "CommitLimit", "CommitPeak", "PhysicalTotal", "PhysicalAvailable",
                 "SystemCache", "KernelTotal", "KernelPaged", "KernelNonpaged", "PageSize")] + [
                 ("HandleCount", ctypes.c_ulong), ("ProcessCount", ctypes.c_ulong), ("ThreadCount", ctypes.c_ulong)]
        info = PerformanceInfo()
        info.cb = ctypes.sizeof(info)
        require(ctypes.windll.psapi.GetPerformanceInfo(ctypes.byref(info), info.cb), "Memory counters unavailable")
        result["commit_headroom_bytes"] = (info.CommitLimit - info.CommitTotal) * info.PageSize
    else:
        result["commit_headroom_bytes"] = memory.available
    return result


def resource_gate():
    value = resource_snapshot()
    require(value["available_bytes"] >= RESOURCE_CONFIG["minimum_available_bytes"], "Available RAM below768MiB")
    require(value["commit_headroom_bytes"] >= RESOURCE_CONFIG["minimum_commit_headroom_bytes"], "Commit headroom below2GiB")
    require(value["disk_free_bytes"] >= RESOURCE_CONFIG["disk_reserve_bytes"] + RESOURCE_CONFIG["maximum_own_bytes"], "Disk reserve insufficient")
    return value


def ongoing_resource_gate():
    import psutil
    value = resource_snapshot()
    require(value["available_bytes"] >= RESOURCE_CONFIG["ongoing_minimum_available_bytes"], "Available RAM below128MiB during extra work")
    require(value["commit_headroom_bytes"] >= RESOURCE_CONFIG["ongoing_minimum_commit_headroom_bytes"], "Commit headroom below1GiB during extra work")
    require(value["disk_free_bytes"] >= RESOURCE_CONFIG["disk_reserve_bytes"], "Disk reserve reached")
    require(psutil.Process().memory_info().rss <= RESOURCE_CONFIG["maximum_process_rss_bytes"], "Extra process RSS above1.5GiB")
    require(sum(p.stat().st_size for p in OUTPUT.rglob("*") if p.is_file()) <= RESOURCE_CONFIG["maximum_own_bytes"], "Own artifact limit reached")
    progress = STUDY_ROOT / "progress.json"
    if progress.exists():
        # The progress receipt is operational metadata, never a scientific panel.
        stage = read_json(progress).get("stage")
        require(stage in ("learning_data_validation", "learning_fits", "learning_fit_audits", "complete"),
                "Do not compete with the source study's simulation stage")
    return value


def configure_compute():
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "1"
    import psutil
    if os.name == "nt":
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
