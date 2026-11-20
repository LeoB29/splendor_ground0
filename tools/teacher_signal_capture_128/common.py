"""Fixed identities, resources and scientific scope for the signal audit."""
from pathlib import Path
import shutil
import time
from tools.proposal_iteration_128.common import (
    ROOT, old, baseline, windows, require, artifact, verified, verify_hash,
)

OUTPUT = ROOT / "outputs/teacher_signal_capture_128"
SOURCE = ROOT / "outputs/proposal_breadth_arena_128"
SOURCE_BINDING = "a8395f535998379a2062ebce8fed79cc6a26e5cd1deb699c96bd4b9dca06deb5"
LEARNING = ROOT / "outputs/proposal_breadth_learning_128"
LEARNING_BINDING = "bf916c18fc25e5be4bb2b7ff13317157d155da82668ae4d8a86308544b5ca72a"
FITS = (161001, 161002, 161003)
CONFIG = {
    "states": {"probe": 18, "main": 336},
    "source_pairs": {"probe": 9, "main": 84},
    "main_pairs_per_fit_split": 14, "main_roots_per_pair": 4,
    "source_splits": ["train", "validation"],
    "selection_tag": "teacher-signal-capture-128-panel-v1",
    "worlds": 128, "assessment_worlds": 128,
    "maximum_candidates": 8, "world_seed_base": 2210000000,
    "bootstrap_seed": 2213000000, "bootstrap_samples": 50000,
    "bootstrap_chunk": 1000, "minimum_progress": .005,
    "audit_seed": 2224000000,
    "audit_world_tag": "teacher-signal-capture-128-audit-world-v1",
    "audit_stress_tag": "teacher-signal-capture-128-audit-stress-v1",
    "replay_worlds_per_block": 8, "maximum_stress_roots": 39,
    "stress_supports": [1, 2, 3], "probe_full_replay": True,
    "full_inner_replay": False, "workers": 8, "torch_threads": 1,
    "device": "cpu", "backend": "serial", "max_turns": 400,
    "repetition_limit": 4, "no_progress_limit": 60,
    "maximum_runtime_hours": 8., "forecast_allowance": 1.35,
    "finalization_allowance_seconds": 300.,
    "disk_reserve_bytes": 5 * 1024**3,
    "maximum_main_continuations": 344064,
    "maximum_probe_continuations": 18432,
    "training": False, "automatic_retry": False, "promotion": False,
    "automatic_successor": False, "scheduled_monitoring": False,
}


class BudgetExceeded(ValueError):
    """Stop at a fixed operational bound while preserving evidence."""


def budget_boundary(base=OUTPUT):
    base = Path(base)
    if (base / "abort.json").exists():
        raise BudgetExceeded("Cooperative abort requested; preserve partial evidence")
    launch = base / "launch.json"
    if not launch.exists():
        require(not base.resolve().is_relative_to(OUTPUT.resolve()), "Production has no launch receipt")
        return
    if time.time() - old.read_json(launch)["epoch_seconds"] > CONFIG["maximum_runtime_hours"] * 3600:
        raise BudgetExceeded("Eight-hour execution ceiling reached")
    if shutil.disk_usage(ROOT).free < CONFIG["disk_reserve_bytes"]:
        raise BudgetExceeded("Five-GiB disk reserve reached")
