"""Frozen settings and resource boundaries for planner-state transfer."""
from pathlib import Path
import shutil
import time
from tools.proposal_iteration_128.common import (
    ROOT, old, baseline, windows, require, artifact, verified, verify_hash,
)

OUTPUT = ROOT / "outputs/planner_state_transfer_128"
SOURCE = ROOT / "outputs/equal_budget_breadth_arena_128/recovery_03"
SOURCE_BINDING = "904b6780bceedca3d620f7a548ba4e903b8558e0a3841f065dfb4eed8d5530c8"
LEARNING = ROOT / "outputs/proposal_breadth_learning_128"
LEARNING_BINDING = "bf916c18fc25e5be4bb2b7ff13317157d155da82668ae4d8a86308544b5ca72a"
FITS = (161001, 161002, 161003)
CONFIG = {
    "states": {"probe": 18, "main": 360},
    "source_pairs": {"probe": 9, "main": 180},
    "selection_tag": "planner-state-transfer-128-panel-v1",
    "worlds": 128, "assessment_worlds": 128,
    "maximum_candidates": 10, "maximum_assessed_choices": 4,
    "world_seed_base": 2180000000,
    "bootstrap_seed": 2183000000, "bootstrap_samples": 50000,
    "bootstrap_chunk": 1000,
    "audit_seed": 2194000000,
    "audit_world_tag": "planner-state-transfer-128-audit-world-v1",
    "audit_stress_tag": "planner-state-transfer-128-audit-stress-v1",
    "replay_worlds_per_block": 8, "maximum_stress_roots": 39,
    "stress_supports": [1, 2, 3],
    "probe_full_replay": True, "full_inner_replay": False,
    "minimum_progress": .005, "superiority_lower_bound": 0.,
    "workers": 8, "torch_threads": 1, "device": "cpu", "backend": "serial",
    "max_turns": 400, "repetition_limit": 4, "no_progress_limit": 60,
    "maximum_runtime_hours": 8., "forecast_allowance": 1.35,
    "finalization_allowance_seconds": 300., "disk_reserve_bytes": 5 * 1024**3,
    "maximum_main_selection_continuations": 460800,
    "maximum_main_assessment_continuations": 184320,
    "maximum_main_continuations": 645120,
    "maximum_probe_continuations": 32256,
    "training": False, "automatic_retry": False, "promotion": False,
    "automatic_successor": False, "scheduled_monitoring": False,
}

class BudgetExceeded(ValueError):
    """Preserve evidence at the fixed resource ceiling."""

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
