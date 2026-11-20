"""Frozen design and a single active-time resource boundary for every stage."""
from pathlib import Path
import math
import shutil
import time

from tools.proposal_iteration_128.common import (
    ROOT, old, baseline, windows, require, artifact, verified, verify_hash,
)
from tools.independent_game_coverage_128_recovery_01 import common as coverage

OUTPUT = ROOT / "outputs/independent_game_scaling_128"
PARENT = ROOT / "outputs/independent_game_coverage_128_recovery_01"
PARENT_BINDING = "1415ca3ff570181ab6d08ef25b4254134cac568902538e754978d6c302c88263"
SOURCE = ROOT / "outputs/teacher_signal_capture_128"
SOURCE_BINDING = "d0105c4178c4b74300aa12ea9ccbf986bb53a1c8cc6eb6bef990bffb54670da1"
FITS = (161001, 161002, 161003)
LEARNING_ARMS = ("small", "medium", "large")
ARMS = ("anchor", "small", "medium", "large", "control", "teacher")
MODEL_ARMS = ARMS[:-1]
CONTRASTS = (
    "large_minus_anchor", "large_minus_small", "large_minus_medium",
    "large_minus_control", "medium_minus_anchor", "small_minus_anchor",
    "control_minus_anchor", "teacher_minus_anchor", "teacher_minus_large",
)
CONFIG = {
    **coverage.CONFIG,
    "train_seeds": dict(coverage.CONFIG["train_seeds"]),
    "added_rows": 11520, "core_rows": 720, "new_label_rows": 10800,
    "sizes": {"small": 720, "medium": 2880, "large": 11520},
    "updates": 7136, "row_exposures_per_fit": 456176,
    "old_exposures_per_fit": 228088, "added_exposures_per_fit": 228088,
    "checkpoint_selection": "fixed_final_update_7136",
    "batch_tag": "independent-game-scaling-128-matched-mixture-v1",
    "partition_tag": "independent-game-scaling-128-nested-v1",
    "resume_checkpoint_interval": 64,
    "diagnostic_updates": [0, 1784, 3568, 5352, 7136],
    "regret_capture_guide": .90,
    "states": {"train": 10800, "probe": 18, "main": 720},
    "maximum_candidates": 12, "maximum_assessed_choices": 6,
    "source_games": {"train": 32400, "probe": 54, "main": 2160},
    "source_seed": {"train": 2405000000, "probe": 2406000000,
                    "main": 2407000000, "test": 2408000000},
    "source_assignment_tag": "independent-game-scaling-128-source-cell-v1",
    "source_stream_tag": "independent-game-scaling-128-source-stream-v1",
    "selection_tag": "independent-game-scaling-128-panel-v1",
    "world_seed_base": 2400000000,
    "world_tag": "independent-game-scaling-128-evaluation-world-v1",
    "teacher_random_tag": "independent-game-scaling-128-teacher-random-v1",
    "label_world_seed_base": 2420000000,
    "label_world_tag": "independent-game-scaling-128-teaching-world-v1",
    "label_probe_rows": 18,
    "label_probe_tag": "independent-game-scaling-128-label-probe-v1",
    "label_audit_tag": "independent-game-scaling-128-label-audit-v1",
    "label_stress_tag": "independent-game-scaling-128-label-stress-v1",
    "label_replay_worlds_per_block": 8,
    "label_unresolved_outcomes": "hold_without_imputation",
    "teacher_label_worlds": 128, "inherited_label_worlds": 128,
    "bootstrap_seed": 2403000000, "audit_seed": 2404000000,
    "audit_world_tag": "independent-game-scaling-128-audit-world-v1",
    "audit_stress_tag": "independent-game-scaling-128-audit-stress-v1",
    "primary_endpoint": "large_minus_anchor",
    "required_control_endpoint": "large_minus_small",
    "maximum_runtime_hours": 168.,
    "pre_label_future_stages_reserve_seconds": 24 * 3600.,
    "pre_label_future_storage_bytes": 32 * 1024**3,
    "source_train_probe_games": 54,
    "source_probe_tag": "independent-game-scaling-128-source-probe-v1",
    "pre_source_future_stages_reserve_seconds": 120 * 3600.,
    "pre_source_future_storage_bytes": 32 * 1024**3,
    "maximum_main_continuations": 720 * 128 * 18,
    "maximum_probe_continuations": 18 * 128 * 18,
    "maximum_new_label_continuations": 10800 * 8 * 128,
    "base_network_frozen": False, "frozen_representation": False,
    "fresh_evaluation_independent_of_development_result": True,
    "old_validation_diagnostic_only": True,
}


class BudgetExceeded(ValueError):
    """Operational hold which preserves completed evidence."""


def active_seconds(base=None):
    base = Path(OUTPUT if base is None else base)
    path = base / "active_run.json" if (base / "active_run.json").exists() else base / "launch.json"
    if not path.exists():
        require(not base.resolve().is_relative_to(OUTPUT.resolve()), "Production launch receipt missing")
        return 0.
    seal_path = path.with_name(path.stem + "_seal.json")
    require(seal_path.exists(), "Active timing completion seal missing")
    seal = old.read_json(seal_path)
    require(verify_hash(seal["receipt"]).resolve() == path.resolve(), "Active timing receipt changed")
    receipt = verified(seal["receipt"])
    prior, epoch = receipt.get("prior_active_seconds", 0.), receipt["epoch_seconds"]
    require(all(type(v) in (float, int) and math.isfinite(v) and v >= 0 for v in (prior, epoch)),
            "Invalid active-time accounting")
    require(receipt["manifest"] == seal["manifest"], "Active timing binding changed")
    if (base / "freeze.json").exists():
        require(receipt["manifest"] == old.read_json(base/"freeze.json")["binding"],
                "Active timing does not belong to frozen experiment")
    return prior + max(0., time.time() - epoch)


def budget_boundary(base=None):
    base = Path(OUTPUT if base is None else base)
    if (base / "abort.json").exists():
        raise BudgetExceeded("Cooperative abort; preserve evidence and diagnose before recovery")
    if active_seconds(base) > CONFIG["maximum_runtime_hours"] * 3600:
        raise BudgetExceeded("Seven-day cumulative active execution ceiling reached; preserve evidence")
    if shutil.disk_usage(ROOT).free < CONFIG["disk_reserve_bytes"]:
        raise BudgetExceeded("Five-GiB disk reserve reached; preserve evidence")
