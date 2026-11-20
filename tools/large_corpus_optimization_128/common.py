"""Frozen design and a single active-time resource boundary for every stage."""
from pathlib import Path
import math
import shutil
import time

from tools.proposal_iteration_128.common import (
    ROOT, old, baseline, windows, require, artifact, verified, verify_hash,
)
from tools.independent_game_scaling_128 import common as scaling

OUTPUT = ROOT / "outputs/large_corpus_optimization_128"
PARENT = ROOT / "outputs/independent_game_scaling_128"
PARENT_BINDING = "85bf68ca676a60610ab7c140afa6793573f3727df6a0d2ada273c35126891ead"
SOURCE = ROOT / "outputs/teacher_signal_capture_128"
SOURCE_BINDING = "d0105c4178c4b74300aa12ea9ccbf986bb53a1c8cc6eb6bef990bffb54670da1"
FITS = (161001, 161002, 161003)
LEARNING_ARMS = ("continued",)
ARMS = ("anchor", "control", "continued", "teacher")
MODEL_ARMS = ARMS[:-1]
CONTRASTS = ("continued_minus_anchor", "continued_minus_control", "control_minus_anchor",
             "teacher_minus_anchor", "teacher_minus_continued")
CONFIG = {
    **scaling.CONFIG,
    "train_seeds": dict(scaling.CONFIG["train_seeds"]),
    "updates": 28544, "start_update": 7136, "extra_updates": 21408,
    "fresh_optimizer": False, "restore_saved_AdamW": True,
    "fit_workers": 1,
    "workers": 4,
    "row_exposures_per_fit": 1824704,
    "old_exposures_per_fit": 912352, "added_exposures_per_fit": 912352,
    "checkpoint_selection": "fixed_final_update_28544",
    "continuation_schedule": "exact_authenticated_7136_plan_repeated_four_times",
    "acquisition_thresholds": {"added": .95, "old": .90},
    "diagnostic_updates": [7136, 14272, 21408, 28544],
    "states": {"probe": 18, "main": 720},
    "source_games": {"probe": 54, "main": 2160},
    "maximum_candidates": 10, "maximum_assessed_choices": 4,
    "source_seed": {"probe": 2606000000, "main": 2607000000, "test": 2608000000},
    "source_assignment_tag": "large-corpus-optimization-128-source-cell-v1",
    "source_stream_tag": "large-corpus-optimization-128-source-stream-v1",
    "selection_tag": "large-corpus-optimization-128-panel-v1",
    "world_seed_base": 2600000000,
    "world_tag": "large-corpus-optimization-128-evaluation-world-v1",
    "teacher_random_tag": "large-corpus-optimization-128-teacher-random-v1",
    "bootstrap_seed": 2603000000, "audit_seed": 2604000000,
    "audit_world_tag": "large-corpus-optimization-128-audit-world-v1",
    "audit_stress_tag": "large-corpus-optimization-128-audit-stress-v1",
    "primary_endpoint": "continued_minus_anchor",
    "required_control_endpoint": "continued_minus_control",
    "maximum_runtime_hours": 48.,
    "maximum_main_continuations": 720 * 128 * 14,
    "maximum_probe_continuations": 18 * 128 * 14,
    "new_label_rows": 0,
    "maximum_new_label_continuations": 0,
    "automatic_successor": False,
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
        raise BudgetExceeded("48-hour cumulative active execution ceiling reached; preserve evidence")
    if shutil.disk_usage(ROOT).free < CONFIG["disk_reserve_bytes"]:
        raise BudgetExceeded("Five-GiB disk reserve reached; preserve evidence")
