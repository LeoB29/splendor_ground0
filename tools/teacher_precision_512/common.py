"""Matched teacher-label precision experiment and active resource boundary."""
from pathlib import Path
import shutil
import time

from tools.proposal_iteration_128.common import (
    ROOT, old, baseline, windows, require, artifact, verified, verify_hash,
)
from tools.independent_game_coverage_128_recovery_01 import common as coverage

OUTPUT = ROOT / "outputs/teacher_precision_512"
PARENT = ROOT / "outputs/independent_game_coverage_128_recovery_01"
PARENT_BINDING = "1415ca3ff570181ab6d08ef25b4254134cac568902538e754978d6c302c88263"
SOURCE = ROOT / "outputs/teacher_signal_capture_128"
SOURCE_BINDING = "d0105c4178c4b74300aa12ea9ccbf986bb53a1c8cc6eb6bef990bffb54670da1"
FITS = (161001, 161002, 161003)
ARMS = ("anchor", "precision", "control", "teacher")
MODEL_ARMS = ARMS[:-1]
CONTRASTS = ("precision_minus_anchor", "precision_minus_control", "control_minus_anchor",
             "teacher_minus_anchor", "teacher_minus_precision")
CONFIG = {
    **coverage.CONFIG,
    "train_seeds": dict(coverage.CONFIG["train_seeds"]),
    "base_network_frozen": False,
    "frozen_representation": False, "value_head_frozen": True,
    "old_validation_diagnostic_only": True,
    "fresh_evaluation_independent_of_development_result": True,
    "source_seed": {"probe": 2306000000, "main": 2305000000, "test": 2308000000},
    "source_assignment_tag": "teacher-precision-512-source-cell-v1",
    "source_stream_tag": "teacher-precision-512-source-stream-v1",
    "selection_tag": "teacher-precision-512-panel-v1",
    "maximum_candidates": 10, "maximum_assessed_choices": 4,
    "world_seed_base": 2300000000,
    "world_tag": "teacher-precision-512-fresh-world-v1",
    "teacher_random_tag": "teacher-precision-512-teacher-random-v1",
    "bootstrap_seed": 2303000000, "audit_seed": 2304000000,
    "audit_world_tag": "teacher-precision-512-audit-world-v1",
    "audit_stress_tag": "teacher-precision-512-audit-stress-v1",
    "primary_endpoint": "precision_minus_anchor",
    "required_control_endpoint": "precision_minus_control",
    "maximum_runtime_hours": 48.,
    "maximum_main_continuations": 645120,
    "maximum_probe_continuations": 32256,
    "resume_checkpoint_interval": 64,
    "teacher_label_worlds": 512, "inherited_label_worlds": 128,
    "label_world_seed_base": 2310000000,
    "label_world_tag": "teacher-precision-512-label-world-v1",
    "label_probe_rows": 18,
    "label_probe_tag": "teacher-precision-512-label-probe-v1",
    "label_audit_tag": "teacher-precision-512-label-audit-v1",
    "label_stress_tag": "teacher-precision-512-label-stress-v1",
    "label_replay_worlds_per_block": 8,
    "label_unresolved_outcomes": "hold_without_imputation",
    "pre_label_future_stages_reserve_seconds": 8 * 3600.,
    "automatic_retry": False,
}


class BudgetExceeded(ValueError):
    """A cooperative operational hold which preserves completed evidence."""


def active_seconds(base=None):
    base = Path(OUTPUT if base is None else base)
    current = base / "active_run.json"
    if current.exists():
        receipt = old.read_json(current)
        return float(receipt["prior_active_seconds"]) + max(0., time.time() - receipt["epoch_seconds"])
    launch = base / "launch.json"
    if launch.exists():
        return max(0., time.time() - old.read_json(launch)["epoch_seconds"])
    return 0.


def budget_boundary(base=None):
    base = Path(OUTPUT if base is None else base)
    if (base / "abort.json").exists():
        raise BudgetExceeded("Cooperative abort; preserve evidence and diagnose before recovery")
    require((base / "launch.json").exists() or not base.resolve().is_relative_to(OUTPUT.resolve()),
            "Production launch receipt missing")
    if active_seconds(base) > CONFIG["maximum_runtime_hours"] * 3600:
        raise BudgetExceeded("Forty-eight-hour active execution ceiling reached; preserve evidence")
    if shutil.disk_usage(ROOT).free < CONFIG["disk_reserve_bytes"]:
        raise BudgetExceeded("Five-GiB disk reserve reached; preserve evidence")
