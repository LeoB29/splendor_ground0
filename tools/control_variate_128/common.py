"""Prospective scope, resource limits and immutable artifact helpers."""
from pathlib import Path
import shutil
import time

from tools.proposal_iteration_128.common import (
    ROOT, old, baseline, windows, require, artifact, verified, verify_hash,
)

OUTPUT = ROOT / "outputs/control_variate_128"
SOURCE = ROOT / "outputs/teacher_signal_capture_128"
SOURCE_BINDING = "d0105c4178c4b74300aa12ea9ccbf986bb53a1c8cc6eb6bef990bffb54670da1"
CODE_PARENT = ROOT / "outputs/planning_opportunity_128"
CODE_PARENT_BINDING = "14c00da467d726f9d95f1178f8250594e35deff81b9791067d54f1b862c146f4"
FITS = (161001, 161002, 161003)
HORIZONS = (4, 8, 16)
CONFIG = {
    "states": {"train": 168, "validation": 168},
    "pairs": {"train": 42, "validation": 42},
    "worlds": 128, "maximum_candidates": 8,
    "horizons": list(HORIZONS), "forced_root_counts_as_ply": True,
    "evaluator": "frozen_143_public_15_feature_ridge",
    "terminal_prefix": "exact_natural_terminal_score",
    "continuation_policy": "unchanged_greedy_035",
    "terminal_target": "original_root_player_score_wdl_plus_one_over_two",
    "reference_action": "old_disjoint_selection_teacher",
    "contrasts": "every_other_recorded_candidate_minus_fixed_teacher",
    "root_weight": "equal_mean_of_nonreference_contrasts_zero_if_none",
    "timing_roots": 18, "timing_worlds": 8,
    "timing_tag": "control-variate-128-timing-v1",
    "timing_seed": 2281000000,
    "minimum_full_samples": 16, "maximum_full_samples": 128,
    "maximum_prefix_samples": 512,
    "prefix_samples_strictly_exceed_full_samples": True,
    "baseline_full_samples": 128,
    "cost_calibration": "train_only_per_phase_cpu_seconds_per_candidate_world",
    "integrated_cost_floor": "max_measured_integrated_and_full_only",
    "allocation_budget": "at_most_128_full_cost_in_each_phase",
    "selection_objective": "product_of_equal_root_mean_variance_and_equal_root_mean_vector_cost",
    "coefficient": "pooled_train_within_root_covariance_over_variance",
    "selection_tie_order": ["efficiency_ratio", "horizon", "n", "M"],
    "selection_ratio_round_digits": 10,
    "interval_scope": "validation_whole_pair_bootstrap_conditional_on_train_timing_and_selected_rule",
    "minimum_variance_cost_reduction": .20,
    "selection_split": "train", "evaluation_split": "validation",
    "selection_before_validation_prefixes": True,
    "bootstrap_samples": 50000, "bootstrap_seed": 2283000000,
    "bootstrap_chunk": 1000,
    "audit_seed": 2284000000,
    "audit_world_tag": "control-variate-128-audit-world-v1",
    "audit_stress_tag": "control-variate-128-audit-stress-v1",
    "replay_worlds_per_root": 8,
    "maximum_stress_roots_per_split": 39,
    "workers": 8, "torch_threads": 1, "device": "cpu", "backend": "serial",
    "maximum_runtime_hours": 2., "forecast_allowance": 1.35,
    "finalization_allowance_seconds": 300.,
    "disk_reserve_bytes": 5 * 1024**3,
    "training": False, "new_source_games": 0,
    "actual_control_variate_player": False,
    "automatic_retry": False, "automatic_successor": False,
    "promotion": False, "scheduled_monitoring": False,
}


class BudgetExceeded(ValueError):
    """Preserve evidence at a declared operational boundary."""


def budget_boundary(base=None):
    base = Path(OUTPUT if base is None else base)
    if (base / "abort.json").exists():
        raise BudgetExceeded("Cooperative abort requested; preserve all evidence")
    launch = base / "launch.json"
    if not launch.exists():
        require(not base.resolve().is_relative_to(OUTPUT.resolve()),
                "Production launch receipt missing")
        return
    if time.time() - old.read_json(launch)["epoch_seconds"] > CONFIG["maximum_runtime_hours"] * 3600:
        raise BudgetExceeded("Two-hour execution ceiling reached; preserve evidence")
    if shutil.disk_usage(ROOT).free < CONFIG["disk_reserve_bytes"]:
        raise BudgetExceeded("Five-GiB disk reserve reached; preserve evidence")
