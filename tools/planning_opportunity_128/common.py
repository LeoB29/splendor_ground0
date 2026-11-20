"""Frozen scope for opportunity-weighted outsider teaching."""
from pathlib import Path
import shutil
import time
from tools.proposal_iteration_128.common import (
    ROOT, old, baseline, windows, require, artifact, verified, verify_hash,
)
from tools.independent_game_coverage_128_recovery_01 import common as coverage

OUTPUT = ROOT / "outputs/planning_opportunity_128"
SOURCE = ROOT / "outputs/teacher_signal_capture_128"
SOURCE_BINDING = "d0105c4178c4b74300aa12ea9ccbf986bb53a1c8cc6eb6bef990bffb54670da1"
PARENT = ROOT / "outputs/independent_game_coverage_128_recovery_01"
PARENT_BINDING = "1415ca3ff570181ab6d08ef25b4254134cac568902538e754978d6c302c88263"
CODE_PARENT = ROOT / "outputs/frozen_representation_128"
CODE_PARENT_BINDING = "5c58a59930136e1b5d94698a852b06d5cc049df4da75ea65e56f14fcafe4f8c7"
FITS = (161001, 161002, 161003)
MODEL_ARMS = ("anchor", "broad", "weighted")
CONFIG = {
    "train_rows": 9461, "validation_rows": 2489, "added_rows": 720,
    "train_source_pairs": 159, "validation_source_pairs": 42,
    "states": 336, "pairs": 84,
    "assessment_train_rows": 168, "assessment_validation_rows": 168,
    "updates": 1784, "batch_size": 64, "row_exposures_per_fit": 114044,
    "old_fraction": .5, "narrow_rows_per_fit": 180,
    "lr": 1e-4, "weight_decay": 1e-4, "gradient_norm": 1.,
    "target_temperature": .05,
    "objective": "recorded_outside_first_three_soft_score_cross_entropy",
    "train_seeds": dict(coverage.CONFIG["train_seeds"]),
    "batch_tag": coverage.CONFIG["batch_tag"],
    "partition_tag": coverage.CONFIG["partition_tag"],
    "trainable_parameters_rule": "all_policy_except_value_head",
    "frozen_representation": False, "value_head_frozen": True,
    "weighting": "positive_best_outsider_minus_best_actual_first_three",
    "weight_normalization": "exact_train_exposure_mean_separately_old_added",
    "weight_normalization_reads_validation": False,
    "soft_targets_unchanged": True, "zero_weight_rows_retained": True,
    "batch_denominator": "all_prescribed_rows", "zero_loss_steps_retained": True,
    "fresh_optimizer": True, "checkpoint_selection": "fixed_final_update_1784",
    "selection_uses_intermediate_checkpoints": False, "historical_replay": False,
    "workers": 3, "fit_workers": 3, "torch_threads": 1,
    "device": "cpu", "dtype": "float32",
    "bootstrap_seed": 2273000000, "bootstrap_samples": 50000,
    "bootstrap_chunk": 1000, "minimum_progress": .005,
    "minimum_added_marginal_regret_capture_each_fit": .50,
    "validation_marginal_regret_improves_both_controls_each_fit": True,
    "planning_nonnegative_both_controls_each_fit": True,
    "primary_endpoint": "weighted_supported_minus_anchor",
    "required_control_endpoint": "weighted_supported_minus_broad_supported",
    "maximum_runtime_hours": 2., "disk_reserve_bytes": 5 * 1024**3,
    "new_simulations": 0, "new_source_games": 0,
    "assessment_used_for_selection": False,
    "automatic_retry": False, "automatic_successor": False,
    "promotion": False, "scheduled_monitoring": False,
}


class BudgetExceeded(ValueError):
    """A bounded hold preserving all partial evidence."""


def budget_boundary(base=None):
    base = Path(OUTPUT if base is None else base)
    if (base / "abort.json").exists():
        raise BudgetExceeded("Cooperative abort; preserve all evidence")
    launch = base / "launch.json"
    if not launch.exists():
        require(not base.resolve().is_relative_to(OUTPUT.resolve()),
                "Production launch receipt missing")
        return
    if time.time() - old.read_json(launch)["epoch_seconds"] > CONFIG["maximum_runtime_hours"] * 3600:
        raise BudgetExceeded("Two-hour execution ceiling reached; preserve evidence")
    if shutil.disk_usage(ROOT).free < CONFIG["disk_reserve_bytes"]:
        raise BudgetExceeded("Five-GiB disk reserve reached; preserve evidence")
