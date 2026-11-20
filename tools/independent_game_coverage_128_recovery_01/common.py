"""Prospective scope and resource boundaries for the coverage experiment."""
from pathlib import Path
import shutil
import time

from tools.proposal_iteration_128.common import (
    ROOT, old, baseline, windows, require, artifact, verified, verify_hash,
)

OUTPUT = ROOT / "outputs/independent_game_coverage_128_recovery_01"
FITS = (161001, 161002, 161003)
ARMS = ("anchor", "narrow", "broad", "lowrate", "soft", "teacher")
MODEL_ARMS = ARMS[:-1]
CONTRASTS = (
    "broad_minus_anchor", "broad_minus_narrow", "broad_minus_lowrate",
    "broad_minus_soft", "narrow_minus_anchor", "narrow_minus_soft",
    "teacher_minus_anchor",
)
CONFIG = {
    "train_rows": 9461, "validation_rows": 2489,
    "train_source_pairs": 159, "validation_source_pairs": 42,
    "added_rows": 720, "narrow_rows_per_fit": 180,
    "partition_tag": "independent-game-coverage-128-four-blocks-v1",
    "old_fraction": .5, "updates": 1784, "batch_size": 64,
    "row_exposures_per_fit": 114044,
    "lr": 1e-4, "weight_decay": 1e-4, "gradient_norm": 1.,
    "value_head_frozen": True, "fresh_optimizer": True,
    "target_temperature": .05,
    "objective": "recorded_outside_first_three_soft_score_cross_entropy",
    "train_seeds": {str(f): 223700001+i for i, f in enumerate(FITS)},
    "batch_tag": "independent-game-coverage-128-matched-mixture-v1",
    "checkpoint_selection": "fixed_final_update_1784",
    "selection_uses_intermediate_checkpoints": False,
    "historical_replay": False,
    "states": {"probe": 18, "main": 360},
    "source_games": {"probe": 54, "main": 1080},
    "source_seed": {"probe": 2256000000, "main": 2255000000, "test": 2258000000},
    "source_assignment_tag": "independent-game-coverage-128-source-cell-v1",
    "source_stream_tag": "independent-game-coverage-128-source-stream-v1",
    "selection_tag": "independent-game-coverage-128-panel-v1",
    "worlds": 128, "assessment_worlds": 128,
    "maximum_candidates": 12, "maximum_assessed_choices": 6,
    "world_seed_base": 2250000000,
    "world_tag": "independent-game-coverage-128-fresh-world-v1",
    "teacher_random_tag": "independent-game-coverage-128-teacher-random-v1",
    "bootstrap_seed": 2253000000, "bootstrap_samples": 50000,
    "bootstrap_chunk": 1000,
    "audit_seed": 2254000000,
    "audit_world_tag": "independent-game-coverage-128-audit-world-v1",
    "audit_stress_tag": "independent-game-coverage-128-audit-stress-v1",
    "replay_worlds_per_block": 8, "maximum_stress_roots": 39,
    "probe_full_replay": True, "full_inner_replay": False,
    "minimum_progress": .005, "superiority_lower_bound": 0.,
    "workers": 8, "fit_workers": 3, "torch_threads": 1,
    "device": "cpu", "dtype": "float32", "backend": "serial",
    "max_turns": 400, "repetition_limit": 4, "no_progress_limit": 60,
    "maximum_runtime_hours": 8., "forecast_allowance": 1.35,
    "finalization_allowance_seconds": 300.,
    "disk_reserve_bytes": 5 * 1024**3,
    "maximum_main_continuations": 829440,
    "maximum_probe_continuations": 41472,
    "training": True, "automatic_retry": False, "promotion": False,
    "automatic_successor": False, "scheduled_monitoring": False,
}


class BudgetExceeded(ValueError):
    """A bounded operational hold, preserving all partial evidence."""


def budget_boundary(base=OUTPUT):
    base = Path(base)
    if (base / "abort.json").exists():
        raise BudgetExceeded("Cooperative abort; preserve evidence")
    launch = base / "launch.json"
    if not launch.exists():
        require(not base.resolve().is_relative_to(OUTPUT.resolve()),
                "Production launch receipt missing")
        return
    if time.time() - old.read_json(launch)["epoch_seconds"] > CONFIG["maximum_runtime_hours"] * 3600:
        raise BudgetExceeded("Eight-hour execution ceiling reached; preserve evidence")
    if shutil.disk_usage(ROOT).free < CONFIG["disk_reserve_bytes"]:
        raise BudgetExceeded("Five-GiB disk reserve reached; preserve evidence")
