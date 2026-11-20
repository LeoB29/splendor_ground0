"""Fixed scientific scope and operational limits for the saved learning curve."""
from pathlib import Path
import shutil
import time
from tools.proposal_iteration_128.common import (
    ROOT, old, baseline, windows, require, artifact, verified, verify_hash,
)

OUTPUT = ROOT / "outputs/saved_teacher_curve_128"
PARENT = ROOT / "outputs/focused_teacher_fit_128"
PARENT_BINDING = "6ac655633e91e87ca03b5905960e72442388069e78ab904b3386508c55608de3"
CHECKPOINTS = (0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 1784)
FITS = (161001, 161002, 161003)
CONFIG = {
    "checkpoints": list(CHECKPOINTS), "fits": list(FITS),
    "states": 336, "pairs": 84, "train_rows": 168, "validation_rows": 168,
    "primary_contrasts": ["focused_outsider_minus_anchor_outsider", "focused_supported_minus_anchor"],
    "simultaneous_family": "validation_12_noninitial_checkpoints_by_2_primary_contrasts",
    "simultaneous_method": "bootstrap_max_absolute_centered_deviation_fixed_se_standardized",
    "simultaneous_confidence": .95,
    "bootstrap_samples": 50000, "bootstrap_seed": 2243000000, "bootstrap_chunk": 500,
    "maximum_runtime_minutes": 30., "disk_reserve_bytes": 5*1024**3,
    "device": "cpu", "torch_threads": 1, "workers": 1,
    "new_training": False, "new_simulations": 0, "new_source_games": 0,
    "checkpoint_selection": False, "best_fit_selection": False,
    "assessment_reused_development": True, "assessment_used_for_selection": False,
    "automatic_retry": False, "automatic_successor": False,
    "promotion": False, "scheduled_monitoring": False,
}


class BudgetExceeded(ValueError):
    pass


def budget_boundary(base=OUTPUT):
    base = Path(base)
    if (base / "abort.json").exists():
        raise BudgetExceeded("Cooperative abort; preserve all evidence")
    receipt = base / "launch.json"
    if not receipt.exists():
        require(not base.resolve().is_relative_to(OUTPUT.resolve()), "Production launch receipt missing")
        return
    if time.time() - old.read_json(receipt)["epoch_seconds"] > CONFIG["maximum_runtime_minutes"] * 60:
        raise BudgetExceeded("Thirty-minute execution ceiling reached")
    if shutil.disk_usage(ROOT).free < CONFIG["disk_reserve_bytes"]:
        raise BudgetExceeded("Five-GiB disk reserve reached")
