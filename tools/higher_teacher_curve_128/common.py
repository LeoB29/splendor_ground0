"""Fixed scientific scope and operational limits for the saved learning curve."""
from pathlib import Path
import shutil
import time
from tools.proposal_iteration_128.common import (
    ROOT, old, baseline, windows, require, artifact, verified, verify_hash,
)

OUTPUT = ROOT / "outputs/higher_teacher_curve_128"
PARENT = ROOT / "outputs/higher_teacher_rate_128"
PARENT_BINDING = "0968f7680f8996373239cd7ad187c654b5267283a075dc79fdeefb97580bdca1"
CHECKPOINTS = (0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 1784)
FITS = (161001, 161002, 161003)
CONFIG = {
    "checkpoints": list(CHECKPOINTS), "fits": list(FITS),
    "states": 336, "pairs": 84, "train_rows": 168, "validation_rows": 168,
    "training_corpus_rows": 9461, "full_validation_rows": 2489,
    "primary_contrasts": ["focused_outsider_minus_anchor_outsider", "focused_supported_minus_anchor",
                          "focused_outsider_minus_lowrate_outsider", "focused_supported_minus_lowrate_supported",
                          "focused_outsider_minus_finalrate_outsider", "focused_supported_minus_finalrate_supported"],
    "simultaneous_family": "validation_12_noninitial_checkpoints_by_6_contrasts_against_anchor_lowrate_finalrate",
    "simultaneous_method": "bootstrap_max_absolute_centered_deviation_fixed_se_standardized",
    "simultaneous_confidence": .95,
    "bootstrap_samples": 50000, "bootstrap_seed": 2247000000, "bootstrap_chunk": 500,
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
