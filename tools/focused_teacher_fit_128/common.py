"""Fixed scope and operational boundaries for the focused learning diagnosis."""
from pathlib import Path
import shutil
import time
from tools.proposal_iteration_128.common import ROOT, old, baseline, windows, require, artifact, verified, verify_hash

OUTPUT = ROOT / "outputs/focused_teacher_fit_128"
SOURCE = ROOT / "outputs/teacher_signal_capture_128"
SOURCE_BINDING = "d0105c4178c4b74300aa12ea9ccbf986bb53a1c8cc6eb6bef990bffb54670da1"
FITS = (161001, 161002, 161003)
CONFIG = {
    "train_rows": 168, "validation_rows": 168, "states": 336, "pairs": 84,
    "updates": 1784, "batch_size": 64, "lr": 1e-5, "weight_decay": 1e-4,
    "gradient_norm": 1., "value_head_frozen": True, "fresh_optimizer": True,
    "checkpoints": [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 1784],
    "checkpoint_selection": "fixed_final_update_1784", "historical_replay": False,
    "objective": "recorded_outside_first_three_winner_cross_entropy",
    "train_seeds": {str(f): 223700001+i for i, f in enumerate(FITS)},
    "batch_tag": "focused-teacher-fit-128-training-v1",
    "device": "cpu", "dtype": "float32", "torch_threads": 1, "workers": 3,
    "maximum_runtime_hours": 2., "disk_reserve_bytes": 5*1024**3,
    "bootstrap_seed": 2233000000, "bootstrap_samples": 50000, "bootstrap_chunk": 1000,
    "regret_capture_guide": .90, "minimum_progress": .005,
    "new_simulations": 0, "new_source_games": 0, "assessment_used_for_selection": False,
    "automatic_retry": False, "automatic_successor": False, "promotion": False,
    "scheduled_monitoring": False,
}


class BudgetExceeded(ValueError):
    pass


def budget_boundary(base=OUTPUT):
    base = Path(base)
    if (base/"abort.json").exists():
        raise BudgetExceeded("Cooperative abort; preserve all evidence")
    launch = base/"launch.json"
    if not launch.exists():
        require(not base.resolve().is_relative_to(OUTPUT.resolve()), "Production launch receipt missing")
        return
    if time.time()-old.read_json(launch)["epoch_seconds"] > CONFIG["maximum_runtime_hours"]*3600:
        raise BudgetExceeded("Two-hour execution ceiling reached")
    if shutil.disk_usage(ROOT).free < CONFIG["disk_reserve_bytes"]:
        raise BudgetExceeded("Five-GiB disk reserve reached")
