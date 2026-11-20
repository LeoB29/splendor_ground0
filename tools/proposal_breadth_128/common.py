"""Frozen configuration and strict artifacts for the authorized breadth screen."""
from pathlib import Path
import shutil
import time

from tools.proposal_iteration_128.common import (
    ROOT, old, baseline, windows, require, artifact, verified, verify_hash,
)

OUTPUT = ROOT / "outputs/proposal_breadth_128"
SOURCE = ROOT / "outputs/proposal_ranking_128"
SOURCE_BINDING = "e838a6a0415b9e5e4749fe79dc2094cfb38a45665f9957817d37f71225178c82"
FITS = (161001, 161002, 161003)
CONFIG = {
    "states": {"probe": 18, "main": 360},
    "source_games": {"probe": 54, "main": 1080},
    "source_seed": {"probe": 2106000000, "main": 2105000000, "test": 2108000000},
    "source_assignment_tag": "proposal-breadth-128-source-cell-v1",
    "source_stream_tag": "proposal-breadth-128-source-stream-v1",
    "selection_tag": "proposal-breadth-128-panel-v1",
    "random_tag": "proposal-breadth-128-random-additions-v1",
    "worlds": 128, "assessment_worlds": 128, "maximum_precision_worlds": 256,
    "maximum_extension_worlds": 128,
    "maximum_candidates": 8, "maximum_assessed_choices": 3,
    "world_seed_base": 2100000000,
    "world_tag": "proposal-breadth-128-fresh-world-v1",
    "bootstrap_seed": 2103000000, "bootstrap_samples": 50000,
    "bootstrap_chunk": 1000,
    "audit_seed": 2104000000,
    "audit_world_tag": "proposal-breadth-128-audit-world-v1",
    "audit_stress_tag": "proposal-breadth-128-audit-stress-v1",
    "replay_worlds_per_block": 8, "maximum_stress_roots": 48,
    "stress_supports": [1, 2, 3, 5, 6, 7],
    "probe_full_replay": True, "full_inner_replay": False,
    "minimum_progress": .005, "superiority_lower_bound": 0.,
    "workers": 8, "torch_threads": 1, "device": "cpu", "backend": "serial",
    "max_turns": 400, "repetition_limit": 4, "no_progress_limit": 60,
    "maximum_runtime_hours": 8., "forecast_allowance": 1.35,
    "finalization_allowance_seconds": 300., "disk_reserve_bytes": 5 * 1024**3,
    "maximum_main_selection_continuations": 368640,
    "maximum_main_extension_continuations": 184320,
    "maximum_main_assessment_continuations": 138240,
    "maximum_main_continuations": 691200,
    "maximum_probe_continuations": 34560,
    "training": False, "automatic_retry": False, "promotion": False,
    "automatic_successor": False, "scheduled_monitoring": False,
}


class BudgetExceeded(ValueError):
    """Resource or fixed-source-availability hold; never an automatic retry."""


def budget_boundary(base=OUTPUT):
    base = Path(base)
    launch = base / "launch.json"
    if not launch.exists():
        require(not base.resolve().is_relative_to(OUTPUT.resolve()), "Production has no launch receipt")
        return
    value = old.read_json(launch)
    if time.time() - value["epoch_seconds"] > CONFIG["maximum_runtime_hours"] * 3600:
        raise BudgetExceeded("Eight-hour execution ceiling reached; preserve partial evidence")
    if shutil.disk_usage(ROOT).free < CONFIG["disk_reserve_bytes"]:
        raise BudgetExceeded("Five-GiB disk reserve reached; preserve partial evidence")
