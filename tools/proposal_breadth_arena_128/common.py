"""Prospective configuration and strict artifact/resource contracts."""
from pathlib import Path
import shutil
import time

from tools.proposal_iteration_128.common import (
    ROOT, old, baseline, windows, require, artifact, verified, verify_hash,
)

OUTPUT = ROOT / "outputs/proposal_breadth_arena_128"
SOURCE = ROOT / "outputs/proposal_breadth_128"
SOURCE_BINDING = "231bdd640ac88af2470c24af38a04e28248a7edeba8e376173df5b356fd27a74"
FITS = (161001, 161002, 161003)
CONFIG = {
    "games": {"probe": 12, "main": 402, "test": 6},
    "environment_seed": {"probe": 2119000000, "main": 2110000000, "test": 2119800000},
    "assignment_tag": "proposal-breadth-arena-128-fit-v1",
    "stream_tag": "proposal-breadth-arena-128-stream-v1",
    "random_tag": "proposal-breadth-arena-128-random-v1",
    "bootstrap_seed": 2113000000, "bootstrap_samples": 50000, "bootstrap_chunk": 1000,
    "audit_seed": 2114000000,
    "audit_world_tag": "proposal-breadth-arena-128-audit-world-v1",
    "audit_stress_tag": "proposal-breadth-arena-128-audit-stress-v1",
    "worlds": 128, "max_candidates": 8, "anchor_candidates": 4,
    "replay_worlds_per_root": 8, "maximum_stress_roots": 171,
    "full_inner_replay": False, "probe_full_replay": False,
    "workers": 8, "torch_threads": 1, "device": "cpu", "backend": "serial", "batch_size": 1,
    "max_turns": 400, "repetition_limit": 4, "no_progress_limit": 60,
    "minimum_score": .55, "superiority_lower_bound": .5,
    "maximum_forecast_hours": 120., "maximum_runtime_hours": 144.,
    "forecast_allowance": 1.25, "finalization_allowance_seconds": 600.,
    "disk_reserve_bytes": 5*1024**3,
    "training": False, "automatic_retry": False, "promotion": False,
    "automatic_successor": False, "scheduled_monitoring": False,
}


class BudgetExceeded(ValueError):
    """A resource hold preserves all partial evidence and never retries."""


def budget_boundary(base=OUTPUT):
    base = Path(base)
    launch = base / "launch.json"
    if not launch.exists():
        require(not base.resolve().is_relative_to(OUTPUT.resolve()), "Production has no launch receipt")
        return
    value = old.read_json(launch)
    if time.time()-value["epoch_seconds"] > CONFIG["maximum_runtime_hours"]*3600:
        raise BudgetExceeded("Six-day execution ceiling reached; preserve partial evidence")
    if shutil.disk_usage(ROOT).free < CONFIG["disk_reserve_bytes"]:
        raise BudgetExceeded("Five-GiB disk reserve reached; preserve partial evidence")
