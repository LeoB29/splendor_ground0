"""Prospective constants and artifact plumbing; no frozen source is modified."""
from pathlib import Path
import time
import shutil

from tools.proposal_iteration_128.common import (
    ROOT, old, baseline, windows, require, artifact, verified, verify_hash,
)

OUTPUT = ROOT / "outputs/proposal_diagnostic_128"
SOURCE = ROOT / "outputs/proposal_iteration_128"
FITS = (161001, 161002, 161003)
SOURCE_BINDING = "1a3c53142516a5d8f08376d0a7227ae9199952a9e077e1309d13fdcdf6d40ace"
CONFIG = {
    "states": {"probe": 12, "main": 201},
    "worlds": 128, "assessment_worlds": 128, "maximum_candidates": 7,
    "workers": 8, "torch_threads": 1, "device": "cpu", "backend": "serial",
    "random_tag": "proposal-diagnostic-128-random-fourth-v1",
    "world_tag": "proposal-diagnostic-128-fresh-world-v1",
    "world_seed_base": 2080000000,
    "bootstrap_seed": 2083000000, "bootstrap_samples": 50000,
    "audit_seed": 2084000000,
    "audit_world_tag": "proposal-diagnostic-128-audit-world-v1",
    "audit_stress_tag": "proposal-diagnostic-128-audit-stress-v1",
    "replay_worlds_per_block": 8, "maximum_stress_roots": 39,
    "probe_full_replay": True, "full_inner_replay": False,
    "maximum_runtime_hours": 8., "forecast_allowance": 1.35,
    "finalization_allowance_seconds": 300.,
    "disk_reserve_bytes": 5 * 1024**3,
    "maximum_main_continuations": 360192,
    "maximum_probe_continuations": 21504,
    "automatic_retry": False, "training": False, "promotion": False,
    "automatic_successor": False, "scheduled_monitoring": False,
}


class BudgetExceeded(ValueError):
    """An operational limit, not a scientific outcome or permission to retry."""


def budget_boundary(base=OUTPUT):
    """Production workers stop at world boundaries; fixtures have no launch file."""
    base = Path(base)
    launch = base / "launch.json"
    if not launch.exists():
        require(base.resolve() != OUTPUT.resolve(), "Production has no launch receipt")
        return
    value = old.read_json(launch)
    if time.time() - value["epoch_seconds"] > CONFIG["maximum_runtime_hours"] * 3600:
        raise BudgetExceeded("Eight-hour execution ceiling reached; preserve partial evidence")
    if shutil.disk_usage(ROOT).free < CONFIG["disk_reserve_bytes"]:
        raise BudgetExceeded("Five-GiB disk reserve reached; preserve partial evidence")
