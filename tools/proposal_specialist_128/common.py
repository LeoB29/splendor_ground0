"""Immutable configuration and artifacts for the authorized role-specific specialist pilot."""
from pathlib import Path
import shutil
import time

from tools.proposal_iteration_128.common import (
    ROOT, old, baseline, windows, require, artifact, verified, verify_hash,
)

OUTPUT = ROOT / "outputs/proposal_specialist_128"
SOURCE = ROOT / "outputs/proposal_iteration_128"
SOURCE_BINDING = "1a3c53142516a5d8f08376d0a7227ae9199952a9e077e1309d13fdcdf6d40ace"
BREADTH_SOURCE = ROOT / "outputs/proposal_breadth_arena_128"
BREADTH_BINDING = "a8395f535998379a2062ebce8fed79cc6a26e5cd1deb699c96bd4b9dca06deb5"
PARENT = ROOT / "outputs/proposal_breadth_learning_128"
PARENT_BINDING = "bf916c18fc25e5be4bb2b7ff13317157d155da82668ae4d8a86308544b5ca72a"
FITS = (161001, 161002, 161003)
CONFIG = {
    "states": {"probe": 18, "main": 360},
    "source_games": {"probe": 54, "main": 1080},
    "source_seed": {"probe": 2136000000, "main": 2135000000, "test": 2138000000},
    "source_assignment_tag": "proposal-specialist-128-source-cell-v1",
    "source_stream_tag": "proposal-specialist-128-source-stream-v1",
    "selection_tag": "proposal-specialist-128-panel-v1",
    "worlds": 128, "assessment_worlds": 128,
    "maximum_candidates": 6, "maximum_assessed_choices": 3,
    "world_seed_base": 2130000000,
    "world_tag": "proposal-specialist-128-fresh-world-v1",
    "bootstrap_seed": 2133000000, "bootstrap_samples": 50000,
    "bootstrap_chunk": 1000,
    "audit_seed": 2134000000,
    "audit_world_tag": "proposal-specialist-128-audit-world-v1",
    "audit_stress_tag": "proposal-specialist-128-audit-stress-v1",
    "replay_worlds_per_block": 8, "maximum_stress_roots": 39,
    "probe_full_replay": True, "full_inner_replay": False,
    "minimum_progress": .005, "superiority_lower_bound": 0.,
    "workers": 8, "fit_workers": 3, "torch_threads": 1,
    "device": "cpu", "backend": "serial",
    "max_turns": 400, "repetition_limit": 4, "no_progress_limit": 60,
    "maximum_runtime_hours": 8., "forecast_allowance": 1.35,
    "finalization_allowance_seconds": 300., "disk_reserve_bytes": 5 * 1024**3,
    "maximum_main_continuations": 414720,
    "maximum_probe_continuations": 20736,
    "training": True, "automatic_retry": False, "promotion": False,
    "exact_identity_shortcut": True,
    "automatic_successor": False, "scheduled_monitoring": False,
}


class BudgetExceeded(ValueError):
    """Resource ceiling or source availability hold; never an automatic retry."""


def budget_boundary(base=OUTPUT):
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
