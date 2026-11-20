"""Immutable configuration and artifact plumbing for one controlled generation."""
from pathlib import Path

from scripts.analysis import learned_proposal_screen as screen

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/proposal_iteration_128"
PRODUCTION_OUTPUT = OUTPUT.resolve()
old, baseline, windows = screen.old, screen.baseline, screen.windows
require = screen.require
FITS = (161001, 161002, 161003)
CONFIG = {
    "games": {"probe_a": 12, "stage_a": 402, "probe_b": 12, "stage_b": 402, "test": 6},
    "environment_seed": {"probe_a": 2079000000, "stage_a": 2070000000,
                         "probe_b": 2079100000, "stage_b": 2071000000, "test": 2079800000},
    "assignment_tag": "proposal-iteration-128-fit-v1",
    "stream_tag": "proposal-iteration-128-stream-v1",
    "bootstrap_seed": {"stage_a": 2073000000, "stage_b": 2073100000, "test": 2073900000},
    "bootstrap_samples": 50000, "bootstrap_chunk": 1000,
    "audit_seed": 2074000000,
    "audit_world_tag": "proposal-iteration-128-audit-world-v1",
    "audit_stress_tag": "proposal-iteration-128-audit-stress-v1",
    "worlds": 128, "max_candidates": 4, "workers": 8, "fit_workers": 3,
    "torch_threads": 1, "device": "cpu", "backend": "serial", "batch_size": 1,
    "max_turns": 400, "repetition_limit": 4, "no_progress_limit": 60,
    "full_inner_replay": False, "replay_worlds_per_root": 8, "maximum_stress_roots": 162,
    "stage_a_minimum_score": .55, "superiority_lower_bound": .5,
    "maximum_forecast_hours_per_stage": 120., "maximum_runtime_hours": 216.,
    "disk_reserve_bytes": 5 * 1024**3,
    "training": True, "automatic_retry": False, "promotion": False,
    "conditional_stage_b": True, "automatic_successor": False, "scheduled_monitoring": False,
}


def artifact(path, **metadata):
    path = Path(path).resolve()
    try:
        name = path.relative_to(ROOT).as_posix()
    except ValueError:
        name = str(path)
    return {"path": name, "sha256": old.file_hash(path), **metadata}


def verify_hash(item):
    path = (ROOT / item["path"]).resolve()
    require(old.file_hash(path) == item["sha256"], f"Changed artifact: {path}")
    return path


def verified(item):
    return old.read_json(verify_hash(item))
