"""Explicit, process-local continuation backend for the historical LCO collector.

Only calls through ``engine.run_block`` use the optional encoder. Root scoring,
selection/proofs and the independent auditor retain their original policy.
Use one collection thread per process; this is not a thread-local policy switch.
The reference backend is the default, and nothing is installed on import.
"""
from __future__ import annotations

from contextlib import contextmanager
from functools import wraps
import threading


_ACTIVE = None


def _noop():
    pass


@contextmanager
def use_backend(engine, name="reference", *, boundary=None, on_position=None):
    """Temporarily opt a real LCO collector into ``reference`` or ``fast035``.

    ``boundary`` runs at block entry and successful exit for either backend,
    and before each fast decision through the already validated Fast035 class.
    The caller may additionally guard the original per-world budget boundary.
    ``on_position`` is an untimed fast-input audit hook, never a root-score hook.
    The yielded dictionary reports attempted/completed block wiring and work.
    Any exception propagates; no silent reference fallback is permitted.
    """
    global _ACTIVE
    if name not in ("reference", "fast035"):
        raise ValueError("Unknown continuation backend")
    if _ACTIVE is not None:
        raise RuntimeError("A continuation backend context is already active")
    from tools.large_corpus_optimization_128 import engine as expected_engine
    if engine is not expected_engine:
        raise ValueError("This adapter supports only the historical LCO collector")
    if boundary is not None and not callable(boundary):
        raise TypeError("Boundary must be callable")
    if on_position is not None and not callable(on_position):
        raise TypeError("Input callback must be callable")
    if name == "reference" and on_position is not None:
        raise ValueError("Input callback requires the fast035 backend")
    guard = boundary if boundary is not None else _noop
    original_block, original_raw = engine.run_block, engine.old._RAW
    if original_raw is None:
        raise ValueError("Initialize the unchanged reference policy first")
    if name == "fast035":
        from tools.simulation_speed_probe_128_v2.encoder import Fast035
        policy = Fast035(original_raw, boundary=guard, on_position=on_position)
    else:
        policy = original_raw
    owner_thread = threading.get_ident()
    usage = {"backend": name, "blocks_attempted": 0, "blocks_completed": 0,
             "candidate_worlds_attempted": 0, "candidate_worlds_completed": 0,
             "blocks": []}

    @wraps(original_block)
    def run_block(state, actions, seeds, block, base):
        if threading.get_ident() != owner_thread:
            raise RuntimeError("Continuation backend used outside its collection thread")
        if engine.old._RAW is not original_raw:
            raise RuntimeError("Reference policy changed while backend context was active")
        guard()
        count = len(actions) * len(seeds)
        row = {"candidate_indices": list(block["candidate_indices"]),
               "worlds": len(seeds), "candidate_worlds": count, "completed": False}
        usage["blocks_attempted"] += 1
        usage["candidate_worlds_attempted"] += count
        usage["blocks"].append(row)
        engine.old._RAW = policy
        try:
            result = original_block(state, actions, seeds, block, base)
        finally:
            engine.old._RAW = original_raw
        row["completed"] = True
        usage["blocks_completed"] += 1
        usage["candidate_worlds_completed"] += count
        guard()
        return result

    _ACTIVE = usage
    engine.run_block = run_block
    try:
        yield usage
    finally:
        engine.run_block = original_block
        _ACTIVE = None
