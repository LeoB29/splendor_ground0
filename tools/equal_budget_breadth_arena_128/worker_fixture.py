"""Actual hidden worker initialization; never runs a game, fit or rollout."""
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import os
from pathlib import Path
import sys
import traceback

import torch

from tools.equal_budget_breadth_arena_128 import common as c, experiment, arena
from tools.proposal_iteration_128 import arena as original_arena
from tools.proposal_ranking_128 import source as ranking_source, engine as ranking_engine
from tools.proposal_breadth_128 import source as breadth_source, engine as breadth_engine
from tools.proposal_breadth_arena_128 import arena as broader_arena
from tools.equal_budget_breadth_128 import source as uniform_source, engine as uniform_engine


def inspect_worker(kind, base):
    base = Path(base)
    assert not base.resolve().is_relative_to(c.OUTPUT.resolve())
    assert c.old.read_json(base / "models_lock.json")["manifest"] == "fixture"
    assert kind in ("games", "audits")
    assert set(arena._MODELS) == set(arena._MODEL_ARTIFACTS)
    for module in (original_arena, ranking_source, ranking_engine, breadth_source, breadth_engine, broader_arena, uniform_source, uniform_engine):
        assert not module._MODELS and not module._MODEL_ARTIFACTS
    assert set(arena._MODELS) == {f"anchor_{fit}" for fit in c.FITS}
    result = {"kind": kind, "health": c.windows.assert_windowless_worker(),
              "models": arena._MODEL_ARTIFACTS, "model_count": len(arena._MODELS),
              "all_eval": all(not model.training for model in arena._MODELS.values()),
              "parameter_counts": {key: sum(p.numel() for p in model.parameters())
                                   for key, model in arena._MODELS.items()},
              "threads": torch.get_num_threads(),
              "deterministic": torch.are_deterministic_algorithms_enabled(),
              "ancestor_registries_unchanged": True}
    print(f"equal-budget breadth arena fixture worker kind={kind} pid={os.getpid()}", flush=True)
    return result


def main(base):
    base = Path(base).resolve()
    assert not base.is_relative_to(c.OUTPUT.resolve())
    assert c.old.read_json(base / "models_lock.json")["manifest"] == "fixture"
    supervisor = c.windows.assert_windowless_worker(base / "supervisor_stdout.txt", base / "supervisor_stderr.txt")
    workers = []
    for kind in ("games", "audits"):
        runtime = c.windows.configure_spawn()
        with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn"),
                                 initializer=experiment.worker_initialize,
                                 initargs=("main", kind, "fixture", base)) as pool:
            value = pool.submit(inspect_worker, kind, base).result(timeout=40)
        workers.append({"runtime": runtime, "result": value})
    c.old.save_new(base / "worker_fixture_complete.json", {"supervisor": supervisor, "workers": workers})


if __name__ == "__main__":
    destination = Path(sys.argv[1])
    try:
        main(destination)
    except BaseException:
        c.old.save_new(destination / "worker_fixture_failure.json", {"traceback": traceback.format_exc()})
        raise
