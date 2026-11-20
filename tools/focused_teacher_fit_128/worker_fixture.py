"""Real hidden supervisor and model-loading worker; no production fit."""
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from pathlib import Path
import sys
import traceback
import torch
from . import common as c, experiment


def inspect(base):
    from splendor_ai.alphazero.strong_policy import StrongPolicyInference
    assert not Path(base).resolve().is_relative_to(c.OUTPUT.resolve())
    refs = c.old.read_json(Path(base)/"fixture_models.json")
    models = {key: StrongPolicyInference(c.verify_hash(ref)).model for key, ref in refs.items()}
    return {"health": c.windows.assert_windowless_worker(), "models": refs,
            "parameter_counts": {key: sum(p.numel() for p in m.parameters()) for key, m in models.items()},
            "all_eval": all(not m.training for m in models.values()), "threads": torch.get_num_threads(),
            "deterministic": torch.are_deterministic_algorithms_enabled()}


def main(base):
    base = Path(base).resolve()
    assert not base.is_relative_to(c.OUTPUT.resolve())
    health = c.windows.assert_windowless_worker(base/"supervisor_stdout.txt", base/"supervisor_stderr.txt")
    runtime = c.windows.configure_spawn()
    with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn"),
                             initializer=experiment.worker_initialize, initargs=("fixture", "fixture", base)) as pool:
        value = pool.submit(inspect, base).result(timeout=40)
    c.old.save_new(base/"worker_fixture_complete.json", {"supervisor": health, "worker": value, "runtime": runtime})


if __name__ == "__main__":
    destination = Path(sys.argv[1])
    try: main(destination)
    except BaseException:
        c.old.save_new(destination/"worker_fixture_failure.json", {"traceback": traceback.format_exc()})
        raise
