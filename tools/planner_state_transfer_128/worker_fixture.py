"""Actual hidden process/initializer check without games or training."""
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from pathlib import Path
import sys
import traceback
import torch
from . import common as c, engine, experiment

def inspect(base):
    assert not Path(base).resolve().is_relative_to(c.OUTPUT.resolve())
    from tools.proposal_breadth_learning_128 import engine as ancestor
    assert not ancestor._MODELS and not ancestor._MODEL_ARTIFACTS
    return {"health":c.windows.assert_windowless_worker(),"models":engine._MODEL_ARTIFACTS,
            "all_eval":all(not m.training for m in engine._MODELS.values()),
            "parameter_counts":{k:sum(p.numel() for p in m.parameters()) for k,m in engine._MODELS.items()},
            "threads":torch.get_num_threads(),"deterministic":torch.are_deterministic_algorithms_enabled()}

def main(base):
    base = Path(base).resolve()
    assert not base.is_relative_to(c.OUTPUT.resolve())
    assert c.old.read_json(base/"models_lock.json")["manifest"] == "fixture"
    health = c.windows.assert_windowless_worker(base/"supervisor_stdout.txt",base/"supervisor_stderr.txt")
    runtime = c.windows.configure_spawn()
    with ProcessPoolExecutor(max_workers=1,mp_context=multiprocessing.get_context("spawn"),
                             initializer=experiment.worker_initialize,initargs=("probe","collection","fixture",base)) as pool:
        value = pool.submit(inspect,base).result(timeout=40)
    c.old.save_new(base/"worker_fixture_complete.json",{"supervisor":health,"worker":value,"runtime":runtime})

if __name__ == "__main__":
    destination = Path(sys.argv[1])
    try: main(destination)
    except BaseException:
        c.old.save_new(destination/"worker_fixture_failure.json",{"traceback":traceback.format_exc()})
        raise
