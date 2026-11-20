"""A real hidden recovery-initialized worker; no scientific collection or fit."""
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path
import sys
import time
import traceback
import torch

from tools.control_variate_128 import common as c, experiment as parent
from . import recovery as r


def inspect(base):
    base=Path(base).resolve()
    assert not base.is_relative_to(r.OUTPUT.resolve())
    assert not base.is_relative_to(r.PARENT.resolve())
    assert c.budget_boundary is r.budget_boundary
    assert parent.worker_initialize is r.worker_initialize
    assert parent.OUTPUT.resolve() == base
    launch={"epoch_seconds":time.time(),"previous_elapsed_seconds":7201.}
    c.old.save_new(base/"launch.json",launch)
    c.budget_boundary(base)
    launch["previous_elapsed_seconds"]=r.MAXIMUM_RUNTIME_SECONDS+1.
    (base/"launch.json").write_text(json.dumps(launch),encoding="utf-8")
    stopped=False
    try:c.budget_boundary(base)
    except c.BudgetExceeded:stopped=True
    assert stopped
    model=c.old._RAW.inference.model if hasattr(c.old._RAW,"inference") else None
    return {"health":c.windows.assert_windowless_worker(),"threads":torch.get_num_threads(),
        "deterministic":torch.are_deterministic_algorithms_enabled(),
        "raw_initialized":c.old._RAW is not None,
        "budget_module":c.budget_boundary.__module__,"budget_name":c.budget_boundary.__name__,
        "maximum_runtime_seconds":r.MAXIMUM_RUNTIME_SECONDS,"passes_after_two_hours":True,
        "stops_after_three_active_hours":stopped,"scientific_config":c.CONFIG,
        "model_eval":None if model is None else not model.training}


def main(base):
    base=Path(base).resolve()
    assert not base.is_relative_to(r.OUTPUT.resolve()) and not base.is_relative_to(r.PARENT.resolve())
    health=c.windows.assert_windowless_worker(base/"supervisor_stdout.txt",base/"supervisor_stderr.txt")
    runtime=c.windows.configure_spawn()
    with ProcessPoolExecutor(max_workers=1,mp_context=multiprocessing.get_context("spawn"),
                             initializer=r.worker_initialize,initargs=("fixture","fixture",base)) as pool:
        result=pool.submit(inspect,base).result(timeout=40)
    c.old.save_new(base/"worker_fixture_complete.json",{"supervisor":health,"worker":result,"runtime":runtime})


if __name__ == "__main__":
    destination=Path(sys.argv[1])
    try:main(destination)
    except BaseException:
        c.old.save_new(destination/"worker_fixture_failure.json",{"traceback":traceback.format_exc()})
        raise
