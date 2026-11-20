"""Isolated process-initialization fixture; never runs a game, fit or rollout."""
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import os
from pathlib import Path
import sys
import traceback

import torch

from tools.proposal_breadth_128 import common as c, experiment, source, engine
from tools.proposal_iteration_128 import arena as ancestor
from tools.proposal_ranking_128 import source as prior_source, engine as prior_engine


def inspect_worker(stage,base):
    base = Path(base)
    assert not base.resolve().is_relative_to(c.OUTPUT.resolve())
    assert c.old.read_json(base/"models_lock.json")["manifest"] == "fixture"
    module = source if stage == "source_main" else engine
    assert stage in ("source_main","main")
    assert set(module._MODELS) == set(module._MODEL_ARTIFACTS)
    assert not ancestor._MODELS and not ancestor._MODEL_ARTIFACTS
    assert not prior_source._MODELS and not prior_source._MODEL_ARTIFACTS
    assert not prior_engine._MODELS and not prior_engine._MODEL_ARTIFACTS
    assert set(module._MODELS) == {f"anchor_{fit}" for fit in c.FITS}
    result = {"stage":stage,"health":c.windows.assert_windowless_worker(),
              "models":module._MODEL_ARTIFACTS,
              "model_count":len(module._MODELS),
              "all_eval":all(not m.training for m in module._MODELS.values()),
              "parameter_counts":{key:sum(p.numel() for p in m.parameters()) for key,m in module._MODELS.items()},
              "threads":torch.get_num_threads(),"deterministic":torch.are_deterministic_algorithms_enabled(),
              "ancestor_registry_unchanged":True}
    print(f"breadth fixture worker stage={stage} pid={os.getpid()}",flush=True)
    return result


def main(base):
    base = Path(base).resolve()
    assert not base.is_relative_to(c.OUTPUT.resolve())
    assert c.old.read_json(base/"models_lock.json")["manifest"] == "fixture"
    supervisor = c.windows.assert_windowless_worker(base/"supervisor_stdout.txt",base/"supervisor_stderr.txt")
    stages = []
    for stage in ("source_main","main"):
        runtime = c.windows.configure_spawn()
        with ProcessPoolExecutor(max_workers=1,mp_context=multiprocessing.get_context("spawn"),
                                 initializer=experiment.worker_initialize,
                                 initargs=(stage,"collection","fixture",base)) as pool:
            value = pool.submit(inspect_worker,stage,base).result(timeout=40)
        stages.append({"runtime":runtime,"result":value})
    c.old.save_new(base/"worker_fixture_complete.json",{"supervisor":supervisor,"stages":stages})


if __name__ == "__main__":
    destination = Path(sys.argv[1])
    try:
        main(destination)
    except BaseException:
        c.old.save_new(destination/"worker_fixture_failure.json",{"traceback":traceback.format_exc()})
        raise
