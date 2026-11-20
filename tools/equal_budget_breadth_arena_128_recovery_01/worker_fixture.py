"""Real hidden recovery worker initialization without gameplay or rollouts."""
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import os
from pathlib import Path
import sys
import traceback

import torch

from tools.equal_budget_breadth_arena_128_recovery_01 import common as c, experiment, arena
from tools.equal_budget_breadth_arena_128 import arena as frozen_arena
from tools.proposal_iteration_128 import arena as original_arena
from tools.proposal_ranking_128 import source as ranking_source, engine as ranking_engine
from tools.proposal_breadth_128 import source as breadth_source, engine as breadth_engine
from tools.equal_budget_breadth_128 import source as uniform_source, engine as uniform_engine

FIXTURE_BINDING = "isolated-recovery-hidden-worker-fixture"


def fixture_base(base):
    base = Path(base).resolve()
    assert not base.is_relative_to(c.SOURCE.resolve())
    assert c.old.read_json(base/"models_lock.json")["manifest"] == "fixture"
    assert c.recovery_binding(base) == FIXTURE_BINDING
    inventory = c.load_inventory(base)
    assert not inventory["games"] and not inventory["roots"]
    assert c.verify_hash(inventory["models"]) == base/"models_lock.json"
    return base


def inspect_worker(kind,base):
    base = fixture_base(base)
    assert kind in ("games","audits")
    assert arena._MODELS is frozen_arena._MODELS and arena._MODEL_ARTIFACTS is frozen_arena._MODEL_ARTIFACTS
    assert set(arena._MODELS) == set(arena._MODEL_ARTIFACTS) == {f"anchor_{fit}" for fit in c.FITS}
    for module in (original_arena,ranking_source,ranking_engine,breadth_source,breadth_engine,
                   uniform_source,uniform_engine):
        assert not module._MODELS and not module._MODEL_ARTIFACTS
    result = {"kind":kind,"health":c.windows.assert_windowless_worker(),
              "models":arena._MODEL_ARTIFACTS,"model_count":len(arena._MODELS),
              "all_eval":all(not model.training for model in arena._MODELS.values()),
              "all_cpu":all(parameter.device.type == "cpu" for model in arena._MODELS.values() for parameter in model.parameters()),
              "parameter_counts":{key:sum(parameter.numel() for parameter in model.parameters()) for key,model in arena._MODELS.items()},
              "threads":torch.get_num_threads(),"deterministic":torch.are_deterministic_algorithms_enabled(),
              "frozen_arena_registry_shared":True,"ancestor_registries_unchanged":True,
              "recovery_binding":c.recovery_binding(base)}
    print(f"recovery fixture worker kind={kind} pid={os.getpid()}",flush=True)
    return result


def main(base):
    base = fixture_base(base)
    supervisor = c.windows.assert_windowless_worker(base/"supervisor_stdout.txt",base/"supervisor_stderr.txt")
    workers = []
    for kind in ("games","audits"):
        runtime = c.windows.configure_spawn()
        with ProcessPoolExecutor(max_workers=1,mp_context=multiprocessing.get_context("spawn"),
                                 initializer=experiment.worker_initialize,
                                 initargs=(kind,FIXTURE_BINDING,base)) as pool:
            result = pool.submit(inspect_worker,kind,base).result(timeout=40)
        workers.append({"runtime":runtime,"result":result})
    c.old.save_new(base/"worker_fixture_complete.json",{"supervisor":supervisor,"workers":workers})


if __name__ == "__main__":
    destination = Path(sys.argv[1])
    try:
        main(destination)
    except BaseException:
        c.old.save_new(destination/"worker_fixture_failure.json",{"traceback":traceback.format_exc()})
        raise
