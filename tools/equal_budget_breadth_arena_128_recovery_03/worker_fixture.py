"""Actual hidden audit-worker initialization, without gameplay or inference."""
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from pathlib import Path
import sys
import traceback
import torch

from . import experiment as e


def initialize(base):
    base=Path(base).resolve()
    assert not base.is_relative_to(e.c.SOURCE.resolve())
    e.OUTPUT=base
    e.worker_initialize('audit-only-worker-fixture')


def inspect():
    return {'health':e.c.windows.assert_windowless_worker(),'threads':torch.get_num_threads(),
            'deterministic':torch.are_deterministic_algorithms_enabled(),
            'models':sorted(e.arena._MODELS),'all_cpu':all(p.device.type=='cpu' for m in e.arena._MODELS.values() for p in m.parameters()),
            'all_eval':all(not m.training for m in e.arena._MODELS.values())}


def main(base):
    base=Path(base).resolve()
    assert not base.is_relative_to(e.c.SOURCE.resolve())
    supervisor=e.c.windows.assert_windowless_worker(base/'supervisor_stdout.txt',base/'supervisor_stderr.txt')
    runtime=e.c.windows.configure_spawn()
    with ProcessPoolExecutor(max_workers=1,mp_context=multiprocessing.get_context('spawn'),initializer=initialize,initargs=(base,)) as pool:
        result=pool.submit(inspect).result(timeout=60)
    e.old.save_new(base/'complete.json',{'supervisor':supervisor,'worker':result,'runtime':runtime})


if __name__=='__main__':
    destination=Path(sys.argv[1])
    try:main(destination)
    except BaseException:
        e.old.save_new(destination/'failure.json',{'traceback':traceback.format_exc()})
        raise
