"""Bounded model-free selection materialization; starts no scientific clock."""
import os
import shutil
import sys
import time
import traceback
import psutil
from . import common as c
from .control import check_clock


def main():
    from tools.future_training_rules_20261021 import verify_runtime_sources
    from . import data
    c.require(not c.SELECTION.exists() and not c.PLAN.exists(), 'Prepare one new selection')
    started, wall = time.monotonic(), time.time()
    last = [0.]
    psutil.Process().cpu_affinity([2])
    def boundary():
        check_clock(started, wall, 3600, time.monotonic(), time.time(), 'model-free preparation')
        if time.monotonic()-last[0] > 5:
            last[0] = time.monotonic()
            c.require(psutil.Process().memory_info().rss < c.MAX_RSS, 'Actual preparation RSS')
            c.require(shutil.disk_usage(c.ROOT).free >= c.MIN_DISK, 'Actual preparation disk floor')
    source = verify_runtime_sources()
    selection = data.prepare_selection(boundary=boundary)
    data.validate_selection(selection, boundary=boundary)
    boundary()
    c.require('torch' not in sys.modules, 'Selection preparation loads no tensor/model runtime')
    item = c.save(c.SELECTION, selection)
    receipt = dict(status='prepared_not_executed', time=c.now(), pid=os.getpid(),
        selection=item, runtime_sources=source, models=0, games=0, optimizer_calls=0,
        monotonic_origin=started, monotonic_end=time.monotonic(), unix_origin=wall, unix_end=time.time(),
        maximum_seconds=3600)
    c.save(c.OUT/'prelaunch/selection_preparation.json', receipt)
    print(c.canonical(receipt), flush=True)
    return 0


if __name__ == '__main__': raise SystemExit(main())
