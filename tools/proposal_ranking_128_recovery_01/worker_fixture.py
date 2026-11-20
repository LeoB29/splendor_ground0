"""Isolated hidden recovery initialization; no games, inference or training."""
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from pathlib import Path
import sys
import traceback

from tools.proposal_ranking_128 import common as c
from tools.proposal_ranking_128 import worker_fixture as original_fixture
from tools.proposal_ranking_128_recovery_01 import recovery

_VALIDATION_CALLS = []


def _checked_fixture(base):
    base = Path(base).resolve()
    # The fixture override is never usable anywhere under production outputs.
    assert not base.is_relative_to((c.ROOT/"outputs").resolve())
    assert c.old.read_json(base/"worker_fixture_guard.json") == {"isolated_recovery_worker_fixture":True}
    assert c.old.read_json(base/"models_lock.json")["manifest"] == "fixture"
    return base


def fixture_initialize(stage,kind,binding,base):
    base = _checked_fixture(base)
    assert binding == "fixture" and (stage,kind) in (("source_main","audits"),("main","collection"))
    recovery.RECOVERY = base/"recovery_01"
    def fixture_validation(check_original=True):
        assert _checked_fixture(base) == base and check_original is False
        _VALIDATION_CALLS.append({"check_original":check_original})
        return "fixture-recovery-binding"
    # Production freeze is intentionally absent only in this guarded fixture.
    # The actual recovery initializer and scientific model-loading functions run.
    recovery.validate_recovery = fixture_validation
    recovery.worker_initialize(stage,kind,binding,base)


def inspect_worker(stage,base):
    base = _checked_fixture(base)
    value = original_fixture.inspect_worker(stage,base)
    value["recovery_folder"] = str(recovery.RECOVERY.resolve())
    value["recovery_validation_calls"] = list(_VALIDATION_CALLS)
    return value


def main(base):
    base = _checked_fixture(base)
    folder = base/"recovery_01"
    folder.mkdir(parents=True,exist_ok=True)
    supervisor = c.windows.assert_windowless_worker(folder/"supervisor_stdout.txt",folder/"supervisor_stderr.txt")
    stages = []
    for stage,kind in (("source_main","audits"),("main","collection")):
        runtime = c.windows.configure_spawn()
        with ProcessPoolExecutor(max_workers=1,mp_context=multiprocessing.get_context("spawn"),
                                 initializer=fixture_initialize,
                                 initargs=(stage,kind,"fixture",base)) as pool:
            value = pool.submit(inspect_worker,stage,base).result(timeout=40)
        stages.append({"stage":stage,"kind":kind,"runtime":runtime,"result":value})
    c.old.save_new(folder/"worker_fixture_complete.json",{"supervisor":supervisor,"stages":stages})


if __name__ == "__main__":
    destination = Path(sys.argv[1])
    try:
        main(destination)
    except BaseException:
        # Validate the isolated destination again before recording fixture errors.
        target = _checked_fixture(destination)/"recovery_01/worker_fixture_failure.json"
        c.old.save_new(target,{"traceback":traceback.format_exc()})
        raise
