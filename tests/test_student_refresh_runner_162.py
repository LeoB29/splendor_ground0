import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import time

import pytest

from splendor_ai.alphazero import run_student_refresh as p


def test_fixed_initializers_and_identity_dependencies():
    manifest = p.identity()
    assert manifest["config"]["device"] == "cpu"
    assert len(manifest["initializers"]) == 3
    assert set(manifest["dependencies"]) == {path.as_posix() for path in p.INVENTORY}
    assert manifest["config"]["fit_workers"] == 3


def test_artifact_tamper_and_duplicate_rejected(tmp_path):
    path = tmp_path / "one.json"
    p.old.save_new(path, {"a": 1})
    item = p.h.artifact(path)
    p.verify_files([item])
    with pytest.raises(AssertionError):
        p.verify_files([item, item])
    path.write_text("tampered", encoding="utf-8")
    with pytest.raises(AssertionError):
        p.verify_files([item])


def test_held_launch_ack_binding_and_timeout_never_relaunch(monkeypatch, tmp_path):
    calls = []
    def fake(*args):
        calls.append(args)
        return {"pid": 111}
    monkeypatch.setattr(p.windows, "detached_launch", fake)
    with pytest.raises(TimeoutError):
        p.held_launch([sys.executable], tmp_path / "missing", tmp_path / "none.json", "fixture", timeout=0)
    assert len(calls) == 1 and (tmp_path / "missing" / "process.json").exists()
    ack = tmp_path / "ack.json"
    p.old.save_new(ack, {"pid": 112, "manifest": "fixture"})
    with pytest.raises(AssertionError):
        p.held_launch([sys.executable], tmp_path / "wrong", ack, "fixture")
    assert len(calls) == 2 and not (tmp_path / "wrong" / "acknowledged.json").exists()


@pytest.mark.skipif(os.name != "nt", reason="Actual Windows launch lifetime and worker integration")
def test_launcher_exits_after_ack_child_survives_and_all_stage_pools_windowless(tmp_path):
    helper = tmp_path / "fixture162.py"
    helper.write_text(textwrap.dedent('''
        from concurrent.futures import ProcessPoolExecutor
        import json
        import multiprocessing
        import os
        from pathlib import Path
        import sys
        import time
        import traceback
        sys.path.insert(0, REPOSITORY_LITERAL)
        sys.path.insert(0, SOURCE_LITERAL)
        from splendor_ai.alphazero import run_student_refresh as p
        ROOT = Path(__file__).parent

        def task(stage, i):
            # Exercise actual production initializer, stream handlers and verifier imports.
            health = p.windows.assert_windowless_worker()
            assert p.auditor().independent_split() == p.h.splits()
            time.sleep(.3)
            print({'stage': stage, 'slot': i}, flush=True)
            return health

        def child():
            health = p.windows.assert_windowless_worker(ROOT/'child_stdout.txt', ROOT/'child_stderr.txt')
            p.old.save_new(ROOT/'ack.json', {'pid': os.getpid(), 'manifest': 'fixture', 'health': health,
                           'second_health': p.windows.assert_windowless_worker()})
            deadline = time.monotonic()+45
            while not (ROOT/'release').exists():
                if time.monotonic() > deadline:
                    raise TimeoutError('test did not release child after launcher exit')
                time.sleep(.05)
            reports = []
            for stage, workers in [('export_fixture', 2), ('fits_fixture', 3), ('audit_fixture', 2)]:
                p.windows.configure_spawn()
                with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
                                         initializer=p.worker_initialize, initargs=(stage, 'fixture', ROOT)) as pool:
                    reports.append([f.result() for f in [pool.submit(task, stage, i) for i in range(workers)]])
            p.old.save_new(ROOT/'done.json', {'reports': reports, 'pid': os.getpid()})

        if __name__ == '__main__':
            try:
                if sys.argv[1] == 'launcher':
                    p.held_launch([sys.executable, '-u', __file__, 'child'], ROOT/'launch_logs', ROOT/'ack.json',
                                  'fixture', cwd=p.REPOSITORY)
                else:
                    child()
            except BaseException:
                p.old.save_new(ROOT/f'failure_{sys.argv[1]}.json', {'traceback': traceback.format_exc()})
                raise
    ''').replace("REPOSITORY_LITERAL", repr(str(p.REPOSITORY))).replace("SOURCE_LITERAL", repr(str(p.REPOSITORY / "src"))), encoding="utf-8")
    with (tmp_path / "launcher_stdout.txt").open("wb") as out, (tmp_path / "launcher_stderr.txt").open("wb") as err:
        launcher = subprocess.Popen([p.windows.configure_spawn()["executable"], "-u", str(helper), "launcher"],
                                    cwd=tmp_path, stdout=out, stderr=err, stdin=subprocess.DEVNULL,
                                    creationflags=subprocess.CREATE_NO_WINDOW)
        assert launcher.wait(timeout=75) == 0, (tmp_path / "launcher_stderr.txt").read_text()
    ack = json.loads((tmp_path / "ack.json").read_text())
    assert ack["health"]["parent_pid"] == launcher.pid
    assert (tmp_path / "launch_logs" / "acknowledged.json").exists()
    assert not (tmp_path / "done.json").exists()
    (tmp_path / "release").write_text("launcher has exited", encoding="utf-8")
    deadline = time.monotonic()+100
    while not (tmp_path / "done.json").exists():
        failures = list(tmp_path.glob("failure_*.json"))
        assert not failures, [path.read_text() for path in failures]
        assert time.monotonic() < deadline, "Windowless stage fixtures timed out"
        time.sleep(.05)
    done = json.loads((tmp_path / "done.json").read_text())
    assert done["pid"] == ack["pid"]
    for check in [ack["health"], ack["second_health"], *[r for stage in done["reports"] for r in stage]]:
        assert Path(check["executable"]).name.lower() == "pythonw.exe"
        assert check["console_window"] == check["console_process_count"] == 0
    healths = list((tmp_path / "workers").rglob("*_health.json"))
    assert len(healths) == 7
    for path in (tmp_path / "workers").rglob("*_stderr.txt"):
        assert path.stat().st_size == 0


def test_selected_dependency_is_bound_once_and_tamper_stays_strict(tmp_path):
    selected = tmp_path / "selected_lock.json"
    p.old.save_new(selected, {"models": {}})
    first = p.stage_dependencies("probe_games", selected, tmp_path)
    assert first == [p.h.artifact(selected)]
    p.verify_files(first)
    upstream = tmp_path / "probe_games_lock.json"
    p.old.save_new(upstream, {"complete": True})
    second = p.stage_dependencies("probe_audit", upstream, tmp_path)
    assert second == [p.h.artifact(upstream), p.h.artifact(selected)]
    p.verify_files(second)
    assert p.stage_dependencies("teacher_probe", upstream, tmp_path) == [p.h.artifact(upstream)]
    selected.write_text("tampered", encoding="utf-8")
    with pytest.raises(AssertionError):
        p.verify_files(first)
    with pytest.raises(AssertionError):
        p.verify_files(second)


def test_collection_to_audit_lock_transition_with_selected_as_upstream(tmp_path, monkeypatch):
    from concurrent.futures import Future

    class ImmediateExecutor:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def submit(self, fn, *args):
            future = Future()
            future.set_result(fn(*args))
            return future

    def dispatch(kind, args):
        index = args[0]
        path = tmp_path / kind / f"item_{index}.json"
        p.old.save_new(path, {"index": index})
        return p.h.artifact(path, index=index)

    monkeypatch.setattr(p, "OUTPUT", tmp_path)
    monkeypatch.setattr(p, "validate_freeze", lambda: "fixture")
    monkeypatch.setattr(p, "ProcessPoolExecutor", ImmediateExecutor)
    monkeypatch.setattr(p, "dispatch", dispatch)
    selected = tmp_path / "selected_lock.json"
    p.old.save_new(selected, {"models": {}})
    p.run_jobs("probe_games", "games", [(0,), (1,)], "fixture", selected)
    first = p.verify_stage("probe_games", "fixture")
    assert len(first["files"]) == 2
    p.run_jobs("probe_audit", "audits", [(0,), (1,)], "fixture", tmp_path / "probe_games_lock.json")
    second = p.verify_stage("probe_audit", "fixture")
    assert len(second["files"]) == 2
    assert len(p.old.read_json(tmp_path / "probe_games_launch.json")["dependencies"]) == 1
    assert len(p.old.read_json(tmp_path / "probe_audit_launch.json")["dependencies"]) == 2
