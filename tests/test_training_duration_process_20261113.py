"""Actual model-free process failure fixtures for the new duration controller."""
import json
from pathlib import Path
import sys
import time

import pytest

from tools.splendorust_training_duration_20261113 import common as c, control


@pytest.fixture
def isolated_output(tmp_path, monkeypatch):
    monkeypatch.setattr(c, 'OUT', tmp_path)
    # Test the controller, not the host's current amount of free storage.
    monkeypatch.setattr(c, 'MIN_DISK', 0)
    return tmp_path


def limits(seconds=15):
    mono, wall = time.monotonic(), time.time()
    return dict(monotonic_origin=mono, unix_origin=wall,
                monotonic_deadline=mono+seconds, unix_deadline=wall+seconds)


def command(code):
    return [sys.executable, '-B', '-u', '-c', code]


def metadata():
    return dict(role='process_fixture', freeze_sha256='1'*64, models=0)


def failed(folder, message):
    receipt = c.read(folder/'actual_exit.json')
    assert receipt['status'] == 'failed'
    assert message in (receipt['error'] or '')
    assert receipt['cleanup_errors'] == []
    if receipt['identity'] is not None:
        assert receipt['actual_exit_observed'] is True
        assert receipt['exact_owned_absence'] is True
        assert not c.exact_alive(receipt['identity'])
        assert all(not c.exact_alive(i) for i in receipt['observed_owned_identities'])
    return receipt


def test_actual_child_zero_exit_and_absence(isolated_output):
    folder = isolated_output/'zero'
    item = control.launch(command('import time; time.sleep(.35)'), folder,
                          limits(), metadata())
    receipt = c.read(c.resolve(item))
    assert receipt['status'] == 'passed' and receipt['exit_code'] == 0
    assert receipt['actual_exit_observed'] is True and receipt['exact_owned_absence'] is True
    assert receipt['error'] is None and receipt['cleanup_errors'] == []
    assert c.read(folder/'owner.json')['windows_job_assigned'] is True
    assert not c.exact_alive(receipt['identity'])


def test_actual_child_nonzero_is_preserved(isolated_output):
    folder = isolated_output/'nonzero'
    with pytest.raises(ValueError, match='Child failed'):
        control.launch(command('import time,sys; time.sleep(.35); sys.exit(7)'),
                       folder, limits(), metadata())
    receipt = failed(folder, '')
    assert receipt['exit_code'] == 7 and receipt['error'] is None


def test_actual_timeout_kills_owned_child(isolated_output):
    folder = isolated_output/'timeout'
    with pytest.raises(ValueError, match='Child failed'):
        control.launch(command('import time; time.sleep(30)'), folder,
                       limits(.8), metadata())
    receipt = failed(folder, 'dual clock expired')
    # Windows kill-on-close can report exit zero. The deadline exception must
    # still classify and preserve failure instead of trusting that number.
    assert receipt['elapsed_seconds'] < 10


def test_receipt_failure_after_popen_cleans_child(isolated_output, monkeypatch):
    folder = isolated_output/'receipt_failure'
    def broken_receipt(path, value):
        raise OSError('injected owner receipt failure')
    monkeypatch.setattr(c, 'atomic_save', broken_receipt)
    with pytest.raises(ValueError, match='Child failed'):
        control.launch(command('import time; time.sleep(30)'), folder,
                       limits(), metadata())
    failed(folder, 'injected owner receipt failure')


def test_job_assignment_failure_cleans_child(isolated_output, monkeypatch):
    from tools.local_player_qualification_128 import clock
    folder = isolated_output/'job_failure'
    def broken_job(pid):
        raise OSError('injected Windows job assignment failure')
    monkeypatch.setattr(clock, '_WindowsJob', broken_job)
    with pytest.raises(ValueError, match='Child failed'):
        control.launch(command('import time; time.sleep(30)'), folder,
                       limits(), metadata())
    failed(folder, 'injected Windows job assignment failure')


def test_actual_rss_limit_preserves_failure(isolated_output):
    folder = isolated_output/'rss'
    with pytest.raises(ValueError, match='Child failed'):
        control.launch(command('import time; time.sleep(30)'), folder,
                       limits(), metadata(), rss=1)
    failed(folder, 'Actual child-tree RSS bound')


def test_expired_clock_never_creates_child(isolated_output, monkeypatch):
    folder = isolated_output/'expired'
    deadline = limits(-1)
    def forbidden(*args, **kwargs):
        raise AssertionError('Popen must not be called after deadline')
    monkeypatch.setattr(control.subprocess, 'Popen', forbidden)
    with pytest.raises(ValueError, match='Child failed'):
        control.launch(command('raise AssertionError'), folder, deadline, metadata())
    receipt = failed(folder, 'dual clock expired')
    assert receipt['identity'] is None and receipt['exit_code'] is None
    assert receipt['actual_exit_observed'] is False


def test_actual_evidence_limit_preserves_failure(isolated_output, monkeypatch):
    folder = isolated_output/'evidence'
    monkeypatch.setattr(c, 'MAX_OUTPUT', 1)
    with pytest.raises(ValueError, match='Child failed'):
        control.launch(command('import time; time.sleep(30)'), folder,
                       limits(), metadata())
    failed(folder, 'Actual new-evidence size bound')


def test_duplicate_output_refuses_before_popen(isolated_output, monkeypatch):
    folder = isolated_output/'duplicate'; folder.mkdir()
    sentinel = folder/'preserved'; sentinel.write_text('unchanged')
    def forbidden(*args, **kwargs):
        raise AssertionError('duplicate attempt must not launch')
    monkeypatch.setattr(control.subprocess, 'Popen', forbidden)
    with pytest.raises(FileExistsError):
        control.launch(command('raise AssertionError'), folder, limits(), metadata())
    assert sentinel.read_text() == 'unchanged'


def test_windows_job_cleans_descendant_after_normal_parent_exit(isolated_output, monkeypatch):
    from tools.splendorust_full_corpus_recovery_20261111 import owner as previous_owner
    folder = isolated_output/'descendant'
    identity_path = isolated_output/'descendant_identity.json'
    observed_path = isolated_output/'descendant_observed.json'
    actual_observer = previous_owner.observe_tree
    def observe_and_acknowledge(identity, observed):
        used = actual_observer(identity, observed)
        if identity_path.exists() and not observed_path.exists():
            descendant = json.loads(identity_path.read_text())
            if descendant in observed.values():
                observed_path.write_text(json.dumps(descendant))
        return used
    monkeypatch.setattr(previous_owner, 'observe_tree', observe_and_acknowledge)
    # Mirror the real worker ownership handshake. Additionally acknowledge actual
    # observation before parent exit, so the test does not rely on host scheduling.
    code = f"""
import subprocess,sys,time,json,psutil
from pathlib import Path
owner=Path({str(folder/'owner.json')!r})
observed=Path({str(observed_path)!r})
deadline=time.monotonic()+10
while not owner.exists():
    assert time.monotonic()<deadline, 'fixture owner handshake expired'
    time.sleep(.01)
child=subprocess.Popen([sys.executable,'-B','-c','import time; time.sleep(30)'],creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
identity=Path({str(identity_path)!r})
temporary=identity.with_suffix('.pending')
temporary.write_text(json.dumps(dict(pid=child.pid,created=psutil.Process(child.pid).create_time())))
temporary.replace(identity)
while not observed.exists():
    assert time.monotonic()<deadline, 'fixture descendant observation expired'
    time.sleep(.01)
time.sleep(1.2)
"""
    item = control.launch(command(code), folder, limits(), metadata())
    receipt = c.read(c.resolve(item)); descendant = json.loads(identity_path.read_text())
    assert receipt['status'] == 'passed' and receipt['exit_code'] == 0
    assert descendant in receipt['observed_owned_identities']
    assert not c.exact_alive(descendant)


@pytest.mark.parametrize('field,value', (
    ('role', 'wrong_role'), ('freeze_sha256', '2'*64),
    ('identity', dict(pid=2147483647,created=0)), ('windows_job_assigned', False),
    ('parent', dict(pid=2147483647,created=0)),
))
def test_handshake_rejects_wrong_owner_binding(isolated_output, field, value):
    folder = isolated_output/'handshake'; folder.mkdir()
    record = dict(identity=c.identity(), parent=c.identity(), role='fit_r0',
                  freeze_sha256='1'*64, windows_job_assigned=True, limits=limits())
    record[field] = value
    c.save(folder/'owner.json', record)
    with pytest.raises(ValueError):
        control.handshake(folder, '1'*64, 'fit_r0')


def test_handshake_accepts_exact_owner_binding(isolated_output):
    folder = isolated_output/'handshake_ok'; folder.mkdir()
    record = dict(identity=c.identity(), parent=c.identity(), role='fit_r0',
                  freeze_sha256='1'*64, windows_job_assigned=True, limits=limits())
    c.save(folder/'owner.json', record)
    assert control.handshake(folder, '1'*64, 'fit_r0') == record


def test_cleanup_accepts_process_disappearance_after_job_close(monkeypatch):
    """Disappearance between exact identity check and enumeration is success."""
    ident = dict(pid=2147483646, created=1.)
    class ProcessHandle:
        returncode = None
        waited = False
        def poll(self): return self.returncode
        def wait(self, timeout=None):
            self.returncode = 0
            self.waited = True
            return 0
    class Job:
        def close(self): pass
    process = ProcessHandle(); errors = []
    checks = iter([True])
    monkeypatch.setattr(c, 'exact_alive', lambda item: next(checks, False))
    def vanished(pid):
        raise control.psutil.NoSuchProcess(pid)
    monkeypatch.setattr(control.psutil, 'Process', vanished)
    control.cleanup_owned(process, Job(), ident, {}, errors)
    assert errors == []
    assert process.waited and process.returncode == 0


@pytest.mark.parametrize('role', ('commit', 'root'))
def test_declared_role_exit_binds_completed_result(isolated_output, role):
    folder = isolated_output/('result_'+role)
    result_path = isolated_output/'completion.json' if role == 'root' else folder/'worker/result.json'
    code = f"""
import json,time
from pathlib import Path
owner=Path({str(folder/'owner.json')!r})
deadline=time.monotonic()+10
while not owner.exists():
    assert time.monotonic()<deadline
    time.sleep(.01)
result=Path({str(result_path)!r})
result.parent.mkdir(parents=True,exist_ok=True)
result.write_text(json.dumps(dict(status='passed',role={role!r})))
"""
    item = control.launch(command(code), folder, limits(), {**metadata(), 'role':role})
    receipt = c.read(c.resolve(item))
    assert receipt['status'] == 'passed' and receipt['exit_code'] == 0
    assert receipt['result_artifact'] == c.artifact(result_path)
    c.verify([receipt['result_artifact']])
    result_path.write_text('{"status":"altered"}')
    with pytest.raises(ValueError, match='Changed input'):
        c.verify([receipt['result_artifact']])


def test_zero_exit_without_declared_result_is_failure(isolated_output):
    folder = isolated_output/'missing_result'
    with pytest.raises(ValueError, match='Child failed'):
        control.launch(command('import time; time.sleep(.35)'), folder,
                       limits(), {**metadata(), 'role':'commit'})
    receipt = failed(folder, 'FileNotFoundError')
    assert receipt['exit_code'] == 0
    assert receipt['result_artifact'] is None
