"""Model-free regression checks for recovery identity and resource admission."""
from copy import deepcopy
import pytest

from tools.splendorust_search256_recovery_20261030 import preparation_support as support


@pytest.fixture
def receipts(monkeypatch):
    paths = [support.OLD_OUT / name for name in
             ('guardian_claim.json', 'watchdog_claim.json', 'launch.json', 'guardian_owner.json')]
    paths += [support.OLD_OUT / 'workers' / worker / name
              for worker in support.INTERRUPTED_WORKERS for name in
              ('initialization.json', 'request-00000016.json', 'placement_admission.json')]
    saved = {path: support._read(path) for path in paths}
    monkeypatch.setattr(support, '_read', lambda path: deepcopy(saved[path]))
    return saved


def test_saved_receipts_bind_exact_fifteen_identities(receipts):
    rows = support.recorded_old_identities()
    assert len(rows) == len({row['pid'] for row in rows}) == 15
    assert rows[:3] == [dict(pid=28928, created=1793309708.9862535),
                       dict(pid=42876, created=1793309709.3052156),
                       dict(pid=31488, created=1793309709.6364348)]
    assert {row['pid'] for row in rows} == {28928, 42876, 31488, 55932, 23316,
        65576, 26824, 59836, 25104, 65532, 32600, 22380, 27508, 4032, 51360}


def test_absence_is_rechecked_and_does_not_invent_exit_codes(receipts, monkeypatch):
    calls = []
    def gone(pid):
        calls.append(pid)
        raise support.psutil.NoSuchProcess(pid)
    monkeypatch.setattr(support.psutil, 'Process', gone)
    first = support.old_owner_identities()
    second = support.verify_old_absence()
    assert first == second and len(calls) == 30
    assert all(set(row) == {'pid', 'created', 'alive'} and row['alive'] is False for row in first)


def test_pid_reuse_proves_original_absent(monkeypatch):
    class Reused:
        def create_time(self): return 1001.
    monkeypatch.setattr(support.psutil, 'Process', lambda pid: Reused())
    assert support._absent(dict(pid=12, created=1000.)) == dict(pid=12, created=1000., alive=False)


def test_same_identity_is_not_absent(monkeypatch):
    class Existing:
        def create_time(self): return 1000.
    monkeypatch.setattr(support.psutil, 'Process', lambda pid: Existing())
    with pytest.raises(ValueError, match='still alive'):
        support._absent(dict(pid=12, created=1000.))


def test_access_denied_does_not_pass_absence(monkeypatch):
    def inaccessible(pid): raise support.psutil.AccessDenied(pid)
    monkeypatch.setattr(support.psutil, 'Process', inaccessible)
    with pytest.raises(support.psutil.AccessDenied):
        support._absent(dict(pid=12, created=1000.))


def test_process_disappearing_during_inspection_is_absent(monkeypatch):
    class Gone:
        def create_time(self): raise support.psutil.NoSuchProcess(12)
    monkeypatch.setattr(support.psutil, 'Process', lambda pid: Gone())
    assert support._absent(dict(pid=12, created=1000.))['alive'] is False


@pytest.mark.parametrize('pid,created', [(True, 1.), (0, 1.), (1, float('nan')),
    (1, float('inf')), (1, True), (1, -1.)])
def test_invalid_identity_rejected(pid, created):
    with pytest.raises(ValueError): support._identity(pid, created)


@pytest.mark.parametrize('mutation', ['wrong_freeze', 'wrong_guardian', 'wrong_native',
    'changed_helper_identity', 'helper_missing', 'wrong_parent', 'wrong_player_quota',
    'wrong_supervisor', 'duplicate_helper', 'missing_startup_identity'])
def test_changed_saved_identity_is_rejected(receipts, mutation):
    folder = support.OLD_OUT / 'workers' / support.INTERRUPTED_WORKERS[0]
    initial = receipts[folder / 'initialization.json']
    last = receipts[folder / 'request-00000016.json']
    if mutation == 'wrong_freeze': initial['result']['freeze_sha256'] = '0' * 64
    elif mutation == 'wrong_guardian': receipts[support.OLD_OUT / 'watchdog_claim.json']['guardian_pid'] += 1
    elif mutation == 'wrong_native': initial['result']['native_pids']['search'] += 1
    elif mutation == 'changed_helper_identity': last['telemetry']['process_tree'][2]['created'] += 1
    elif mutation == 'helper_missing': last['telemetry']['process_tree'].pop()
    elif mutation == 'wrong_parent': last['telemetry']['process_tree'][2]['parent_pid'] += 1
    elif mutation == 'wrong_player_quota': initial['result']['request_quotas'] = [128]
    elif mutation == 'wrong_supervisor': receipts[folder / 'placement_admission.json']['supervisor']['created'] += 1
    elif mutation == 'duplicate_helper': last['telemetry']['process_tree'][2]['pid'] = last['telemetry']['process_tree'][3]['pid']
    elif mutation == 'missing_startup_identity': initial.pop('identity')
    with pytest.raises(ValueError): support.recorded_old_identities()


def test_actual_prior_counts_leave_no_slack_for_864():
    proof = support.old_tree_counts()
    assert proof['used_worker_trees'] == 2 + 486 == 488
    assert proof['maximum_new_worker_trees'] == 2 + 164 * 2 + 48 == 378
    assert proof['maximum_combined_worker_trees'] == 866
    assert proof['maximum_combined_worker_trees'] > 864
    assert proof['completed_graph_parities'] == proof['used_graph_builds'] - 1


def test_changed_prior_tree_counts_fail_closed(monkeypatch):
    original = support._tree_count
    def changed(path):
        value = original(path)
        if path == support.OLD_OUT: value['worker_trees'] += 1
        return value
    monkeypatch.setattr(support, '_tree_count', changed)
    with pytest.raises(ValueError, match='lifecycle counts'): support.old_tree_counts()


def test_identity_support_has_no_player_or_runtime_import():
    import ast
    tree = ast.parse(support.Path(support.__file__).read_text())
    imports = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
    names = [alias.name for node in imports for alias in node.names]
    assert not set(names) & {'torch', 'common', 'worker', 'runner', 'transport', 'adapters'}
