"""Model-free counterexamples for the 20 Nov launch-gate correction only.

Real-evidence tests are read-only: they open accepted receipts, never write
outside pytest's temporary folder, and never start a model, game or study.
"""
import json
import os
from pathlib import Path
from types import SimpleNamespace
import pytest
from . import common as c, owner, preparation, schedule


def accepted_plan():
    return dict(supersedes=preparation.original_unlaunched(),
                admission=dict(required_receipts=preparation.admission_receipts(),
                               accepted_collection_freeze=c.artifact(c.ACCEPTED_COLLECTION/'freeze.json'),
                               training_freeze=c.artifact(c.TRAINING_OUT/'freeze.json')))


@pytest.fixture
def no_other_studies(monkeypatch):
    monkeypatch.setattr(owner, 'other_study_processes', lambda: [])


def test_original_gate_reproduces_the_refusal_on_real_files():
    """The defect being corrected: the frozen 15 Nov gate cannot pass today."""
    from tools.splendorust_parallel_throughput_20261115 import common as old_c, owner as old_owner
    plan = old_c.read(old_c.PLAN)
    authorization = dict(explicit_user_launch=True, machine_quiet_confirmed=True, plan=old_c.artifact(old_c.PLAN))
    with pytest.raises(ValueError, match='Collection still incomplete: outputs/splendorust_collection4096_20261115_01/'):
        old_owner.launch_admission(plan, authorization)


def test_corrected_gate_passes_on_real_accepted_evidence(no_other_studies):
    result = owner.accepted_predecessors(accepted_plan())
    assert result['status'] == 'passed' and result['receipts'] == 6


def test_plan_binds_only_existing_accepted_receipts():
    receipts = preparation.admission_receipts()
    paths = [r['path'] for r in receipts]
    assert len(paths) == len(set(paths)) == 6
    assert all((c.ROOT/p).exists() for p in paths)
    assert not any(p.startswith('outputs/splendorust_collection4096_20261115_01/') for p in paths)
    assert {'outputs/splendorust_collection4096_resume_20261118_02/final_audit/worker/result.json',
            'outputs/splendorust_training4096_20261120_01/observed_launcher_exit.json'} <= set(paths)


CASES = [(i, key) for i, receipt in enumerate(preparation.admission_receipts()) for key in receipt['equals']]


@pytest.mark.parametrize('index,key', CASES)
def test_every_bound_receipt_value_is_enforced(monkeypatch, no_other_studies, index, key):
    target = c.normalized(c.ROOT/preparation.admission_receipts()[index]['path'])
    real_read = c.read

    def altered(path):
        value = real_read(path)
        if c.normalized(path) == target:
            value = dict(value)
            value[key] = 'tampered'
        return value
    monkeypatch.setattr(c, 'read', altered)
    with pytest.raises(ValueError, match='Predecessor acceptance/exit'):
        owner.accepted_predecessors(accepted_plan())


def test_missing_receipt_refuses_as_incomplete(no_other_studies):
    plan = accepted_plan()
    plan['admission']['required_receipts'].append(dict(path='outputs/does_not_exist_20261120/observed_owner_exit.json', equals={}))
    with pytest.raises(ValueError, match='Predecessor still incomplete'):
        owner.accepted_predecessors(plan)


def test_missing_bound_key_refuses_even_when_expected_value_is_none(monkeypatch, no_other_studies):
    target = c.normalized(c.ACCEPTED_COLLECTION/'final_audit/worker/result.json')
    real_read = c.read

    def without_error(path):
        value = real_read(path)
        if c.normalized(path) == target:
            value = {k: v for k, v in value.items() if k != 'error'}
        return value
    monkeypatch.setattr(c, 'read', without_error)
    with pytest.raises(ValueError, match='Predecessor acceptance/exit: .*final_audit/worker/result.json error'):
        owner.accepted_predecessors(accepted_plan())


@pytest.mark.parametrize('which', ['collection_launcher', 'collection_observer', 'collection_root',
                                   'training_launcher', 'training_observer', 'training_root'])
def test_surviving_predecessor_process_refuses(monkeypatch, no_other_studies, which):
    collection = c.read(c.ACCEPTED_COLLECTION_LAUNCH/'observed_launcher_exit.json')
    observation = c.read(c.ACCEPTED_COLLECTION_LAUNCH/'launch_observation.json')
    croot = c.read(c.ACCEPTED_COLLECTION/'launch/execution.json')
    training = c.read(c.TRAINING_OUT/'observed_launcher_exit.json')
    texit = c.read(c.TRAINING_OUT/'launch/actual_exit.json')
    alive = dict(collection_launcher=collection['launcher_identity'], collection_observer=observation['observer_identity'],
                 collection_root=croot['observed_owned_identities'][-1],
                 training_launcher=training['launcher_identity'],
                 training_observer=training['observer_identity'], training_root=texit['identity'])[which]
    real = owner.require_absent

    def check(identity):
        if identity == alive:
            raise ValueError('PARALLEL_THROUGHPUT: Exact previous process still alive')
        return real(identity)
    monkeypatch.setattr(owner, 'require_absent', check)
    with pytest.raises(ValueError, match='still alive'):
        owner.accepted_predecessors(accepted_plan())


def fake_processes(monkeypatch, rows, me=10, parents=(9, 8)):
    monkeypatch.setattr(owner.psutil, 'Process', lambda *a: SimpleNamespace(
        pid=me, parents=lambda: [SimpleNamespace(pid=p) for p in parents]))
    monkeypatch.setattr(owner.psutil, 'process_iter', lambda attrs: iter(
        [SimpleNamespace(info=dict(pid=pid, cmdline=line)) for pid, line in rows]))


def test_other_study_process_blocks_launch(monkeypatch):
    fake_processes(monkeypatch, [
        (10, ['python', '-m', c.PACKAGE+'.launcher']),                       # this launcher
        (9, ['powershell', '-Command', 'python -m tools.splendorust_x.run']),  # its parent shell
        (20, ['python', '-m', c.PACKAGE+'.owner']),                          # same pilot package
        (21, ['C:\\py\\python.exe', 'C:\\repo\\tools\\splendorust_parallel_throughput_20261120\\fixture.py']),
        (22, None), (23, ['firefox.exe']),
        (30, ['python', '-m', 'tools.splendorust_training4096_20261120.owner']),
        (31, ['python', 'C:\\repo\\tools\\splendorust_parallel_throughput_20261115\\launcher.py'])])
    found = owner.other_study_processes()
    assert [x['pid'] for x in found] == [30, 31]


def test_own_package_exemption_is_exact(monkeypatch):
    fake_processes(monkeypatch, [
        (40, ['python', '-m', c.PACKAGE+'_02.owner']),
        (41, ['python', 'C:\\repo\\tools\\splendorust_parallel_throughput_20261120x\\owner.py']),
        (42, ['python', '-m', c.PACKAGE])])
    assert [x['pid'] for x in owner.other_study_processes()] == [40, 41]


def test_study_started_during_scan_refuses_before_any_claim(tmp_path, monkeypatch):
    from . import topology
    monkeypatch.setattr(c, 'OUT', tmp_path/'out')
    monkeypatch.setattr(c, 'PLAN', c.OUT/'freeze.json')
    plan = dict(design=preparation.design(), live_schedule=[], collection=dict(games=dict(specifications=[]),
                engineering_games=[]), inputs=[], topology={}, placement={}, **c.IDENTITY)
    (tmp_path/'out').mkdir()
    c.PLAN.write_text(json.dumps(plan), encoding='utf-8')
    monkeypatch.setattr(c, 'artifact', lambda path: dict(path=str(path), bytes=0, sha256=c.sha(path)))
    order = []
    monkeypatch.setattr(owner, 'accepted_predecessors', lambda p: order.append('predecessors'))
    monkeypatch.setattr(schedule, 'registry_inventory', lambda t: order.append('scan'))
    monkeypatch.setattr(topology, 'verify_startup', lambda t, p: order.append('topology'))
    monkeypatch.setattr(owner, 'other_study_processes', lambda: [dict(pid=50, cmdline='tools.splendorust_new.owner')])
    authorization = dict(explicit_user_launch=True, machine_quiet_confirmed=True, plan=c.artifact(c.PLAN))
    with pytest.raises(ValueError, match='became active'):
        owner.launch_admission(plan, authorization)
    assert order == ['predecessors', 'scan', 'topology']
    assert sorted(p.name for p in (tmp_path/'out').iterdir()) == ['freeze.json']


def reservation_workspace(tmp_path, monkeypatch):
    from .live.schedule import expected_spec
    monkeypatch.setattr(c, 'OUT', tmp_path/'out')
    monkeypatch.setattr(c, 'PREP', tmp_path/'prep')
    rows = dict(live=[expected_spec(i) for i in range(16)], collection_schedule=schedule.collection_specs(),
                qualification_schedule=schedule.qualification_specs(), seed_inventory=dict(status='passed'))
    reservation = {k: c.save(c.OUT/'reserved'/(k+'.json'), v) for k, v in rows.items()}
    c.save(c.OUT/'prepared.json', dict(reservation=reservation))
    return reservation


def test_final_validation_binds_the_frozen_reservation(tmp_path, monkeypatch):
    reservation = reservation_workspace(tmp_path, monkeypatch)
    seen = []
    monkeypatch.setattr(schedule, 'registry_inventory', lambda targets, boundary: seen.append(len(targets)) or
                        dict(checked=[dict(bytes=7)], collisions=[], unicode_semantic_fallbacks=[], discovery='walk'))
    item = schedule.final_validation()
    receipt = c.read(c.resolve(item))
    assert seen == [50] and receipt['status'] == 'passed' and receipt['target_count'] == 50
    assert receipt['seed_inventory'] == reservation['seed_inventory'] and receipt['collisions'] == []


def test_final_validation_collision_saves_no_passing_receipt(tmp_path, monkeypatch):
    reservation_workspace(tmp_path, monkeypatch)

    def collide(targets, boundary):
        raise ValueError('PARALLEL_THROUGHPUT: Setup collision; never redraw')
    monkeypatch.setattr(schedule, 'registry_inventory', collide)
    with pytest.raises(ValueError, match='collision'):
        schedule.final_validation()
    assert not (tmp_path/'prep'/'reservation_final_validation.json').exists()


def test_python_discovery_skips_junctions(tmp_path, monkeypatch):
    import _winapi
    monkeypatch.setattr(c, 'ROOT', tmp_path)
    outside = tmp_path/'elsewhere'
    outside.mkdir()
    (outside/'x_freeze.json').write_text('{}', encoding='utf-8')
    (tmp_path/'outputs').mkdir()
    (tmp_path/'outputs'/'own_freeze.json').write_text('{}', encoding='utf-8')
    try:
        _winapi.CreateJunction(str(outside), str(tmp_path/'outputs'/'link'))
    except OSError as error:
        pytest.skip('junction creation unavailable: '+repr(error))
    assert schedule.registered_files() == ['outputs/own_freeze.json']


def current_console():
    return c.artifact((Path(os.environ['SystemRoot'])/'System32/conhost.exe').resolve())


def test_console_pin_follows_current_system32_host():
    prior = c.read(c.ROOT/'outputs/splendorust_sixworld512_20261114_02/freeze.json')
    inherited = prior['console_executable']
    assert inherited in prior['worker_inputs']
    plan = preparation.rebind_console(dict(console_executable=dict(inherited),
                                           inherited_worker_inputs=list(prior['worker_inputs']),
                                           worker_inputs=list(prior['worker_inputs'])))
    assert plan['console_executable'] == current_console()
    for key in ('inherited_worker_inputs', 'worker_inputs'):
        assert current_console() in plan[key] and len(plan[key]) == len(prior['worker_inputs'])
        assert [x for x in plan[key] if x != current_console()] == [x for x in prior['worker_inputs'] if x != inherited]
    preparation.unique(plan['worker_inputs']+list(c.nested_artifacts(plan)))
    stale = [a for a in c.nested_artifacts(plan) if a['sha256'] == inherited['sha256'] and a != current_console()]
    assert stale == []
    if current_console() != inherited:
        assert plan['console_rebinding']['inherited_sha256'] == inherited['sha256']
        assert plan['console_rebinding']['current_sha256'] == current_console()['sha256']


def test_console_pin_unchanged_records_no_rebinding():
    plan = preparation.rebind_console(dict(console_executable=current_console()))
    assert plan == dict(console_executable=current_console())


def test_console_pin_refuses_any_other_executable(tmp_path):
    other = dict(path=str(tmp_path/'conhost.exe'), bytes=1, sha256='0'*64)
    with pytest.raises(ValueError, match='Same pinned System32 console host path'):
        preparation.rebind_console(dict(console_executable=other))


def test_note_rebinding_binds_final_bytes_and_keeps_prepared_hash_inert():
    current = c.artifact(c.CORRECTION)
    stale = dict(current, sha256='0'*64, bytes=1)
    plan = preparation.rebind_note(dict(correction=stale))
    assert plan['correction'] == current
    assert plan['correction_at_preparation']['prepared_sha256'] == '0'*64
    assert list(c.nested_artifacts(plan)) == [current]
    assert preparation.rebind_note(dict(correction=current)) == dict(correction=current)
    with pytest.raises(ValueError, match='Same correction note path'):
        preparation.rebind_note(dict(correction=dict(current, path='docs/other.md')))


def test_active_other_study_refuses_admission(monkeypatch):
    monkeypatch.setattr(owner, 'other_study_processes', lambda: [dict(pid=30, cmdline='tools.splendorust_x')])
    with pytest.raises(ValueError, match='Another study process is active'):
        owner.accepted_predecessors(accepted_plan())


def test_superseded_original_must_stay_unlaunched(tmp_path, monkeypatch):
    frozen = tmp_path/'freeze.json'
    frozen.write_text('{"frozen":true}\n', encoding='utf-8')
    monkeypatch.setattr(c, 'ORIGINAL_OUT', tmp_path)
    monkeypatch.setattr(c, 'ORIGINAL_FREEZE_SHA', c.sha(frozen))
    monkeypatch.setattr(c, 'artifact', lambda path: dict(path=str(path), bytes=1, sha256=c.sha(path)))
    assert preparation.original_unlaunched()['launched'] is False
    (tmp_path/'launcher_claim.json').write_text('{}', encoding='utf-8')
    with pytest.raises(ValueError, match='never launched: launcher_claim.json'):
        preparation.original_unlaunched()
    (tmp_path/'launcher_claim.json').unlink()
    frozen.write_text('{"frozen":false}\n', encoding='utf-8')
    with pytest.raises(ValueError, match='freeze unchanged'):
        preparation.original_unlaunched()


def test_gate_order_unchanged_before_predecessor_reads(monkeypatch):
    called = []
    monkeypatch.setattr(owner, 'accepted_predecessors', lambda plan: called.append(1))
    plan = dict(design=preparation.design(), **c.IDENTITY)
    with pytest.raises(ValueError, match='Later explicit'):
        owner.launch_admission(plan, {})
    assert called == []


def test_python_discovery_matches_rg_scope(tmp_path, monkeypatch):
    monkeypatch.setattr(c, 'ROOT', tmp_path)
    files = {'outputs/a/x_manifest_1.json': True, 'outputs/c/deep/d/selection_x.json': True,
             'outputs/b/freeze.json': True, 'outputs/b/notes.json': False, 'outputs/b/Freeze.JSON': False,
             'outputs/b/.z_freeze.json': False, 'outputs/.hidden/y_schedule.json': False,
             'outputs/b/schedule.json.gz': False, 'outputs/b/reservation_receipt.json': True}
    for name in files:
        (tmp_path/name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path/name).write_text('{}', encoding='utf-8')
    assert schedule.registered_files() == sorted(n for n, keep in files.items() if keep)


def test_python_discovery_covers_every_original_rg_registered_file():
    """Real registry: nothing the 15 Nov rg scan checked is missed today."""
    original = c.read(c.ORIGINAL_OUT/'reserved/seed_inventory.json')
    earlier = [x['path'] for x in original['checked']]
    assert len(earlier) == 9342
    now = set(schedule.registered_files())
    missed = [p for p in earlier if (c.ROOT/p).exists() and p not in now]
    assert missed == []
