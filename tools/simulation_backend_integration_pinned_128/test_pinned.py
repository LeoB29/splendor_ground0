"""Checks for causal ordering and isolation of the controlled integration."""
from collections import Counter
import json

import pytest

from . import integration as h


def test_every_root_has_two_adjacent_opposite_order_pairs_before_any_fault():
    timed = h.SCHEDULE[:12]
    assert all(t['kind'] == 'collect' for t in timed)
    assert Counter((t['index'], t['backend']) for t in timed) == {
        (i, b): 2 for i in range(3) for b in ('reference', 'fast035')}
    first, second = timed[:6], timed[6:]
    assert [t['index'] for t in first] == [0, 0, 1, 1, 2, 2]
    assert [t['index'] for t in second] == [2, 2, 1, 1, 0, 0]
    for a, b in zip(first[::2], first[1::2]):
        assert a['index'] == b['index']
        assert (a['backend'], b['backend']) == ('reference', 'fast035')
    for a, b in zip(second[::2], second[1::2]):
        assert a['index'] == b['index']
        assert (a['backend'], b['backend']) == ('fast035', 'reference')
    assert len(h.SCHEDULE) == 18
    assert len({t['key'] for t in h.SCHEDULE}) == len(h.SCHEDULE)
    fault, recovery = h.SCHEDULE[12:14]
    assert fault['kind'] == 'interrupt' and recovery['kind'] == 'collect'
    assert fault['index'] == recovery['index'] == 1
    assert fault['backend'] == recovery['backend'] == 'fast035'
    assert [t['kind'] for t in h.SCHEDULE[14:]] == ['audit'] * 3 + ['summary']


def test_output_cannot_replace_main_or_any_prior_study():
    for name in ('structured_proposer_128', 'structured_proposer_128_recovery_01',
                 'simulation_backend_integration_128_01',
                 'simulation_backend_diagnosis_pinned_128_01',
                 'simulation_speed_validation_128_02'):
        with pytest.raises(ValueError):
            h.safe_output(h.ROOT / 'outputs' / name)
    with pytest.raises(ValueError):
        h.safe_output(h.ROOT / 'outputs/simulation_backend_integration_pinned_128_01/child')


class FakeProcess:
    pid = 12345

    def __init__(self):
        self.mask = [0, 1, 2, 3]
        self.changes = []

    def cpu_affinity(self, mask=None):
        if mask is not None:
            self.mask = list(mask)
            self.changes.append(list(mask))
        return list(self.mask)


@pytest.mark.parametrize('fail_inside', [False, True])
def test_pin_restores_only_supplied_worker_and_keeps_receipt_on_exception(tmp_path, fail_inside):
    own, untouched = FakeProcess(), FakeProcess()
    path = tmp_path / 'affinity.json'

    def execute():
        with h.pin_worker(0, path, process=own, current_processor=lambda: own.mask[0]):
            assert own.mask == [0]
            assert untouched.mask == [0, 1, 2, 3]
            if fail_inside:
                raise RuntimeError('deliberate simulated worker failure')

    if fail_inside:
        with pytest.raises(RuntimeError, match='deliberate simulated worker failure'):
            execute()
    else:
        execute()
    assert own.mask == [0, 1, 2, 3]
    assert own.changes == [[0], [0, 1, 2, 3]]
    assert not untouched.changes
    assert isinstance(json.loads(path.read_text(encoding='utf-8')), dict)


def test_pin_refuses_unavailable_core_without_altering_affinity(tmp_path):
    own = FakeProcess()
    with pytest.raises((RuntimeError, ValueError)):
        with h.pin_worker(12, tmp_path / 'affinity.json', process=own,
                          current_processor=lambda: own.mask[0]):
            pytest.fail('Worker must not run on a core outside its existing mask')
    assert own.mask == [0, 1, 2, 3]


@pytest.mark.parametrize('fail_inside', [False, True])
def test_dispatch_overrides_never_escape_the_new_process_scope(fail_inside):
    names = ('MODULE', 'SCHEDULE', 'load_plan', 'completed', 'finish', 'WorkerGuard', 'save', 'CONFIG')
    before = {name: getattr(h.h, name) for name in names}

    def execute():
        with h.controller_scope():
            assert h.h.MODULE == h.MODULE
            assert h.h.SCHEDULE is h.SCHEDULE
            assert h.h.load_plan is h.load_plan
            assert h.h.CONFIG is before['CONFIG']
            if fail_inside:
                raise RuntimeError('dispatch failure')

    if fail_inside:
        with pytest.raises(RuntimeError, match='dispatch failure'):
            execute()
    else:
        execute()
    for name, value in before.items():
        assert getattr(h.h, name) is value


def test_completion_holes_and_unknown_jobs_cannot_reorder_resume(tmp_path):
    folder = tmp_path / 'completed'
    folder.mkdir()
    assert h.validate_prefix(tmp_path) == 0
    first = folder / (h.SCHEDULE[0]['key'] + '.json')
    first.write_text('{}')
    assert h.validate_prefix(tmp_path) == 1
    hole = folder / (h.SCHEDULE[2]['key'] + '.json')
    hole.write_text('{}')
    with pytest.raises(ValueError, match='prefix'):
        h.validate_prefix(tmp_path)
    hole.unlink()
    (folder / 'unregistered.json').write_text('{}')
    with pytest.raises(ValueError, match='prefix'):
        h.validate_prefix(tmp_path)


def synthetic_completed(fast_seconds):
    return {t['key']: {
        'receipt': {'net_wall_seconds': (fast_seconds[t['repeat']] if t['backend'] == 'fast035' else 10)
                    if t.get('timed') else 1_000_000},
        'result': {'continuations': 128, 'cpu_seconds': 1, 'collection_seconds': 1,
                   'summary': {'fixture': True}}}
        for t in h.SCHEDULE}


@pytest.mark.parametrize('fast_seconds,accepted', [([7, 8], True), ([9, 11], False), ([12, 11], False), ([5, 12], True)])
def test_net_gate_pools_both_repeats_and_excludes_fault_recovery_audits(monkeypatch, tmp_path, fast_seconds, accepted):
    monkeypatch.setattr(h, 'completed', lambda *_: synthetic_completed(fast_seconds))
    monkeypatch.setattr(h, 'artifact', lambda path: {'path': str(path)})
    monkeypatch.setattr(h.h, 'active_used', lambda _: 999)
    result = h.finish(tmp_path, {'binding': 'fixture', 'target_cpu': {'logical_processor_index': 0}})
    assert result['net_reference_seconds'] == 60
    assert result['net_fast_seconds'] == 3 * sum(fast_seconds)
    assert result['optional_backend_accepted'] is accepted
    assert [r['net_fast_seconds'] for r in result['repeats']] == [3 * x for x in fast_seconds]
    assert len(result['pairs']) == 6
    assert result['clean_timed_collections'] == 12
    assert result['production_enabled'] is False


def test_auditor_root_identity_accepts_collector_metadata_but_not_changed_bytes():
    # These are the distinct artifact shapes produced by the unchanged real
    # collector and auditor. Only optional counters are absent from the latter.
    collector = {'path': 'attempt/data/test/roots/state_0000.json', 'sha256': 'sealed',
                 'index': 0, 'continuations': 1536, 'proof_continuations': 0}
    auditor = {'path': collector['path'], 'sha256': collector['sha256']}
    assert h.same_artifact_identity(collector, auditor)
    assert h.same_artifact_identity(auditor, collector)
    assert not h.same_artifact_identity(collector, {**auditor, 'sha256': 'changed'})
    assert not h.same_artifact_identity(collector, {**auditor, 'path': 'other.json'})
    with pytest.raises(KeyError):
        h.same_artifact_identity(collector, {'path': collector['path']})


def science_fixture():
    return {t['key']: {'result': {'task': t, 'science_sha256': 'root-' + str(t['index']),
                                 'row': {'fixture_root': t['index']}}}
            for t in h.SCHEDULE if t['kind'] == 'collect'}


@pytest.mark.parametrize('key', ['r0_fast035_0', 'r1_fast035_2', 'r1_reference_1', 'recovery_1'])
@pytest.mark.parametrize('field', ['science_sha256', 'row'])
def test_every_repeat_backend_and_recovery_must_match_in_science_and_statistics(key, field):
    found = science_fixture()
    h.validate_scientific_agreement(found)
    found[key]['result'][field] = 'changed'
    with pytest.raises(ValueError, match='scientific outputs differ'):
        h.validate_scientific_agreement(found)
