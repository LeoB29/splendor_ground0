"""No models, games, performance loops, or real affinity changes."""
from copy import deepcopy

import pytest

from . import topology as t


def fixture_topology():
    cores = [{'core_index': index, 'flags': 1, 'efficiency_class': 1,
              'logical_processors': [2 * index, 2 * index + 1]} for index in range(6)]
    cores += [{'core_index': cpu - 6, 'flags': 0, 'efficiency_class': 0,
               'logical_processors': [cpu]} for cpu in range(12, 20)]
    rows = [{'id': 256 + cpu, 'group': 0, 'logical_processor_index': cpu,
             'core_index': min(core['logical_processors']),
             'efficiency_class': core['efficiency_class']}
            for core in cores for cpu in core['logical_processors']]
    return dict(api=t.API, cpu_set_api=t.CPU_SET_API, cores=cores,
                cpu_sets=rows, allowed_logical_processors=list(range(20)))


def test_selects_physical_cores_not_siblings():
    placement = t.choose_placement(fixture_topology())
    assert placement['model_cpus'] == [0, 2, 4, 6]
    assert placement['audit_cpus'] == [8]
    assert placement['controller_cpus'] == list(range(12, 20))
    assert placement['validation']['model_physical_masks'] == [[0, 1], [2, 3], [4, 5], [6, 7]]


@pytest.mark.parametrize('bad_role,bad_value', [('model_cpus', [0, 1, 4, 6]),
                                              ('audit_cpus', [7]),
                                              ('controller_cpus', [10])])
def test_invalid_physical_placement(bad_role, bad_value):
    host = fixture_topology()
    placement = t.choose_placement(host)
    placement[bad_role] = bad_value
    with pytest.raises(ValueError):
        t.validate_placement(placement, host)


def test_api_ordinals_may_differ_but_membership_must_match():
    host = fixture_topology()
    assert t.validate_topology(host)['physical_core_count'] == 14
    host['cpu_sets'][1]['core_index'] = 999
    with pytest.raises(ValueError, match='APIs disagree'):
        t.validate_topology(host)


@pytest.mark.parametrize('mutation', ['overlap', 'duplicate', 'class', 'group', 'smt', 'unknown_allowed'])
def test_rejects_inconsistent_topology(mutation):
    host = fixture_topology()
    if mutation == 'overlap': host['cores'][1]['logical_processors'] = [0, 3]
    if mutation == 'duplicate': host['cpu_sets'].append(deepcopy(host['cpu_sets'][0]))
    if mutation == 'class': host['cpu_sets'][1]['efficiency_class'] = 0
    if mutation == 'group': host['cpu_sets'][0]['group'] = 1
    if mutation == 'smt': host['cores'][0]['flags'] = 0
    if mutation == 'unknown_allowed': host['allowed_logical_processors'].append(32)
    with pytest.raises(ValueError):
        t.validate_topology(host)


def test_restricted_original_mask_requires_five_available_pcores():
    host = fixture_topology()
    host['allowed_logical_processors'] = [0, 2, 4, 6, 12]
    with pytest.raises(ValueError, match='Five distinct'):
        t.choose_placement(host)
    host['allowed_logical_processors'].append(9)
    assert t.choose_placement(host)['audit_cpus'] == [9]


def test_startup_requires_unchanged_map_and_original_mask(monkeypatch):
    original = fixture_topology()
    placement = t.choose_placement(original)
    current = deepcopy(original)
    monkeypatch.setattr(t, 'read_topology', lambda: current)
    assert t.verify_startup(original, placement) == original
    current['allowed_logical_processors'] = list(range(12, 20))
    with pytest.raises(ValueError, match='before narrowing'):
        t.verify_startup(original, placement)


class FakeProcess:
    def __init__(self):
        self.mask = [12, 13]
        self.history = []

    def cpu_affinity(self, mask=None):
        if mask is not None:
            self.mask = list(mask)
            self.history.append(self.mask)
        return list(self.mask)


def test_live_then_fresh_replay_admission_and_restore():
    process = FakeProcess()
    for pin in (0, 8):
        with t.service_launch_affinity(list(range(20)), [12, 13], [pin], process):
            assert pin in process.cpu_affinity()
        assert process.cpu_affinity() == [12, 13]
    assert process.history == [list(range(20)), [12, 13], list(range(20)), [12, 13]]


def test_restore_after_failed_admission():
    process = FakeProcess()
    with pytest.raises(RuntimeError, match='constructor failure'):
        with t.service_launch_affinity(list(range(20)), [12, 13], [6], process):
            raise RuntimeError('constructor failure')
    assert process.cpu_affinity() == [12, 13]


def test_invalid_masks_change_nothing():
    process = FakeProcess()
    with pytest.raises(ValueError):
        with t.service_launch_affinity([12, 13], [12, 13], [6], process):
            pass
    assert process.history == []
