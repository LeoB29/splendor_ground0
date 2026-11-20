"""Synthetic, model-free checks for nested recovery provenance and RAM waiting."""
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import analyze_live_game_operational_recovery_20261020_02 as a


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def stamp(value): return datetime.fromtimestamp(value, timezone.utc).isoformat()


def memory_receipt(name='games/g021/t0011', start=100., before=0., count=0, low_first=False):
    initial = name == 'initial'; needs = initial or low_first
    times = [10.*i for i in range(14 if low_first else 13 if initial else 1)]
    mono_start = start+100.; rows = []
    for index, elapsed in enumerate(times):
        good = not low_first or index > 0
        rows.append({'epoch_seconds': start+elapsed, 'monotonic_seconds': mono_start+elapsed, 'elapsed_seconds': elapsed,
            'available_physical_bytes': 2*1024**3 if good else 1, 'available_commit_bytes': 4*1024**3,
            'admissible': good, 'consecutive_admissible_samples': index if low_first else index+1,
            'remaining_seconds': 40000.-elapsed})
    end = times[-1]+.001
    end_mono, end_epoch = mono_start+end, start+end
    charge = max(end_mono-mono_start, end_epoch-start, 0.)
    return {'schema': 'operational_memory_admission_v2', 'time': stamp(end_epoch), 'name': name,
        'status': 'passed', 'failure_category': None, 'failure_detail': None,
        'initial': initial, 'stability_required': needs, 'started_epoch_seconds': start,
        'finished_epoch_seconds': end_epoch, 'started_monotonic': mono_start, 'finished_monotonic': end_mono,
        'elapsed_seconds': end_mono-mono_start, 'charged_seconds': charge, 'prior_admission_count': count,
        'cumulative_before_seconds': before, 'cumulative_after_seconds': before+charge,
        'maximum_wait_seconds': 7200, 'maximum_cumulative_wait_seconds': 14400,
        'minimum_available_physical_bytes': 2*1024**3, 'minimum_available_commit_bytes': 4*1024**3,
        'sample_interval_seconds': 10, 'stable_seconds': 120, 'required_consecutive_samples': 13,
        'reserve_seconds': 600 if name == 'analysis' or name.startswith('audits/') else 21600,
        'request_seconds': 0 if name == 'analysis' else 600 if name.startswith('audits/') else 120,
        'initialization_seconds': 60, 'readings': rows, 'no_worker_during_admission': True, 'outcomes_closed': True}


def test_exact_three_segment_partition_retains_all_existing_player_failures():
    for game in a.core.GAMES:
        for ply in range(400):
            expected = ('original' if a.first.original_move(game, ply) else
                        'recovery01' if game in a.core.GAME_ORDER[:16] or game == 21 and ply < 11 else 'recovery02')
            assert a.segment_name(game, ply) == expected
    assert a.segment_name(11, 14) == 'original'
    assert a.segment_name(11, 15) == 'recovery01'
    assert a.segment_name(21, 10) == 'recovery01'
    assert a.segment_name(21, 11) == a.segment_name(23, 0) == 'recovery02'


@pytest.mark.parametrize('game,ply', [(True, 0), (21, True), (5, 0), (24, 0), (21, -1), (21, 400)])
def test_no_unplanned_or_boolean_move_identity(game, ply):
    with pytest.raises(ValueError): a.segment_name(game, ply)


def failure():
    return {'correct': False, 'failure_category': 'ResourceHold', 'id': 'games/g021/t0011', 'player': 'opponent',
        'status': 'incomplete', 'total_seconds': .1, 'observer': {'schema': 'external_screen_observation_v1',
        'observation': {'turn_index': 11, 'current_player': 1}}}


@pytest.mark.parametrize('damage', [None, 'init', 'request', 'completed', 'actor', 'turn', 'context', 'player'])
def test_second_unstarted_hold_is_not_a_retried_timed_decision(damage):
    value = failure()
    if damage in ('init', 'request'): value[damage] = {}
    elif damage == 'completed': value['status'] = 'completed'
    elif damage == 'actor': value['observer']['observation']['current_player'] = 0
    elif damage == 'turn': value['observer']['observation']['turn_index'] = 10
    elif damage == 'context': value['observer']['context'] = a.core.context_for(21, 11)
    elif damage == 'player': value['player'] = 'local'
    if damage is None: a.check_failure_attempt(value)
    else:
        with pytest.raises(ValueError): a.check_failure_attempt(value)


@pytest.mark.parametrize('name,low', [('initial', False), ('games/g021/t0011', False),
    ('games/g021/t0011', True), ('audits/games/g006/t0001', False), ('analysis', False), ('analysis', True)])
def test_fast_and_two_minute_stable_memory_paths(name, low):
    value = memory_receipt(name, low_first=low)
    assert a.check_memory_receipt(value, name, {'epoch_seconds': 90., 'monotonic_start': 190.}) == value['finished_epoch_seconds']
    assert len(value['readings']) == (14 if low else 13 if name == 'initial' else 1)


def test_backward_wall_adjustment_does_not_refund_monotonic_wait():
    value = memory_receipt('initial', start=1000.)
    for row in value['readings'][1:]: row['epoch_seconds'] -= 300.
    value['finished_epoch_seconds'] -= 300.
    value['charged_seconds'] = value['elapsed_seconds']
    value['cumulative_after_seconds'] = value['charged_seconds']
    a.check_memory_receipt(value, 'initial', {'epoch_seconds': 90., 'monotonic_start': 190.})
    assert value['charged_seconds'] >= 120.
    value['charged_seconds'] = value['cumulative_after_seconds'] = 0.
    with pytest.raises(ValueError):
        a.check_memory_receipt(value, 'initial', {'epoch_seconds': 90., 'monotonic_start': 190.})


@pytest.mark.parametrize('damage', ['threshold', 'commit', 'streak', 'spacing', 'short_stability', 'early_stop',
    'extra_sample', 'boolean', 'request', 'reserve', 'charged', 'clock_pause', 'episode', 'cumulative', 'worker',
    'wrong_name', 'failure', 'before_launch'])
def test_memory_gates_reject_relaxed_thresholds_clock_or_stability(damage):
    name = 'initial'; value = memory_receipt(name); launch = {'epoch_seconds': 90., 'monotonic_start': 190.}
    if damage == 'threshold': value['minimum_available_physical_bytes'] //= 2
    elif damage == 'commit': value['readings'][-1]['available_commit_bytes'] -= 1
    elif damage == 'streak': value['readings'][-1]['consecutive_admissible_samples'] -= 1
    elif damage == 'spacing': value['readings'][1]['monotonic_seconds'] -= 1
    elif damage == 'short_stability':
        value['readings'][-1]['monotonic_seconds'] -= .01
        value['readings'][-1]['elapsed_seconds'] -= .01
    elif damage == 'early_stop': value['readings'].pop()
    elif damage == 'extra_sample': value['readings'].append(deepcopy(value['readings'][-1]))
    elif damage == 'boolean': value['readings'][0]['available_physical_bytes'] = True
    elif damage == 'request': value['request_seconds'] = 0
    elif damage == 'reserve': value['readings'][0]['remaining_seconds'] = 21779.
    elif damage == 'charged': value['charged_seconds'] = 0.
    elif damage == 'clock_pause': value['finished_epoch_seconds'] += 100.
    elif damage == 'episode': value['charged_seconds'] = 7200.
    elif damage == 'cumulative': value['cumulative_before_seconds'] = 14400.
    elif damage == 'worker': value['no_worker_during_admission'] = False
    elif damage == 'wrong_name': value['name'] = 'analysis'
    elif damage == 'failure': value['failure_category'] = 'ResourceHold'
    else: launch['epoch_seconds'] = 101.
    with pytest.raises(ValueError): a.check_memory_receipt(value, name, launch)


@pytest.fixture
def paths(tmp_path, monkeypatch):
    original = tmp_path/'outputs/original'; source = tmp_path/'outputs/recovery01'; base = tmp_path/'outputs/recovery02'
    for path in (original, source, base): path.mkdir(parents=True)
    monkeypatch.setattr(a, 'ROOT', tmp_path); monkeypatch.setattr(a, 'SOURCE', source); monkeypatch.setattr(a, 'ORIGINAL', original)
    def desc(path):
        path = Path(path)
        return {'path': path.resolve().relative_to(tmp_path).as_posix(), 'sha256': a.file_hash(path), 'bytes': path.stat().st_size}
    monkeypatch.setattr(a, 'artifact', desc)
    audit = a.Audit(base); audit.launch = {'epoch_seconds': 90., 'monotonic_start': 190.}
    def bound(item, expected=None):
        path = (tmp_path/item['path']).resolve()
        a.require(path.is_relative_to(tmp_path) and (expected is None or path == Path(expected).resolve()) and desc(path) == item,
                  'Synthetic immutable artifact')
        return path
    for segment in audit.segments.values(): segment.bound = bound
    audit.bound = bound
    return audit, original, source, base, desc


def test_only_declared_transitive_source_aliases_are_accepted(paths):
    audit, original, source, base, desc = paths
    old, middle, new = original/'move.json', source/'move.json', base/'move.json'
    for path in (old, middle, new): write(path, {'unchanged': True})
    audit.imports[new] = desc(middle); audit.parent_imports[middle] = desc(old)
    for origin in (old, middle, new): assert audit.copied(desc(origin), new) == new
    unknown = original/'unrelated.json'; write(unknown, {'unchanged': True})
    with pytest.raises(ValueError): audit.copied(desc(unknown), new)
    del audit.parent_imports[middle]
    with pytest.raises(ValueError): audit.copied(desc(old), new)
    write(new, {'changed': True})
    with pytest.raises(ValueError): audit.copied(desc(middle), new)


def save_gate(audit, value, desc):
    path = audit.base/'memory_admissions'/(value['name']+'.json'); path.parent.mkdir(parents=True, exist_ok=True)
    log = path.with_suffix('.jsonl')
    log.write_text(''.join(json.dumps(row)+'\n' for row in value['readings']), encoding='utf-8')
    value['readings_log'] = {'path': log.relative_to(audit.base).as_posix(), 'sha256': a.file_hash(log), 'bytes': log.stat().st_size}
    write(path, value)
    return desc(path)


@pytest.mark.parametrize('damage', [None, 'wall_backward', 'cumulative', 'count', 'overlap', 'log', 'extra_gate', 'analysis_before_gate'])
def test_serial_gate_accounting_includes_analysis_and_rejects_unbound_waiting(paths, monkeypatch, damage):
    audit, _, _, _, desc = paths; before = 0.; start = 100.
    names = ['initial', 'games/g021/t0011', 'audits/games/g006/t0001', 'analysis']
    last = None
    for index, name in enumerate(names):
        value = memory_receipt(name, start=start, before=before, count=index)
        if damage == 'cumulative' and index == 2:
            value['cumulative_before_seconds'] += 1.; value['cumulative_after_seconds'] += 1.
        if damage == 'count' and index == 2: value['prior_admission_count'] = 1
        if damage == 'overlap' and index == 2:
            value = memory_receipt(name, start=start-2, before=before, count=index)
        if damage == 'wall_backward' and index >= 2:
            for key in ('started_epoch_seconds', 'finished_epoch_seconds'): value[key] -= 2.
            for reading in value['readings']: reading['epoch_seconds'] -= 2.
        descriptor = save_gate(audit, value, desc)
        if name != 'analysis': audit.memory_gate(descriptor, name)
        before += value['charged_seconds']; start = value['finished_monotonic']-100.+1.; last = value
    if damage == 'log': (audit.base/'memory_admissions/analysis.jsonl').write_text('{}\n', encoding='utf-8')
    if damage == 'extra_gate': save_gate(audit, memory_receipt('games/g023/t0000', start=start, before=before, count=4), desc)
    import psutil
    monkeypatch.setattr(psutil, 'Process', lambda: SimpleNamespace(create_time=lambda:
        last['finished_epoch_seconds']-1 if damage == 'analysis_before_gate' else start))
    if damage in (None, 'wall_backward'):
        audit.memory_inventory()
        assert audit.memory_summary['admissions'] == 4 and audit.memory_summary['stability_required_admissions'] == 1
        assert audit.memory_summary['charged_seconds'] == pytest.approx(before)
    else:
        with pytest.raises(ValueError): audit.memory_inventory()


@pytest.mark.parametrize('label,player,auditing,damage', [
    ('original', 'local', False, None), ('recovery01', 'local', False, None), ('recovery02', 'opponent', False, None),
    ('recovery02', 'audit', True, None), ('recovery01', 'local', False, 'module'),
    ('recovery02', 'local', False, 'module'), ('recovery02', 'local', False, 'before_gate')])
def test_three_receipt_modules_and_per_worker_gate_timestamps(paths, label, player, auditing, damage):
    audit, _, _, _, desc = paths; segment = audit.segments[label]
    audit.prior_admission_epoch = 200.; audit.boundary = lambda: None
    names = []
    def gate(descriptor, name): names.append(name); return 500.
    audit.memory_gate = gate
    name = 'g006-t0001' if auditing else 'games/g021/t0011'
    folder = segment.base/('audits' if auditing else 'workers')/name
    module = 'tools.splendorust_120_qualification_20261018.worker' if player == 'opponent' else segment.module
    created = 100. if label == 'original' else 300. if label == 'recovery01' else 600.
    if damage == 'before_gate': created = 499.
    frozen = {'executable': 'python.exe', 'topology': {'selected_logical_processor': 0}}
    init = {'status': 'ok', 'elapsed_seconds': 1., 'command': ['python.exe', '-B', '-u', '-m', module],
        'affinity': [0], 'topology': frozen['topology'], 'thread_environment': {k: '1' for k in a.core.THREADS},
        'identity': {'pid': 999, 'created': created}, 'telemetry': {'process_tree': [
            {'role': 'python', 'pid': 999, 'created': created, 'executable': 'python.exe'}]},
        'result': {'pid': 999, 'threads': 1}}
    if damage == 'module': init['command'][-1] = 'wrong.worker'
    receipt = {'status': 'ok', 'elapsed_seconds': 1., 'result': {'synthetic': True}}
    write(folder/'initialization.json', init); write(folder/'request-00000001.json', receipt)
    row = {'initialization': {k: v for k, v in init.items() if k not in a.core.METADATA and k != 'result'},
        'clock': {k: v for k, v in receipt.items() if k != 'result'}, 'memory_admission': {'synthetic': True},
        'initialization_artifact': desc(folder/'initialization.json'), 'receipt': desc(folder/'request-00000001.json')}
    segment.frozen = frozen; segment.cpu = 0; clocks = []
    segment.check_clock = lambda value, request, seconds, cpu: clocks.append(seconds)
    segment.process_identity = lambda *args: None
    if damage:
        with pytest.raises(ValueError): segment.receipts(row, name, player, auditing)
    else:
        segment.receipts(row, name, player, auditing)
        assert clocks == [60, 600 if auditing else 120]
        assert names == (['audits/games/g006/t0001' if auditing else name] if label == 'recovery02' else [])


@pytest.mark.parametrize('clock', ['wall', 'monotonic'])
def test_new_twelve_hour_bound_does_not_rewrite_prior_twenty_four_hour_bound(paths, monkeypatch, clock):
    audit = paths[0]
    monkeypatch.setattr(a.time, 'time', lambda: 90.+(43200 if clock == 'wall' else 1))
    monkeypatch.setattr(a.time, 'monotonic', lambda: 190.+(43200 if clock == 'monotonic' else 1))
    with pytest.raises(ValueError): audit.boundary()
    assert a.first.CAP == 86400


def test_failure_diagnostic_is_sealed_without_outcome_summary(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(a.sys, 'argv', ['analyzer', '--base', str(tmp_path)])
    def fail(self): raise ValueError('SEALED_SYNTHETIC_PAYLOAD')
    monkeypatch.setattr(a.Audit, 'run', fail)
    with pytest.raises(ValueError): a.main()
    assert 'SEALED_SYNTHETIC_PAYLOAD' in a.read(tmp_path/'sealed/independent_analysis_failure_01.json')['traceback']
    assert 'SEALED_SYNTHETIC_PAYLOAD' not in capsys.readouterr().out
