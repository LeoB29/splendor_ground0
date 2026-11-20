"""Model-free controller regressions for segment order, preservation and clocks."""
import ast
from copy import deepcopy
import importlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

os.environ.setdefault('INDEPENDENT_DATA_REPETITION', '0')
control = importlib.import_module('tools.splendorust_value_targets_20261110.strength.control')
c = control.c
ROOT = Path(__file__).resolve().parents[1]


def test_exact_reused_game_transport_replay_functions():
    old = ast.parse((ROOT / 'tools/splendorust_selfplay_pilot_resume2_20261103/assessment/control.py').read_text())
    new = ast.parse((ROOT / 'tools/splendorust_value_targets_20261110/strength/control.py').read_text())
    before = {n.name: n for n in old.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    after = {n.name: n for n in new.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    names = ('actual_after', 'decision', 'validate_row', 'play_game', 'expected_worker_plan', 'read_games',
             'SavedRows', 'replay_selected', 'verify_replays', 'close_workers', 'collect_main',
             'load_inputs', 'current_worker_names', 'source_move_path', 'game_folder', 'verify_closed_before')
    for name in names:
        assert ast.dump(before[name], include_attributes=False) == ast.dump(after[name], include_attributes=False), name


def test_declared_qualification_and_ordinary_lifecycle_bounds():
    rows = control.lifecycle_inventory()
    assert len(rows) == 406
    assert set(n for n in rows if n.startswith('qualification_')) == {'qualification_control', 'qualification_candidate'}
    assert rows['qualification_candidate']['ordinary_requests'] == 0
    assert rows['main_g000_control']['ordinary_requests'] == 10000
    assert rows['main_g003_control']['ordinary_requests'] == 7000
    assert rows['main_g199_control']['ordinary_requests'] == 1000
    assert rows['replay_control']['ordinary_requests'] == 96
    assert all(v['startup_requests'] == 2 and v['quota'] == 128 for v in rows.values())


@pytest.mark.parametrize('phase,index', [('main_block', True), ('main_block', 0.0), ('main_block', -10),
    ('main_block', 1), ('main_block', 200), ('operational', 0), ('saved', False), (False, None), ('other', None)])
def test_segment_identity_rejects_bad_types_and_nonfixed_indices(phase, index):
    with pytest.raises(ValueError):
        control.segment_id(phase, index)


def test_segment_inventory_is_fixed_and_bounded():
    rows = control._expected_segments()
    assert len(rows) == 23
    assert rows[:2] == [('qualification', None), ('operational', None)]
    assert rows[-1] == ('replay', None)
    assert [start for phase, start in rows if phase == 'main_block'] == list(range(0, 200, 10))
    assert control.segment_id('main_block', 190) == 'main_190'


def test_contract_no_forecast_or_ram_stops():
    value = control.contract()
    for key in ('initial_available_ram_bytes', 'minimum_available_ram_bytes', 'initial_ram_stability_seconds',
                'maximum_ram_wait_seconds', 'automatic_retries', 'failed_decision_retries', 'game_replacements'):
        assert value[key] == 0
    assert value['resource_forecasts_may_stop'] is False
    assert value['one_second_gate'] is False
    assert value['seven_hundred_ms_gate'] is False
    assert value['main_games'] == 200 and value['normal_lifecycles'] == 46
    assert value['seconds_per_move'] == 10 and value['internal_seconds'] == 9


def test_freeze_repetition_bool_rejected():
    value = dict(control.contract(), profiles={'control': {}, 'candidate': {}})
    assert control.validate_freeze_contract(value)
    value['repetition'] = bool(c.REPETITION)
    with pytest.raises(ValueError, match='Strict fixed repetition'):
        control.validate_freeze_contract(value)


@pytest.mark.parametrize('case', ['quota_float', 'stage_float', 'ordinary_bool'])
def test_freeze_contract_rejects_python_equal_type_aliases(case):
    value = dict(control.contract(), profiles={'control': {}, 'candidate': {}})
    if case == 'quota_float':
        value['routes']['candidate'] = 128.0
    elif case == 'stage_float':
        value['stage_seconds'] = dict(value['stage_seconds'], main=float(value['stage_seconds']['main']))
    else:
        value['declared_worker_lifecycles']['qualification_control']['ordinary_requests'] = False
    with pytest.raises(ValueError, match='Exact typed'):
        control.validate_freeze_contract(value)


def test_envelope_filtered_view_retains_global_guard(monkeypatch, tmp_path):
    monkeypatch.setattr(c, 'OUT', tmp_path / 'r0' / 'assessment')
    own = c.OUT / 'workers' / 'main_g000_control'
    peer = tmp_path / 'r1' / 'assessment' / 'workers' / 'main_g000_control'
    calls = []
    root = SimpleNamespace(evidence={'baseline': 17, 'workers': {own: 123, peer: 456}},
                           resident={'actual_global_owner': 1}, guard=lambda: calls.append('global'))
    view = control.AssessmentEnvelope(root)
    assert view.evidence == {'baseline': 17, 'workers': {own: 123}}
    assert view.resident == root.resident
    view.guard()
    assert calls == ['global']


def test_main_forfeit_closes_workers_before_next_fixed_game(monkeypatch):
    starts, ends, played = [], [], []
    workers = {}
    specs = [{'stage': 'main', 'game_index': i} for i in range(10)]

    def begin(spec, current, *_):
        assert not current
        starts.append(spec['game_index'])
        current['pair'] = spec['game_index']

    def play(spec, current, *_):
        assert current
        played.append(spec['game_index'])
        return dict(spec, reason='forfeit' if spec['game_index'] == 2 else 'natural')

    def close(current, *_):
        ends.append(played[-1])
        current.clear()

    monkeypatch.setattr(control, 'begin_block', begin)
    monkeypatch.setattr(control, 'play_game', play)
    monkeypatch.setattr(control, 'close_workers', close)
    result = control.collect_main(specs, workers, None, None, None)
    assert starts == [0, 3] and ends == [2, 9]
    assert played == list(range(10)) and len(result) == 10 and not workers


def test_both_operational_cohorts_required(monkeypatch, tmp_path):
    monkeypatch.setattr(c, 'STUDY_OUT', tmp_path)
    values = {}
    for rep in (0, 1):
        path = tmp_path / 'strength' / f'r{rep}' / 'segments' / 'operational' / 'complete.json'
        values[path] = dict(status='segment_complete', repetition=rep, games=6, resident_workers_after=0)
    monkeypatch.setattr(c, 'read', lambda path: deepcopy(values[Path(path)]))
    monkeypatch.setattr(c, 'artifact', lambda path: {'path': str(path)})
    assert len(control._both_operational_passed()) == 2
    values[next(p for p in values if 'r1' in p.parts)]['games'] = 5
    with pytest.raises(ValueError, match='Both six-game'):
        control._both_operational_passed()


def _fake_segment_history(monkeypatch, tmp_path):
    out = tmp_path / 'assessment'
    monkeypatch.setattr(c, 'OUT', out)
    monkeypatch.setattr(c, 'ROOT', tmp_path)
    def read(path):
        return json.loads(Path(path).read_text())
    def artifact(path):
        return {'path': Path(path).relative_to(tmp_path).as_posix(), 'bytes': Path(path).stat().st_size,
                'sha256': 'fixed_' + Path(path).name}
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
    monkeypatch.setattr(c, 'read', read)
    monkeypatch.setattr(c, 'artifact', artifact)
    monkeypatch.setattr(c, 'verify', lambda items: None)
    monkeypatch.setattr(control, '_absent', lambda identity: True)
    write(out / 'freeze.json', {'repetition': c.REPETITION})
    claim_path = tmp_path / 'claim.json'
    write(claim_path, {'monotonic_origin': 1000., 'unix_origin': 100000.})
    claim = artifact(claim_path)
    env = SimpleNamespace(start=1000., wall_start=100000., claim=claim, guard=lambda: None)
    clocks = {}
    for phase, delta in [('strength', 0), ('strength_replay', 3000)]:
        cap = c.parent.STAGES[phase]
        value = dict(stage=phase, claim=claim, cap_seconds=cap, monotonic=1000. + delta, unix_time=100000. + delta,
                     monotonic_deadline=min(1000. + delta + cap, 1000. + c.COLLECTION_CUTOFF_SECONDS),
                     unix_deadline=min(100000. + delta + cap, 100000. + c.COLLECTION_CUTOFF_SECONDS))
        path = tmp_path / 'stages' / f'{phase}_start.json'
        write(path, value)
        clocks[phase] = (path, value)
    for number, (phase, block_start) in enumerate(control._expected_segments()):
        name = control.segment_id(phase, block_start)
        clock_name = c.STAGE_MAP['main' if phase == 'main_block' else phase]
        path, clock = clocks[clock_name]
        offset = 10 if phase == 'qualification' else 20 if phase == 'operational' else 30 + (block_start or 0)
        start = dict(segment=name, phase=phase, repetition=c.REPETITION, block_start=block_start,
            local_freeze=artifact(out / 'freeze.json'), owner={'pid': 98765, 'created': 123.}, active_decision=False,
            deadline_reset=False, root_claim=claim, original_monotonic_origin=1000., original_unix_origin=100000.,
            root_phase_clock=artifact(path), segment_claim=None, monotonic=clock['monotonic'] + offset,
            unix_time=clock['unix_time'] + offset, monotonic_deadline=clock['monotonic_deadline'],
            unix_deadline=clock['unix_deadline'])
        start_path = out / 'segments' / name / 'start.json'
        write(start_path, start)
        end = dict(start, status='segment_complete', resident_workers_after=0, failed_decision_retries=0,
            game_replacements=0, start=artifact(start_path), games=6 if phase == 'operational' else 10 if phase == 'main_block' else 0,
            replays=12 if phase == 'replay' else 0, monotonic=start['monotonic'] + 1, unix_time=start['unix_time'] + 1)
        write(out / 'segments' / name / 'complete.json', end)
    return env, out, read, write


def test_saved_segment_clocks_preserve_one_root_origin(monkeypatch, tmp_path):
    env, out, read, write = _fake_segment_history(monkeypatch, tmp_path)
    assert len(control.verify_segments(env, {})) == 23
    path = out / 'segments' / 'main_010' / 'start.json'
    value = read(path)
    value['original_monotonic_origin'] += 1
    write(path, value)
    with pytest.raises(ValueError, match='original whole-study origins'):
        control.verify_segments(env, {})


def test_saved_segment_cannot_extend_deadline(monkeypatch, tmp_path):
    env, out, read, write = _fake_segment_history(monkeypatch, tmp_path)
    path = out / 'segments' / 'main_010' / 'start.json'
    value = read(path)
    value['monotonic_deadline'] += 10
    write(path, value)
    with pytest.raises(ValueError, match='nonreset root phase'):
        control.verify_segments(env, {})


def test_saved_segment_fails_on_still_live_owner(monkeypatch, tmp_path):
    env, _, _, _ = _fake_segment_history(monkeypatch, tmp_path)
    monkeypatch.setattr(control, '_absent', lambda identity: False)
    with pytest.raises(ValueError, match='controller actual identity still alive'):
        control.verify_segments(env, {})


def test_qualification_no_forecast_refusal():
    node = ast.parse((ROOT / 'tools/splendorust_value_targets_20261110/strength/control.py').read_text())
    qualify = next(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == 'qualify')
    assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == 'remaining'
                   for n in ast.walk(qualify))


def test_worker_to_segment_binding():
    assert control._worker_segment('qualification_control') == 'qualification'
    assert control._worker_segment('operational_g000_candidate') == 'operational'
    assert control._worker_segment('main_g013_candidate') == 'main_010'
    assert control._worker_segment('replay_control') == 'replay'


def test_prepare_wires_exact_model_free_import_without_a_fit(monkeypatch, tmp_path):
    from tools.splendorust_value_targets_20261110.strength import checkpoint
    monkeypatch.setattr(c, 'OUT', tmp_path)
    monkeypatch.setattr(control.design, 'prepare', lambda: {'schedule': {'path': 'fixed'},
        'initial_states': {'path': 'initials'}, 'replay_rule': {'path': 'rule'}})
    monkeypatch.setattr(control, 'build_plan', lambda: {'qualification': {'fixed': True}})
    monkeypatch.setattr(c, 'save', lambda path, value: {'path': str(path), 'saved': value})
    imported = dict(candidate_manifest={'path': 'accepted96'},
        checkpoint_bindings={'candidate': {'native_reference': {'path': 'exact_state'}}, 'control': {'checkpoint': 'parent'}},
        export_proof={'path': 'source_byte_proof'}, parent_export_proof={'path': 'parent_byte_proof'})
    monkeypatch.setattr(checkpoint, 'prepare_candidate', lambda **kwargs: deepcopy(imported))
    seen = []
    def profiles(manifest, reference):
        seen.append((manifest, reference))
        return {'profiles': {'control': 'parent', 'candidate': 'onepass'}}
    monkeypatch.setattr(control.adapters, 'prepare_profiles', profiles)
    value = control.prepare()
    assert seen == [({'path': 'accepted96'}, {'path': 'exact_state'})]
    assert value['candidate_manifest'] == imported['candidate_manifest']
    assert value['checkpoint_bindings'] == imported['checkpoint_bindings']
    assert value['qualification_plan']['saved'] == {'fixed': True}


def test_checkpoint_binding_refuses_substituted_candidate_before_writes(monkeypatch, tmp_path):
    monkeypatch.setattr(c, 'OUT', tmp_path)
    written = []
    monkeypatch.setattr(c, 'save', lambda *args: written.append(args))
    freeze = {}
    monkeypatch.setattr(control, '_prepared', lambda root: {'candidate_manifest': {'path': 'fixed96'}})
    with pytest.raises(ValueError, match='Accepted fixed checkpoint only'):
        control.bind_checkpoint(freeze, {'path': 'another_checkpoint'}, SimpleNamespace(guard=lambda: None))
    assert not written


def test_operational_failure_is_preserved_without_replacement(monkeypatch, tmp_path):
    monkeypatch.setattr(c, 'OUT', tmp_path)
    saved, played, closed = [], [], []
    monkeypatch.setattr(c, 'read', lambda path: {})
    monkeypatch.setattr(c, 'save', lambda path, value: saved.append((Path(path), value)))
    monkeypatch.setattr(control, 'verify_runtime', lambda *args: None)
    monkeypatch.setattr(control, 'load_inputs', lambda *args: ([], {}))
    monkeypatch.setattr(control, '_segment_start', lambda *args: {'path': 'immutable_start'})
    monkeypatch.setattr(control, '_prior_complete', lambda *args: None)
    monkeypatch.setattr(control.design, 'make_schedule', lambda stage: [{'stage': stage, 'game_index': i} for i in range(6)])
    child = SimpleNamespace(_closed=False)
    def qualify(stage, start, workers, *_):
        workers['operational_g000_candidate'] = child
    def play(spec, *_):
        played.append(spec['game_index'])
        return dict(spec, reason='forfeit')
    monkeypatch.setattr(control, 'qualify_pair', qualify)
    monkeypatch.setattr(control, 'play_game', play)
    monkeypatch.setattr(control.transport, 'close_and_audit', lambda child, *_: closed.append(child))
    envelope = SimpleNamespace(evidence={'workers': {}}, guard=lambda: None)
    with pytest.raises(ValueError, match='Any operational failure stops'):
        control.run_segment(envelope, {}, 'operational')
    assert played == [0] and closed == [child]
    assert len(saved) == 1 and saved[0][0].name == 'failure.json'
    assert saved[0][1]['source_start'] == {'path': 'immutable_start'}
    assert not any(p.name == 'complete.json' for p, _ in saved)


def test_failed_main_game_never_reenters_same_move_or_game(monkeypatch):
    specs = [{'stage': 'main', 'game_index': i} for i in range(10, 20)]
    workers, called, starts = {}, [], []
    def begin(spec, current, *_):
        starts.append(spec['game_index']); current['worker'] = 1
    def play(spec, *_):
        called.append(spec['game_index'])
        return dict(spec, reason='forfeit')
    monkeypatch.setattr(control, 'begin_block', begin)
    monkeypatch.setattr(control, 'play_game', play)
    monkeypatch.setattr(control, 'close_workers', lambda current, *_: current.clear())
    control.collect_main(specs, workers, None, None, None)
    assert called == starts == list(range(10, 20))
