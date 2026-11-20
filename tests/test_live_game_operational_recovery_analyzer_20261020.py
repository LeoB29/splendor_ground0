"""Model-free recovery tests; no player, native executable or neural inference."""
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import analyze_live_game_operational_recovery_20261020 as a


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def stamp(value):
    return datetime.fromtimestamp(value, timezone.utc).isoformat()


def failure_row():
    return {'correct': False, 'failure_category': 'ResourceHold', 'id': 'games/g011/t0015',
        'observer': {'context': a.core.context_for(11, 15)}, 'player': 'local',
        'status': 'incomplete', 'total_seconds': .157}


def source_controls():
    launch = {'epoch_seconds': 100., 'time': stamp(100), 'pid': 77}
    progress = {'status': 'preserved_hold', 'failure': {'category': 'ResourceHold', 'detail_path': 'sealed/failure.json'},
        'completed_games': 5, 'validation_passed': True, 'outcomes_closed': True, 'strength_analysis_games': 0,
        'larger_data_deferred': True, 'elapsed_seconds': 500., 'time': stamp(600)}
    completion = {**progress, 'worker_processes_closed': True, 'progress': {'bound': True}}
    execution = {'exit_code': 1, 'outcomes_closed': True, 'runner_identity': {'pid': 77, 'created': 99.}}
    return completion, progress, execution, launch


def admission(readings=None):
    rows = readings or [{'elapsed_seconds': float(10*i), 'available_physical_bytes': 2*1024**3,
        'available_commit_bytes': 4*1024**3, 'admissible': True, 'consecutive_admissible_samples': i+1} for i in range(4)]
    elapsed = rows[-1]['elapsed_seconds']+.001
    return {'schema': 'operational_recovery_memory_admission_v1', 'time': stamp(100+elapsed),
        'status': 'passed', 'failure_category': None, 'minimum_available_physical_bytes': 2*1024**3,
        'minimum_available_commit_bytes': 4*1024**3, 'required_consecutive_samples': 4,
        'sample_interval_seconds': 10, 'maximum_wait_seconds': 600, 'elapsed_seconds': elapsed,
        'readings': rows, 'outcomes_closed': True, 'no_player_requests_during_admission': True}


def test_fixed_schedule_partition_never_replaces_original_forfeits():
    assert [s['game'] for s in a.core.schedule()][:6] == [6, 8, 10, 7, 9, 11]
    for game in a.core.GAMES:
        for ply in range(400):
            assert a.original_move(game, ply) is (game in (6, 8, 10, 7, 9) or game == 11 and ply < 15)
    assert not a.original_move(11, 15)


@pytest.mark.parametrize('game,ply', [(True, 0), (11, True), (5, 0), (24, 0), (11, -1), (11, 400), (11., 15)])
def test_invalid_recovery_identity_rejected(game, ply):
    with pytest.raises(ValueError): a.original_move(game, ply)


def test_original_failure_is_an_unstarted_hold_not_a_forfeit():
    assert a.check_failure_attempt(failure_row()) is None
    assert a.check_source_completion(*source_controls()) is None


@pytest.mark.parametrize('damage', ['init', 'request', 'receipt', 'result', 'completed', 'forfeit', 'wrong_context', 'wrong_id'])
def test_original_failed_position_cannot_hide_an_attempted_decision(damage):
    row = failure_row()
    if damage in ('init', 'request', 'receipt', 'result'): row[damage] = {}
    elif damage == 'completed': row['status'] = 'completed'
    elif damage == 'forfeit': row['status'] = 'forfeit_timeout'
    elif damage == 'wrong_context': row['observer']['context']['turn'] = 14
    else: row['id'] = 'games/g011/t0014'
    with pytest.raises(ValueError): a.check_failure_attempt(row)


@pytest.mark.parametrize('damage', ['accepted', 'six_games', 'failure', 'open', 'winner_use', 'not_closed', 'exit0', 'pid', 'elapsed'])
def test_source_hold_and_original_clock_cannot_be_relabelled(damage):
    complete, progress, execution, launch = source_controls()
    if damage == 'accepted': complete['status'] = progress['status'] = 'completed'
    elif damage == 'six_games': complete['completed_games'] = progress['completed_games'] = 6
    elif damage == 'failure': complete['failure'] = progress['failure'] = None
    elif damage == 'open': complete['outcomes_closed'] = progress['outcomes_closed'] = False
    elif damage == 'winner_use': complete['strength_analysis_games'] = progress['strength_analysis_games'] = 5
    elif damage == 'not_closed': complete['worker_processes_closed'] = False
    elif damage == 'exit0': execution['exit_code'] = 0
    elif damage == 'pid': execution['runner_identity']['pid'] += 1
    else: complete['elapsed_seconds'] = progress['elapsed_seconds'] = 10.
    with pytest.raises(ValueError): a.check_source_completion(complete, progress, execution, launch)


def test_admission_independently_reconstructs_thresholds_spacing_and_streak():
    value = admission()
    assert a.check_admission(value, {'epoch_seconds': 100.}) == pytest.approx(130.001)
    rows = deepcopy(value['readings'])
    rows[0].update(available_physical_bytes=1, admissible=False, consecutive_admissible_samples=0)
    for i in range(1, 4): rows[i]['consecutive_admissible_samples'] = i
    rows.append({**rows[-1], 'elapsed_seconds': 40., 'consecutive_admissible_samples': 4})
    assert a.check_admission(admission(rows), {'epoch_seconds': 100.}) == pytest.approx(140.001)


@pytest.mark.parametrize('damage', ['physical', 'commit', 'boolean', 'streak', 'spacing', 'span', 'late', 'early_launch',
                                   'threshold', 'requests', 'failure', 'extra_sample', 'missing_sample'])
def test_admission_failure_or_relaxed_stability_is_rejected(damage):
    value = admission(); launch = {'epoch_seconds': 100.}
    if damage == 'physical': value['readings'][-1]['available_physical_bytes'] -= 1
    elif damage == 'commit': value['readings'][-1]['available_commit_bytes'] -= 1
    elif damage == 'boolean': value['readings'][0]['available_physical_bytes'] = True
    elif damage == 'streak': value['readings'][1]['consecutive_admissible_samples'] = 3
    elif damage == 'spacing': value['readings'][1]['elapsed_seconds'] = 9.
    elif damage == 'span': value['readings'][-1]['elapsed_seconds'] = 29.99
    elif damage == 'late': value['elapsed_seconds'] = 600.
    elif damage == 'early_launch': launch['epoch_seconds'] = 101.
    elif damage == 'threshold': value['minimum_available_physical_bytes'] = 1024**3
    elif damage == 'requests': value['no_player_requests_during_admission'] = False
    elif damage == 'failure': value['failure_category'] = 'ResourceHold'
    elif damage == 'extra_sample':
        value['readings'].append({**value['readings'][-1], 'elapsed_seconds': 40., 'consecutive_admissible_samples': 5})
        value['elapsed_seconds'] = 40.001; value['time'] = stamp(140.001)
    else: value['readings'].pop()
    with pytest.raises(ValueError): a.check_admission(value, launch)


@pytest.mark.parametrize('clock', ['wall', 'monotonic'])
def test_both_fresh_clocks_bound_verification_without_rebinding_original_constants(tmp_path, monkeypatch, clock):
    audit = a.Audit(tmp_path); audit.launch = {'epoch_seconds': 100., 'monotonic_start': 200.}
    monkeypatch.setattr(a.time, 'time', lambda: 100.+(86400 if clock == 'wall' else 1))
    monkeypatch.setattr(a.time, 'monotonic', lambda: 200.+(86400 if clock == 'monotonic' else 1))
    with pytest.raises(ValueError): audit.boundary()
    assert a.core.BASE == a.SOURCE


@pytest.fixture
def paths(tmp_path, monkeypatch):
    source = tmp_path/'outputs/source'; base = tmp_path/'outputs/recovery'
    source.mkdir(parents=True); base.mkdir(parents=True)
    monkeypatch.setattr(a, 'ROOT', tmp_path); monkeypatch.setattr(a, 'SOURCE', source)
    def desc(path):
        path = Path(path)
        return {'path': path.resolve().relative_to(tmp_path).as_posix(), 'sha256': a.file_hash(path), 'bytes': path.stat().st_size}
    monkeypatch.setattr(a, 'artifact', desc)
    audit = a.Audit(base)
    def bound(item, expected=None):
        path = (tmp_path/item['path']).resolve()
        a.require(path.is_relative_to(tmp_path) and (expected is None or path == Path(expected).resolve()) and desc(path) == item,
                  'Synthetic artifact binding')
        return path
    audit.bound = bound
    return audit, source, base, desc


def test_copy_alias_accepts_only_the_exact_bound_source_and_equal_bytes(paths):
    audit, source, base, desc = paths
    src = source/'move.json'; dst = base/'move.json'; write(src, {'original': True}); dst.write_bytes(src.read_bytes())
    audit.imports[dst.resolve()] = desc(src)
    assert audit.copied(desc(src), dst) == dst
    assert audit.copied(desc(dst), dst) == dst
    other = source/'other.json'; write(other, {'original': True})
    with pytest.raises(ValueError): audit.copied(desc(other), dst)
    write(dst, {'original': False})
    with pytest.raises(ValueError): audit.copied(desc(src), dst)


def synthetic_lineage(paths):
    audit, source, base, desc = paths
    model = source.parent.parent/'frozen_kernel.py'; write(model, {'code_identity': True})
    frozen = {'inputs': [desc(model)], 'executable': 'python.exe', 'topology': {'selected_logical_processor': 0}}
    write(source/'freeze.json', frozen)
    complete, progress, execution, launch = source_controls()
    launch['freeze'] = desc(source/'freeze.json')
    command = ['python.exe', '-B', '-u', '-m', 'tools.live_game_operational_20261019.runner', 'run']
    execution['command'] = command
    write(source/'launch.json', launch); write(source/'operational_progress.json', progress)
    complete['progress'] = desc(source/'operational_progress.json')
    for name, value in (('completion.json', complete), ('runner_execution.json', execution),
        ('runner_process.json', {'identity': execution['runner_identity'], 'command': command,
                               'freeze_sha256': a.file_hash(source/'freeze.json')}),
        ('final_input_check.json', {'all_hashes_and_sizes_match': True, 'files': 1}),
        ('coordinator_hold.json', {'status': 'preserved_hold'}), ('sealed/failure.json', {'traceback': 'synthetic hold'}),
        ('decision_attempts/games/g011/t0015.json', failure_row())):
        write(source/name, value)
    imports = []
    for name, dest in (('decision_attempts/games/g011/t0015.json', 'preserved/incomplete_g011_t0015.json'),
                       ('sealed/failure.json', 'preserved/first_attempt_failure.json')):
        target = base/dest; target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes((source/name).read_bytes())
        imports.append({'source': desc(source/name), 'destination': desc(target)})
    snapshot = {'schema': 'operational_first_attempt_snapshot_v1', 'source_base': source.relative_to(a.ROOT).as_posix(),
        'files': [desc(p) for p in source.rglob('*') if p.is_file()], 'outcomes_closed': True}
    write(base/'source_snapshot.json', snapshot)
    plan = {'schema': 'live_operational_recovery_plan_v1', 'source_base': source.relative_to(a.ROOT).as_posix(),
        'completed_games': list(a.COMPLETED_GAMES), 'partial_game': {'game': 11, 'next_ply': 15, 'state_sha256': '0'*64},
        'validation_reused': True, 'source_snapshot': desc(base/'source_snapshot.json'), 'imports': imports,
        'preserved_failure': [desc(source/name) for name in ('sealed/failure.json', 'completion.json',
                                                          'coordinator_hold.json', 'decision_attempts/games/g011/t0015.json')]}
    for key, name in (('source_freeze', 'freeze.json'), ('source_launch', 'launch.json'), ('source_completion', 'completion.json'),
                      ('source_runner_execution', 'runner_execution.json')):
        plan[key] = desc(source/name)
    audit.plan = plan; audit.frozen = frozen
    items = {p.resolve(): desc(p) for p in a.ROOT.rglob('*') if p.is_file()}
    return audit, items


def test_complete_source_snapshot_and_hold_preservation_are_authenticated(paths, monkeypatch):
    audit, items = synthetic_lineage(paths)
    monkeypatch.setattr(a, 'SOURCE_FREEZE_SHA256', a.file_hash(a.SOURCE/'freeze.json'))
    audit.imports_and_lineage(items)
    assert audit.first_lineage['original_failure_retained'] and audit.first_lineage['validation_requests_reused'] == 12
    assert audit.first_lineage['next_unstarted_ply'] == 15


@pytest.mark.parametrize('damage', ['changed_original', 'missing_source', 'extra_source', 'old_input_rebound',
    'changed_copy', 'duplicate_copy', 'wrong_prefix', 'wrong_resume', 'missing_hold', 'old_request', 'wrong_core'])
def test_lineage_cannot_refreeze_changed_prior_evidence_or_replace_prefix(paths, monkeypatch, damage):
    audit, items = synthetic_lineage(paths)
    monkeypatch.setattr(a, 'SOURCE_FREEZE_SHA256', a.file_hash(a.SOURCE/'freeze.json'))
    if damage == 'changed_original': write(a.SOURCE/'sealed/failure.json', {'changed': True})
    elif damage == 'missing_source': del items[(a.SOURCE/'sealed/failure.json').resolve()]
    elif damage == 'extra_source': write(a.SOURCE/'new_file.json', {'late': True})
    elif damage == 'old_input_rebound':
        path = (a.ROOT/audit.frozen['inputs'][0]['path']).resolve()
        items[path] = {**items[path], 'sha256': '0'*64}
    elif damage == 'changed_copy': write(audit.base/'preserved/first_attempt_failure.json', {'changed': True})
    elif damage == 'duplicate_copy': audit.plan['imports'].append(deepcopy(audit.plan['imports'][0]))
    elif damage == 'wrong_prefix': audit.plan['completed_games'] = [6, 8, 10, 7]
    elif damage == 'wrong_resume': audit.plan['partial_game']['next_ply'] = 14
    elif damage == 'missing_hold': audit.plan['preserved_failure'].pop()
    elif damage == 'old_request': write(a.SOURCE/'workers/games/g011/t0015/request-00000001.json', {'unexpected': True})
    else: audit.frozen = {**audit.frozen, 'topology': {'selected_logical_processor': 1}}
    with pytest.raises(ValueError): audit.imports_and_lineage(items)


def synthetic_game(paths, monkeypatch):
    audit, source, base, desc = paths
    spec = next(s for s in a.core.schedule() if s['game'] == 11)
    def state(ply): return {'ruleset_id': a.core.RULESET, 'catalogue_sha256': a.core.CATALOGUE,
        'state': {'turn_index': ply, 'current_player': ply%2, 'terminal': ply == 16, 'winner': 0 if ply == 16 else None}}
    audit.launch = {'epoch_seconds': 0., 'monotonic_start': 0.}
    audit.boundary = lambda: None; audit.catalogue = {}
    audit.initial_snapshot = lambda *args: state(0)
    audit.ledger = SimpleNamespace(validate_ledger=lambda _: None, check_observer=lambda *args: None)
    audit.rules = SimpleNamespace(OBSERVATION_SCHEMA='lawful_synthetic')
    audit.plan = {'partial_game': {'game': 11, 'next_ply': 15, 'state_sha256': a.digest(state(15))}}
    monkeypatch.setattr(a.core, 'hidden_variant', lambda value, _: deepcopy(value))
    source_folder = source/'sealed/games/011'; folder = base/'sealed/games/011'
    initial = {**spec, 'initial': state(0), 'initial_sha256': a.digest(state(0))}
    write(source_folder/'initial.json', initial); write(folder/'initial.json', initial)
    audit.imports[(folder/'initial.json').resolve()] = desc(source_folder/'initial.json')
    calls = []
    def decide(label):
        def decision(row, snapshot, name, player, stage):
            calls.append((label, row['ply']))
            return {'decision': {'record': {'candidates': {'actions': [{'synthetic': True}]}}}}, state(row['ply']+1)
        return decision
    audit.old.decision = decide('original'); audit.new.decision = decide('recovery')
    moves = []
    for ply in range(16):
        player = 'local' if ply%2 == spec['local_seat'] else 'opponent'
        observer = {'schema': 'lawful_synthetic', 'observation': {'turn_index': ply}}
        if player == 'local': observer['context'] = a.core.context_for(11, ply)
        row = {**{k: spec[k] for k in ('game', 'pair', 'fit')}, 'ply': ply, 'actor': ply%2,
            'before_sha256': a.digest(state(ply)), 'id': f'games/g011/t{ply:04d}', 'player': player,
            'observer': observer, 'initialization': {'elapsed_seconds': 1.},
            'clock': {'elapsed_seconds': 1.}, 'total_seconds': 2., 'status': 'completed'}
        path = folder/'moves'/f'{ply:04d}.json'; write(path, row)
        write(base/'decision_attempts'/(row['id']+'.json'), row)
        if ply < 15:
            old_path = source_folder/'moves'/f'{ply:04d}.json'; write(old_path, row)
            original_decision = source/'decision_attempts'/(row['id']+'.json'); write(original_decision, row)
            audit.imports[path.resolve()] = desc(old_path)
            audit.imports[(base/'decision_attempts'/(row['id']+'.json')).resolve()] = desc(original_decision)
        moves.append(desc(path))
    audit.failed_attempt = {'observer': read_observer(base/'sealed/games/011/moves/0015.json')}
    result = {**spec, 'initial_sha256': a.digest(state(0)), 'moves': moves, 'complete': True,
              'reason': 'natural', 'winner': 0, 'final': state(16)}
    write(folder/'result.json', result)
    write(base/'resume_check.json', {'game': 11, 'next_ply': 15, 'state_sha256': a.digest(state(15)),
        'prefix_transitions': 15, 'no_completed_request_repeated': True, 'outcomes_closed': True})
    return audit, spec, calls


def read_observer(path):
    return json.loads(path.read_text(encoding='utf-8'))['observer']


def test_combined_replay_uses_fifteen_original_moves_then_exact_new_position(paths, monkeypatch):
    audit, spec, calls = synthetic_game(paths, monkeypatch)
    audit.game(spec)
    assert calls == [('original', i) for i in range(15)]+[('recovery', 15)]
    assert audit.resume_checked and audit.actual_transitions == 16
    assert audit.segment_transitions == {'original': 15, 'recovery': 1}
    assert audit.game_costs[0]['segments']['original']['attempted_decisions'] == 15
    assert audit.game_costs[0]['segments']['recovery']['attempted_decisions'] == 1


@pytest.mark.parametrize('origin,player,is_audit,damage', [
    ('original', 'local', False, None), ('original', 'audit', True, None),
    ('recovery', 'local', False, None), ('recovery', 'opponent', False, None),
    ('recovery', 'audit', True, None), ('original', 'local', False, 'module'),
    ('recovery', 'local', False, 'module'), ('recovery', 'local', False, 'before_admission')])
def test_segment_receipts_retain_own_module_and_new_admission_timestamp(paths, monkeypatch, origin, player, is_audit, damage):
    audit, source, base, desc = paths; audit.admission_epoch = 500.; audit.boundary = lambda: None
    segment = audit.old if origin == 'original' else audit.new
    name = 'synthetic'; folder = segment.base/('audits' if is_audit else 'workers')/name
    module = ('tools.splendorust_120_qualification_20261018.worker' if player == 'opponent'
              else a.ORIGINAL_MODULE if origin == 'original' else a.MODULE)
    created = 100. if origin == 'original' else 600.
    if damage == 'before_admission': created = 499.
    freeze = {'executable': 'python.exe', 'topology': {'selected_logical_processor': 0}}
    process = {'role': 'python', 'pid': 999, 'created': created, 'executable': 'python.exe'}
    init = {'status': 'ok', 'elapsed_seconds': 1., 'command': ['python.exe', '-B', '-u', '-m', module],
        'affinity': [0], 'topology': freeze['topology'], 'thread_environment': {k: '1' for k in a.core.THREADS},
        'identity': {'pid': 999, 'created': created}, 'telemetry': {'process_tree': [process]},
        'result': {'pid': 999, 'threads': 1}}
    if damage == 'module': init['command'][-1] = 'wrong.worker'
    receipt = {'status': 'ok', 'result': {'synthetic': True}, 'elapsed_seconds': 1.}
    write(folder/'initialization.json', init); write(folder/'request-00000001.json', receipt)
    row = {'initialization': {k: v for k, v in init.items() if k not in a.core.METADATA and k != 'result'},
        'clock': {k: v for k, v in receipt.items() if k != 'result'},
        'initialization_artifact': desc(folder/'initialization.json'), 'receipt': desc(folder/'request-00000001.json')}
    segment.bound = audit.bound; segment.frozen = freeze; segment.cpu = 0
    calls = []
    segment.check_clock = lambda value, request, seconds, cpu: calls.append((request, seconds, cpu))
    segment.process_identity = lambda *args: None
    if damage:
        with pytest.raises(ValueError): segment.receipts(row, name, player, is_audit)
    else:
        assert segment.receipts(row, name, player, is_audit) == receipt
        assert calls == [(None, 60, 0), ('request-00000001', 600 if is_audit else 120, 0)]


@pytest.mark.parametrize('split_stress', [False, True])
def test_one_combined_manifest_selects_stress_across_both_clock_segments(paths, monkeypatch, split_stress):
    audit, source, base, desc = paths
    monkeypatch.setattr(a.core, 'artifact', desc)
    roots = []
    for game, ply in ((6, 25), (13, 25)):
        observer = {'context': a.core.context_for(game, ply),
            'observation': {'turn_index': ply, 'current_player': 0, 'players': [{'reserves': []}, {'reserves': []}]}}
        path = base/f'sealed/games/{game:03d}/moves/{ply:04d}.json'; write(path, {'synthetic_root': [game, ply]})
        row = {'fit': 161001, 'observer': observer, 'receipt': {'preserved': game == 6}, 'trace': {'same': True}}
        roots.append({'id': f'g{game:03d}-t{ply:04d}', 'game': game, 'ply': ply, 'observer': observer,
                      'candidate_support': 4, 'row': row, 'path': path, 'wrapper': {}})
    cells, full = a.core.stress_selection(roots)
    assert len(full) == 1
    entries = []
    for root in roots:
        row = root['row']; identity = row['observer']['context']['identity']; sample = a.core.sample_worlds(identity)
        mode = 'full' if root['id'] in full or split_stress else 'sample'
        entries.append({'id': root['id'], 'game': root['game'], 'ply': root['ply'], 'fit': row['fit'],
            'identity': identity, 'observer_sha256': a.digest(row['observer']), 'move': desc(root['path']),
            'receipt': row['receipt'], 'trace': row['trace'], 'candidate_support': 4, 'mode': mode,
            'world_indices': list(range(128)) if mode == 'full' else sample, 'sample_world_indices': sample,
            'sample_hashes': [{'world_index': wi, 'sha256': a.digest([a.core.AUDIT_DOMAIN, 'world', a.core.AUDIT_SEED, identity, wi])}
                              for wi in sample],
            'stress_priority_sha256': a.digest([a.core.AUDIT_DOMAIN, 'stress', a.core.AUDIT_SEED, identity]),
            'stress_cells': a.core.root_cells(row['observer'], 4)})
    write(base/'audit_manifest.json', {'schema': 'operational_continuation_manifest_v1', 'entries': entries,
        'cells': cells, 'full_stress_ids': [r['id'] for r in roots] if split_stress else full})
    audit.new.roots = roots; audit.new.local_recorded_worlds = 256; audit.new.local_recorded_continuations = 1024
    calls = []
    def proof(name, row, wrapper, mode):
        calls.append((name, mode)); worlds = 128 if mode == 'full' else 8
        return {'worlds': worlds, 'continuations': 4*worlds, 'transitions': 4*worlds, 'raw_choices': 0}
    audit.new.proof = proof
    if split_stress:
        with pytest.raises(ValueError): audit.new.manifest()
        assert not calls
    else:
        coverage = audit.new.manifest()
        assert len(calls) == 2 and coverage['full_stress_roots'] == coverage['sample_only_roots'] == 1
        assert coverage['recorded_worlds'] == 256 and coverage['replayed_union']['worlds'] == 136


@pytest.mark.parametrize('damage', ['boundary', 'observer', 'before', 'duplicate', 'missing', 'relabelled', 'resume_receipt'])
def test_boundary_mutation_duplicate_or_replacement_never_counts_as_continuation(paths, monkeypatch, damage):
    audit, spec, calls = synthetic_game(paths, monkeypatch)
    folder = audit.base/'sealed/games/011'
    if damage == 'boundary': audit.plan['partial_game']['state_sha256'] = '0'*64
    elif damage == 'observer': audit.failed_attempt['observer']['unexpected'] = True
    elif damage == 'relabelled': del audit.imports[(folder/'moves/0014.json').resolve()]
    elif damage == 'resume_receipt':
        value = a.read(audit.base/'resume_check.json'); value['no_completed_request_repeated'] = False
        write(audit.base/'resume_check.json', value)
    else:
        game = a.read(folder/'result.json')
        if damage == 'duplicate': game['moves'].insert(15, deepcopy(game['moves'][14]))
        elif damage == 'missing': game['moves'].pop(14)
        else:
            path = folder/'moves/0015.json'; row = a.read(path); row['before_sha256'] = '0'*64; write(path, row)
            game['moves'][15] = a.artifact(path)
        write(folder/'result.json', game)
    with pytest.raises(ValueError): audit.game(spec)
    assert ('recovery', 15) not in calls


def test_no_closed_game_outcomes_in_sanitized_main_failure(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(a.sys, 'argv', ['analyzer', '--base', str(tmp_path)])
    def failure(self): raise ValueError('SEALED_SYNTHETIC_CONTENT')
    monkeypatch.setattr(a.Audit, 'run', failure)
    with pytest.raises(ValueError): a.main()
    assert 'SEALED_SYNTHETIC_CONTENT' in a.read(tmp_path/'sealed/independent_analysis_failure_01.json')['traceback']
    assert 'SEALED_SYNTHETIC_CONTENT' not in capsys.readouterr().out
    assert not (tmp_path/'independent_analysis.json').exists()
