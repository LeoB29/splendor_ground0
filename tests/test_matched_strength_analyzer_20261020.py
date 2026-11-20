"""Synthetic audit regressions and existing receipt schemas; no inference/games."""
from copy import deepcopy
from datetime import datetime, timezone
from types import SimpleNamespace
import json
from pathlib import Path

import pytest

from tools import analyze_matched_strength_20261020 as a
from tools.analyze_external_strength_screen_20261018_02 import check_clock, check_diagnostics


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def initialization():
    base = a.ROOT/'outputs/external_strength_screen_20261018_02'
    value = a.read(base/'workers/qualification/initialization.json')
    freeze = a.read(base/'freeze.json'); freeze['executable'] = value['command'][0]
    module = 'tools.splendorust_120_qualification_20261018.worker'
    value['command'][-1] = module
    saved = {k: v for k, v in value.items() if k not in a.METADATA and k != 'result'}
    return value, saved, freeze, module


def test_real_initialization_enrichment_is_validated_without_model_execution():
    value, saved, freeze, module = initialization()
    a.check_initialization(value, saved, freeze, module)
    check_clock(value, None, 60, freeze['topology']['selected_logical_processor'])
    audit = a.Audit(a.BASE); audit.frozen = freeze; audit.profile = a.read(a.ROOT/a.OPPONENT_PROFILE)
    audit.process_identity([value, deepcopy(value)], 'opponent')


@pytest.mark.parametrize('damage', ['command', 'affinity', 'topology', 'threads', 'identity', 'ready_pid', 'extra', 'clock'])
def test_enriched_initialization_rejects_metadata_or_copy_changes(damage):
    value, saved, freeze, module = initialization(); saved = deepcopy(saved)
    if damage == 'command': value['command'][-1] += '_other'
    elif damage == 'affinity': value['affinity'] = [999]
    elif damage == 'topology': value['topology'] = {**value['topology'], 'extra': True}
    elif damage == 'threads': value['thread_environment']['OMP_NUM_THREADS'] = '2'
    elif damage == 'identity': value['identity']['created'] += 1.
    elif damage == 'ready_pid': value['result']['pid'] += 1
    elif damage == 'extra': value['unknown_supervisor_field'] = True
    else: saved['receipt_elapsed_seconds'] += .001
    with pytest.raises(ValueError): a.check_initialization(value, saved, freeze, module)


@pytest.mark.parametrize('code', [1, -1, False, '0', None])
def test_preflight_requires_real_zero_exit_code(code):
    with pytest.raises(ValueError): a.check_preflight({'returncode': code})


def test_schedule_public_context_profile_and_samples_are_fixed():
    a.check_preflight({'returncode': 0})
    rows = a.schedule({'schema': 'matched_strength_private_seed_material_v1', 'private_nonce': '1'*64}); assert [r['game'] for r in rows] == list(a.GAME_ORDER)
    for pair in range(30):
        both = [r for r in rows if r['pair'] == pair]
        assert len({r['seed'] for r in both}) == 1 and {r['local_seat'] for r in both} == {0, 1}
        assert {r['fit'] for r in both} == {a.FITS[pair%3]}
    context = a.context_for(9, 37); sample = a.sample_worlds(context['identity'])
    assert sample == sorted(sample) and len(set(sample)) == 8
    expected = sorted(range(128), key=lambda wi: (a.digest([a.AUDIT_DOMAIN, 'world', a.AUDIT_SEED, context['identity'], wi]), wi))[:8]
    assert sample == sorted(expected)
    assert a.local_profile()['worlds'] == 128 and a.local_profile()['candidates'] == 4
    assert a.local_profile()['parent_profile_id'] == a.LOCAL_PROFILE
    with pytest.raises(ValueError): a.context_for(True, 0)
    with pytest.raises(ValueError): a.context_for(1, 400)


def root(game=6, ply=24, support=2):
    observer = {'context': a.context_for(game, ply), 'observation': {'current_player': 0, 'turn_index': ply,
        'players': [{'reserves': [{'source': 'deck'}]}, {'reserves': []}]}}
    return {'id': f'g{game:03d}-t{ply:04d}', 'game': game, 'ply': ply, 'observer': observer, 'candidate_support': support}


def test_stress_minimum_is_identity_based_and_overlap_is_deduplicated():
    roots = [root(ply=24), root(ply=25), root(ply=26, support=4)]
    cells, selected = a.stress_selection(roots)
    assert len(cells) == 81
    ordinary = cells['ordinary:161001:1:0:own']
    assert ordinary['eligible'] == [r['id'] for r in roots]
    expected = min(roots, key=lambda r: (a.digest([a.AUDIT_DOMAIN, 'stress', a.AUDIT_SEED, r['observer']['context']['identity']]), r['game'], r['ply']))
    assert ordinary['selected'] == expected['id']
    assert selected == sorted(set(v['selected'] for v in cells.values() if v['selected']))
    with pytest.raises(ValueError): a.stress_selection(roots+[roots[0]])


def proof_fixture(mode):
    observer = root()['observer']; identity = observer['context']['identity']; support = 2
    decision = {'record': {'candidates': {'actions': [{}, {}]}, 'worlds': [{'outcomes': [{'turns': 2}, {'turns': 3}]} for _ in range(128)]}}
    wrapper = {'decision': decision, 'trace': {'sha256': '1'*64, 'bytes': 19}}
    indices = list(range(128)) if mode == 'full' else a.sample_worlds(identity)
    proof = {'status': 'passed', 'profile_id': a.LOCAL_PROFILE, 'ruleset_id': a.RULESET, 'catalogue_sha256': a.CATALOGUE,
        'observer_sha256': a.digest(observer), 'result_sha256': a.digest(wrapper), 'decision_sha256': a.digest(decision),
        'trace_sha256': '1'*64, 'trace_bytes': 19, 'root_identity': identity, 'fit': 161001, 'mode': mode,
        'recorded_worlds': 128, 'recorded_continuations': 256, 'candidate_support': support,
        'audited_world_indices': indices, 'audited_worlds': len(indices), 'audited_continuations': 2*len(indices),
        'audited_transitions': 5*len(indices), 'raw035_choices_reexecuted': 3*len(indices),
        'proposal_model_reexecuted': True, 'proposal_selection_independent': True, 'all_world_records_checked': True,
        'all_root_checks_passed': True, 'all_continuations_replayed': mode == 'full', 'old_outcomes_used_as_ground_truth': False,
        'transition_oracle': 'frozen_external_screen_dictionary_ledger',
        'shared_dependencies': ['qualified corrected model/feature kernels', 'corrected legal enumeration/catalogue/serialization',
                                'frozen independent world/proposal/trajectory audit primitives']}
    return proof, observer, wrapper


@pytest.mark.parametrize('mode', ['sample', 'full'])
def test_proofs_bind_exact_replay_union_and_raw_choice_counts(mode):
    proof, observer, wrapper = proof_fixture(mode)
    result = a.check_proof(proof, observer, wrapper, mode)
    assert result['worlds'] == (8 if mode == 'sample' else 128)
    assert result['raw_choices'] == result['transitions']-result['continuations']


@pytest.mark.parametrize('damage', ['world', 'transition', 'raw_choice', 'trace', 'observer', 'result', 'all_replayed', 'oracle', 'extra'])
def test_proof_does_not_promote_sampling_or_accept_unbound_results(damage):
    proof, observer, wrapper = proof_fixture('sample')
    if damage == 'world': proof['audited_world_indices'][0] = 999
    elif damage == 'transition': proof['audited_transitions'] += 1
    elif damage == 'raw_choice': proof['raw035_choices_reexecuted'] += 1
    elif damage == 'trace': proof['trace_sha256'] = '0'*64
    elif damage == 'observer': proof['observer_sha256'] = '0'*64
    elif damage == 'result': proof['result_sha256'] = '0'*64
    elif damage == 'all_replayed': proof['all_continuations_replayed'] = True
    elif damage == 'oracle': proof['transition_oracle'] = 'model-only'
    else: proof['unearned_claim'] = True
    with pytest.raises(ValueError): a.check_proof(proof, observer, wrapper, 'sample')


@pytest.mark.parametrize('damage', ['none', 'sample_hash', 'stress_priority', 'cell', 'mode', 'missing', 'extra'])
def test_exact_manifest_and_prescribed_priority_hashes(tmp_path, monkeypatch, damage):
    value = root(); path = tmp_path/'move.json'; write(path, {})
    monkeypatch.setattr(a, 'artifact', lambda p: {'path': str(p), 'sha256': a.file_hash(p), 'bytes': p.stat().st_size})
    row = {'fit': 161001, 'observer': value['observer'], 'receipt': {'fixed': 'receipt'}, 'trace': {'fixed': 'trace'}}
    value.update(row=row, path=path, wrapper={})
    cells, full = a.stress_selection([value]); identity = row['observer']['context']['identity']; sample = a.sample_worlds(identity)
    entry = {'id': value['id'], 'game': 6, 'ply': 24, 'fit': 161001, 'identity': identity,
        'observer_sha256': a.digest(row['observer']), 'move': a.artifact(path), 'receipt': row['receipt'], 'trace': row['trace'],
        'candidate_support': 2, 'mode': 'full', 'world_indices': list(range(128)), 'sample_world_indices': sample,
        'sample_hashes': [{'world_index': wi, 'sha256': a.digest([a.AUDIT_DOMAIN, 'world', a.AUDIT_SEED, identity, wi])} for wi in sample],
        'stress_priority_sha256': a.digest([a.AUDIT_DOMAIN, 'stress', a.AUDIT_SEED, identity]), 'stress_cells': a.root_cells(row['observer'], 2)}
    manifest = {'schema': 'operational_continuation_manifest_v1', 'entries': [entry], 'cells': cells, 'full_stress_ids': full}
    if damage == 'sample_hash': entry['sample_hashes'][0]['sha256'] = '0'*64
    elif damage == 'stress_priority': entry['stress_priority_sha256'] = '0'*64
    elif damage == 'cell': cells['ordinary:161001:1:0:own']['selected'] = None
    elif damage == 'mode': entry['mode'] = 'sample'
    elif damage == 'missing': manifest['entries'] = []
    elif damage == 'extra': manifest['entries'].append(deepcopy(entry))
    write(tmp_path/'audit_manifest.json', manifest)
    audit = a.Audit(tmp_path); audit.roots = [value]
    audit.proof = lambda *args: {'worlds': 128, 'continuations': 256, 'transitions': 512, 'raw_choices': 256}
    if damage == 'none': assert audit.manifest()['full_stress_roots'] == 1
    else:
        with pytest.raises(ValueError, match='manifest'): audit.manifest()


@pytest.mark.parametrize('status', ['completed', 'internal_budget_exhausted', 'timeout', 'worker_error'])
def test_full_opponent_decision_receipts_bind_result_clock_and_successor(tmp_path, monkeypatch, status):
    from tools.external_strength_screen_20261018 import rules, audit as ledger
    base = a.ROOT/'outputs/external_strength_screen_20261018_02'
    old = a.read(base/'dynamic_qualification/002.json'); snapshot = a.read(base/'fixtures.json')[0]['snapshot']
    init, saved_init, freeze, module = initialization(); profile = a.read(a.ROOT/a.OPPONENT_PROFILE)
    clock = deepcopy(old['clock']); clock['request_id'] = 'request-00000001'
    result = clock['result']; result['status'] = 'completed'; result['profile_sha256'] = a.OPPONENT_SHA256
    result['diagnostics']['profile_id'] = profile['profile_id']
    for phase in result['diagnostics']['phases']:
        if not phase['heuristic_phase']: phase.update(simulations=6400, leaf_calls=6401)
    result['diagnostics']['inference_calls'] = result['diagnostics']['inference_rows'] = 6401
    if status == 'internal_budget_exhausted':
        result = {k: v for k, v in result.items() if k not in ('status', 'action', 'diagnostics')}
        result.update(status=status, internal_budget_seconds=108, opponent_id='splendorust', profile_id=profile['profile_id'],
                      reason='Internal fixed-work deadline exceeded'); clock['result'] = result
    if status in ('timeout', 'worker_error'):
        clock.update(status=status, accepted=False, result=None, receipt_elapsed_seconds=None, received_before_deadline=False)
    name = 'games/g006/t0000'; folder = tmp_path/'workers'/name
    write(folder/'initialization.json', init); write(folder/'request-00000001.json', clock)
    monkeypatch.setattr(a, 'artifact', lambda p: {'path': str(p), 'sha256': a.file_hash(p), 'bytes': p.stat().st_size})
    row = {'id': name, 'player': 'opponent', 'observer': old['observer'], 'initialization': saved_init,
        'clock': {k: v for k, v in clock.items() if k != 'result'},
        'initialization_artifact': a.artifact(folder/'initialization.json'), 'receipt': a.artifact(folder/'request-00000001.json'),
        'correct': status == 'completed', 'status': 'completed' if status == 'completed' else 'forfeit_'+status}
    if status not in ('timeout', 'worker_error'): row['result_sha256'] = a.digest(result)
    if status == 'completed': row['after_sha256'] = a.digest(ledger.transition(snapshot, result['action']))
    path = tmp_path/'decision_attempts'/(name+'.json'); write(path, row)
    audit = a.Audit(tmp_path); audit.boundary = lambda: None; audit.frozen = freeze
    audit.memory_gate = lambda *args: 0.
    row['memory_admission'] = {'synthetic': True}; write(path, row)
    audit.profile = profile; audit.cpu = freeze['topology']['selected_logical_processor']
    audit.rules = rules; audit.ledger = ledger; audit.check_clock = check_clock; audit.check_diagnostics = check_diagnostics
    def bound(item, expected=None):
        path = Path(item['path']); assert path == expected and a.artifact(path) == item; return path
    audit.bound = bound
    if status == 'worker_error':
        with pytest.raises(ValueError, match='hard error'): audit.decision(row, snapshot, name, 'opponent', 'games')
    else:
        outcome = audit.decision(row, snapshot, name, 'opponent', 'games')
        assert (outcome is not None) == (status == 'completed')
        if status == 'completed':
            row['result_sha256'] = '0'*64; write(path, row)
            with pytest.raises(ValueError, match='result digest'): audit.decision(row, snapshot, name, 'opponent', 'games')
    if status == 'timeout':
        row['clock']['accepted'] = True; write(path, row)
        with pytest.raises(ValueError, match='Copied'): audit.decision(row, snapshot, name, 'opponent', 'games')


@pytest.mark.parametrize('damage', ['none', 'wrong_result', 'nonterminal_natural', 'referee_forfeit', 'terminal_forfeit', 'final_state'])
def test_sealed_natural_and_forfeit_endings_are_checked_without_returning_outcomes(damage):
    state = {'state': {'terminal': True, 'winner': 1}}; game = {'complete': True, 'final': deepcopy(state), 'reason': 'natural', 'winner': 1}
    failure = None
    if damage == 'wrong_result': game['winner'] = 0
    elif damage == 'nonterminal_natural': state['state']['terminal'] = False; game['final'] = deepcopy(state)
    elif damage in ('referee_forfeit', 'terminal_forfeit'):
        failure = ('forfeit_worker_error' if damage == 'referee_forfeit' else 'forfeit_timeout', 0, 'local')
        game.update(reason=failure[0], failing_player='local')
        if damage == 'referee_forfeit': state['state']['terminal'] = False; game['final'] = deepcopy(state)
    elif damage == 'final_state': game['final']['state']['winner'] = 0
    if damage == 'none': assert a.check_game_ending(game, state, failure) is None
    else:
        with pytest.raises(ValueError): a.check_game_ending(game, state, failure)
    state = {'state': {'terminal': False, 'winner': None}}
    game = {'complete': True, 'final': state, 'reason': 'forfeit_timeout', 'winner': 1, 'failing_player': 'local'}
    assert a.check_game_ending(game, state, ('forfeit_timeout', 0, 'local')) is None


def test_failure_diagnostics_preserved_only_in_sealed_file(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(a.sys, 'argv', ['analyzer', '--base', str(tmp_path)])
    def fail(self): raise ValueError('SEALED_TEST_PAYLOAD')
    monkeypatch.setattr(a.Audit, 'run', fail)
    with pytest.raises(ValueError): a.main()
    assert 'SEALED_TEST_PAYLOAD' in a.read(tmp_path/'sealed/independent_analysis_failure_01.json')['traceback']
    captured = capsys.readouterr(); assert 'SEALED_TEST_PAYLOAD' not in captured.out+captured.err
    assert not (tmp_path/'independent_analysis.json').exists()


def test_independent_hidden_variant_matches_existing_referee_transformation():
    from tools.external_strength_screen_20261018 import rules
    from tools.external_strength_screen_20261018_02.runner import hidden_variant
    catalogue = a.read(a.ROOT/'outputs/external_strength_screen_20261018_02/catalogue.json')
    for fixture in a.TRANSFER_INDICES:
        snapshot = a.read(a.PRIOR_LOCAL/'fixtures'/f'{fixture:02d}.json')['snapshot']
        assert a.hidden_variant(snapshot, catalogue) == rules.serialize_snapshot(hidden_variant(rules.deserialize_snapshot(snapshot)))


def test_new_contexts_and_validation_producer_match_independent_contract(monkeypatch):
    from tools.matched_strength_20261020 import common, runner
    monkeypatch.setattr(common, 'config', lambda: deepcopy(a.CONFIG))
    assert a.schedule({'schema': 'matched_strength_private_seed_material_v1', 'private_nonce': '1'*64}) == common.make_schedule({'schema': 'matched_strength_private_seed_material_v1', 'private_nonce': '1'*64})
    assert len(a.schedule({'schema': 'matched_strength_private_seed_material_v1', 'private_nonce': '1'*64})) == 60 and len({v['seed'] for v in a.schedule({'schema': 'matched_strength_private_seed_material_v1', 'private_nonce': '1'*64})}) == 30
    assert {fit: sum(v['fit'] == fit for v in a.schedule({'schema': 'matched_strength_private_seed_material_v1', 'private_nonce': '1'*64})) for fit in a.FITS} == dict.fromkeys(a.FITS, 20)
    cases = runner.build_validation(); assert len(cases) == 12
    auditor = a.Audit(a.BASE)
    for index, case in enumerate(cases):
        fixture = (0, 6, 12)[index//4]; pilot_case = (2, 6, 10)[index//4]
        expected_mode = 'pilot_transfer' if index%4 < 2 else 'operational'
        assert case['fixture'] == fixture and case['fit'] == a.FITS[index//4] and case['mode'] == expected_mode
        assert case['reference_receipt']['path'] == (a.PRIOR_PILOT/'workers/validation'/f'v{pilot_case:03d}'/'request-00000001.json').relative_to(a.ROOT).as_posix()
        auditor.context(case['observer'])
        if expected_mode == 'pilot_transfer':
            old = a.read(a.PRIOR_PILOT/'decision_attempts/validation'/f'v{pilot_case:03d}.json')['observer']
            assert case['observer']['context'] == {**old['context'], 'mode': 'pilot_transfer', 'fixture_index': fixture}
            assert case['observer']['observation'] == old['observation']
        else: assert case['observer']['context'] == a.context_for((1, 2, 5)[index//4], case['snapshot']['state']['turn_index'])
        if index%2: assert case['observer'] == cases[index-1]['observer'] and case['snapshot'] != cases[index-1]['snapshot']


@pytest.mark.parametrize('damage', ['fit', 'stream', 'namespace', 'identity', 'fixture', 'unknown_mode'])
def test_pilot_transfer_rejects_mutations_to_preserved_public_context(damage):
    old = a.read(a.PRIOR_PILOT/'decision_attempts/validation/v002.json')['observer']
    observer = deepcopy(old); observer['context'].update(mode='pilot_transfer', fixture_index=0)
    if damage == 'fit': observer['context']['fit'] = 161003
    elif damage == 'stream': observer['context']['stream'] = '0'*64
    elif damage == 'namespace': observer['context']['namespace'] = a.LIVE_NAMESPACE
    elif damage == 'identity': observer['context']['identity'][1] = 6
    elif damage == 'fixture': observer['context']['fixture_index'] = 1
    else: observer['context']['mode'] = 'legacy_transfer'
    with pytest.raises(ValueError): a.Audit(a.BASE).context(observer)


@pytest.mark.parametrize('game', [-1, 60, 120, True, 6.0])
def test_operational_context_does_not_reuse_pilot_or_unplanned_games(game):
    with pytest.raises(ValueError): a.context_for(game, 0)


def test_prospective_dual_clock_cap_protects_final_reserve(monkeypatch):
    audit = a.Audit(a.BASE); audit.config = deepcopy(a.CONFIG)
    audit.launch = {'epoch_seconds': 100., 'monotonic_start': 200.}
    monkeypatch.setattr(a.time, 'time', lambda: 100.+259200-600-.1)
    monkeypatch.setattr(a.time, 'monotonic', lambda: 200.+259200-600-.1); audit.boundary()
    monkeypatch.setattr(a.time, 'monotonic', lambda: 200.+259200-600)
    with pytest.raises(ValueError, match='dual-clock'): audit.boundary()
    monkeypatch.setattr(a.time, 'monotonic', lambda: 201.)
    monkeypatch.setattr(a.time, 'time', lambda: 100.+259200-600)
    with pytest.raises(ValueError, match='dual-clock'): audit.boundary()


@pytest.mark.parametrize('damage', [None, 'envelope', 'state', 'state_and_hash', 'seed', 'hash'])
def test_initial_world_literal_schema_and_full_equality_regression(tmp_path, monkeypatch, damage):
    from tools.external_strength_screen_20261018 import rules, audit as ledger
    from tools.analyze_external_strength_screen_20261018_02 import initial_snapshot
    spec = a.schedule({'schema': 'matched_strength_private_seed_material_v1', 'private_nonce': '1'*64})[0]
    snapshot = rules.serialize_snapshot(rules.CorrectedEnv(seed=int(spec['seed'])).initial_state())
    assert set(snapshot) == {'schema', 'ruleset_id', 'catalogue_sha256', 'state'}
    value = {**spec, 'initial': snapshot, 'initial_sha256': a.digest(snapshot)}
    if damage == 'envelope': value['initial']['ruleset'] = value['initial'].pop('ruleset_id')
    elif damage in ('state', 'state_and_hash'):
        value['initial']['state']['bank_tokens']['gold'] += 1
        if damage == 'state_and_hash': value['initial_sha256'] = a.digest(value['initial'])
    elif damage == 'seed': value['seed'] = str(int(spec['seed'])+1)
    elif damage == 'hash': value['initial_sha256'] = '0'*64
    write(tmp_path/'sealed/games/000/initial.json', value)
    audit = a.Audit(tmp_path); audit.initial_snapshot = initial_snapshot; audit.ledger = ledger
    audit.catalogue = a.read(a.ROOT/'outputs/external_strength_screen_20261018_02/catalogue.json')
    reader = a.read
    class InitialComparisonPassed(Exception): pass
    def read_until_game(path):
        if Path(path).name == 'result.json': raise InitialComparisonPassed()
        return reader(path)
    monkeypatch.setattr(a, 'read', read_until_game)
    if damage is None:
        with pytest.raises(InitialComparisonPassed): audit.game(spec)
    else:
        with pytest.raises(ValueError, match='Independent paired initial world'): audit.game(spec)


def pilot_lineage_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(a, 'ROOT', tmp_path); pilot = tmp_path/'outputs/pilot'; monkeypatch.setattr(a, 'PRIOR_PILOT', pilot)
    first = tmp_path/'model.bin'; second = tmp_path/'repair.py'
    write(first, {'fixed': 'model identity only'}); write(second, {'fixed': 'repair source'})
    write(pilot/'freeze.json', {'inputs': [a.artifact(first)]})
    repair = {'inputs': [a.artifact(second)], 'original_freeze': a.artifact(pilot/'freeze.json')}
    for key, filename in (('original_analyzer', 'original_analyzer.py'), ('original_completion', 'completion.json'),
                          ('original_final_input_check', 'final_input_check.json'), ('original_launch', 'launch.json')):
        write(pilot/filename, {'preserved': key}); repair[key] = a.artifact(pilot/filename)
    failures = [pilot/name for name in ('execution01.log', 'execution01.json', 'sealed/failure01.json')]
    for path in failures: write(path, {'preserved': str(path)})
    repair['preserved_failure'] = [a.artifact(path) for path in failures]
    write(pilot/'analyzer_repair_02/freeze.json', repair)
    write(pilot/'independent_analysis_v2.json', {'status': 'passed', 'outcomes_closed': True, 'strength_analysis_games': 0})
    report = tmp_path/'docs/result.md'; write(report, {'accepted': 'immutable report'})
    current = tmp_path/'AGENT_CURRENT.md'; handoff = tmp_path/'AGENT_HANDOFF.md'
    write(current, {'historical': True}); write(handoff, {'historical': True})
    final = {'status': 'accepted', 'outcomes_closed': True, 'strength_analysis_games': 0,
        'accepted_analysis_sha256': a.file_hash(pilot/'independent_analysis_v2.json'),
        'original_freeze_sha256': a.file_hash(pilot/'freeze.json'),
        'repair_freeze_sha256': a.file_hash(pilot/'analyzer_repair_02/freeze.json'),
        'artifacts': [a.artifact(path) for path in (report, current, handoff, pilot/'independent_analysis_v2.json')]}
    write(pilot/'finalization.json', final)
    write(current, {'subsequently_updated': True}); write(handoff, {'subsequently_updated': True})
    write(pilot/'sealed/evidence.json', {'this_body_must_not_be_read': True})
    hashes = {name: a.file_hash(pilot/name) for name in a.PILOT_HASHES}
    monkeypatch.setattr(a, 'PILOT_HASHES', hashes)
    descriptors = [a.artifact(pilot/name) for name in hashes]
    frozen = {p.resolve(): a.artifact(p) for p in tmp_path.rglob('*') if p.is_file()}
    reader = a.read
    def no_sealed_read(path):
        assert 'sealed' not in Path(path).parts
        return reader(path)
    monkeypatch.setattr(a, 'read', no_sealed_read)
    return pilot, first, second, descriptors, frozen


def test_prior_lineage_binds_accepted_freezes_and_all_evidence_without_reading_outcomes(tmp_path, monkeypatch):
    _, _, _, descriptors, frozen = pilot_lineage_fixture(tmp_path, monkeypatch)
    result = a.check_pilot_lineage(descriptors, frozen)
    assert result['all_four_lineage_artifacts_authenticated'] and result['all_original_repair_and_pilot_evidence_preserved']
    assert result['prior_strength_analysis_games'] == 0


@pytest.mark.parametrize('damage', ['lineage_order', 'lineage_hash', 'missing_original', 'rebound_original', 'missing_repair', 'missing_evidence',
    'changed_accepted_output', 'changed_failure_log', 'changed_original_completion', 'changed_original_launch', 'unaccepted', 'opened_outcomes', 'strength_used'])
def test_prior_lineage_rejects_missing_changed_or_unaccepted_evidence(tmp_path, monkeypatch, damage):
    pilot, first, second, descriptors, frozen = pilot_lineage_fixture(tmp_path, monkeypatch)
    if damage == 'lineage_order': descriptors.reverse()
    elif damage == 'lineage_hash': descriptors[0]['sha256'] = '0'*64
    elif damage == 'missing_original': del frozen[first.resolve()]
    elif damage == 'rebound_original': frozen[first.resolve()] = {**frozen[first.resolve()], 'sha256': '0'*64}
    elif damage == 'missing_repair': del frozen[second.resolve()]
    elif damage == 'missing_evidence': del frozen[(pilot/'sealed/evidence.json').resolve()]
    elif damage in ('changed_accepted_output', 'changed_failure_log', 'changed_original_completion', 'changed_original_launch'):
        path = {'changed_accepted_output': tmp_path/'docs/result.md', 'changed_failure_log': pilot/'execution01.log',
                'changed_original_completion': pilot/'completion.json', 'changed_original_launch': pilot/'launch.json'}[damage]
        write(path, {'changed_before_new_freeze': True}); frozen[path.resolve()] = a.artifact(path)
    else:
        final = a.read(pilot/'finalization.json')
        if damage == 'unaccepted': final['status'] = 'pending'
        elif damage == 'opened_outcomes': final['outcomes_closed'] = False
        else: final['strength_analysis_games'] = 6
        write(pilot/'finalization.json', final)
        a.PILOT_HASHES['finalization.json'] = a.file_hash(pilot/'finalization.json')
        descriptors = [a.artifact(pilot/name) for name in a.PILOT_HASHES]
    with pytest.raises(ValueError): a.check_pilot_lineage(descriptors, frozen)


@pytest.mark.parametrize('field,value', [('pair_count',31),('execution_cap_seconds',259201),
    ('bootstrap_seed',202610206702),('cumulative_memory_wait_seconds',21601),('confidence_level',.9)])
def test_exact_prospective_config_cannot_be_changed(field,value):
    assert a.check_config(a.CONFIG) == a.CONFIG
    changed = deepcopy(a.CONFIG); changed[field] = value
    with pytest.raises(ValueError): a.check_config(changed)


def test_private_referee_material_does_not_change_public_search_context():
    first={'schema':'matched_strength_private_seed_material_v1','private_nonce':'1'*64}
    second={**first,'private_nonce':'2'*64}
    assert {s['seed'] for s in a.schedule(first)}.isdisjoint({s['seed'] for s in a.schedule(second)})
    context=a.context_for(1,0)
    assert set(context)=={'mode','namespace','pair','game','turn','fit','stream','identity'}
    assert context['stream']==a.digest([a.SEARCH_TAG,a.LIVE_NAMESPACE,1])
    for invalid in ({}, {**first,'private_nonce':'x'*64}, {**first,'seed':1}):
        with pytest.raises(ValueError): a.schedule(invalid)


@pytest.mark.parametrize('seat,winner,expected', [(0,None,1),(0,0,2),(0,1,0),(1,0,0),(1,1,2)])
def test_verified_natural_score_mapping(seat,winner,expected):
    row=a.score_row({'game':0,'pair':0,'fit':161001,'local_seat':seat},{'reason':'natural','winner':winner})
    assert row['local_score_halfpoints']==expected and row['forfeiting_player'] is None


def test_genuine_timeout_is_retained_as_loss_and_other_leg_remains_planned():
    spec=a.schedule({'schema':'matched_strength_private_seed_material_v1','private_nonce':'1'*64})[0]
    state={'state':{'terminal':False,'winner':None}}
    game={'complete':True,'final':state,'reason':'forfeit_timeout','winner':1-spec['local_seat'],'failing_player':'local'}
    a.check_game_ending(game,state,('forfeit_timeout',spec['local_seat'],'local'))
    assert a.score_row(spec,game)['local_score_halfpoints']==0
    assert len(a.GAMES)==60 and 1 in a.GAMES


def test_failed_verification_never_calls_statistics(tmp_path,monkeypatch):
    from tools.matched_strength_20261020 import statistics
    audit=a.Audit(tmp_path); audit.schedule=[{'game':i} for i in range(60)]
    audit.freeze=lambda:None; audit.validation=lambda:None; audit.memory_gate=lambda *args:None
    audit.game=lambda *args:None
    monkeypatch.setattr(a,'artifact',lambda *args:{})
    def failed(): raise ValueError('incomplete verified manifest')
    audit.manifest=failed
    monkeypatch.setattr(statistics,'statistics',lambda *args,**kwargs:pytest.fail('Statistics before verification'))
    with pytest.raises(ValueError,match='manifest'):audit.run()


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
            'remaining_seconds': 250000.-elapsed})
    end = times[-1]+.001
    end_mono, end_epoch = mono_start+end, start+end
    charge = max(end_mono-mono_start, end_epoch-start, 0.)
    return {'schema': 'operational_memory_admission_v2', 'time': stamp(end_epoch), 'name': name,
        'status': 'passed', 'failure_category': None, 'failure_detail': None,
        'initial': initial, 'stability_required': needs, 'started_epoch_seconds': start,
        'finished_epoch_seconds': end_epoch, 'started_monotonic': mono_start, 'finished_monotonic': end_mono,
        'elapsed_seconds': end_mono-mono_start, 'charged_seconds': charge, 'prior_admission_count': count,
        'cumulative_before_seconds': before, 'cumulative_after_seconds': before+charge,
        'maximum_wait_seconds': 7200, 'maximum_cumulative_wait_seconds': 21600,
        'minimum_available_physical_bytes': 2*1024**3, 'minimum_available_commit_bytes': 4*1024**3,
        'sample_interval_seconds': 10, 'stable_seconds': 120, 'required_consecutive_samples': 13,
        'reserve_seconds': 600 if name in ('initial','analysis') else 4200 if name.startswith('audits/') else 64800,
        'request_seconds': 0 if name == 'initial' else 3600 if name == 'analysis' else 600 if name.startswith('audits/') else 120,
        'initialization_seconds': 60, 'readings': rows, 'no_worker_during_admission': True, 'outcomes_closed': True}


@pytest.mark.parametrize('name,low',[('initial',False),('validation/v000',False),('games/g000/t0000',True),
    ('audits/g000-t0000',False),('audits/validation/v000',False),('analysis',False),('analysis',True)])
def test_memory_admission_keeps_work_and_final_analysis_reserves(name,low):
    value=memory_receipt(name,low_first=low)
    a.check_memory_receipt(value,name,{'epoch_seconds':90.,'monotonic_start':190.},a.CONFIG)


@pytest.mark.parametrize('damage',['streak','threshold','clock','reserve','cumulative','request','extra','charge'])
def test_memory_tampering_rejected(damage):
    value=memory_receipt('initial')
    if damage=='streak':value['readings'][-1]['consecutive_admissible_samples']-=1
    elif damage=='threshold':value['minimum_available_physical_bytes']//=2
    elif damage=='clock':value['finished_epoch_seconds']+=100
    elif damage=='reserve':value['readings'][-1]['remaining_seconds']=659
    elif damage=='cumulative':value['cumulative_before_seconds']=21600.
    elif damage=='request':value['request_seconds']=120
    elif damage=='extra':value['readings'].append(deepcopy(value['readings'][-1]))
    else:value['charged_seconds']=0.
    with pytest.raises(ValueError):a.check_memory_receipt(value,'initial',{'epoch_seconds':90.,'monotonic_start':190.},a.CONFIG)


def test_memory_wall_backwards_never_refunds_monotonic_wait():
    value=memory_receipt('initial',start=1000.)
    for row in value['readings'][1:]:row['epoch_seconds']-=300
    value['finished_epoch_seconds']-=300
    value['charged_seconds']=value['elapsed_seconds'];value['cumulative_after_seconds']=value['charged_seconds']
    a.check_memory_receipt(value,'initial',{'epoch_seconds':90.,'monotonic_start':190.},a.CONFIG)
    assert value['charged_seconds']>=120


@pytest.mark.parametrize('damage',[None,'count','cumulative','overlap','extra','before_analysis','log'])
def test_exact_memory_inventory_serial_charge_and_analysis_gate(tmp_path,monkeypatch,damage):
    monkeypatch.setattr(a,'ROOT',tmp_path)
    audit=a.Audit(tmp_path/'run');audit.launch={'epoch_seconds':90.,'monotonic_start':190.};audit.config=deepcopy(a.CONFIG)
    names=['initial','validation/v000','audits/validation/v000','games/g000/t0000','audits/g000-t0000','analysis']
    before=0.;start=100.;last=None
    for index,name in enumerate(names):
        value=memory_receipt(name,start=start,before=before,count=index)
        if damage=='count' and index==2:value['prior_admission_count']=1
        if damage=='cumulative' and index==2:
            value['cumulative_before_seconds']+=1;value['cumulative_after_seconds']+=1
        if damage=='overlap' and index==2:value=memory_receipt(name,start=start-2,before=before,count=index)
        path=audit.base/'memory_admissions'/(name+'.json');path.parent.mkdir(parents=True,exist_ok=True)
        log=path.with_suffix('.jsonl');log.write_text(''.join(json.dumps(row)+'\n' for row in value['readings']))
        value['readings_log']={'path':log.relative_to(audit.base).as_posix(),'sha256':a.file_hash(log),'bytes':log.stat().st_size}
        write(path,value)
        if name!='analysis':audit.memory_gate(a.artifact(path),name)
        before+=value['charged_seconds'];start=value['finished_epoch_seconds']+1;last=value
    if damage=='extra':write(audit.base/'memory_admissions/unplanned.json',{})
    if damage=='log':(audit.base/'memory_admissions/analysis.jsonl').write_text('{}\n')
    import psutil
    monkeypatch.setattr(psutil,'Process',lambda:SimpleNamespace(create_time=lambda:last['finished_epoch_seconds']+(-1 if damage=='before_analysis' else 1)))
    if damage is None:
        audit.memory_inventory();assert audit.memory_summary['admissions']==6
        assert audit.memory_summary['charged_seconds']==pytest.approx(before)
    else:
        with pytest.raises(ValueError):audit.memory_inventory()


@pytest.mark.parametrize('damage',[None,'source_rebound','accepted_rebound','missing_file','extra_file','hold'])
def test_full_closed_operational_lineage_retains_accepted_hashes(tmp_path,monkeypatch,damage):
    old=tmp_path/'old';accepted=tmp_path/'accepted';base=tmp_path/'new'
    for path in (old,accepted,base):path.mkdir()
    monkeypatch.setattr(a,'ROOT',tmp_path);monkeypatch.setattr(a,'PRIOR_COMPLETED',old);monkeypatch.setattr(a,'PRIOR_ACCEPTANCE',accepted)
    code=tmp_path/'oldcode.py';write(code,{'fixed':'code'})
    write(old/'freeze.json',{'inputs':[a.artifact(code)]});write(old/'sealed.json',{'must_never_be_opened':True})
    inventory={'files':[a.artifact(p) for p in old.iterdir()]};write(accepted/'inventory.json',inventory)
    write(accepted/'repair_freeze.json',{'source_inventory':a.artifact(accepted/'inventory.json'),'inputs':[a.artifact(code)]})
    report=tmp_path/'oldreport.md';write(report,{'fixed':'accepted report'})
    write(accepted/'finalization.json',{'status':'operational_checks_passed_with_preserved_timeouts','games':18,
        'preserved_local_timeout_forfeits':4,'outcomes_closed':True,'strength_analysis_games':0,'artifacts':[a.artifact(report)]})
    write(accepted/'finalization_validation.json',{'status':'passed','outcomes_closed':True,'finalization':a.artifact(accepted/'finalization.json')})
    monkeypatch.setattr(a,'PRIOR_FREEZE_SHA',a.file_hash(old/'freeze.json'))
    monkeypatch.setattr(a,'PRIOR_ACCEPTANCE_SHA',a.file_hash(accepted/'finalization_validation.json'))
    audit=a.Audit(base);audit.frozen={'prior_acceptance':a.artifact(accepted/'finalization_validation.json')}
    items={p.resolve():a.artifact(p) for p in tmp_path.rglob('*') if p.is_file()}
    if damage in ('source_rebound','accepted_rebound'):
        target=code if damage=='source_rebound' else report
        write(target,{'changed_before_new_freeze':True});items[target.resolve()]=a.artifact(target)
    elif damage=='missing_file':del items[(old/'sealed.json').resolve()]
    elif damage=='extra_file':write(old/'extra.json',{})
    elif damage=='hold':write(accepted/'hold.json',{})
    reader=a.read
    def closed_reader(path):
        assert Path(path)!=old/'sealed.json'
        return reader(path)
    monkeypatch.setattr(a,'read',closed_reader)
    if damage is None:assert audit.accepted_history(items)['old_outcomes_opened'] is False
    else:
        with pytest.raises(ValueError):audit.accepted_history(items)


@pytest.mark.parametrize('damage',[None,'public_namespace','skip','missing','collision','unfrozen'])
def test_seed_inventory_private_namespace_complete_scan_no_skips(tmp_path,monkeypatch,damage):
    monkeypatch.setattr(a,'ROOT',tmp_path)
    base=tmp_path/'outputs/new';old=tmp_path/'outputs/old';base.mkdir(parents=True);old.mkdir()
    audit=a.Audit(base);audit.schedule=a.schedule({'schema':'matched_strength_private_seed_material_v1','private_nonce':'1'*64})
    target=old/'schedule.json';write(target,{'seed':1})
    if damage=='collision':write(target,{'nested':[audit.schedule[0]['seed']]})
    value={'namespace':a.PRIVATE_NAMESPACE,'distinct_seeds':30,'collisions':[],
        'checked':[a.artifact(target)],'skipped_oversize':[]}
    paths={target.resolve()}
    if damage=='public_namespace':value['namespace']=a.LIVE_NAMESPACE
    elif damage=='skip':value['skipped_oversize']=value['checked'];value['checked']=[]
    elif damage=='missing':write(old/'manifest.json',{})
    elif damage=='unfrozen':paths=set()
    write(base/'seed_collision_check.json',value)
    if damage is None:audit.seed_collisions(paths)
    else:
        with pytest.raises(ValueError):audit.seed_collisions(paths)



def test_fresh_worker_identity_cannot_be_reused_in_another_folder(tmp_path,monkeypatch):
    init,saved,freeze,_=initialization()
    init['command'][-1]='tools.matched_strength_20261020.worker'
    audit=a.Audit(tmp_path);audit.frozen=freeze;audit.cpu=0
    audit.check_clock=lambda *args:None;audit.process_identity=lambda *args:None;audit.memory_gate=lambda *args:0.
    audit.bound=lambda item,expected:expected
    monkeypatch.setattr(a,'artifact',lambda p:{'path':str(p),'sha256':a.file_hash(p),'bytes':Path(p).stat().st_size})
    def make(name):
        folder=tmp_path/'workers'/name
        receipt={'status':'timeout','result':None}
        write(folder/'initialization.json',init);write(folder/'request-00000001.json',receipt)
        return {'initialization':saved,'initialization_artifact':{},'receipt':{},'memory_admission':{},'clock':{'status':'timeout'}}
    first=make('games/g000/t0000');second=make('games/g001/t0000')
    audit.receipts(first,'games/g000/t0000','local')
    audit.receipts(first,'games/g000/t0000','local')
    with pytest.raises(ValueError,match='Fresh independent process'):
        audit.receipts(second,'games/g001/t0000','local')



def illegal_fixture(tmp_path,monkeypatch,player):
    from tools.external_strength_screen_20261018 import rules
    from tools.matched_strength_20261020 import runner
    from tools.analyze_external_strength_screen_20261018_02 import main_action
    from splendor_ai.engine.actions import Action,ActionType
    profile=a.read(a.ROOT/a.OPPONENT_PROFILE)
    snapshot=rules.serialize_snapshot(rules.CorrectedEnv(seed=71).initial_state())
    observer=rules.observation_payload(rules.deserialize_snapshot(snapshot))
    if player=='local':observer={**observer,'schema':a.LOCAL_SCHEMA,'context':a.context_for(1,0)}
    legal=[rules.serialize_action(x) for x in rules.legal_actions(rules.deserialize_snapshot(snapshot))]
    action=rules.serialize_action(Action(action_type=ActionType.PASS) if player=='local' else
        Action(action_type=ActionType.TAKE_TOKENS,take_tokens=('white','blue','green','red')))
    assert action not in legal
    trace=tmp_path/'workers/games/g001/t0000/continuations.jsonl.gz'
    trace.parent.mkdir(parents=True);trace.write_bytes(b'synthetic complete trace bytes')
    if player=='local':
        record={'root_identity':observer['context']['identity'],'profile':a.LOCAL_PROFILE,'ruleset_id':a.RULESET,
            'catalogue_sha256':a.CATALOGUE,'proposal':{'fit':161001,'model_id':'anchor_161001'},'planner':'reference',
            'candidates':{'actions':[action]},'choice':0,'worlds':[{'outcomes':[{}]} for _ in range(128)]}
        result={'decision':{'action':action,'record':record},'trace':{'bytes':trace.stat().st_size,
            'sha256':a.file_hash(trace),'records':128},'profile_id':a.LOCAL_PROFILE,'ruleset_id':a.RULESET,
            'catalogue_sha256':a.CATALOGUE,'observer_sha256':a.digest(observer)}
    else:
        result={'action':action,'status':'completed','player':'opponent','observer_sha256':a.digest(observer),
            'legal_actions_sha256':a.digest(legal),'legal_count':len(legal),'ruleset':a.RULESET,
            'catalogue_sha256':a.CATALOGUE,'profile_sha256':a.OPPONENT_SHA256,
            'diagnostics':{'opponent_id':'splendorust','profile_id':profile['profile_id'],
                'observer_seed':int(a.digest(observer)[:16],16),'phases':[{'action':main_action(action),
                    'simulations':6400,'leaf_calls':6401,
                    'native_legal_count':len({a.canonical(main_action(item)) for item in legal}),'heuristic_phase':False}],
                'inference_calls':6401,'inference_rows':6401,'forced_pass':False}}
    monkeypatch.setattr(a,'ROOT',tmp_path)
    audit=a.Audit(tmp_path);audit.rules=rules;audit.profile=profile;audit.check_diagnostics=check_diagnostics
    row={'id':'games/g001/t0000','observer':observer,'result_sha256':a.digest(result),'trace':a.artifact(trace)}
    receipt={'result':result,'received_before_deadline':True}
    return audit,row,snapshot,receipt,result,legal,trace,runner


@pytest.mark.parametrize('player',['local','opponent'])
def test_illegal_action_requires_retained_recognized_timely_evidence(tmp_path,monkeypatch,player):
    audit,row,snapshot,receipt,result,legal,trace,runner=illegal_fixture(tmp_path,monkeypatch,player)
    assert runner.response_has_recognized_illegal_action(result,player,row['observer'],legal,trace)
    audit.illegal_response(row,snapshot,receipt,player)
    receipt['received_before_deadline']=False
    with pytest.raises(ValueError,match='Timely'):audit.illegal_response(row,snapshot,receipt,player)


@pytest.mark.parametrize('damage',['observer','profile','work','choice','selected','trace','fit','legal'])
def test_illegal_local_classification_does_not_mask_wrapper_or_work_failure(tmp_path,monkeypatch,damage):
    audit,row,snapshot,receipt,result,legal,trace,runner=illegal_fixture(tmp_path,monkeypatch,'local')
    if damage=='observer':result['observer_sha256']='0'*64
    elif damage=='profile':result['decision']['record']['profile']='other'
    elif damage=='work':result['decision']['record']['worlds'].pop()
    elif damage=='choice':result['decision']['record']['choice']=True
    elif damage=='selected':result['decision']['record']['candidates']['actions'][0]=legal[0]
    elif damage=='trace':result['trace']['sha256']='0'*64
    elif damage=='fit':result['decision']['record']['proposal']['fit']=161002
    else:
        result['decision']['action']=legal[0];result['decision']['record']['candidates']['actions'][0]=legal[0]
    row['result_sha256']=a.digest(result)
    assert not runner.response_has_recognized_illegal_action(result,'local',row['observer'],legal,trace)
    with pytest.raises((ValueError,KeyError)):audit.illegal_response(row,snapshot,receipt,'local')


@pytest.mark.parametrize('damage',['observer','work','phase','legal_count','seed','action','empty_phases'])
def test_illegal_opponent_classification_preserves_identity_and_work_holds(tmp_path,monkeypatch,damage):
    audit,row,snapshot,receipt,result,legal,trace,runner=illegal_fixture(tmp_path,monkeypatch,'opponent')
    if damage=='observer':result['observer_sha256']='0'*64
    elif damage=='work':result['diagnostics']['phases'][0]['simulations']=1
    elif damage=='phase':result['diagnostics']['phases'][0]['leaf_calls']=0
    elif damage=='legal_count':result['legal_count']+=1
    elif damage=='seed':result['diagnostics']['observer_seed']+=1
    elif damage=='empty_phases':
        result['diagnostics'].update(phases=[],inference_calls=0,inference_rows=0)
    else:result['action']['tier']=True
    row['result_sha256']=a.digest(result)
    assert not runner.response_has_recognized_illegal_action(result,'opponent',row['observer'],legal,trace)
    with pytest.raises((ValueError,KeyError)):audit.illegal_response(row,snapshot,receipt,'opponent')
