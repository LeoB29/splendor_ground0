"""Synthetic audit regressions and existing receipt schemas; no inference/games."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from tools import analyze_live_game_operational_20261019 as a
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
    rows = a.schedule(); assert [r['game'] for r in rows] == [6, 8, 10, 7, 9, 11, 12, 14, 16, 13, 15, 17, 18, 20, 22, 19, 21, 23]
    for pair in range(3, 12):
        both = [r for r in rows if r['pair'] == pair]
        assert len({r['seed'] for r in both}) == 1 and {r['local_seat'] for r in both} == {0, 1}
        assert {r['fit'] for r in both} == {a.FITS[(pair-3)%3]}
    context = a.context_for(9, 37); sample = a.sample_worlds(context['identity'])
    assert sample == sorted(sample) and len(set(sample)) == 8
    expected = sorted(range(128), key=lambda wi: (a.digest([a.AUDIT_DOMAIN, 'world', a.AUDIT_SEED, context['identity'], wi]), wi))[:8]
    assert sample == sorted(expected)
    assert a.local_profile()['worlds'] == 128 and a.local_profile()['candidates'] == 4
    assert a.local_profile()['parent_profile_id'] == 'four_candidate128_corrected_rules_20261018_v1'
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
    audit = a.Audit(tmp_path); audit.launch = {'epoch_seconds': a.time.time()}; audit.frozen = freeze
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


def test_new_contexts_and_validation_producer_match_independent_contract():
    from tools.live_game_operational_20261019 import common, runner
    assert a.schedule() == common.make_schedule()
    assert len(a.schedule()) == 18 and len({v['seed'] for v in a.schedule()}) == 9
    assert {fit: sum(v['fit'] == fit for v in a.schedule()) for fit in a.FITS} == dict.fromkeys(a.FITS, 6)
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
        else: assert case['observer']['context'] == a.context_for((6, 9, 10)[index//4], case['snapshot']['state']['turn_index'])
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


@pytest.mark.parametrize('game', [-1, 0, 5, 24, 30, True, 6.0])
def test_operational_context_does_not_reuse_pilot_or_unplanned_games(game):
    with pytest.raises(ValueError): a.context_for(game, 0)


def test_prospective_twenty_four_hour_boundary_is_not_a_new_allowance_per_stage(monkeypatch):
    audit = a.Audit(a.BASE); audit.launch = {'epoch_seconds': 100.}
    monkeypatch.setattr(a.time, 'time', lambda: 100.+86400-.1); audit.boundary()
    monkeypatch.setattr(a.time, 'time', lambda: 100.+86400)
    with pytest.raises(ValueError, match='twenty-four'): audit.boundary()


@pytest.mark.parametrize('damage', [None, 'envelope', 'state', 'state_and_hash', 'seed', 'hash'])
def test_initial_world_literal_schema_and_full_equality_regression(tmp_path, monkeypatch, damage):
    from tools.external_strength_screen_20261018 import rules, audit as ledger
    from tools.analyze_external_strength_screen_20261018_02 import initial_snapshot
    spec = a.schedule()[0]
    snapshot = rules.serialize_snapshot(rules.CorrectedEnv(seed=int(spec['seed'])).initial_state())
    assert set(snapshot) == {'schema', 'ruleset_id', 'catalogue_sha256', 'state'}
    value = {**spec, 'initial': snapshot, 'initial_sha256': a.digest(snapshot)}
    if damage == 'envelope': value['initial']['ruleset'] = value['initial'].pop('ruleset_id')
    elif damage in ('state', 'state_and_hash'):
        value['initial']['state']['bank_tokens']['gold'] += 1
        if damage == 'state_and_hash': value['initial_sha256'] = a.digest(value['initial'])
    elif damage == 'seed': value['seed'] = str(int(spec['seed'])+1)
    elif damage == 'hash': value['initial_sha256'] = '0'*64
    write(tmp_path/'sealed/games/006/initial.json', value)
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
