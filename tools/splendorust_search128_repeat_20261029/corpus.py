"""Prospective, finite qualification of the actual Entity policy/search workers.

No import loads a model. Production calls belong inside the frozen stage clock.
The frozen 23 October structural fixtures are reused as corrected-rule
counterexamples, never as a claim that the new entry point is already qualified.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import struct
from pathlib import Path

from tools import future_training_rules_20261021 as guard
from tools.external_strength_screen_20261018 import audit, rules
from tools.external_strength_screen_20261018_02 import bridge
from tools.move_evaluation_20261023 import qualification as reference

ROOT = Path(__file__).resolve().parents[2]
PRIOR = ROOT / 'outputs/external_strength_screen_20261018_02'
PANEL = PRIOR / 'fixtures.json'
PARITY_RECEIPT = PRIOR / 'qualification/source_attempt_02.json'
MODEL_BIN = ROOT / ('outputs/external_opponent_search_20261018_02/splendorust/'
                   'web/public/models/57f6e227f8ac0382b7fa67dba6b58ec6663d1f635c7b9f583937867cd6692deb.bin')
ARMS = ('search', 'reference')


def require(value, message):
    if not value:
        raise ValueError('One-second qualification: ' + message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def artifact(path):
    path = Path(path)
    return {'path': path.relative_to(ROOT).as_posix(), 'bytes': path.stat().st_size,
            'sha256': guard.sha256(path)}


def hidden_variant(state):
    """The unchanged earlier screen's hidden-only permutation."""
    variant = state.copy_shallow()
    for tier in (1, 2, 3):
        variant.hidden_tier_decks[tier].reverse()
    opponent = variant.players[1 - state.current_player]
    for index, card in enumerate(opponent.reserved_cards):
        if opponent.reserved_card_source(index) == 'deck' and variant.hidden_tier_decks[card.tier]:
            opponent.reserved_cards[index], variant.hidden_tier_decks[card.tier][0] = (
                variant.hidden_tier_decks[card.tier][0], card)
    return variant


def build_fixtures():
    return reference.build_fixtures()




def build_plan(quota=64):
    """Materialize all fixture identities, counts and ordering before launch."""
    require(type(quota) is int and quota in (64, 128), 'challenger quota64or128')
    original = read(PANEL)
    require(len(original) == 23, 'unchanged panel count')
    fixtures = []
    for index, item in enumerate(original):
        state = guard.decode_training_snapshot(item['snapshot'])
        changed = hidden_variant(state)
        observer = rules.observation_payload(state)
        require(rules.observation_payload(changed) == observer, 'variant changes lawful observation')
        variant = rules.serialize_snapshot(changed)
        require(variant != item['snapshot'], 'variant did not change hidden referee state')
        audit.validate_ledger(variant)
        fixtures.append({'fixture_index': index, 'fixture_id': item['source']['root_id'],
                         'original': item['snapshot'], 'hidden_variant': variant,
                         'observer_sha256': digest(observer),
                         'prior_search_reference': artifact(PRIOR / 'dynamic_qualification' / f'{4 * index + 2:03d}.json')})
    counterexamples = build_fixtures()
    seat_one = reference._initial()
    seat_one.current_player = seat_one.start_player = 1
    return {'schema': 'splendorust_search128_structural_suite_v1',
            'challenger_quota': quota, 'reference_quota': 16,
            'ruleset_id': guard.RULESET_ID, 'catalogue_sha256': guard.CATALOGUE_SHA256,
            'turn_contract': guard.TURN_CONTRACT, 'panel_source': artifact(PANEL),
            'fixtures': fixtures,
            'counterexamples': counterexamples,
            'alternate_starting_seat': rules.serialize_snapshot(seat_one),
            'actual_complete_decisions': {'fixture_choices': 30, 'private_repeats': 2,
                'alternate_starting_seat': 2, 'mixed_final_round_trajectory': 3,
                'frozen_puct16_reference': 15, 'total': 52},
            'native_operations': {'catalogue': 2, 'search_inspect': 28, 'forced_auxiliary': 6,
                'fresh_numeric_parity': 46, 'total': 82},
            'worker_rejection_cases': rejection_case_names(),
            'worker_rejection_attempts': 2 * len(rejection_case_names()),
            'work_accounting': 'native leaf evaluations and actual host neural callbacks are distinct; exact branch-aware validation',
            'regression_cases': ['turn123', 'turn124'],
            'forensic_evidence': 'append each full request and returned result before its assertions; preserve failures and partial callbacks',
            'parity': {'old_receipt': artifact(PARITY_RECEIPT), 'model_bin': artifact(MODEL_BIN),
                'fresh_checks': 46, 'max_abs_delta': .0005,
                'old_receipt_scope': 'source provenance; historical external Torch bytes not fully bound there',
                'fresh_scope': 'fixed additional entry-point parity under the newly frozen runtime'},
            'training_loader_qualified': False}




def scientific_result(arm, action, trace):
    """Exclude capture/envelope fields while retaining actual PUCT scientific work."""
    require(arm in ARMS, 'unknown scientific arm')
    keys = ('opponent_id', 'profile_id', 'observer_seed', 'forced_pass',
            'inference_calls', 'inference_rows')
    normalized = {key: trace[key] for key in keys}
    phase_keys = ('action', 'heuristic_phase', 'leaf_calls', 'native_legal_count', 'simulations')
    normalized['phases'] = [{key: phase[key] for key in phase_keys} for phase in trace['phases']]
    return {'action': action, 'diagnostics': normalized}

def callback_receipts(trace):
    """New wrapper evidence compared across repeats, outside old trace schema."""
    return [{key: phase[key] for key in ('neural_callbacks', 'neural_callback_count', 'neural_callback_rows')}
            for phase in trace['phases']]


def validate_scientific_decision(arm, observer, action, trace, quota=None):
    """Independent PUCT/compound checks bound to an externally supplied quota."""
    from .adapters import profile_id
    from .scientific import validate_search_work
    require(arm in ARMS, 'unknown arm')
    if arm == 'reference':
        require(quota is None or (type(quota) is int and quota == 16), 'reference quota changed')
        quota = 16
    else:
        require(type(quota) is int and quota in (64, 128), 'external challenger quota required')
    require(trace['profile_id'] == profile_id(quota) and trace['opponent_id'] == 'splendorust',
            'search scientific identity')
    state = rules.state_from_observation(observer)
    legal = rules.legal_actions(state)
    selected = rules.deserialize_action(action)
    require(selected in legal, 'returned complete action is illegal')
    expected_seed = int(digest(observer)[:16], 16)
    require(trace['observer_seed'] == expected_seed, 'observer seed differs')
    mains = bridge.main_actions(observer)
    forced = mains == [{'kind': 'Pass'}]
    require(trace['forced_pass'] is forced, 'forced-pass flag')
    if forced:
        require(not trace['phases'] and trace['inference_calls'] == trace['inference_rows'] == 0,
                'forced pass used neural search')
        return {'forced_pass': True, 'simulations': 0, 'inference_calls': 0}
    phases = trace['phases']
    require(bool(phases), 'missing Main phase')
    require(phases[0]['action'] in mains, 'Main action is not legal')
    require(phases[0]['action'] == bridge._main(selected), 'Main/complete-action disagreement')
    require(phases[0]['native_legal_count'] == len(mains), 'Main legal count differs')
    phase_cursor = [1]
    def select_aux(projection, choices):
        require(phase_cursor[0] < len(phases), 'missing auxiliary phase')
        phase = phases[phase_cursor[0]]
        phase_cursor[0] += 1
        require(phase['action'] in choices and phase['native_legal_count'] == len(choices), 'auxiliary support')
        return phase['action']
    resolved = bridge.resolve_compound(observer, phases[0]['action'], select_aux)
    require(resolved == selected and phase_cursor[0] == len(phases), 'complete compound response differs')
    for index, phase in enumerate(phases):
        heuristic = index > 0 or len(mains) == 1 or state.turn_index >= 124
        require(phase['heuristic_phase'] is heuristic, 'search branch')
        require(type(phase['simulations']) is int and phase['simulations'] == (0 if heuristic else quota),
                'search simulation quota')
        require(type(phase['leaf_calls']) is int and phase['leaf_calls'] >= 0 and
                (not heuristic or phase['leaf_calls'] == 0), 'search leaf counts')
    work = validate_search_work(trace, observer, quota=quota)
    return {'forced_pass': False, 'simulations': sum(p.get('simulations', 0) for p in phases),
            'inference_calls': trace['inference_calls'], 'search_work': work}



def rejection_case_names():
    return ['extra_request_field', 'unknown_arm', 'missing_observer', 'bare_historical_observation',
            'missing_schema', 'old_schema', 'missing_ruleset', 'old_ruleset', 'missing_catalogue',
            'wrong_catalogue', 'referee_seed', 'hidden_decks', 'opponent_private_card']


def worker_rejections(request_callback, private_state, *, evidence_callback=lambda row: None):
    observer = rules.observation_payload(private_state)
    rows = []
    for arm in ARMS:
        for name in rejection_case_names():
            request = {'player': arm, 'observer': deepcopy(observer)}
            view = request['observer']
            if name == 'extra_request_field': request['seed'] = 1
            elif name == 'unknown_arm': request['player'] = 'puct6400'
            elif name == 'missing_observer': del request['observer']
            elif name == 'bare_historical_observation': request['observer'] = view['observation']
            elif name.startswith('missing_'): del view[{'schema': 'schema', 'ruleset': 'ruleset_id', 'catalogue': 'catalogue_sha256'}[name[8:]]]
            elif name == 'old_schema': view['schema'] = 'historical'
            elif name == 'old_ruleset': view['ruleset_id'] = 'historical'
            elif name == 'wrong_catalogue': view['catalogue_sha256'] = '0' * 64
            elif name == 'referee_seed': view['observation']['seed'] = 1
            elif name == 'hidden_decks': view['observation']['hidden_tier_decks'] = {}
            elif name == 'opponent_private_card':
                seat = 1 - private_state.current_player
                view['observation']['players'][seat]['reserves'][0]['card_id'] = private_state.players[seat].reserved_cards[0].card_id
            evidence = {'case_id': 'worker_rejection/' + name, 'arm': arm,
                        'operation': 'ordinary_worker_request', 'request': request}
            def attempt():
                evidence_callback({**evidence, 'event': 'started'})
                try:
                    result = request_callback(request)
                except BaseException as error:
                    evidence_callback({**evidence, 'event': 'rejected', 'error': repr(error)})
                    raise
                evidence_callback({**evidence, 'event': 'result', 'result': result})
                return result
            rows.append({'arm': arm, 'case': reference._rejected(attempt, name)})
    return rows


def model_free_checks(quota=64):
    """Separate preparation tests, explicitly not model-bearing admission."""
    source_receipt = guard.verify_runtime_sources()
    fixtures = build_fixtures()
    rows, phases = [], set()
    for item in fixtures:
        state = guard.decode_training_snapshot(item['snapshot'])
        count, covered = reference._compound_all(state)
        phases.update(covered)
        rows.append({'id': item['id'], 'atomic_transitions': count})
    require(phases == {'Payment', 'Return', 'Noble'}, 'atomic phase coverage')
    plan = build_plan(quota)
    return {'status': 'model_free_checks_passed', 'production_entrypoint_qualified': False,
            'runtime_sources': source_receipt, 'fixtures': rows, 'structural_fixture_count': len(plan['fixtures']),
            'semantic_edges': reference._semantic_edges(fixtures),
            'strict_identity_rejections': reference.identity_rejections(fixtures[0])}


def _actual_encoded_costs(state, trace, *, corrected_face=False):
    """Independently check corrected faces in the actual host root input tensor."""
    if trace['forced_pass'] or not trace['phases'][0]['neural_callback_count']:
        require(not corrected_face, 'corrected face did not reach actual neural encoder')
        return {'neural_input': False, 'visible_cards_checked': 0}
    tokens = trace['phases'][0]['inference'][0]['tokens']
    require(len(tokens) == 31 * 48 and all(type(x) in (int, float) and math.isfinite(x) for x in tokens),
            'actual entity root tensor shape')
    def f32(value):
        return struct.unpack('<f', struct.pack('<f', value))[0]
    checked = []
    for tier in (1, 2, 3):
        for slot, card in enumerate(state.visible_tier_cards[tier]):
            index = 4 * (tier - 1) + slot
            actual = tokens[(3 + index) * 48:(3 + index) * 48 + 5]
            expected = [f32(f32(card.cost.get(color, 0)) * f32(.1)) for color in bridge.COLORS]
            require(actual == expected, 'actual model entity input card costs')
            checked.append(card.card_id)
    return {'neural_input': True, 'visible_cards_checked': len(checked),
            'card_ids': checked, 'actual_root_tokens_sha256': digest(tokens)}


def _native_catalogues(adapters):
    rows = []
    for arm in ARMS:
        actual = adapters[arm].exchange({'op': 'catalogue'})
        require(len(actual['cards']) == 90 and len(actual['nobles']) == 10, 'native catalogue size')
        for index, card_id in enumerate(bridge.RUST_CARD_IDS):
            card = rules.corrected_cards_by_id()[card_id]
            require(actual['cards'][index] == {'tier': card.tier - 1, 'bonus': bridge.COLORS.index(card.bonus_color),
                'points': card.points, 'cost': [card.cost.get(c, 0) for c in bridge.COLORS]}, 'native card face ' + card_id)
        for index, noble_id in enumerate(bridge.RUST_NOBLE_IDS):
            noble = rules.corrected_nobles_by_id()[noble_id]
            require(actual['nobles'][index] == [noble.requirement.get(c, 0) for c in bridge.COLORS], 'native noble')
        rows.append({'arm': arm, 'cards_matched': 90, 'nobles_matched': 10, 'catalogue_response_sha256': digest(actual)})
    return rows


def _fresh_parity(search, boundary):
    """Authenticate old dependencies, then repeat the fixed native/host check."""
    old = read(PARITY_RECEIPT)
    require(old['status'] == 'passed' and old['cards_matched'] == 90 and old['nobles_matched'] == 10,
            'historical source parity status')
    for item in old['profile_artifacts']:
        require(artifact(ROOT / item['path']) == item, 'old parity source dependency changed: ' + item['path'])
    rows = []
    for index, fixture in enumerate(read(PANEL)):
        boundary()
        observer = rules.observation_payload(guard.decode_training_snapshot(fixture['snapshot']))
        native = bridge.encode_observation(observer)
        response = search.exchange({'op': 'parity', 'observation': native,
            'seed': int(digest(observer)[:16], 16), 'model_bin': str(MODEL_BIN)})
        delta = response['max_abs_delta']
        require(type(delta) in (float, int) and math.isfinite(delta) and 0 <= delta <= .0005, 'native/host numeric parity')
        rows.append({'fixture_index': index, 'observer_sha256': digest(observer), 'response': response})
    return {'status': 'passed', 'checks': len(rows), 'max_abs_tolerance': .0005, 'rows': rows,
            'old_source_receipt': artifact(PARITY_RECEIPT), 'authenticated_old_source_dependencies': len(old['profile_artifacts']),
            'runtime_scope': 'fresh comparison under current frozen runtime; not proof of historical external library byte identity'}


def _auxiliary_checks(adapters, fixtures, boundary):
    wanted = {'face_T1_05': {'kind': 'BuyVisible', 'index': 0},
              'gold_returns': {'kind': 'ReserveDeck', 'tier': 0},
              'multiple_nobles': {'kind': 'Take', 'tokens': [1, 1, 1, 0, 0]}}
    rows, covered = [], set()
    for fixture in fixtures:
        if fixture['id'] not in wanted:
            continue
        state = guard.decode_training_snapshot(fixture['snapshot'])
        observer = rules.observation_payload(state)
        by_arm = {}
        for arm in ARMS:
            phases = []
            def select_aux(projection, choices):
                boundary()
                phase = projection['phase']['kind']
                require(phase in {'Payment', 'Return', 'Noble'}, 'forced nonauxiliary projection')
                request = {'op': 'select',
                    'observation': projection, 'legal': choices, 'seed': int(digest(observer)[:16], 16)}
                request['iterations'] = adapters[arm].profile['iterations']
                result = adapters[arm].exchange(request)
                require(result['action'] in choices and result['simulations'] == result['leaf_calls'] == 0,
                        'forced auxiliary action/work')
                require(result['heuristic_phase'] is True, 'search auxiliary branch')
                phases.append({'phase': phase, 'projection_sha256': digest(projection), 'result': result})
                covered.add(phase)
                return result['action']
            action = bridge.resolve_compound(observer, wanted[fixture['id']], select_aux)
            reference._step_check(state, action)
            by_arm[arm] = {'action': rules.serialize_action(action), 'phases': phases}
        require(by_arm['search']['action'] == by_arm['reference']['action'], 'arms differ on unchanged auxiliary heuristic')
        require([(p['phase'], p['projection_sha256'], p['result']['action']) for p in by_arm['search']['phases']] ==
                [(p['phase'], p['projection_sha256'], p['result']['action']) for p in by_arm['reference']['phases']],
                'auxiliary projections or choices differ')
        rows.append({'id': fixture['id'], 'forced_main': wanted[fixture['id']], 'arms': by_arm})
    require(covered == {'Payment', 'Return', 'Noble'}, 'actual auxiliary coverage')
    require(sum(len(row['arms'][arm]['phases']) for row in rows for arm in ARMS) == 6, 'fixed auxiliary request count')
    return rows


class _RecordedAdapter:
    """Record native/reference returns before the consuming checker can fail.

    This delegates to the same adapter methods without changing their inputs,
    arithmetic or native operation. The observer/snapshot binding is forensic
    context, never supplied to the native player.
    """
    def __init__(self, adapter, arm, fixtures, evidence_callback):
        self._adapter = adapter
        self._arm = arm
        self._record = evidence_callback
        self._fixtures = {}
        for fixture in fixtures:
            observer = rules.observation_payload(guard.decode_training_snapshot(fixture['snapshot']))
            self._fixtures[digest(observer)] = fixture
            self._fixtures[digest(bridge.encode_observation(observer))] = fixture

    def __getattr__(self, name):
        method = getattr(self._adapter, name)
        if name not in ('exchange', 'inspect', 'native_reference', 'reference_choose'):
            return method
        def recorded(*args, **kwargs):
            request = ({'native_request': args[0]} if name == 'exchange' else
                       {'observer': args[0], 'arguments': list(args[1:]), 'options': kwargs})
            observer = request.get('observer', request.get('native_request', {}).get('observation'))
            fixture = self._fixtures.get(digest(observer)) if observer is not None else None
            native_kind = request.get('native_request', {}).get('op', name)
            evidence = {'case_id': fixture['id'] if fixture else 'native/' + native_kind + '/' + digest(request)[:16],
                        'arm': self._arm, 'operation': name, 'request': request}
            if fixture is not None:
                evidence['before_snapshot'] = fixture['snapshot']
            self._record({**evidence, 'event': 'started'})
            try:
                result = method(*args, **kwargs)
            except BaseException as error:
                self._record({**evidence, 'event': 'failed', 'error': repr(error)})
                raise
            saved = ({'action': rules.serialize_action(result[0]), 'diagnostics': result[1]}
                     if name == 'reference_choose' else result)
            self._record({**evidence, 'event': 'result', 'result': saved})
            return result
        return recorded


def _run_original_entrypoint_checks(adapters, *, boundary=lambda: None, request_callback=None,
                          evidence_callback=lambda row: None):
    """Run inside an initialized authenticated worker, never during preparation.

    request_callback is the worker's ordinary handler, so the exact decoder,
    legal support and response envelope are exercised. The worker authenticates
    sources once at startup before invoking this function.
    """
    from . import scientific as independent
    require(request_callback is not None and set(adapters) == set(ARMS), 'actual worker callback required')
    quota = adapters['search'].profile['iterations']
    require(adapters['reference'].profile['iterations'] == 16, 'reference quota')
    plan = build_plan(quota)
    fixtures = plan['counterexamples']
    evidence_fixtures = fixtures + [{'id': 'panel/' + fixture['fixture_id'], 'snapshot': fixture['original']}
                                  for fixture in plan['fixtures']]
    adapters = {arm: _RecordedAdapter(adapter, arm, evidence_fixtures, evidence_callback)
                for arm, adapter in adapters.items()}
    counts = {'complete_decisions': 0}
    def decide(arm, state, case_id):
        boundary()
        before = rules.serialize_snapshot(state)
        decoded = guard.decode_training_snapshot(before)
        observer = rules.observation_payload(decoded)
        audit.check_observer(before, observer)
        request = {'player': arm, 'observer': observer}
        evidence = {'case_id': case_id, 'arm': arm, 'operation': 'ordinary_worker_request',
                    'before_snapshot': before, 'request': request}
        evidence_callback({**evidence, 'event': 'started'})
        try:
            result = request_callback(request)
        except BaseException as error:
            evidence_callback({**evidence, 'event': 'failed', 'error': repr(error)})
            raise
        evidence_callback({**evidence, 'event': 'result', 'result': result})
        require(result['player'] == arm and result['observer_sha256'] == digest(observer), 'worker result binding')
        independent.validate_result(result, observer, arm, expected_quota=quota if arm == 'search' else 16)
        checked = validate_scientific_decision(arm, observer, result['action'], result['diagnostics'],
                                               quota=quota if arm == 'search' else 16)
        after = reference._step_check(decoded, rules.deserialize_action(result['action']))
        evidence_callback({**evidence, 'event': 'validated', 'check': checked,
                           'after_snapshot': rules.serialize_snapshot(after)})
        counts['complete_decisions'] += 1
        return result, after

    rows, reference_reproductions, privacy, regressions, native_inspections = [], [], [], [], []
    encoded_cost_checks = []
    catalogues = _native_catalogues(adapters)
    for fixture in fixtures:
        state = guard.decode_training_snapshot(fixture['snapshot'])
        observer = rules.observation_payload(state)
        results = {}
        for arm in ARMS:
            result, _ = decide(arm, state, fixture['id'])
            results[arm] = result
            encoded = _actual_encoded_costs(state, result['diagnostics'],
                                           corrected_face='corrected_face' in fixture['tags'])
            encoded_cost_checks.append({'case_id': fixture['id'], 'arm': arm, **encoded})
            evidence_callback({'event': 'validated', 'operation': 'actual_entity_encoded_costs',
                               'case_id': fixture['id'], 'arm': arm, 'result': encoded})
        if fixture['id'] in ('turn123', 'turn124'):
            for arm in ARMS:
                actual_quota = quota if arm == 'search' else 16
                search_work = independent.validate_search_work(results[arm]['diagnostics'], observer, quota=actual_quota)
                expected_calls = int(fixture['id'] == 'turn123')
                require(search_work['inference_calls'] == expected_calls, 'late boundary host neural callback regression')
                require(search_work['simulations'] == (actual_quota if expected_calls else 0),
                        'late boundary simulation regression')
                if expected_calls:
                    require(search_work['leaf_evaluations'] > search_work['inference_calls'],
                            'turn123 did not exercise non-neural searched leaves')
                regressions.append({'case_id': fixture['id'], 'arm': arm, 'work': search_work})
                evidence_callback({'event': 'validated', 'operation': 'late_boundary_regression',
                                   'case_id': fixture['id'], 'arm': arm, 'result': search_work})
        if fixture['id'] != 'forced_pass':
            for arm in ARMS:
                inspected = adapters[arm].exchange({'op': 'inspect', 'observation': bridge.encode_observation(observer),
                                                    'seed': int(digest(observer)[:16], 16)})
                require(inspected['legal'] == bridge.main_actions(observer), 'search native ordered legal support')
                native_inspections.append({'case_id': fixture['id'], 'arm': arm, 'inspection': inspected})
        # Invoke the original unbound PUCT16 choose path again, with fresh agents.
        expected_action, expected_trace = adapters['reference'].reference_choose(observer, diagnostic_capture=True)
        counts['complete_decisions'] += 1
        require(scientific_result('reference', results['reference']['action'], results['reference']['diagnostics']) ==
                scientific_result('reference', rules.serialize_action(expected_action), expected_trace),
                'frozen PUCT16 reference changed')
        require(callback_receipts(results['reference']['diagnostics']) == callback_receipts(expected_trace),
                'reference host callback reproduction changed')
        reference_reproductions.append({'id': fixture['id'], 'trace': expected_trace})
        if 'corrected_face' in fixture['tags']:
            search_input = results['search']['diagnostics']['phases'][0]['inference'][0]['tokens']
            reference_input = results['reference']['diagnostics']['phases'][0]['inference'][0]['tokens']
            require(search_input == reference_input, 'same corrected face encoded differently across quotas')
        if fixture['id'] == 'private_reserves':
            changed = hidden_variant(state)
            require(rules.observation_payload(changed) == observer and rules.serialize_snapshot(changed) != fixture['snapshot'],
                    'private variant changed public information')
            for arm in ARMS:
                repeated, _ = decide(arm, changed, fixture['id'] + '/hidden_variant')
                require(scientific_result(arm, repeated['action'], repeated['diagnostics']) ==
                        scientific_result(arm, results[arm]['action'], results[arm]['diagnostics']), 'private hidden dependence')
                require(callback_receipts(repeated['diagnostics']) == callback_receipts(results[arm]['diagnostics']),
                        'private hidden dependence in actual host callbacks')
                privacy.append({'arm': arm, 'equal': True})
        rows.append({'id': fixture['id'], 'observer_sha256': digest(observer), 'results': results})

    require(len(native_inspections) == 28, 'native ordered support inspection count')
    native = {'status': 'passed', 'checks': native_inspections, 'count': 28,
              'scope': 'actual PUCT native ordered legal sets, corrected catalogue, host tokens and branch counts'}
    auxiliary = _auxiliary_checks(adapters, fixtures, boundary)
    seat_one = guard.decode_training_snapshot(plan['alternate_starting_seat'])
    starting_seat = [{'arm': arm, 'result': decide(arm, seat_one, 'alternate_starting_seat')[0]} for arm in ARMS]
    trajectory_state = guard.decode_training_snapshot(next(f['snapshot'] for f in fixtures if f['id'] == 'noble_final_round'))
    trajectory_state.current_player, trajectory_state.turn_index = 1, 1
    trajectory = []
    for step, arm in enumerate(('reference', 'search', 'reference')):
        require(not trajectory_state.terminal, 'mixed trajectory ended prematurely')
        before = rules.serialize_snapshot(trajectory_state)
        result, trajectory_state = decide(arm, trajectory_state, 'mixed_final_round/' + str(step))
        trajectory.append({'arm': arm, 'before': before, 'result': result,
                           'after': rules.serialize_snapshot(trajectory_state)})
    require(trajectory_state.terminal and guard.require_official_terminal_target(
        rules.serialize_snapshot(trajectory_state), 'completed') == trajectory_state.winner, 'actual final-round termination')
    private_state = guard.decode_training_snapshot(next(f['snapshot'] for f in fixtures if f['id'] == 'private_reserves'))
    rejections = worker_rejections(request_callback, private_state, evidence_callback=evidence_callback)
    parity = {arm: _fresh_parity(adapters[arm], boundary) for arm in ARMS}
    require(counts['complete_decisions'] == plan['actual_complete_decisions']['total'], 'fixed actual decision count')
    return {'status': 'entrypoint_counterexamples_passed', 'ruleset_id': guard.RULESET_ID,
            'catalogue_sha256': guard.CATALOGUE_SHA256, 'turn_contract': guard.TURN_CONTRACT,
            'challenger_quota': quota, 'reference_quota': 16,
            'counts': counts, 'fixtures': rows, 'reference_reproductions': reference_reproductions, 'native_catalogues': catalogues, 'native_contract': native,
            'actual_encoded_costs': encoded_cost_checks, 'private_repeats': privacy, 'alternate_starting_seat': starting_seat, 'actual_auxiliary': auxiliary,
            'late_boundary_regressions': regressions,
            'actual_mixed_final_round': trajectory, 'worker_rejections': rejections,
            'strict_snapshot_rejections': reference.identity_rejections(fixtures[0]),
            'semantic_edges': reference._semantic_edges(fixtures), 'fresh_numeric_parity': parity,
            'training_loader_qualified': False,
            'remaining_gate': 'separately frozen finite live qualification and operational checks'}


def run_entrypoint_checks(adapters, *, boundary=lambda:None, request_callback=None, evidence_callback=lambda row:None):
    """Original52 plus15 declared same-quota direct128 native/host reproductions."""
    summary=_run_original_entrypoint_checks(adapters,boundary=boundary,request_callback=request_callback,evidence_callback=evidence_callback)
    quota=adapters['search'].profile['iterations'];direct=[]
    if quota==128:
        fixtures={f['id']:f for f in build_fixtures()}
        for row in summary['fixtures']:
            boundary();fixture=fixtures[row['id']]
            observer=rules.observation_payload(guard.decode_training_snapshot(fixture['snapshot']))
            evidence=dict(operation='direct_same_quota_choose',arm='search',case_id=row['id'],
                request=dict(observer=observer,quota=128),before_snapshot=fixture['snapshot'])
            evidence_callback(dict(evidence,event='started'))
            try:action,trace=adapters['search'].direct_choose(observer,diagnostic_capture=True)
            except BaseException as error:
                evidence_callback(dict(evidence,event='failed',error=repr(error)));raise
            result=dict(action=rules.serialize_action(action),diagnostics=trace)
            evidence_callback(dict(evidence,event='result',result=result))
            expected=row['results']['search']
            require(digest(result)==digest(dict(action=expected['action'],diagnostics=expected['diagnostics'])),
                'Direct actual128 native/host action and full callback output parity')
            direct.append(dict(id=row['id'],result=result,exact_callback_arrays_and_hashes=True))
        require(len(direct)==15,'Fixed direct128 inventory')
    summary.update(original_counts={'complete_decisions':52},direct_same_quota=direct,
        counts={'complete_decisions':52+len(direct)},execution_schema='splendorust_search128_strength_execution_v1',
        execution_route='research' if quota==128 else 'frozen',world_pool=3)
    return summary
