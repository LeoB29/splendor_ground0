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
ARMS = ('search', 'greedy')


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


def timed_schedule():
    rows = []
    for pass_index in (0, 1):
        for fixture_index in (range(23) if pass_index == 0 else range(22, -1, -1)):
            arms = list(ARMS if fixture_index % 2 == 0 else reversed(ARMS))
            variants = ['original', 'hidden_variant']
            if pass_index:
                arms.reverse(); variants.reverse()
            for arm in arms:
                for variant in variants:
                    rows.append({'index': len(rows), 'fixture_index': fixture_index,
                                 'pass': pass_index, 'arm': arm, 'variant': variant})
    return rows


def build_plan():
    """Materialize all fixture identities, counts and ordering before launch."""
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
    schedule = [{**item, 'fixture_id': fixtures[item['fixture_index']]['fixture_id']}
                for item in timed_schedule()]
    return {'schema': 'splendorust_one_second_qualification_plan_v2',
            'ruleset_id': guard.RULESET_ID, 'catalogue_sha256': guard.CATALOGUE_SHA256,
            'turn_contract': guard.TURN_CONTRACT, 'panel_source': artifact(PANEL),
            'fixtures': fixtures, 'timed_requests': schedule,
            'timed_request_count': 184, 'engineering_margin_seconds': .7,
            'counterexamples': counterexamples,
            'alternate_starting_seat': rules.serialize_snapshot(seat_one),
            'actual_complete_decisions': {'fixture_choices': 30, 'private_repeats': 2,
                'alternate_starting_seat': 2, 'mixed_final_round_trajectory': 3,
                'frozen_greedy_reference': 15, 'total': 52},
            'native_operations': {'catalogue': 2, 'search_inspect': 14,
                'greedy_contract_and_reference': 56, 'forced_auxiliary': 6,
                'fresh_numeric_parity': 23, 'total': 101},
            'worker_rejection_cases': rejection_case_names(),
            'worker_rejection_attempts': 2 * len(rejection_case_names()),
            'work_accounting': 'native leaf evaluations and actual host neural callbacks are distinct; exact branch-aware validation',
            'regression_cases': ['turn123', 'turn124'],
            'forensic_evidence': 'append each full request and returned result before its assertions; preserve failures and partial callbacks',
            'parity': {'old_receipt': artifact(PARITY_RECEIPT), 'model_bin': artifact(MODEL_BIN),
                'fresh_checks': 23, 'max_abs_delta': .0005,
                'old_receipt_scope': 'source provenance; historical external Torch bytes not fully bound there',
                'fresh_scope': 'fixed additional entry-point parity under the newly frozen runtime'},
            'training_loader_qualified': False}


def state_for_request(plan, spec):
    fixture = plan['fixtures'][spec['fixture_index']]
    require(fixture['fixture_id'] == spec['fixture_id'], 'request fixture identity')
    return guard.decode_training_snapshot(fixture[spec['variant']])


def scientific_result(arm, action, trace):
    """Exclude declared capture/envelope fields, retaining actual scientific work."""
    if arm == 'search':
        keys = ('opponent_id', 'profile_id', 'observer_seed', 'forced_pass',
                'inference_calls', 'inference_rows')
        normalized = {key: trace[key] for key in keys}
        phase_keys = ('action', 'heuristic_phase', 'leaf_calls', 'native_legal_count', 'simulations')
        normalized['phases'] = [{key: phase[key] for key in phase_keys} for phase in trace['phases']]
    else:
        normalized = deepcopy(trace)
        # Capturing raw input/output arrays is diagnostic transport, not a new
        # scientific result; their SHA256 values remain in the common record.
        normalized.pop('deployment_wrapper_id', None)
        for phase in normalized['phases']:
            for callback in phase['inference']:
                callback.pop('tokens', None); callback.pop('output83', None)
    return {'action': action, 'diagnostics': normalized}


def callback_receipts(trace):
    """New wrapper evidence compared across repeats, outside old trace schema."""
    return [{key: phase[key] for key in ('neural_callbacks', 'neural_callback_count', 'neural_callback_rows')}
            for phase in trace['phases']]


def validate_scientific_decision(arm, observer, action, trace):
    """Independent checks applied to every accepted panel/game decision."""
    require(arm in ARMS, 'unknown arm')
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
        require(not trace['phases'] and trace['inference_calls'] == 0, 'forced pass used neural search')
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
        if arm == 'greedy':
            require(phase['phase'] == projection['phase']['kind'] and
                    phase['observation_sha256'] == digest(projection) and
                    phase['legal_sha256'] == digest(choices), 'auxiliary projection binding')
        return phase['action']
    resolved = bridge.resolve_compound(observer, phases[0]['action'], select_aux)
    require(resolved == selected and phase_cursor[0] == len(phases), 'complete compound response differs')
    if arm == 'search':
        require(trace['profile_id'] == 'entity_onehot_puct16_canonical_atomic_prereveal_cpu_v1', 'search profile')
        require(trace['opponent_id'] == 'splendorust', 'search identity')
        for index, phase in enumerate(phases):
            heuristic = index > 0 or len(mains) == 1 or state.turn_index >= 124
            require(phase['heuristic_phase'] is heuristic, 'search branch')
            require(type(phase['simulations']) is int and phase['simulations'] == (0 if heuristic else 16),
                    'search simulation quota')
            require(type(phase['leaf_calls']) is int and phase['leaf_calls'] >= 0 and
                    (not heuristic or phase['leaf_calls'] == 0), 'search leaf counts')
        # Native leaf() increments before its late-turn heuristic branch. The
        # host network is only called by the neural branch, so these counters
        # intentionally differ when a searched continuation reaches turn124.
        from .audit import validate_search_work
        work = validate_search_work(trace, observer)
    else:
        require(trace['simulations'] == 0 and trace['observer_sha256'] == digest(observer), 'greedy identity/work')
        require(trace['action_sha256'] == digest(action), 'greedy complete-action binding')
        total_calls = 0
        for index, phase in enumerate(phases):
            if index:
                require(phase['branch'] == 'auxiliary_heuristic' and not phase['inference'], 'greedy auxiliary')
                continue
            retained = phase['retained_indices']
            require(isinstance(retained, list) and retained == sorted(set(retained)) and bool(retained)
                    and all(type(v) is int and 0 <= v < len(mains) for v in retained), 'greedy retained support')
            filtered = [mains[i] for i in retained]
            require(phase['canonical_indices'] == [reference.canonical_index(a) for a in filtered], 'greedy canonical map')
            require(type(phase['selected_index']) is int and 0 <= phase['selected_index'] < len(filtered), 'greedy selection index')
            require(phase['action'] == filtered[phase['selected_index']], 'greedy selection binding')
            expected_branch = ('single_choice' if len(filtered) == 1 else
                               'late_heuristic' if state.turn_index >= 124 else 'neural_greedy')
            require(phase['branch'] == expected_branch, 'greedy branch')
            neural = expected_branch == 'neural_greedy'
            require(len(phase['inference']) == int(neural), 'greedy forward quota')
            if neural:
                logits = phase['canonical_logits']
                require(len(logits) == 67 and all(type(v) in (int, float) and math.isfinite(v) for v in logits), 'greedy logits')
                require(digest(logits) == phase['policy_logits_sha256'], 'greedy logits hash')
                winner = max(range(len(filtered)), key=lambda i: logits[reference.canonical_index(filtered[i])])
                require(phase['selected_index'] == winner, 'greedy first maximum')
            else:
                require(phase.get('canonical_logits') is None and phase['policy_logits_sha256'] is None, 'heuristic logits')
            total_calls += int(neural)
        require(trace['inference_calls'] == total_calls, 'greedy inference accounting')
    return {'forced_pass': False, 'simulations': sum(p.get('simulations', 0) for p in phases),
            'inference_calls': trace['inference_calls'],
            **({'search_work': work} if arm == 'search' else {})}


def validate_timed_panel(rows, fixtures=None):
    """Require the exact fixed schedule, historical PUCT agreement and all repeats."""
    plan_fixtures = (fixtures['fixtures'] if isinstance(fixtures, dict) else fixtures)
    if plan_fixtures is None:
        plan_fixtures = build_plan()['fixtures']
    schedule = timed_schedule()
    require(len(rows) == len(schedule) == 184, 'incomplete timed panel')
    seen = {}
    for row, spec in zip(rows, schedule, strict=True):
        for key in ('arm', 'variant', 'pass'):
            require(row[key] == spec[key], 'changed panel order: ' + key)
        index = spec['fixture_index']
        fixture = plan_fixtures[index]
        require(row.get('fixture_index', index) == index and row['fixture_id'] == fixture['fixture_id'], 'changed fixture order')
        elapsed = row['elapsed_seconds']
        require(type(elapsed) in (int, float) and math.isfinite(elapsed) and 0 <= elapsed <= .7, 'timing engineering margin')
        state = guard.decode_training_snapshot(fixture[spec['variant']])
        observer = rules.observation_payload(state)
        action, trace = row['action'], row['trace']
        validate_scientific_decision(spec['arm'], observer, action, trace)
        reference._step_check(state, rules.deserialize_action(action))
        result = scientific_result(spec['arm'], action, trace)
        repeated_result = {'scientific_result': result,
                           'host_callback_receipts': callback_receipts(trace) if spec['arm'] == 'search' else None}
        key = (index, spec['arm'])
        if key in seen:
            require(repeated_result == seen[key], 'hidden/repeated input changed scientific result or callbacks')
        else:
            seen[key] = repeated_result
        if spec['arm'] == 'search':
            old = read(ROOT / fixture['prior_search_reference']['path'])
            require(artifact(ROOT / fixture['prior_search_reference']['path']) == fixture['prior_search_reference'], 'historical reference changed')
            require(old['observer'] == observer and old['clock']['status'] == 'ok', 'historical fixture observer/status')
            old_result = old['clock']['result']
            require(result == scientific_result('search', old_result['action'], old_result['diagnostics']), 'saved PUCT16 decision/work changed')
    return {'status': 'passed', 'requests': 184, 'roots': 23, 'scientific_repeat_groups': 46,
            'independent_transitions': 184, 'historical_search_comparisons': 92,
            'maximum_receipt_seconds': max(row['elapsed_seconds'] for row in rows)}


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


def model_free_checks():
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
    plan = build_plan()
    return {'status': 'model_free_checks_passed', 'production_entrypoint_qualified': False,
            'runtime_sources': source_receipt, 'fixtures': rows, 'timed_request_count': len(plan['timed_requests']),
            'semantic_edges': reference._semantic_edges(fixtures),
            'strict_identity_rejections': reference.identity_rejections(fixtures[0])}


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
                request = {'op': 'greedy' if arm == 'greedy' else 'select',
                    'observation': projection, 'legal': choices, 'seed': int(digest(observer)[:16], 16)}
                if arm == 'search': request['iterations'] = 16
                result = adapters[arm].exchange(request)
                require(result['action'] in choices and result['simulations'] == result['leaf_calls'] == 0,
                        'forced auxiliary action/work')
                if arm == 'greedy': require(result['branch'] == 'auxiliary_heuristic', 'greedy auxiliary branch')
                else: require(result['heuristic_phase'] is True, 'search auxiliary branch')
                phases.append({'phase': phase, 'projection_sha256': digest(projection), 'result': result})
                covered.add(phase)
                return result['action']
            action = bridge.resolve_compound(observer, wanted[fixture['id']], select_aux)
            reference._step_check(state, action)
            by_arm[arm] = {'action': rules.serialize_action(action), 'phases': phases}
        require(by_arm['search']['action'] == by_arm['greedy']['action'], 'arms differ on unchanged auxiliary heuristic')
        require([(p['phase'], p['projection_sha256'], p['result']['action']) for p in by_arm['search']['phases']] ==
                [(p['phase'], p['projection_sha256'], p['result']['action']) for p in by_arm['greedy']['phases']],
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


def run_entrypoint_checks(adapters, *, boundary=lambda: None, request_callback=None,
                          evidence_callback=lambda row: None):
    """Run inside an initialized authenticated worker, never during preparation.

    request_callback is the worker's ordinary handler, so the exact decoder,
    legal support and response envelope are exercised. The worker authenticates
    sources once at startup before invoking this function.
    """
    from . import audit as independent
    require(request_callback is not None and set(adapters) == set(ARMS), 'actual worker callback required')
    plan = build_plan()
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
        independent.validate_result(result, observer, arm)
        checked = validate_scientific_decision(arm, observer, result['action'], result['diagnostics'])
        after = reference._step_check(decoded, rules.deserialize_action(result['action']))
        evidence_callback({**evidence, 'event': 'validated', 'check': checked,
                           'after_snapshot': rules.serialize_snapshot(after)})
        counts['complete_decisions'] += 1
        return result, after

    rows, greedy_rows, privacy, regressions = [], [], [], []
    catalogues = _native_catalogues(adapters)
    for fixture in fixtures:
        state = guard.decode_training_snapshot(fixture['snapshot'])
        observer = rules.observation_payload(state)
        results = {}
        for arm in ARMS:
            result, _ = decide(arm, state, fixture['id'])
            results[arm] = result
        if fixture['id'] in ('turn123', 'turn124'):
            search_work = independent.validate_search_work(results['search']['diagnostics'], observer)
            expected_calls = int(fixture['id'] == 'turn123')
            require(search_work['inference_calls'] == expected_calls, 'late boundary host neural callback regression')
            require(search_work['simulations'] == (16 if expected_calls else 0), 'late boundary simulation regression')
            if expected_calls:
                require(search_work['leaf_evaluations'] > search_work['inference_calls'],
                        'turn123 did not exercise non-neural searched leaves')
            regressions.append({'case_id': fixture['id'], 'work': search_work})
            evidence_callback({'event': 'validated', 'operation': 'late_boundary_regression',
                               'case_id': fixture['id'], 'arm': 'search', 'result': search_work})
        if fixture['id'] != 'forced_pass':
            inspected = adapters['search'].exchange({'op': 'inspect', 'observation': bridge.encode_observation(observer),
                                                    'seed': int(digest(observer)[:16], 16)})
            require(inspected['legal'] == bridge.main_actions(observer), 'search native ordered legal support')
        # Unbound frozen explicit greedy operation is the reference, not PUCT0.
        expected_action, expected_trace = adapters['greedy'].reference_choose(observer, diagnostic_capture=True)
        counts['complete_decisions'] += 1
        require(scientific_result('greedy', results['greedy']['action'], results['greedy']['diagnostics']) ==
                scientific_result('greedy', rules.serialize_action(expected_action), expected_trace), 'frozen explicit greedy changed')
        greedy_rows.append({'id': fixture['id'], 'greedy': results['greedy']['diagnostics']})
        if 'corrected_face' in fixture['tags']:
            search_input = results['search']['diagnostics']['phases'][0]['inference'][0]['tokens']
            greedy_input = results['greedy']['diagnostics']['phases'][0]['inference'][0]['tokens']
            require(search_input == greedy_input, 'same corrected face encoded differently for search/greedy')
        if fixture['id'] == 'private_reserves':
            changed = hidden_variant(state)
            require(rules.observation_payload(changed) == observer and rules.serialize_snapshot(changed) != fixture['snapshot'],
                    'private variant changed public information')
            for arm in ARMS:
                repeated, _ = decide(arm, changed, fixture['id'] + '/hidden_variant')
                require(scientific_result(arm, repeated['action'], repeated['diagnostics']) ==
                        scientific_result(arm, results[arm]['action'], results[arm]['diagnostics']), 'private hidden dependence')
                if arm == 'search':
                    require(callback_receipts(repeated['diagnostics']) == callback_receipts(results[arm]['diagnostics']),
                            'private hidden dependence in actual host callbacks')
                privacy.append({'arm': arm, 'equal': True})
        rows.append({'id': fixture['id'], 'observer_sha256': digest(observer), 'results': results})

    native = reference.verify_native_contract(adapters['greedy'], fixtures, greedy_rows, boundary=boundary)
    auxiliary = _auxiliary_checks(adapters, fixtures, boundary)
    seat_one = guard.decode_training_snapshot(plan['alternate_starting_seat'])
    starting_seat = [{'arm': arm, 'result': decide(arm, seat_one, 'alternate_starting_seat')[0]} for arm in ARMS]
    trajectory_state = guard.decode_training_snapshot(next(f['snapshot'] for f in fixtures if f['id'] == 'noble_final_round'))
    trajectory_state.current_player, trajectory_state.turn_index = 1, 1
    trajectory = []
    for step, arm in enumerate(('greedy', 'search', 'greedy')):
        require(not trajectory_state.terminal, 'mixed trajectory ended prematurely')
        before = rules.serialize_snapshot(trajectory_state)
        result, trajectory_state = decide(arm, trajectory_state, 'mixed_final_round/' + str(step))
        trajectory.append({'arm': arm, 'before': before, 'result': result,
                           'after': rules.serialize_snapshot(trajectory_state)})
    require(trajectory_state.terminal and guard.require_official_terminal_target(
        rules.serialize_snapshot(trajectory_state), 'completed') == trajectory_state.winner, 'actual final-round termination')
    private_state = guard.decode_training_snapshot(next(f['snapshot'] for f in fixtures if f['id'] == 'private_reserves'))
    rejections = worker_rejections(request_callback, private_state, evidence_callback=evidence_callback)
    parity = _fresh_parity(adapters['search'], boundary)
    require(counts['complete_decisions'] == plan['actual_complete_decisions']['total'], 'fixed actual decision count')
    return {'status': 'entrypoint_counterexamples_passed', 'ruleset_id': guard.RULESET_ID,
            'catalogue_sha256': guard.CATALOGUE_SHA256, 'turn_contract': guard.TURN_CONTRACT,
            'counts': counts, 'fixtures': rows, 'native_catalogues': catalogues, 'native_contract': native,
            'private_repeats': privacy, 'alternate_starting_seat': starting_seat, 'actual_auxiliary': auxiliary,
            'late_boundary_regressions': regressions,
            'actual_mixed_final_round': trajectory, 'worker_rejections': rejections,
            'strict_snapshot_rejections': reference.identity_rejections(fixtures[0]),
            'semantic_edges': reference._semantic_edges(fixtures), 'fresh_numeric_parity': parity,
            'training_loader_qualified': False,
            'remaining_gate': '184 externally timed requests and their full scientific/transition checks'}
