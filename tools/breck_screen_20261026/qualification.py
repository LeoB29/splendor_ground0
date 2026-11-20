"""Prospective corrected-rule qualification for the pinned Breck policy.

Building this panel loads no model and plays no game. The old corrected fixtures
are structural counterexamples, not historical training targets. Model-bearing
entrypoint checks may run only after the new attempt has been frozen.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

from tools import future_training_rules_20261021 as guard
from tools.external_strength_screen_20261018 import audit, rules
from tools.move_evaluation_20261023 import qualification as structural

ROOT = Path(__file__).resolve().parents[2]
SAVED_PANEL = ROOT / 'outputs/external_strength_screen_20261018_02/fixtures.json'
FEATURE_ATOL = 1e-7
FEATURE_RTOL = 0.0
LOGIT_ATOL = 1e-6
LOGIT_RTOL = 1e-6
CASE_COUNT = 64
TIMED_REQUEST_COUNT = 256
REPRODUCTION_QUOTA = 48
REPRODUCTIONS_PER_ROLE = 24
FORENSIC_MAX_BYTES = 4 * 1024 * 1024
FORENSIC_MAX_RECORDS = 512


def require(value, message):
    if not value:
        raise ValueError('Breck qualification: ' + message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def artifact(path):
    path = Path(path)
    data = path.read_bytes()
    return {'path': path.relative_to(ROOT).as_posix(), 'bytes': len(data),
            'sha256': hashlib.sha256(data).hexdigest()}


def swapped_seats(state):
    changed = state.copy_shallow()
    changed.players.reverse()
    for seat, player in enumerate(changed.players):
        player.player_id = seat
    changed.current_player = 1 - state.current_player
    changed.start_player = 1 - state.start_player
    if changed.winner is not None:
        changed.winner = 1 - changed.winner
    return changed


def hidden_variant(state):
    """Change only unknown reserves/deck order, without borrowing hidden inputs."""
    changed = state.copy_shallow()
    opponent = changed.players[1 - changed.current_player]
    for tier in (1, 2, 3):
        changed.hidden_tier_decks[tier].reverse()
    for index, card in enumerate(opponent.reserved_cards):
        if opponent.reserved_card_source(index) == 'deck':
            deck = changed.hidden_tier_decks[card.tier]
            if deck:
                opponent.reserved_cards[index], deck[0] = deck[0], card
    require(rules.observation_payload(changed) == rules.observation_payload(state),
            'hidden variant changes lawful observation')
    return changed


def _reserve(state, seat, tier, source):
    if source == 'visible':
        card = state.visible_tier_cards[tier].pop(0)
        if state.hidden_tier_decks[tier]:
            state.visible_tier_cards[tier].append(state.hidden_tier_decks[tier].pop(0))
    else:
        card = state.hidden_tier_decks[tier].pop(0)
    state.deck_counts[tier] = len(state.hidden_tier_decks[tier])
    state.players[seat].reserve_card(card, source)


def _tie(fewer_cards):
    state = structural._initial(1)
    available = list(rules.corrected_cards_by_id().values())
    for seat, points in enumerate(((5, 5, 4, 1), (5, 5, 5) if fewer_cards else (5, 5, 4, 1))):
        cards = []
        for value in points:
            card = next(c for c in available if c.points == value)
            available.remove(card)
            cards.append(card)
        structural._give_cards(state, seat, cards)
    require([p.score for p in state.players] == [15, 15], 'tie setup scores')
    state.pending_round_end = True
    return state


def _extra_cases():
    values = [('initial', structural._initial(), ['initial'])]
    private = structural._initial()
    for seat in (0, 1):
        for tier in (1, 2, 3):
            _reserve(private, seat, tier, 'deck')
    values.append(('reserve_caps', private, ['three_reserves', 'private_reserves', 'all_tiers']))
    mixed = structural._initial()
    for seat in (0, 1):
        for tier, source in ((1, 'visible'), (2, 'deck'), (3, 'visible')):
            _reserve(mixed, seat, tier, source)
    values.append(('mixed_reserve_sources', mixed, ['public_reserves', 'private_reserves', 'reserve_order']))
    for tier in (1, 2, 3):
        depleted = structural._initial(1)
        structural._give_cards(depleted, 0, list(depleted.hidden_tier_decks[tier]))
        depleted.pending_round_end = any(p.score >= 15 for p in depleted.players)
        require(not depleted.hidden_tier_decks[tier], 'depleted tier fixture')
        values.append((f'depleted_tier_{tier}', depleted, ['depleted_tier', str(tier)]))
    for remaining in (1, 2):
        scarce = structural._initial()
        allocations = [{}, {}]
        for index, color in enumerate(structural.COLORS[remaining:]):
            allocations[index % 2][color] = 4
        for seat, amounts in enumerate(allocations):
            structural._tokens(scarce, seat, amounts)
        values.append((f'scarce_{remaining}_colors', scarce, ['scarce_bank', str(remaining)]))
    threshold = structural._initial()
    structural._tokens(threshold, 1, {'white': 1})
    values.append(('double_take_threshold', threshold, ['bank_three_vs_four']))
    values.append(('equal_score_equal_cards', _tie(False), ['final_round', 'shared_win']))
    values.append(('equal_score_fewer_cards', _tie(True), ['final_round', 'fewer_cards_tiebreak']))
    require(len(values) == 11, 'extra case count')
    return values


def _mapping_witnesses():
    """141 bounded structural index checks; no neural inference is involved."""
    from . import adapter
    rows = []
    for descriptor in adapter.ACTION_TABLE:
        index, family = descriptor['index'], descriptor['family']
        state = structural._initial()
        if family.startswith('take_'):
            gain = descriptor['take_tokens']
            actor_total = 10 - len(gain) + descriptor['return_count']
            restricted = family in ('take_one', 'take_two')
            removed = {color: (4 if restricted and color not in gain else 0) for color in structural.COLORS}
            extra = max(0, actor_total - sum(removed.values()))
            for color in structural.COLORS:
                minimum = (4 if family == 'take_double' and color in gain else int(color in gain))
                available = 4 - minimum - removed[color]
                amount = min(extra, available)
                removed[color] += amount
                extra -= amount
            require(extra == 0, 'cannot allocate take witness')
            allocations, remaining = [{}, {}], actor_total
            for color, amount in removed.items():
                own = min(remaining, amount)
                allocations[0][color], allocations[1][color] = own, amount - own
                remaining -= own
            require(remaining == 0, 'witness actor token count')
            for seat, amounts in enumerate(allocations):
                structural._tokens(state, seat, amounts)
        elif family.startswith('buy_'):
            for tier in (1, 2, 3):
                _reserve(state, 0, tier, 'deck')
            structural._give_cards(state, 0, [card for tier in (1, 2, 3)
                                             for card in state.hidden_tier_decks[tier]])
            state.pending_round_end = any(p.score >= 15 for p in state.players)
            state.start_player, state.turn_index = 1, 1  # Actor0 is the final reply, not a completed round.
        elif family == 'forced_pass_extension':
            state = structural._pass()
        before = rules.serialize_snapshot(state)
        guard.decode_training_snapshot(before)
        observer = rules.observation_payload(state)
        prepared = adapter.prepare(observer)
        action = prepared['actions'][index]
        require(action is not None, f'index{index} has no corrected witness')
        after = _step(before, action)
        rows.append({'index': index, 'descriptor': deepcopy(descriptor), 'before': before,
                     'observer': observer, 'action': action, 'after': after})
    return rows


def _case(index, name, state, tags, origin):
    original = rules.serialize_snapshot(state)
    guard.decode_training_snapshot(original)
    changed = rules.serialize_snapshot(hidden_variant(state))
    guard.decode_training_snapshot(changed)
    observer = rules.observation_payload(state)
    return {'case_index': index, 'case_id': name, 'tags': tags, 'origin': origin,
            'original': original, 'hidden_variant': changed, 'observer': observer,
            'observer_sha256': digest(observer), 'hidden_state_changed': changed != original,
            'hidden_variant_scope': ('distinct_hidden_state' if changed != original else
                                     'no_hidden_alternative_in_exhausted_structural_case')}


def build_plan():
    """The exact finite inventory, independent of any model outputs."""
    cases = []
    for fixture in structural.build_fixtures():
        state = guard.decode_training_snapshot(fixture['snapshot'])
        for swapped in (False, True):
            current = swapped_seats(state) if swapped else state
            cases.append(_case(len(cases), fixture['id'] + ('_swapped' if swapped else ''),
                               current, list(fixture['tags']) + ['seat_swapped' if swapped else 'original_seat'],
                               'frozen_corrected_structural_counterexample'))
    saved = json.loads(SAVED_PANEL.read_text(encoding='utf-8'))
    require(len(saved) == 23, 'saved structural panel count')
    for index, item in enumerate(saved):
        state = guard.decode_training_snapshot(item['snapshot'])
        cases.append(_case(len(cases), f'saved_corrected_{index:02d}', state,
                           ['saved_corrected_structure'], {'meaning': 'structural input only; no historical targets',
                            'original_source': item['source']}))
    for name, state, tags in _extra_cases():
        cases.append(_case(len(cases), name, state, tags, 'new_ledger_valid_structural_counterexample'))
    require(len(cases) == CASE_COUNT, 'case count')
    requests = []
    for repeat in (0, 1):
        order = cases if repeat == 0 else list(reversed(cases))
        for case in order:
            variants = ('original', 'hidden_variant') if repeat == 0 else ('hidden_variant', 'original')
            for variant in variants:
                requests.append({'index': len(requests), 'case_index': case['case_index'],
                                 'case_id': case['case_id'], 'variant': variant, 'repeat': repeat,
                                 'observer': case['observer'], 'before': case[variant]})
    plan = {'schema': 'breck_corrected_qualification_plan_v1', 'ruleset_id': guard.RULESET_ID,
            'catalogue_sha256': guard.CATALOGUE_SHA256, 'turn_contract': guard.TURN_CONTRACT,
            'cases': cases, 'requests': requests, 'case_count': CASE_COUNT,
            'timed_request_count': TIMED_REQUEST_COUNT, 'ordinary_arrival_exclusive_seconds': .7,
            'ordinary_deadline_seconds': 1.0,
            'source_inputs': [artifact(SAVED_PANEL), artifact(Path(structural.__file__))],
            'numeric_tolerances': {'features': {'atol': FEATURE_ATOL, 'rtol': FEATURE_RTOL},
                                   'logits': {'atol': LOGIT_ATOL, 'rtol': LOGIT_RTOL}},
            'mapping_witnesses': _mapping_witnesses(),
            'bundled_startup': {'complete_handler_decisions': CASE_COUNT, 'reference_forwards': CASE_COUNT,
                               'deadline_seconds': 300, 'ordinary_timed_requests': 0,
                               'rejection_cases': rejection_case_names(),
                               'forensic_max_bytes': FORENSIC_MAX_BYTES,
                               'forensic_max_records': FORENSIC_MAX_RECORDS},
            'reproductions': {'maximum': REPRODUCTION_QUOTA, 'per_role': REPRODUCTIONS_PER_ROLE,
                             'selection': 'deterministic outcome-and-time-blind cells; report missing cells'},
            'structural_reachability_claim': False, 'training_loader_qualified': False}
    validate_plan(plan)
    return plan


def validate_plan(plan):
    require(plan['schema'] == 'breck_corrected_qualification_plan_v1', 'plan schema')
    require((plan['ruleset_id'], plan['catalogue_sha256'], plan['turn_contract']) ==
            (guard.RULESET_ID, guard.CATALOGUE_SHA256, guard.TURN_CONTRACT), 'plan rule identity')
    require(plan['numeric_tolerances'] == {'features': {'atol': FEATURE_ATOL, 'rtol': FEATURE_RTOL},
                                         'logits': {'atol': LOGIT_ATOL, 'rtol': LOGIT_RTOL}}, 'frozen numerical tolerances')
    require(plan['case_count'] == len(plan['cases']) == CASE_COUNT and
            plan['timed_request_count'] == len(plan['requests']) == TIMED_REQUEST_COUNT, 'plan counts')
    require(len({c['case_id'] for c in plan['cases']}) == CASE_COUNT, 'duplicate case identity')
    for index, case in enumerate(plan['cases']):
        require(case['case_index'] == index, 'case index order')
        before = guard.decode_training_snapshot(case['original'])
        changed = guard.decode_training_snapshot(case['hidden_variant'])
        require(rules.observation_payload(before) == rules.observation_payload(changed) == case['observer'],
                'case lawful observation differs')
        require(digest(case['observer']) == case['observer_sha256'], 'observer hash')
        require(case['hidden_state_changed'] is (case['original'] != case['hidden_variant']), 'hidden difference label')
    expected = []
    for repeat in (0, 1):
        order = plan['cases'] if repeat == 0 else list(reversed(plan['cases']))
        for case in order:
            variants = ('original', 'hidden_variant') if repeat == 0 else ('hidden_variant', 'original')
            for variant in variants:
                expected.append({'index': len(expected), 'case_index': case['case_index'],
                                 'case_id': case['case_id'], 'variant': variant, 'repeat': repeat,
                                 'observer': case['observer'], 'before': case[variant]})
    require(plan['requests'] == expected, 'request inventory/order changed')
    require({key for c in plan['cases'] for key in c['tags']} >= set(rules.CORRECTED_COSTS) |
            {'nonpurchase_noble', 'multiple_nobles', 'noble_final_round', 'gold_returns',
             'private_reserves', 'forced_pass', 'depleted_tier', 'shared_win', 'fewer_cards_tiebreak'},
            'mandatory counterexample coverage')
    require(len(plan['mapping_witnesses']) == 141 and
            [w['index'] for w in plan['mapping_witnesses']] == list(range(141)), 'mapping witness inventory')
    require(plan['bundled_startup']['rejection_cases'] == rejection_case_names(), 'rejection inventory')
    return {'status': 'plan_validated', 'cases': CASE_COUNT, 'ordinary_timed_requests': TIMED_REQUEST_COUNT,
            'distinct_hidden_pairs': sum(c['hidden_state_changed'] for c in plan['cases'])}


def assert_numeric_close(actual, expected, *, atol, rtol, size, label):
    actual, expected = list(actual), list(expected)
    require(len(actual) == len(expected) == size, label + ' shape')
    deltas = []
    for left, right in zip(actual, expected, strict=True):
        require(type(left) in (float, int) and type(right) in (float, int) and
                math.isfinite(left) and math.isfinite(right), label + ' non-finite/non-numeric value')
        delta = abs(left - right)
        require(delta <= atol + rtol * abs(right), label + ' differs beyond frozen tolerance')
        deltas.append(delta)
    return {'count': size, 'maximum_absolute_delta': max(deltas, default=0.0), 'atol': atol, 'rtol': rtol}


def _step(before, action):
    state = guard.decode_training_snapshot(before)
    chosen = rules.deserialize_action(action)
    require(chosen in rules.legal_actions(state), 'returned complete action outside corrected support')
    after = rules.serialize_snapshot(rules.CorrectedEnv().step(state, chosen))
    guard.check_training_transition(before, action, after)
    return after


def validate_qualification_rows(rows, plan):
    """Validate complete original-clock receipts; failures cannot be dropped."""
    from . import adapter
    validate_plan(plan)
    require(len(rows) == len(plan['requests']), 'incomplete qualification panel')
    repeats, margins = {}, []
    for row, spec in zip(rows, plan['requests'], strict=True):
        require(all(row.get(k) == v for k, v in spec.items()), 'changed timed request/order')
        clock = row['clock']
        require(clock['status'] == 'ok' and isinstance(clock.get('result'), dict), 'ordinary request failed')
        elapsed = clock.get('receipt_elapsed_seconds')
        require(type(elapsed) in (int, float) and math.isfinite(elapsed) and 0 <= elapsed <= 1,
                'missing/invalid trusted receipt elapsed')
        require(clock.get('received_before_deadline') is True, 'response not received before real deadline')
        result = clock['result']
        adapter.validate_result(result, spec['observer'])
        after = _step(spec['before'], result['action'])
        if 'after' in row:
            require(row['after'] == after, 'saved complete successor differs')
        if 'action' in row:
            require(row['action'] == result['action'], 'saved action differs from original clock result')
        science = {'action': result['action'], 'diagnostics': result['diagnostics']}
        if spec['case_id'] in repeats:
            require(repeats[spec['case_id']] == science, 'hidden/referee/repetition dependence')
        else:
            repeats[spec['case_id']] = science
        if elapsed >= .7:
            margins.append({'index': spec['index'], 'case_id': spec['case_id'], 'seconds': elapsed})
    return {'status': 'qualified' if not margins else 'timing_not_qualified',
            'qualified': not margins, 'requests': len(rows), 'independent_transitions': len(rows),
            'repeat_groups': len(repeats), 'timing_margin_failures': margins,
            'maximum_arrival_seconds': max(row['clock']['receipt_elapsed_seconds'] for row in rows)}


def rejection_case_names():
    return ['extra_request_field', 'unknown_player', 'missing_observer', 'bare_historical_observation',
            'missing_schema', 'old_schema', 'missing_ruleset', 'old_ruleset', 'missing_catalogue',
            'wrong_catalogue', 'referee_seed', 'hidden_decks', 'opponent_private_card']


def rejection_requests(plan):
    case = next(c for c in plan['cases'] if c['case_id'] == 'private_reserves')
    private = guard.decode_training_snapshot(case['original'])
    rows = []
    for name in rejection_case_names():
        request = {'player': 'breck', 'observer': deepcopy(case['observer'])}
        view = request['observer']
        if name == 'extra_request_field': request['seed'] = 1
        elif name == 'unknown_player': request['player'] = 'unqualified'
        elif name == 'missing_observer': del request['observer']
        elif name == 'bare_historical_observation': request['observer'] = view['observation']
        elif name.startswith('missing_'):
            del view[{'schema': 'schema', 'ruleset': 'ruleset_id', 'catalogue': 'catalogue_sha256'}[name[8:]]]
        elif name == 'old_schema': view['schema'] = 'historical'
        elif name == 'old_ruleset': view['ruleset_id'] = 'historical'
        elif name == 'wrong_catalogue': view['catalogue_sha256'] = '0' * 64
        elif name == 'referee_seed': view['observation']['seed'] = 1
        elif name == 'hidden_decks': view['observation']['hidden_tier_decks'] = {}
        elif name == 'opponent_private_card':
            seat = 1 - private.current_player
            view['observation']['players'][seat]['reserves'][0]['card_id'] = private.players[seat].reserved_cards[0].card_id
        rows.append({'case_id': name, 'request': request})
    return rows


def validate_model_free_contract(plan, *, boundary=lambda: None):
    """All source mappings and independent successors, without learned calls."""
    from . import adapter, reference
    validate_plan(plan)
    mapping = reference.validate_mapping_contract()
    for witness in plan['mapping_witnesses']:
        boundary()
        index = witness['index']
        require(witness['descriptor'] == adapter.ACTION_TABLE[index], 'frozen index descriptor changed')
        state = guard.decode_training_snapshot(witness['before'])
        require(rules.observation_payload(state) == witness['observer'], 'mapping witness observer')
        prepared = adapter.prepare(witness['observer'])
        require(prepared['source_mask'].tolist() == reference.source_mask(witness['observer']).tolist(),
                'mapping witness source legality differs')
        require(prepared['actions'][index] == witness['action'] and prepared['legal_mask'][index],
                'mapping witness action changed')
        require(_step(witness['before'], witness['action']) == witness['after'], 'mapping successor changed')
    semantic = structural._semantic_edges(structural.build_fixtures())
    tie_checks = []
    for case_id, expected in [('equal_score_equal_cards', None), ('equal_score_fewer_cards', 1)]:
        case = next(c for c in plan['cases'] if c['case_id'] == case_id)
        state = guard.decode_training_snapshot(case['original'])
        options = [a for a in rules.legal_actions(state) if a.action_type.name == 'TAKE_TOKENS' and a.noble_id is None]
        require(options, 'missing score-preserving final reply')
        after = _step(case['original'], rules.serialize_action(options[0]))
        terminal = guard.decode_training_snapshot(after)
        require(terminal.terminal and terminal.winner == expected, 'independent tie convention')
        tie_checks.append({'case_id': case_id, 'winner': expected, 'after_sha256': digest(after)})
    return {'status': 'model_free_contract_passed', 'static_mapping': mapping,
            'enabled_index_witnesses': 141, 'independent_mapping_transitions': 141,
            'semantic_edges': semantic, 'terminal_ties': tie_checks,
            'training_loader_qualified': False, 'actual_player_qualified': False}


class _Forensics:
    def __init__(self, folder):
        self.path = Path(folder) / 'breck_entrypoint_forensics.jsonl'
        self.stream = self.path.open('xb')
        self.records = self.bytes = 0

    def write(self, value):
        raw = (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()
        require(self.records + 1 <= FORENSIC_MAX_RECORDS and self.bytes + len(raw) <= FORENSIC_MAX_BYTES,
                'bounded startup forensics exhausted; preserve prefix')
        self.stream.write(raw)
        self.stream.flush()
        self.records += 1
        self.bytes += len(raw)

    def close(self):
        self.stream.close()


def run_entrypoint_checks(context, request_callback, *, boundary=lambda: None):
    """Actual bundled handler/source checks; caller owns the frozen300s clock.

    Every request/result is flushed before its assertions. Unexpected failures
    retain that prefix. This performs64 real handler decisions and64 independent
    reference forwards, additional to the256 ordinary externally timed calls.
    """
    from . import adapter, reference
    plan = context['qualification_plan']
    validate_plan(plan)
    forensic = _Forensics(context['folder'])
    try:
        contract = validate_model_free_contract(plan, boundary=boundary)
        reference_policy = reference.load_policy()  # Explicit model-bearing startup, never pre-freeze.
        parity, covered = [], set()
        for case in plan['cases']:
            boundary()
            observer = case['observer']
            payload = {'player': 'breck', 'observer': observer}
            forensic.write({'event': 'started', 'case_id': case['case_id'], 'request': payload})
            result = request_callback(payload)
            forensic.write({'event': 'returned', 'case_id': case['case_id'], 'result': result})
            adapter.validate_result(result, observer)
            prepared = adapter.prepare(observer)
            source_features = reference.encode(observer)
            source_mask = reference.source_mask(observer)
            require(prepared['source_mask'].tolist() == source_mask.tolist(), 'actual source legal mask differs')
            features = assert_numeric_close(result['diagnostics']['features'], source_features.tolist(),
                atol=FEATURE_ATOL, rtol=FEATURE_RTOL, size=251, label='actual source features')
            source_logits = reference.predict(source_features, policy=reference_policy)
            forensic.write({'event': 'reference_returned', 'case_id': case['case_id'],
                            'features': source_features.tolist(), 'logits': source_logits.tolist(),
                            'source_mask': source_mask.tolist()})
            logits = assert_numeric_close(result['diagnostics']['logits'], source_logits.tolist(),
                atol=LOGIT_ATOL, rtol=LOGIT_RTOL, size=141, label='actual source logits')
            supported = [i for i, action in enumerate(prepared['actions']) if action is not None]
            source_selected = max(supported, key=lambda i: float(source_logits[i]))
            require(source_selected == result['diagnostics']['selected_index'] and
                    prepared['actions'][source_selected] == result['action'],
                    'source numerical parity changed exact masked first-maximum decision')
            _step(case['original'], result['action'])
            covered.update(i for i, action in enumerate(prepared['actions']) if action is not None)
            parity.append({'case_id': case['case_id'], 'features': features, 'logits': logits,
                           'exact_features': result['diagnostics']['features'] == source_features.tolist(),
                           'result_sha256': digest(result)})
        rejects = []
        for case in rejection_requests(plan):
            boundary()
            forensic.write({'event': 'rejection_started', **case})
            try:
                result = request_callback(case['request'])
            except (ValueError, KeyError, TypeError) as error:
                forensic.write({'event': 'rejected', 'case_id': case['case_id'], 'error': repr(error)})
                rejects.append(case['case_id'])
            else:
                forensic.write({'event': 'unexpected_acceptance', 'case_id': case['case_id'], 'result': result})
                raise ValueError('Breck qualification: malformed worker request accepted: ' + case['case_id'])
        require(rejects == rejection_case_names(), 'actual rejection inventory')
        return {'status': 'breck_entrypoint_checks_passed', 'ruleset_id': guard.RULESET_ID,
                'catalogue_sha256': guard.CATALOGUE_SHA256, 'turn_contract': guard.TURN_CONTRACT,
                'plan_sha256': digest(plan), 'complete_handler_decisions': 64, 'reference_forwards': 64,
                'numeric_tolerances': deepcopy(plan['numeric_tolerances']),
                'rejection_count': len(rejects), 'rejections': rejects, 'parity': parity,
                'mapping_contract': contract, 'timed_panel_enabled_indices': sorted(covered),
                'timed_panel_not_enabled_indices': sorted(set(range(141)) - covered),
                'forensics': {'path': forensic.path.name, 'bytes': forensic.bytes, 'records': forensic.records,
                              'sha256': hashlib.sha256(forensic.path.read_bytes()).hexdigest()},
                'remaining_gate': '256 original-clock decisions, operational games and complete evidence audit',
                'training_loader_qualified': False}
    except BaseException as error:
        try:
            forensic.write({'event': 'entrypoint_failed', 'error': repr(error)})
        except BaseException:
            pass  # Preserve the primary failure and every already-flushed byte.
        raise
    finally:
        forensic.close()


def validate_entrypoint_result(result, plan, forensic_path, *, boundary=lambda: None):
    """Reread the bounded startup proof without redoing a learned forward.

    The original response/forensic stream must also be bound by the owning raw
    transport and frozen runtime evidence. Saved reference logits are checked,
    not presented as freshly reproduced inference.
    """
    from . import adapter, reference
    validate_plan(plan)
    require(result['status'] == 'breck_entrypoint_checks_passed' and result['plan_sha256'] == digest(plan),
            'startup status/plan identity')
    require((result['ruleset_id'], result['catalogue_sha256'], result['turn_contract']) ==
            (guard.RULESET_ID, guard.CATALOGUE_SHA256, guard.TURN_CONTRACT), 'startup rules identity')
    require(result['complete_handler_decisions'] == result['reference_forwards'] == 64 and
            result['rejection_count'] == 13 and result['rejections'] == rejection_case_names(), 'startup counts')
    require(result['numeric_tolerances'] == plan['numeric_tolerances'], 'startup numerical tolerance changed')
    path = Path(forensic_path)
    require(path.name == 'breck_entrypoint_forensics.jsonl', 'startup forensic basename')
    raw = path.read_bytes()
    require(0 < len(raw) <= FORENSIC_MAX_BYTES and raw.endswith(b'\n'), 'startup forensic bound/termination')
    events = [json.loads(line) for line in raw.decode('utf-8').splitlines()]
    require(len(events) == 64*3 + 13*2, 'startup forensic event inventory')
    require(result['forensics'] == {'path': path.name, 'bytes': len(raw), 'records': len(events),
                                   'sha256': hashlib.sha256(raw).hexdigest()}, 'startup forensic binding')
    expected_parity, covered = [], set()
    cursor = 0
    for case in plan['cases']:
        boundary()
        start, returned, source = events[cursor:cursor+3]
        cursor += 3
        require(start == {'event': 'started', 'case_id': case['case_id'],
                          'request': {'player': 'breck', 'observer': case['observer']}}, 'startup request differs')
        require(set(returned) == {'event', 'case_id', 'result'} and returned['event'] == 'returned' and
                returned['case_id'] == case['case_id'], 'startup returned event differs')
        actual = returned['result']
        adapter.validate_result(actual, case['observer'])
        require(set(source) == {'event', 'case_id', 'features', 'logits', 'source_mask'} and
                source['event'] == 'reference_returned' and source['case_id'] == case['case_id'],
                'startup source event differs')
        prepared = adapter.prepare(case['observer'])
        require(source['features'] == reference.encode(case['observer']).tolist(), 'saved source features changed')
        require(source['source_mask'] == prepared['source_mask'].tolist() ==
                reference.source_mask(case['observer']).tolist(), 'saved source mask changed')
        features = assert_numeric_close(actual['diagnostics']['features'], source['features'],
            atol=FEATURE_ATOL, rtol=FEATURE_RTOL, size=251, label='saved source features')
        logits = assert_numeric_close(actual['diagnostics']['logits'], source['logits'],
            atol=LOGIT_ATOL, rtol=LOGIT_RTOL, size=141, label='saved source logits')
        supported = [i for i, action in enumerate(prepared['actions']) if action is not None]
        selected = max(supported, key=lambda i: source['logits'][i])
        require(selected == actual['diagnostics']['selected_index'] and
                prepared['actions'][selected] == actual['action'], 'saved source exact decision differs')
        _step(case['original'], actual['action'])
        covered.update(supported)
        expected_parity.append({'case_id': case['case_id'], 'features': features, 'logits': logits,
                                'exact_features': actual['diagnostics']['features'] == source['features'],
                                'result_sha256': digest(actual)})
    require(result['parity'] == expected_parity, 'startup parity summary differs')
    for case in rejection_requests(plan):
        start, rejected = events[cursor:cursor+2]
        cursor += 2
        require(start == {'event': 'rejection_started', **case}, 'startup rejection input differs')
        require(set(rejected) == {'event', 'case_id', 'error'} and rejected['event'] == 'rejected' and
                rejected['case_id'] == case['case_id'] and isinstance(rejected['error'], str) and rejected['error'],
                'startup rejection result differs')
    require(result['timed_panel_enabled_indices'] == sorted(covered) and
            result['timed_panel_not_enabled_indices'] == sorted(set(range(141))-covered), 'enabled coverage differs')
    require(result['mapping_contract'] == validate_model_free_contract(plan, boundary=boundary),
            'startup independent mapping/semantic receipt differs')
    require(result['training_loader_qualified'] is False, 'unqualified training claim')
    return {'status': 'breck_entrypoint_receipt_verified', 'complete_handler_decisions': 64,
            'saved_reference_forwards_checked': 64, 'new_model_forwards': 0,
            'actual_rejection_records': 13, 'independent_mapping_transitions': 141,
            'forensic_sha256': hashlib.sha256(raw).hexdigest(), 'plan_sha256': digest(plan)}
