"""Bounded entry-point counterexamples for the fixed move-evaluation study.

The model-free check is explicitly not production qualification. The runtime
entry point uses the actual evaluator decoder/sampler/raw policy and the new
greedy adapter. It is invoked only inside the supervisor's frozen clock.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import tempfile

from tools import future_training_rules_20261021 as guard
from tools.external_strength_screen_20261018 import audit, rules
from tools.external_strength_screen_20261018_02 import bridge

COLORS = bridge.COLORS


def require(value, message):
    if not value:
        raise ValueError("Move-evaluation qualification: " + message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def _initial(turn=0):
    state = rules.CorrectedEnv(seed=20261023).initial_state()
    state.turn_index = turn
    state.current_player = turn % 2
    return state


def _tokens(state, seat, amounts):
    for color, amount in amounts.items():
        require(0 <= amount <= state.bank_tokens[color], "fixture token allocation")
        state.players[seat].tokens[color] += amount
        state.bank_tokens[color] -= amount


def _give_cards(state, seat, cards):
    ids = {c.card_id for c in cards}
    require(len(ids) == len(cards), "fixture duplicate card grant")
    for tier in (1, 2, 3):
        state.visible_tier_cards[tier] = [c for c in state.visible_tier_cards[tier] if c.card_id not in ids]
        state.hidden_tier_decks[tier] = [c for c in state.hidden_tier_decks[tier] if c.card_id not in ids]
        while len(state.visible_tier_cards[tier]) < 4 and state.hidden_tier_decks[tier]:
            state.visible_tier_cards[tier].append(state.hidden_tier_decks[tier].pop(0))
        state.deck_counts[tier] = len(state.hidden_tier_decks[tier])
    player = state.players[seat]
    for card in cards:
        player.purchased_cards.append(card)
        player.bonuses[card.bonus_color] += 1
        player.score += card.points


def _eligible(state, seat, nobles):
    cards = list(rules.corrected_cards_by_id().values())
    owned = {c.card_id for p in state.players for c in p.purchased_cards}
    wanted = []
    for color in COLORS:
        needed = max(n.requirement.get(color, 0) for n in nobles)
        count = max(0, needed - state.players[seat].bonuses[color])
        choices = sorted((c for c in cards if c.points == 0 and c.bonus_color == color
                          and c.card_id not in owned), key=lambda c: c.card_id)
        require(len(choices) >= count, "fixture zero-point supply")
        wanted.extend(choices[:count])
    _give_cards(state, seat, wanted)


def _face(card_id):
    state = _initial()
    card = rules.corrected_cards_by_id()[card_id]
    groups = (state.visible_tier_cards[card.tier], state.hidden_tier_decks[card.tier])
    group, index = next((g, i) for g in groups for i, c in enumerate(g) if c.card_id == card_id)
    group[index], state.visible_tier_cards[card.tier][0] = state.visible_tier_cards[card.tier][0], group[index]
    tokens = {c: min(n, 4) for c, n in card.cost.items()}
    tokens['gold'] = sum(max(0, n - 4) for n in card.cost.values())
    if card_id == 'T1_05':
        tokens['gold'] += 1  # Explicit coloured/gold alternative-payment fixture.
    _tokens(state, 0, tokens)
    return state


def _nobles(multiple=False, finish=False):
    state = _initial()
    if finish:
        cards = list(rules.corrected_cards_by_id().values())
        _give_cards(state, 0, [c for c in cards if c.points == 5][:2] +
                    [c for c in cards if c.points == 4][:1])
        require(state.players[0].score == 14, "fixture pre-noble score")
    _eligible(state, 0, state.nobles[:2 if multiple else 1])
    return state


def _private():
    state = _initial()
    for seat, tier in ((0, 1), (1, 2)):
        card = state.hidden_tier_decks[tier].pop(0)
        state.deck_counts[tier] -= 1
        state.players[seat].reserved_cards.append(card)
        state.players[seat].reserved_card_sources.append('deck')
    return state


def _pass(single_buy=False):
    # Ledger-valid structural final-round fixtures; no reachability assertion.
    state = _initial(1)
    cards = list(rules.corrected_cards_by_id().values())
    reserved = None
    if single_buy:
        reserved = next(c for c in cards if all(c.cost.get(color, 0) <=
            {'green': 2, 'red': 4, 'black': 4}.get(color, 0) for color in COLORS))
    _give_cards(state, 0, [c for c in cards if c != reserved])
    if reserved:
        for tier in (1, 2, 3):
            state.visible_tier_cards[tier] = []
            state.hidden_tier_decks[tier] = []
            state.deck_counts[tier] = 0
        state.players[1].reserved_cards = [reserved]
        state.players[1].reserved_card_sources = ['deck']
    _tokens(state, 0, {'white': 4, 'blue': 4, 'green': 2})
    _tokens(state, 1, {'green': 2, 'red': 4, 'black': 4})
    state.pending_round_end = True
    return state


def build_fixtures():
    """Return a fixed 15-case, independently ledger-valid structural panel."""
    values = [(f'face_{key}', _face(key), ['corrected_face', key]) for key in rules.CORRECTED_COSTS]
    values += [('noble_nonpurchase', _nobles(), ['nonpurchase_noble']),
               ('multiple_nobles', _nobles(multiple=True), ['multiple_nobles']),
               ('noble_final_round', _nobles(finish=True), ['noble_final_round']),
               ('private_reserves', _private(), ['private_reserves']),
               ('forced_pass', _pass(), ['forced_pass']),
               ('single_buy', _pass(single_buy=True), ['single_choice']),
               ('turn123', _initial(123), ['turn123']),
               ('turn124', _initial(124), ['turn124'])]
    returns = _initial()
    _tokens(returns, 0, {'white': 3, 'blue': 3, 'green': 3, 'gold': 1})
    values.append(('gold_returns', returns, ['gold_returns']))
    out = []
    for name, state, tags in values:
        snapshot = rules.serialize_snapshot(state)
        audit.validate_ledger(snapshot)
        out.append({'id': name, 'tags': tags, 'snapshot': snapshot})
    require(len(out) == 15, "fixture count")
    return out


def _private_variant(state):
    changed = state.copy_shallow()
    other = changed.players[1 - state.current_player]
    for index, card in enumerate(other.reserved_cards):
        if other.reserved_card_source(index) == 'deck':
            deck = changed.hidden_tier_decks[card.tier]
            other.reserved_cards[index], deck[0] = deck[0], card
    for deck in changed.hidden_tier_decks.values():
        deck.reverse()
    return changed


def _step_check(state, action):
    require(action in rules.legal_actions(state), 'policy/fixture action outside complete corrected support')
    before = rules.serialize_snapshot(state)
    after = rules.serialize_snapshot(rules.CorrectedEnv().step(state, action))
    guard.check_training_transition(before, rules.serialize_action(action), after)
    restored = guard.decode_training_snapshot(after)
    if restored.terminal:
        require(guard.require_official_terminal_target(after, 'completed') == restored.winner,
                'terminal score differs')
    return restored


def _rejected(call, label):
    try:
        call()
    except (ValueError, KeyError, TypeError):
        return label
    raise ValueError('Move-evaluation qualification: accepted malformed ' + label)


def identity_rejections(fixture):
    snapshot = fixture['snapshot']
    rejected = [_rejected(lambda: guard.decode_training_snapshot(snapshot['state']), 'bare historical snapshot')]
    for field, value in [('schema', 'legacy'), ('ruleset_id', 'historical'), ('catalogue_sha256', '0' * 64)]:
        bad = deepcopy(snapshot); bad[field] = value
        rejected.append(_rejected(lambda bad=bad: guard.decode_training_snapshot(bad), field))
    metadata = {'schema': 'corrected_training_data_v1', 'ruleset_id': guard.RULESET_ID,
                'catalogue_sha256': guard.CATALOGUE_SHA256, 'turn_contract': guard.TURN_CONTRACT,
                'origin': 'generated_under_declared_rules', 'data_sha256': digest('fixture')}
    for field in metadata:
        bad = deepcopy(metadata); del bad[field]
        rejected.append(_rejected(lambda bad=bad: guard.require_data_contract(bad), 'missing ' + field))
    with tempfile.TemporaryDirectory(prefix='move_eval_contract_') as folder:
        path = Path(folder) / 'fixture'
        path.write_bytes(b'not the declared data')
        rejected.append(_rejected(lambda: guard.verify_data_file(path, metadata), 'data hash mismatch'))
    for reason in ('completed', 'forced_pass_deadlock', 'max_turns', 'no_progress_cutoff'):
        rejected.append(_rejected(lambda reason=reason: guard.require_official_terminal_target(snapshot, reason),
                                  'unfinished target ' + reason))
    return rejected


def _encoded_costs(state):
    """Inspect real feature offsets, including every concrete legal action."""
    from splendor_ai.encoding import encode_legal_action_features, encode_public_observation_tensor
    legal = rules.legal_actions(state)
    observation = encode_public_observation_tensor(state, state.current_player)
    features = encode_legal_action_features(state, legal, state.current_player)
    checked = []
    for tier in (1, 2, 3):
        for slot, card in enumerate(state.visible_tier_cards[tier]):
            expected = [card.cost.get(color, 0) / 7.0 for color in COLORS]
            start = observation.sections['board'][0] + (4 * (tier - 1) + slot) * 13 + 8
            require(list(observation.vector[start:start + 5]) == expected, 'actual observation encoded card costs')
            for action, row in zip(legal, features, strict=True):
                if action.tier == tier and action.market_index == slot:
                    offset = row.sections['target_card'][0] + 8
                    require(list(row.vector[offset:offset + 5]) == expected, 'actual legal-action encoded costs')
            checked.append(card.card_id)
    return {'observation_sha256': digest(observation.vector),
            'features_sha256': digest([f.vector for f in features]), 'visible_cards': checked,
            'legal_count': len(legal)}


def _compound_all(state):
    observer = rules.observation_payload(state)
    phases, count = set(), 0
    for expected in rules.legal_actions(state):
        def select_aux(observation, choices):
            phase = observation['phase']['kind']; phases.add(phase)
            require(phase != 'Main', 'auxiliary projection reached Main')
            if phase == 'Payment':
                target = {'kind': 'Pay', 'tokens': bridge._vector(expected.spend_tokens, COLORS)}
            elif phase == 'Return':
                target = {'kind': 'Return', 'tokens': bridge._vector(expected.return_tokens)}
            else:
                require(phase == 'Noble', 'unknown auxiliary phase')
                target = {'kind': 'Noble', 'index': bridge.NOBLE_TO_RUST[expected.noble_id]}
            require(target in choices, 'atomic auxiliary choice absent from bridge')
            return target
        action = bridge.resolve_compound(observer, bridge._main(expected), select_aux)
        require(action == expected, 'compound resolver changed concrete representative')
        _step_check(state, action)
        count += 1
    return count, phases


def _semantic_edges(fixtures):
    states = {f['id']: guard.decode_training_snapshot(f['snapshot']) for f in fixtures}
    for name in ('noble_nonpurchase', 'multiple_nobles', 'noble_final_round'):
        state = states[name]
        options = [a for a in rules.legal_actions(state) if a.action_type.name == 'TAKE_TOKENS']
        require(options and all(a.noble_id is not None for a in options), 'nonpurchase misses compulsory noble')
        if name == 'multiple_nobles':
            require(len({a.noble_id for a in options}) >= 2, 'missing eligible noble alternative')
        after = _step_check(state, options[0])
        require(after.players[0].score == state.players[0].score + 3, 'nonpurchase noble points')
        if name == 'noble_final_round':
            require(after.pending_round_end and not after.terminal, 'noble did not trigger final round')
            final = _step_check(after, rules.legal_actions(after)[0])
            require(final.terminal, 'final reply did not naturally terminate')
            _rejected(lambda: guard.require_official_terminal_target(rules.serialize_snapshot(final),
                                                                     'forced_pass_deadlock'), 'false terminal label')
    require(bridge.main_actions(rules.observation_payload(states['forced_pass'])) == [{'kind': 'Pass'}],
            'forced-pass fixture support')
    mains = bridge.main_actions(rules.observation_payload(states['single_buy']))
    require(len(mains) == 1 and mains[0]['kind'] == 'BuyReserved', 'single-choice fixture support')
    returns = rules.legal_actions(states['gold_returns'])
    require(any('gold' in a.return_tokens for a in returns), 'gold-return alternative absent')
    require(any('gold' in a.spend_tokens for a in rules.legal_actions(states['face_T2_14'])),
            'gold-payment alternative absent')
    return ['six_faces', 'nonpurchase_noble', 'multiple_nobles', 'noble_final_round',
            'gold_payment', 'gold_return', 'private_reserves', 'forced_pass', 'single_choice', 'turn123', 'turn124']


def model_free_checks():
    """No model load or inference; preparation coverage, never admission alone."""
    sources = guard.verify_runtime_sources()
    fixtures = build_fixtures()
    rows, phases = [], set()
    for fixture in fixtures:
        state = guard.decode_training_snapshot(fixture['snapshot'])
        observer = rules.observation_payload(state)
        rebuilt = rules.state_from_observation(observer)
        require(rules.observation_payload(rebuilt) == observer, 'observer roundtrip')
        count, observed = _compound_all(state); phases.update(observed)
        rows.append({'id': fixture['id'], 'snapshot_sha256': digest(fixture['snapshot']),
                     'checked_atomic_transitions': count, 'encoding': _encoded_costs(rebuilt)})
    require({'Payment', 'Return', 'Noble'} <= phases, 'auxiliary coverage incomplete')
    return {'status': 'model_free_checks_passed', 'production_entrypoint_qualified': False,
            'runtime_sources': sources, 'ruleset_id': guard.RULESET_ID,
            'catalogue_sha256': guard.CATALOGUE_SHA256, 'turn_contract': guard.TURN_CONTRACT,
            'fixtures': rows, 'semantic_edges': _semantic_edges(fixtures),
            'rejections': identity_rejections(fixtures[0]), 'auxiliary_phases': sorted(phases)}


def _forward_probe(model, state, legal):
    """Check tensors actually delivered to a loaded model, without replacing it."""
    import torch
    from splendor_ai.alphazero.strong_policy import encode_position
    expected, _, _ = encode_position(state, model.config, legal)
    seen = []
    def check(_module, inputs):
        observation, features, mask = inputs
        require(torch.equal(observation, expected['observation'][None]), 'actual model observation tensor differs')
        require(torch.equal(features, expected['features'][None]), 'actual model action tensor differs')
        require(mask.shape == (1, len(legal)) and bool(mask.all()), 'incomplete actual legal mask')
        seen.append({'observation_sha256': digest(observation.tolist()),
                     'features_sha256': digest(features.tolist()), 'legal_count': len(legal)})
    return model.register_forward_pre_hook(check), seen


def _context(state, fit):
    from tools.corrected_planner_qualification_20261018 import local
    pair = next(p for p in range(201) if local.historical._public_fit_and_stream(p)[0] == fit)
    _, stream = local.historical._public_fit_and_stream(pair)
    return {'fit': fit, 'stream': stream, 'identity': ['main', 2 * pair, state.turn_index]}


def run_entrypoint_checks(greedy, raw_callback, *, boundary=lambda: None):
    """Actual model-bearing qualification, called only after frozen launch.

    The owning worker must call verify_runtime_sources once at startup; this
    function does not call it again. Every returned move is independently audited.
    """
    from tools.move_evaluation_20261023 import evaluator
    from tools.corrected_planner_qualification_20261018 import local
    require(local._INITIALIZED, 'actual proposal/raw models not initialized')
    fixtures = build_fixtures()
    rows, proposals, privacy = [], [], []
    for fixture in fixtures:
        boundary()
        state = guard.decode_training_snapshot(fixture['snapshot'])
        observer = rules.observation_payload(state)
        decoded = evaluator.decode_observer(observer)
        require(rules.observation_payload(decoded) == observer, 'actual decoder differs')
        world = evaluator.sample_world(decoded, 'move-evaluation-entrypoint|' + fixture['id'])
        require(rules.observation_payload(world) == observer, 'actual sampler changed lawful view')
        local.assert_corrected_cards(world)
        encoding = _encoded_costs(world)
        legal = rules.legal_actions(world)
        handle, seen = _forward_probe(local._RAW._model, world, legal)
        try:
            raw_action, raw_diag = raw_callback(observer)
        finally:
            handle.remove()
        require(bool(seen), 'raw production route did not reach checked encoder')
        _step_check(world, raw_action)
        action, trace = greedy.choose(observer, diagnostic_capture=True)
        _step_check(world, action)
        rows.append({'id': fixture['id'], 'observer_sha256': digest(observer),
                     'sampled_world_sha256': digest(rules.serialize_snapshot(world)), 'encoding': encoding,
                     'raw_model_inputs': seen, 'raw_action': rules.serialize_action(raw_action), 'raw': raw_diag,
                     'greedy_action': rules.serialize_action(action), 'greedy': trace})
        if 'corrected_face' in fixture['tags']:
            for fit in local.FITS:
                boundary()
                model, artifact = local.registered_model(f'anchor_{fit}')
                handle, seen = _forward_probe(model, decoded, rules.legal_actions(decoded))
                try:
                    proposal = local.proposal_for_root(decoded, _context(decoded, fit),
                        rules.legal_actions(decoded), local.score_actions(decoded, rules.legal_actions(decoded)))
                finally:
                    handle.remove()
                require(bool(seen) and proposal['fit'] == fit, 'actual proposal encoder route not exercised')
                proposals.append({'id': fixture['id'], 'fit': fit, 'model_sha256': artifact['sha256'],
                                  'model_inputs': seen, 'proposal_sha256': digest(proposal)})
        if fixture['id'] == 'private_reserves':
            changed = _private_variant(state)
            other = rules.observation_payload(changed)
            require(other == observer and rules.serialize_snapshot(changed) != fixture['snapshot'],
                    'privacy variant is not hidden-only')
            repeated, repeated_trace = greedy.choose(other, diagnostic_capture=True)
            require(repeated == action and repeated_trace == trace, 'greedy private state dependence')
            alternate = evaluator.sample_world(evaluator.decode_observer(other), 'move-evaluation-entrypoint|' + fixture['id'])
            require(rules.serialize_snapshot(alternate) == rules.serialize_snapshot(world), 'sampler hidden state dependence')
            privacy.append({'id': fixture['id'], 'scientific_trace_sha256': digest(trace), 'equal': True})
    trajectory = verify_actual_trajectory(greedy, fixtures, boundary=boundary)
    native = verify_native_contract(greedy, fixtures, rows, boundary=boundary)
    auxiliary = verify_actual_auxiliary(greedy, fixtures, boundary=boundary)
    return {'status': 'entrypoint_counterexamples_passed', 'ruleset_id': guard.RULESET_ID,
            'catalogue_sha256': guard.CATALOGUE_SHA256, 'turn_contract': guard.TURN_CONTRACT,
            'fixtures': rows, 'proposal_encoder_checks': proposals, 'privacy': privacy,
            'semantic_edges': _semantic_edges(fixtures), 'rejections': identity_rejections(fixtures[0]),
            'native_contract': native, 'actual_auxiliary': auxiliary,
            'actual_mixed_trajectory': trajectory, 'training_loader_qualified': False,
            'qualification_scope': 'finite entry-point counterexamples; timed panel and complete replay still required'}


def canonical_index(action):
    """Separate Python reference for the pinned 67-action indexing."""
    kind = action['kind']
    if kind == 'Take':
        values = action['tokens']
        return 32 + values.index(2) if 2 in values else sum(1 << i for i, n in enumerate(values) if n)
    offsets = {'ReserveVisible': (37, 'index'), 'ReserveDeck': (49, 'tier'),
               'BuyVisible': (52, 'index'), 'BuyReserved': (64, 'index')}
    require(kind in offsets, 'not a canonical Main policy action')
    offset, key = offsets[kind]
    return offset + action[key]


def verify_actual_trajectory(greedy, fixtures, *, boundary=lambda: None):
    """Exercise forced root -> greedy opposing turn -> raw own turn -> terminal."""
    from tools.move_evaluation_20261023 import evaluator
    from tools.corrected_planner_qualification_20261018 import local
    initial = guard.decode_training_snapshot(next(f['snapshot'] for f in fixtures
                                                   if f['id'] == 'noble_final_round'))
    # Player0 has14points and a compulsory noble; player1 moves immediately
    # before that visit, then receives the legal final-round response.
    initial.current_player, initial.turn_index = 1, 1
    initial = guard.decode_training_snapshot(rules.serialize_snapshot(initial))
    forced = next(a for a in rules.legal_actions(initial) if a.action_type.name == 'TAKE_TOKENS')
    result, trace = evaluator.trajectory(initial, forced, greedy, capture=True, boundary=boundary)
    require([row['policy'] for row in trace] == ['forced', 'greedy_entity', 'raw035'],
            'actual trajectory dispatch did not cover both continuation policies')
    state = initial
    for row in trace:
        boundary()
        require(row['state_hash'] == local.old.digest(local.serialize_state_snapshot(state)), 'trajectory state hash')
        action = rules.deserialize_action(row['action'])
        state = _step_check(state, action)
    require(state.terminal and result['reason'] == 'completed' and result['winner'] == state.winner and
            result['turns'] == 3 and result['final_state_sha256'] == local.old.digest(local.serialize_state_snapshot(state)),
            'actual mixed trajectory terminal/scoring mismatch')
    require(guard.require_official_terminal_target(rules.serialize_snapshot(state), 'completed') == result['winner'],
            'actual mixed trajectory unofficial outcome')
    return {'status': 'passed', 'initial_snapshot_sha256': digest(rules.serialize_snapshot(initial)),
            'result': result, 'trace': trace, 'independent_transitions': 3}


def verify_actual_auxiliary(greedy, fixtures, *, boundary=lambda: None):
    """Force bounded Main choices so all three actual auxiliary paths execute.

    Main interventions here are labelled qualification fixtures, never study
    choices. Each projection is passed to the same native greedy operation,
    whose non-Main path is the unchanged upstream auxiliary heuristic.
    """
    wanted = {'face_T1_05': {'kind': 'BuyVisible', 'index': 0},
              'gold_returns': {'kind': 'ReserveDeck', 'tier': 0},
              'multiple_nobles': {'kind': 'Take', 'tokens': [1, 1, 1, 0, 0]}}
    rows, covered = [], set()
    for fixture in fixtures:
        if fixture['id'] not in wanted:
            continue
        boundary()
        state = guard.decode_training_snapshot(fixture['snapshot'])
        observer = rules.observation_payload(state)
        phases = []
        def select_aux(projection, choices):
            phase = projection['phase']['kind']
            require(phase in {'Payment', 'Return', 'Noble'}, 'non-auxiliary forced projection')
            result = greedy.exchange({'op': 'greedy', 'observation': projection, 'legal': choices,
                                      'seed': int(digest(observer)[:16], 16)}, capture=True)
            require(result['branch'] == 'auxiliary_heuristic' and result['simulations'] == 0 and
                    result['leaf_calls'] == 0 and not greedy.callbacks and result['canonical_logits'] is None,
                    'auxiliary projection reached neural/search')
            require(result['action'] in choices and result['filtered'] == choices and
                    result['action'] == choices[result['selected_index']], 'invalid auxiliary response')
            covered.add(phase)
            phases.append({'phase': phase, 'projection_sha256': digest(projection), 'response': result})
            return result['action']
        action = bridge.resolve_compound(observer, wanted[fixture['id']], select_aux)
        _step_check(state, action)
        rows.append({'id': fixture['id'], 'forced_main': wanted[fixture['id']], 'phases': phases,
                     'action': rules.serialize_action(action)})
    require(covered == {'Payment', 'Return', 'Noble'}, 'actual auxiliary branch coverage incomplete')
    return {'status': 'passed', 'phases': sorted(covered), 'fixtures': rows}


def canonical_logits(policy81):
    """Separate Python 81-to-67 mapping; mask list frozen from public contract."""
    require(len(policy81) == 81 and all(math.isfinite(x) for x in policy81), 'bad reference logits')
    # Upstream TAKE_MASKS uses lexicographic combinations, not numeric mask order;
    # exact public constants are asserted by the native contract below.
    from itertools import combinations
    masks = [sum(1 << i for i in combo) for count in (1, 2, 3) for combo in combinations(range(5), count)]
    out = [-1e8] * 67
    for index, mask in enumerate(masks):
        out[mask] = float(policy81[30 + index])
    out[32:37] = policy81[55:60]
    out[37:49] = policy81[12:24]
    out[49:52] = policy81[24:27]
    out[52:64] = policy81[:12]
    out[64:67] = policy81[27:30]
    return out


def verify_native_contract(greedy, fixtures, runtime_rows, *, boundary=lambda: None):
    """Compare actual native policy inputs/masks to a separate Python reference."""
    import torch
    checks, branches = [], set()
    by_id = {row['id']: row for row in runtime_rows}
    for fixture in fixtures:
        boundary()
        state = guard.decode_training_snapshot(fixture['snapshot'])
        observer = rules.observation_payload(state)
        runtime = by_id[fixture['id']]['greedy']
        for phase in runtime['phases']:
            branches.add(phase['branch'])
        if runtime['forced_pass']:
            require(fixture['id'] == 'forced_pass' and not runtime['phases'] and runtime['inference_calls'] == 0,
                    'forced pass entered neural/native action map')
            checks.append({'id': fixture['id'], 'forced_pass_no_inference': True})
            continue
        contract = greedy.inspect(observer)
        require(set(contract) == {'kind', 'legal', 'filtered', 'canonical_indices', 'transfer_features', 'public_context'}
                and contract['kind'] == 'result', 'native contract schema')
        legal, filtered = bridge.main_actions(observer), contract['filtered']
        require(contract['legal'] == legal and filtered and all(a in legal for a in filtered), 'native Main legality')
        retained = [legal.index(a) for a in filtered]
        require(retained == sorted(set(retained)), 'native legal mask order')
        require(contract['canonical_indices'] == [canonical_index(a) for a in filtered], 'independent canonical map')
        main = runtime['phases'][0]
        require(main['phase'] == 'Main' and main['retained_indices'] == retained and
                main['canonical_indices'] == contract['canonical_indices'], 'actual policy legal mask differs')
        feature = contract['transfer_features']
        if state.turn_index < 124:
            require(len(feature) == 392, 'native transfer feature length')
            for tier in (1, 2, 3):
                for slot, card in enumerate(state.visible_tier_cards[tier]):
                    index = 4 * (tier - 1) + slot
                    start = (1 + 2 * index) * 7
                    require(feature[start:start + 5] == [card.cost.get(c, 0) for c in COLORS],
                            'actual native transfer card costs')
        else:
            require(feature is None and main['branch'] == 'late_heuristic' and runtime['inference_calls'] == 0,
                    'turn124 entered neural encoder')
        for pattern in ([float(i - 40) for i in range(81)], [0.0] * 81,
                        [1000.0 if i >= 60 else -float(i) for i in range(81)]):
            reference = greedy.native_reference(observer, pattern)
            expected = canonical_logits(pattern)
            winner = max(range(len(filtered)), key=lambda i: expected[canonical_index(filtered[i])])
            require(reference['filtered'] == filtered and reference['canonical_indices'] == contract['canonical_indices']
                    and reference['canonical_logits'] == expected and reference['selected_index'] == winner
                    and reference['action'] == filtered[winner], 'native policy mapping/mask/tie differs')
        if main['branch'] == 'neural_greedy':
            require(len(main['inference']) == 1, 'non-single neural evaluation')
            callback = main['inference'][0]
            tokens = torch.tensor(callback['tokens'], dtype=torch.float32).reshape(31, 48)
            require(tokens.shape == (31, 48) and bool(torch.isfinite(tokens).all()), 'actual entity input shape')
            for tier in (1, 2, 3):
                for slot, card in enumerate(state.visible_tier_cards[tier]):
                    index = 4 * (tier - 1) + slot
                    expected_cost = torch.tensor([card.cost.get(c, 0) for c in COLORS], dtype=torch.float32) * .1
                    require(torch.equal(tokens[3 + index, :5], expected_cost), 'actual model entity input card costs')
            logits = canonical_logits(callback['output83'][:81])
            require(torch.equal(torch.tensor(logits, dtype=torch.float32),
                                torch.tensor(main['canonical_logits'], dtype=torch.float32)), 'actual policy output map')
            selected = max(range(len(filtered)), key=lambda i: logits[canonical_index(filtered[i])])
            require(main['selected_index'] == selected and main['action'] == filtered[selected], 'actual greedy first maximum')
        if 'corrected_face' in fixture['tags']:
            require(main['branch'] == 'neural_greedy', 'corrected face did not reach actual native neural input')
        checks.append({'id': fixture['id'], 'contract_sha256': digest(contract),
                       'mask_reference_patterns': 3, 'actual_branch': main['branch']})
    require({'neural_greedy', 'single_choice', 'late_heuristic'} <= branches, 'native branch coverage incomplete')
    return {'status': 'passed', 'checks': checks, 'actual_branches': sorted(branches),
            'reference': 'separate Python canonical mapping, first maximum and encoded cost checks',
            'safe_filter_trust_boundary': 'authenticated upstream safe_choices, independently invoked contract operation'}
