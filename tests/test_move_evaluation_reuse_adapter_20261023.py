"""Model-free parity and isolation checks; no model/native constructor is called."""
import ast
from collections import Counter
from copy import deepcopy
from dataclasses import FrozenInstanceError
import inspect
from types import MethodType

import pytest

from tools import future_training_rules_20261021 as guard
from tools.external_strength_screen_20261018 import rules
from tools.external_strength_screen_20261018_02 import bridge as old
from tools.move_evaluation_20261023 import qualification
from tools.move_evaluation_20261023 import greedy_adapter as reference
from tools.move_evaluation_reuse_20261023 import prepared
from tools.move_evaluation_reuse_20261023 import greedy_adapter as reuse


@pytest.fixture(scope='module')
def fixtures():
    return {f['id']: f for f in qualification.build_fixtures()}


def observer(fixture):
    return rules.observation_payload(guard.decode_training_snapshot(fixture['snapshot']))


def context(payload):
    value = prepared.prepare(payload)
    native = prepared.encode_prepared(value)
    main = prepared.main_prepared(value)
    return value, native, main


def requested_aux(expected, records, *, mutate=False):
    def select(native, choices):
        records.append(deepcopy((native, choices)))
        phase = native['phase']['kind']
        selected = ({'kind': 'Pay', 'tokens': old._vector(expected.spend_tokens, old.COLORS)}
                    if phase == 'Payment' else
                    {'kind': 'Return', 'tokens': old._vector(expected.return_tokens)}
                    if phase == 'Return' else
                    {'kind': 'Noble', 'index': old.NOBLE_TO_RUST[expected.noble_id]})
        assert selected in choices
        if mutate:
            native.clear()
            choices.clear()
        return selected
    return select


@pytest.mark.parametrize('name', [f['id'] for f in qualification.build_fixtures()])
def test_native_and_main_match_frozen_order(fixtures, name):
    payload = observer(fixtures[name])
    value, native, main = context(payload)
    assert native == old.encode_observation(payload)
    assert main == old.main_actions(payload)
    assert value.legal == tuple(rules.legal_actions(rules.state_from_observation(payload)))
    assert value.observer_sha256 == reference.digest(payload)


def test_every_atomic_choice_has_identical_projection_order_and_resolution(fixtures):
    seen_types, phases, returns, corrected_faces = set(), set(), set(), set()
    for name, fixture in fixtures.items():
        payload = observer(fixture)
        value, native, main = context(payload)
        if name.startswith('face_'):
            corrected_faces.add(name.removeprefix('face_'))
        for expected in value.legal:
            before, after = [], []
            selected = old._main(expected)
            frozen_action = old.resolve_compound(payload, selected, requested_aux(expected, before))
            actual = prepared.resolve_prepared(value, selected, requested_aux(expected, after))
            assert actual == expected == frozen_action
            assert after == before
            assert prepared.encode_prepared(value) == native
            assert prepared.main_prepared(value) == main
            seen_types.add(expected.action_type.name)
            phases.update(record[0]['phase']['kind'] for record in after)
            returns.update(expected.return_tokens)
    assert seen_types == {'TAKE_TOKENS', 'BUY_VISIBLE', 'BUY_RESERVED', 'RESERVE_VISIBLE', 'RESERVE_DECK', 'PASS'}
    assert phases == {'Payment', 'Return', 'Noble'}
    assert 'gold' in returns
    assert corrected_faces == set(rules.CORRECTED_COSTS)


def test_input_and_returned_native_objects_cannot_mutate_context(fixtures):
    payload = observer(fixtures['private_reserves'])
    pristine = deepcopy(payload)
    value = prepared.prepare(payload)
    payload['observation']['bank_tokens'].clear()
    payload['observation']['players'][0]['reserves'].clear()
    payload['observation']['visible_tier_cards'].clear()
    native = prepared.encode_prepared(value)
    main = prepared.main_prepared(value)
    original_native, original_main = deepcopy(native), deepcopy(main)
    native['players'][0].clear()
    native['market'].clear()
    main.clear()
    assert prepared.encode_prepared(value) == original_native == old.encode_observation(pristine)
    assert prepared.main_prepared(value) == original_main == old.main_actions(pristine)
    assert value.observer_sha256 == reference.digest(pristine)
    with pytest.raises(FrozenInstanceError):
        value.legal = ()
    with pytest.raises(FrozenInstanceError):
        value.legal[0].noble_id = 'N01'


@pytest.mark.parametrize('name', ['face_T1_05', 'gold_returns', 'multiple_nobles'])
def test_auxiliary_callback_mutations_do_not_change_cached_support(fixtures, name):
    value, native, main = context(observer(fixtures[name]))
    found = False
    for expected in value.legal:
        records = []
        actual = prepared.resolve_prepared(value, old._main(expected), requested_aux(expected, records, mutate=True))
        assert actual == expected
        assert prepared.encode_prepared(value) == native
        assert prepared.main_prepared(value) == main
        if records:
            found = True
            break
    assert found


def test_blind_reserve_and_future_deck_order_are_absent(fixtures):
    state = guard.decode_training_snapshot(fixtures['private_reserves']['snapshot'])
    changed = qualification._private_variant(state)
    assert rules.serialize_snapshot(state) != rules.serialize_snapshot(changed)
    payload, variant = rules.observation_payload(state), rules.observation_payload(changed)
    assert payload == variant
    a, an, am = context(payload)
    b, bn, bm = context(variant)
    assert (a.observer_sha256, an, am, a.legal) == (b.observer_sha256, bn, bm, b.legal)
    other = payload['observation']['players'][1]['reserves'][0]
    assert other['card_id'] is None
    assert bn['players'][1]['reserved'][0]['card'] == old.NONE


def test_starting_seat_one_normalization():
    state = rules.CorrectedEnv(seed=120).initial_state()
    state.start_player = state.current_player = 1
    payload = rules.observation_payload(state)
    _, native, main = context(payload)
    assert native == old.encode_observation(payload)
    assert main == old.main_actions(payload)
    assert native['viewer'] == native['current'] == 0


def noble_pass_payload(fixtures):
    state = guard.decode_training_snapshot(fixtures['forced_pass']['snapshot'])
    state.current_player = 0
    state.turn_index = 0
    return rules.observation_payload(state)


def test_forced_pass_still_resolves_compulsory_noble(fixtures):
    payload = noble_pass_payload(fixtures)
    value, _, main = context(payload)
    assert main == [{'kind': 'Pass'}]
    assert len(value.legal) > 1 and all(a.noble_id for a in value.legal)
    for expected in value.legal:
        records, old_records = [], []
        result = prepared.resolve_prepared(value, main[0], requested_aux(expected, records))
        reference_result = old.resolve_compound(payload, main[0], requested_aux(expected, old_records))
        assert result == reference_result == expected
        assert records == old_records and records[0][0]['phase']['kind'] == 'Noble'


def test_resolution_requires_completed_preparation_and_valid_choices(fixtures):
    value = prepared.prepare(observer(fixtures['multiple_nobles']))
    with pytest.raises(ValueError, match='precede resolution'):
        prepared.resolve_prepared(value, {'kind': 'Pass'}, lambda *_: None)
    prepared.encode_prepared(value)
    with pytest.raises(ValueError, match='precede resolution'):
        prepared.resolve_prepared(value, {'kind': 'Pass'}, lambda *_: None)
    main = prepared.main_prepared(value)
    with pytest.raises(ValueError, match='invalid Main'):
        prepared.resolve_prepared(value, {'kind': 'Pass'}, lambda *_: None)
    with pytest.raises(ValueError, match='invalid auxiliary'):
        prepared.resolve_prepared(value, main[0], lambda *_: {'kind': 'Invalid'})


@pytest.mark.parametrize('field,value', [('schema', 'wrong'), ('ruleset_id', 'wrong'),
                                       ('catalogue_sha256', '0'*64), ('extra', {})])
def test_bad_envelopes_fail_closed(fixtures, field, value):
    payload = observer(fixtures['private_reserves'])
    payload[field] = value
    with pytest.raises((ValueError, KeyError, TypeError)):
        prepared.prepare(payload)


def test_disclosed_opponent_blind_identity_is_rejected(fixtures):
    state = guard.decode_training_snapshot(fixtures['private_reserves']['snapshot'])
    payload = rules.observation_payload(state)
    payload['observation']['players'][1]['reserves'][0]['card_id'] = state.players[1].reserved_cards[0].card_id
    with pytest.raises(ValueError):
        prepared.prepare(payload)


def fake_opponent(cls, mutation=None):
    """Synthetic transport only; deliberately bypass every model/native initializer."""
    instance = cls.__new__(cls)
    instance.boundary = lambda: None
    instance.callbacks = []

    def exchange(self, request, *, capture=False):
        native, legal = request['observation'], request['legal']
        main = native['phase']['kind'] == 'Main'
        branch = ('auxiliary_heuristic' if not main else 'single_choice' if len(legal) == 1
                  else 'late_heuristic' if native['turns'] >= 124 else 'neural_greedy')
        neural = branch == 'neural_greedy'
        self.callbacks = [{'synthetic': True, 'input_sha256': reference.digest(native)}] if neural else []
        if mutation:
            mutation()
        return {'kind': 'result', 'action': deepcopy(legal[0]), 'branch': branch,
                'native_legal_count': len(legal), 'filtered': deepcopy(legal),
                'canonical_indices': [reference.canonical_index(a) for a in legal] if main else [],
                'selected_index': 0, 'canonical_logits': [0.0]*67 if neural else None,
                'simulations': 0, 'leaf_calls': int(neural)}
    instance.exchange = MethodType(exchange, instance)
    return instance


@pytest.mark.parametrize('capture', [False, True])
def test_complete_synthetic_choose_matches_frozen_science(fixtures, capture):
    branches = set()
    for fixture in fixtures.values():
        payload = observer(fixture)
        expected = fake_opponent(reference.GreedyOpponent).choose(payload, diagnostic_capture=capture)
        actual = fake_opponent(reuse.GreedyOpponent).choose(payload, diagnostic_capture=capture)
        assert actual == expected
        branches.update(p['branch'] for p in actual[1]['phases'])
    payload = noble_pass_payload(fixtures)
    expected = fake_opponent(reference.GreedyOpponent).choose(payload, diagnostic_capture=capture)
    actual = fake_opponent(reuse.GreedyOpponent).choose(payload, diagnostic_capture=capture)
    assert actual == expected
    assert actual[1]['forced_pass'] and actual[1]['inference_calls'] == 0
    assert [p['phase'] for p in actual[1]['phases']] == ['Noble']
    branches.update(p['branch'] for p in actual[1]['phases'])
    assert branches == {'neural_greedy', 'single_choice', 'late_heuristic', 'auxiliary_heuristic'}


def test_mid_choice_input_mutation_retains_original_support_and_hash(fixtures):
    payload = observer(fixtures['private_reserves'])
    expected = fake_opponent(reference.GreedyOpponent).choose(deepcopy(payload))
    def mutate():
        payload.clear()
    actual = fake_opponent(reuse.GreedyOpponent, mutate).choose(payload)
    assert actual == expected and payload == {}


@pytest.mark.parametrize('forced', [False, True])
def test_one_preparation_and_no_legacy_reconstruction(fixtures, monkeypatch, forced):
    payload = noble_pass_payload(fixtures) if forced else observer(fixtures['face_T1_05'])
    counts = Counter()
    for module, name in [(prepared, name) for name in ('prepare', 'encode_prepared', 'main_prepared', 'resolve_prepared')] + [
            (rules, 'state_from_observation'), (rules, 'legal_actions')]:
        original = getattr(module, name)
        def wrapped(*args, _name=name, _original=original, **kwargs):
            counts[_name] += 1
            return _original(*args, **kwargs)
        monkeypatch.setattr(module, name, wrapped)
    def forbidden(*_args, **_kwargs):
        raise AssertionError('legacy observer preparation called')
    for name in ('_state_and_legal', 'encode_observation', 'main_actions', 'resolve_compound'):
        monkeypatch.setattr(old, name, forbidden)
    fake_opponent(reuse.GreedyOpponent).choose(payload)
    assert counts == Counter(dict.fromkeys(('prepare', 'encode_prepared', 'main_prepared', 'resolve_prepared',
                                           'state_from_observation', 'legal_actions'), 1))


def test_scientific_selection_and_inherited_arithmetic_are_unchanged():
    def selected(module):
        tree = ast.parse(inspect.getsource(module))
        return next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'select')
    assert ast.dump(selected(reference), include_attributes=False) == ast.dump(selected(reuse), include_attributes=False)
    for name in ('__init__', '_infer', 'exchange', 'inspect', 'native_reference'):
        assert getattr(reuse.GreedyOpponent, name) is getattr(reference.GreedyOpponent, name)
    assert reuse.PROFILE_ID == reference.PROFILE_ID
    assert reuse.IMPLEMENTATION_ID == prepared.IMPLEMENTATION_ID
    assert reuse.IMPLEMENTATION_ID != reuse.PROFILE_ID
