"""Bounded semantic fixtures: no checkpoint load or experimental-panel access.

Trajectory fixtures take at most six synthetic steps with a tiny deterministic
scorer.  They exercise the unchanged termination machinery and process-local
policy restoration, not strength or throughput.
"""
from __future__ import annotations

import copy
from types import SimpleNamespace

import numpy as np
import pytest

from splendor_ai.encoding.action_features import encode_legal_action_features
from splendor_ai.engine.actions import Action, ActionType
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.engine.state import (Card, Noble, RESERVE_SOURCE_DECK,
                                     RESERVE_SOURCE_LEGACY_UNKNOWN, RESERVE_SOURCE_VISIBLE)
from tools.simulation_speed_probe_128_v2 import encoder
from . import collector


def private_state(actor):
    state = SplendorEnv(seed=217).initial_state()
    state.current_player = actor
    state.turn_index = 41 + actor
    for player in state.players:
        player.bonuses.update(white=3, blue=3, green=3, red=3, black=3)
        player.tokens.update(white=1, blue=1, green=1, red=1, black=1, gold=2)
    state.players[actor].reserve_card(Card("own", 1, "red", 1, {"white": 1}), RESERVE_SOURCE_DECK)
    state.players[1-actor].reserve_card(Card("hidden", 2, "blue", 2, {"green": 5}), RESERVE_SOURCE_DECK)
    state.players[1-actor].reserve_card(Card("public", 3, "white", 4, {"black": 7}), RESERVE_SOURCE_VISIBLE)
    return state


def assert_reference(state, actions, actor=None):
    expected = np.asarray([value.vector for value in encode_legal_action_features(state, actions, actor)],
                          dtype=np.float32).reshape(len(actions), 79)
    np.testing.assert_array_equal(encoder.action_array(state, actions, actor), expected)


@pytest.mark.parametrize("actor", [0, 1])
@pytest.mark.parametrize("private_source", [RESERVE_SOURCE_DECK, RESERVE_SOURCE_LEGACY_UNKNOWN])
def test_every_observer_ignores_opponent_private_identity_and_deck_order(actor, private_source):
    state = private_state(actor)
    state.players[1-actor].reserved_card_sources[0] = private_source
    changed = copy.deepcopy(state)
    changed.players[1-actor].reserved_cards[0] = Card("different-hidden", 2, "red", 1, {"white": 6})
    for cards in changed.hidden_tier_decks.values():
        cards.reverse()
    actions = SplendorEnv().legal_actions(state)
    assert actions == SplendorEnv().legal_actions(changed)
    for original, alternative in zip(encoder.arrays(state, actions), encoder.arrays(changed, actions), strict=True):
        np.testing.assert_array_equal(original, alternative)
    assert_reference(state, actions)
    assert_reference(changed, actions)


@pytest.mark.parametrize("actor", [0, 1])
def test_own_private_and_opponent_public_cards_remain_in_lawful_inputs(actor):
    state = private_state(actor)
    actions = SplendorEnv().legal_actions(state)
    own_changed = copy.deepcopy(state)
    own_changed.players[actor].reserved_cards[0] = Card("different-own", 1, "green", 3, {"blue": 2})
    public_changed = copy.deepcopy(state)
    public_changed.players[1-actor].reserved_cards[1] = Card("different-public", 3, "green", 5, {"white": 7})
    baseline_observation, baseline_features = encoder.arrays(state, actions)
    own_observation, own_features = encoder.arrays(own_changed, actions)
    public_observation, _ = encoder.arrays(public_changed, actions)
    assert not np.array_equal(baseline_observation, own_observation)
    assert not np.array_equal(baseline_features, own_features)
    assert not np.array_equal(baseline_observation, public_observation)
    assert_reference(own_changed, actions)
    assert_reference(public_changed, actions)


@pytest.mark.parametrize("actor", [0, 1])
def test_explicit_observer_override_and_encoder_do_not_mutate_state(actor):
    state = private_state(actor)
    actions = SplendorEnv().legal_actions(state)
    state.current_player = 1 - actor
    before = copy.deepcopy(state)
    assert_reference(state, actions, actor)
    assert state == before


def test_multiple_legal_noble_choices_gold_payments_and_returns_are_exact():
    state = private_state(0)
    state.nobles = [Noble("noble-a", 3, {"white": 3}), Noble("noble-b", 3, {"blue": 3})]
    state.players[0].tokens.update(white=2, blue=2, green=1, red=1, black=1, gold=3)
    actions = SplendorEnv().legal_actions(state)
    assert {action.noble_id for action in actions} >= {"noble-a", "noble-b"}
    assert any(action.return_tokens for action in actions)
    assert any("gold" in action.spend_tokens for action in actions)
    assert_reference(state, list(reversed(actions)))


def test_supplied_support_memory_boundary_is_exact_and_never_silent_fallback():
    state = SplendorEnv(seed=34).initial_state()
    action = Action(ActionType.PASS)
    # Repeated supplied actions stress the encoder's allocation limit; this is
    # deliberately not claimed to be a reachable 4096-legal-action position.
    actions = [action] * encoder.MAX_ACTIONS
    actual = encoder.action_array(state, actions)
    assert actual.shape == (4096, 79) and actual.dtype == np.float32
    one = encoder.action_array(state, [action])[0]
    np.testing.assert_array_equal(actual, np.repeat(one[None], 4096, axis=0))
    with pytest.raises(ValueError, match="memory bound"):
        encoder.action_array(state, actions + [action])


class TinyModel:
    def __init__(self, mode="first", failure=None):
        self.config = SimpleNamespace(observation_size=295, action_feature_size=79)
        self.training = False
        self.mode, self.failure, self.calls = mode, failure, 0

    def __call__(self, observation, features, mask):
        import torch
        self.calls += 1
        if self.failure is not None:
            raise self.failure
        scores = torch.zeros(mask.shape, dtype=torch.float32)
        if self.mode == "noop":
            scores = 2 * features[:, :, encoder.TYPE_INDEX[ActionType.TAKE_TOKENS]] - features[:, :, 54:60].abs().sum(2)
        return scores, torch.zeros((1,), dtype=torch.float32)


class TinyOriginal:
    def __init__(self, mode="first", failure=None):
        self._model, self._device = TinyModel(mode, failure), "cpu"
        self.mode = mode

    def choose_action(self, env, state, legal_actions=None):
        actions = env.legal_actions(state) if legal_actions is None else legal_actions
        if not actions:
            return None
        if self.mode == "noop":
            return next(action for action in actions if action.action_type == ActionType.TAKE_TOKENS
                        and sorted(action.take_tokens) == sorted(action.return_tokens))
        return actions[0]


def exhausted_state():
    state = SplendorEnv(seed=91).initial_state()
    state.visible_tier_cards = {1: [], 2: [], 3: []}
    state.hidden_tier_decks = {1: [], 2: [], 3: []}
    state.deck_counts = {1: 0, 2: 0, 3: 0}
    state.bank_tokens = {key: 0 for key in state.bank_tokens}
    state.nobles = []
    return state


def repetition_state():
    state = exhausted_state()
    # Conserved 4-per-colour / 5-gold totals, each player holding ten tokens.
    for player in state.players:
        player.tokens.update(white=2, blue=2, green=2, red=1, black=1, gold=2)
    state.bank_tokens.update(red=2, black=2, gold=1)
    return state


@pytest.mark.parametrize("kind,reason,turns", [
    ("terminal", "completed", 0), ("forced_pass", "forced_pass_deadlock", 2),
    ("turn_cap", "max_turns", 1), ("repetition", "repetition_cutoff", 6),
])
def test_rare_termination_trace_equivalence_with_tiny_policy(kind, reason, turns):
    from splendor_ai.alphazero import run_afterstate_advantage as old
    state = repetition_state() if kind == "repetition" else exhausted_state()
    if kind == "terminal":
        state.terminal, state.winner = True, 1
    if kind == "turn_cap":
        state.turn_index = 399
    original = TinyOriginal("noop" if kind == "repetition" else "first")
    sentinel = old._RAW
    reference = collector.trajectory_with_policy(old, original, state, include_trace=True)
    observed = []
    def check_inputs(position, legal, actual):
        import torch
        expected = encoder.reference_tensors(position, legal)
        assert all(torch.equal(a, b) for a, b in zip(actual, expected, strict=True))
        observed.append(position.turn_index)
    fast = encoder.Fast035(original, on_position=check_inputs)
    actual = collector.trajectory_with_policy(old, fast, state, include_trace=True)
    assert old._RAW is sentinel
    assert actual == reference
    assert actual["reason"] == reason and actual["turns"] == turns
    assert len(observed) == turns
    if kind == "repetition":
        assert actual["winner"] is None


def test_no_progress_branch_with_temporary_small_fixture_limit(monkeypatch):
    from splendor_ai.alphazero import run_afterstate_advantage as old
    saved_config, sentinel = old.CONFIG, old._RAW
    # Exercise the branch without spending sixty synthetic moves or altering
    # the frozen production value. This tests branch equivalence, not its dose.
    with monkeypatch.context() as patch:
        patch.setattr(old, "CONFIG", {**old.CONFIG, "no_progress_limit": 3})
        state = repetition_state()
        original = TinyOriginal("noop")
        reference = collector.trajectory_with_policy(old, original, state, include_trace=True)
        actual = collector.trajectory_with_policy(old, encoder.Fast035(original), state, include_trace=True)
        assert actual == reference
        assert actual["reason"] == "no_progress_cutoff" and actual["turns"] == 3
    assert old.CONFIG is saved_config and old._RAW is sentinel


def test_defensive_stalled_branch_never_calls_policy(monkeypatch):
    from splendor_ai.alphazero import run_afterstate_advantage as old
    saved_env, sentinel = old.SplendorEnv, old._RAW
    original = TinyOriginal()
    # Real nonterminal Splendor emits PASS instead of an empty list. Injecting
    # this impossible response tests fail-closed branch preservation only.
    with monkeypatch.context() as patch:
        patch.setattr(old, "SplendorEnv", lambda: SimpleNamespace(legal_actions=lambda state: []))
        state = exhausted_state()
        reference = collector.trajectory_with_policy(old, original, state, include_trace=True)
        actual = collector.trajectory_with_policy(old, encoder.Fast035(original), state, include_trace=True)
        assert actual == reference and actual["reason"] == "stalled" and actual["turns"] == 0
        assert original._model.calls == 0
    assert old.SplendorEnv is saved_env and old._RAW is sentinel


def test_empty_terminal_choice_never_invokes_model_and_preserves_guard():
    state = exhausted_state()
    state.terminal = True
    original = TinyOriginal()
    calls = []
    fast = encoder.Fast035(original, boundary=lambda: calls.append("guard"))
    assert fast.choose_action(SplendorEnv(), state) is None
    assert calls == ["guard"] and original._model.calls == 0
    assert encoder.action_array(state, []).shape == (0, 79)


@pytest.mark.parametrize("point", ["boundary", "callback", "model", "unsupported_shape"])
def test_any_failure_restores_process_local_scorer_and_is_not_hidden(point):
    from splendor_ai.alphazero import run_afterstate_advantage as old
    sentinel = old._RAW
    error = RuntimeError("deliberate isolated fixture failure")
    def fail(*args):
        raise error
    original = TinyOriginal(failure=error if point == "model" else None)
    fast = encoder.Fast035(original, boundary=fail if point == "boundary" else lambda: None,
                           on_position=fail if point == "callback" else None)
    state = exhausted_state()
    before = copy.deepcopy(state)
    with pytest.raises((RuntimeError, ValueError)) as caught:
        with collector.policy_scope(old, fast):
            assert old._RAW is fast
            if point == "unsupported_shape":
                old._RAW.choose_action(SplendorEnv(), state, [Action(ActionType.PASS)] * 4097)
            else:
                old.trajectory(state)
    assert old._RAW is sentinel
    assert state == before
    if point != "unsupported_shape":
        assert caught.value is error


def test_policy_scopes_are_nested_and_restore_exact_prior_identity():
    from splendor_ai.alphazero import run_afterstate_advantage as old
    sentinel, outer, inner = old._RAW, object(), object()
    with collector.policy_scope(old, outer):
        assert old._RAW is outer
        with collector.policy_scope(old, inner):
            assert old._RAW is inner
        assert old._RAW is outer
    assert old._RAW is sentinel


@pytest.mark.parametrize("mutation", ["device", "training", "observation", "action"])
def test_fast_wrapper_rejects_nonreference_configuration(mutation):
    original = TinyOriginal()
    if mutation == "device":
        original._device = "cuda"
    elif mutation == "training":
        original._model.training = True
    elif mutation == "observation":
        original._model.config.observation_size = 296
    else:
        original._model.config.action_feature_size = 80
    with pytest.raises(ValueError):
        encoder.Fast035(original)
