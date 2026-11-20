import importlib.util
import json
from pathlib import Path
import random

import numpy as np
import pytest
import torch

from splendor_ai.alphazero.afterstate_149 import Afterstate, AdvantageModel, deterministic_part, features, public_card_status
from splendor_ai.alphazero import run_afterstate_advantage as p
from splendor_ai.bots.hidden_information import determinize_for_observer
from splendor_ai.engine.actions import Action, ActionType
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.engine.state import RESERVE_SOURCE_DECK, RESERVE_SOURCE_VISIBLE
from splendor_ai.encoding import encode_public_observation
from splendor_ai.training.replay import serialize_state_snapshot


def positions():
    env = SplendorEnv(seed=149)
    initial = env.initial_state()
    rich = initial.copy_shallow()
    rich.players[0].bonuses = {c: 5 for c in rich.players[0].bonuses}
    rich.players[0].score = 14
    rich.current_player = rich.start_player = 0
    last = rich.copy_shallow()
    last.start_player = 1
    reserve = initial.copy_shallow()
    card = reserve.visible_tier_cards[1].pop()
    reserve.players[0].reserve_card(card, RESERVE_SOURCE_VISIBLE)
    reserve.players[0].tokens = {c: (5 if c == "gold" else 1) for c in reserve.players[0].tokens}
    dry = rich.copy_shallow()
    dry.hidden_tier_decks = {tier: [] for tier in (1, 2, 3)}
    dry.deck_counts = {tier: 0 for tier in (1, 2, 3)}
    return [initial, rich, last, reserve, dry]


@pytest.mark.parametrize("index", range(5))
def test_all_legal_actions_compose_without_mutating_source(index):
    state = positions()[index]
    before = serialize_state_snapshot(state)
    env = SplendorEnv()
    for action in env.legal_actions(state):
        after = deterministic_part(state, action)
        assert isinstance(after, Afterstate)
        card = None if after.chance_kind is None else state.hidden_tier_decks[after.chance_tier][0].card_id
        assert serialize_state_snapshot(after.resolve(card)) == serialize_state_snapshot(env.step(state, action))
    assert serialize_state_snapshot(state) == before


@pytest.mark.parametrize("kind", [ActionType.BUY_VISIBLE, ActionType.RESERVE_VISIBLE, ActionType.RESERVE_DECK])
def test_every_chance_card_matches_engine_with_that_card_first(kind):
    state = positions()[1]
    action = next(a for a in SplendorEnv().legal_actions(state) if a.action_type == kind and a.tier == 1)
    after = deterministic_part(state, action)
    for card in state.hidden_tier_decks[1]:
        world = state.copy_shallow()
        world.hidden_tier_decks[1].remove(card)
        world.hidden_tier_decks[1].insert(0, card)
        assert serialize_state_snapshot(after.resolve(card.card_id)) == serialize_state_snapshot(SplendorEnv().step(world, action))


def test_pending_private_reservation_is_owner_only_after_resolution():
    state = positions()[0]
    action = next(a for a in SplendorEnv().legal_actions(state) if a.action_type == ActionType.RESERVE_DECK)
    pending = deterministic_part(state, action)
    with pytest.raises(ValueError):
        pending.resolve()
    with pytest.raises(ValueError):
        pending.resolve("not-a-card")
    with pytest.raises(TypeError):
        deterministic_part(pending, action)
    card = state.hidden_tier_decks[action.tier][0]
    resolved = pending.resolve(card.card_id)
    assert resolved.players[0].reserved_card_source(0) == RESERVE_SOURCE_DECK
    assert encode_public_observation(resolved, 0)["self"]["reserved_cards"] == [card.card_id]
    assert encode_public_observation(resolved, 1)["opponent"]["reserved_cards"] == [None]
    assert resolved.deck_counts[action.tier] == state.deck_counts[action.tier] - 1
    assert resolved.players[0].tokens["gold"] == 1


@pytest.mark.parametrize("arm", p.ARMS)
def test_features_ignore_future_order_and_opponent_private_identity(arm):
    state = positions()[0]
    opp = state.players[1]
    opp.reserve_card(state.hidden_tier_decks[2].pop(), RESERVE_SOURCE_DECK)
    state.deck_counts[2] -= 1
    for action in SplendorEnv().legal_actions(state):
        expected = features(state, action, arm)
        for seed in range(3):
            world = determinize_for_observer(state, 0, random.Random(seed))
            assert torch.equal(expected, features(world, action, arm))
    assert public_card_status(state, 0)[::6].count(1.) == sum(state.deck_counts.values()) + 1


def test_known_consequences_and_terminal_noble_information_are_encoded():
    state = positions()[2]
    action = next(a for a in SplendorEnv().legal_actions(state) if a.noble_id is not None)
    after = deterministic_part(state, action)
    assert after._partial.terminal
    assert after._partial.players[0].nobles
    assert not torch.equal(features(state, action, "control"), features(state, action, "afterstate"))


def test_forced_pass_and_empty_deck_have_no_chance():
    state = positions()[0]
    state.bank_tokens = {c: 0 for c in state.bank_tokens}
    state.visible_tier_cards = {t: [] for t in (1, 2, 3)}
    state.hidden_tier_decks = {t: [] for t in (1, 2, 3)}
    state.deck_counts = {t: 0 for t in (1, 2, 3)}
    action, = SplendorEnv().legal_actions(state)
    assert action.action_type == ActionType.PASS
    after = deterministic_part(state, action)
    assert after.chance_kind is None
    assert serialize_state_snapshot(after.resolve()) == serialize_state_snapshot(SplendorEnv().step(state, action))
    with pytest.raises(ValueError):
        after.resolve("x")


def test_equal_initialization_exact_zero_baseline_and_gradient():
    torch.set_num_threads(1)
    torch.manual_seed(4)
    x = torch.randn(3, 4, 12)
    torch.manual_seed(5)
    control = AdvantageModel(12)
    torch.manual_seed(5)
    after = AdvantageModel(12)
    assert torch.equal(control(x), after(x))
    assert torch.equal(control(x)[:, 0], torch.zeros(3))
    loss = (control(x)[:, 1:] - 1).square().mean()
    loss.backward()
    assert control.network[0].weight.grad.abs().sum() > 0


def test_padding_and_baseline_do_not_inflate_mse_denominator():
    pred = torch.tensor([[0., 2., 90., 90.], [0., 90., 90., 90.]])
    mask = torch.tensor([[False, True, False, False], [False, False, False, False]])
    assert p.state_loss(pred, torch.zeros_like(pred), mask).tolist() == [4., 0.]


def test_seeds_are_disjoint_and_not_training_seed_dependent():
    seeds = [{p.world_seed(split, i, w) for i in range(p.CONFIG["states"][split]) for w in range(p.CONFIG["worlds"][split])}
             for split in ("train", "validation", "test")]
    assert not seeds[0] & seeds[1] and not seeds[0] & seeds[2] and not seeds[1] & seeds[2]


def test_complete_action_round_trip():
    for state in positions():
        for action in SplendorEnv().legal_actions(state):
            assert p.decode_action(p.serialize_action(action)) == action


def test_source_eligibility_has_seat_phase_and_private_restriction():
    state = positions()[0]
    assert p.eligible(state, 0) and not p.eligible(state, 1)
    state.turn_index, state.current_player = 40, 1
    assert p.eligible(state, 5)
    state.players[0].reserve_card(state.hidden_tier_decks[1].pop(), RESERVE_SOURCE_DECK)
    assert not p.eligible(state, 5)


def test_selected_outcome_statistics_do_not_use_empirical_best_action():
    q = torch.tensor([[0., .25, -.5, 1.]] * 192)
    mask = torch.tensor([[False, True, True, True]] * 192)
    pred = {arm: [{"seed": seed, "choices": [1 if arm == "afterstate" else 0]*192,
                   "advantages": [([0., .25, -.5, 1.] if arm == "afterstate" else [0., -.25, -.5, 0.])]*192}
                  for seed in p.CONFIG["training_seeds"]] for arm in p.ARMS}
    result = p.statistics(q, q, mask, pred)
    assert result["contrasts"]["afterstate_minus_control"]["mean"] == .25
    assert result["statistical_pass"]
    spec = importlib.util.spec_from_file_location("audit149", Path("scripts/analysis/verify_afterstate_advantage_149.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.independent_stats(q, q, mask, pred, result)


def test_width_or_precision_failure_cannot_advance():
    q = torch.tensor([[0., .125, 0., 0.], [0., -.0625, 0., 0.]] * 96)
    mask = torch.tensor([[False, True, True, True]] * 192)
    pred = {arm: [{"choices": [1 if arm == "afterstate" else 0]*192, "advantages": [[0., 0., 0., 0.]]*192}
                 for _ in range(3)] for arm in p.ARMS}
    result = p.statistics(q, q, mask, pred)
    assert not result["statistical_pass"]  # equal MSE cannot pass even with selected gain


def test_full_fit_reproduction_on_synthetic_advantages():
    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(149)
    x = torch.randn(16, 4, 12, generator=generator)
    y = x[..., 0].tanh() - x[:, :1, 0].tanh()
    data = {"control": x, "afterstate": x.clone(), "y": y,
            "mask": torch.tensor([[False, True, True, True]]*16)}
    a = p.fit_model("control", 149001, data, data, updates=32)
    b = p.fit_model("afterstate", 149001, data, data, updates=32)
    assert a["history"] == b["history"]
    assert a["history"][-1]["validation_mse"] < a["history"][0]["validation_mse"]
    for name in a["selected_state"]:
        assert torch.equal(a["selected_state"][name], b["selected_state"][name])


def test_continuation_collector_matches_independent_auditor(monkeypatch):
    class FirstLegal:
        def choose_action(self, env, state, legal=None):
            return (legal if legal is not None else env.legal_actions(state))[0]

        def score_actions(self, state, legal):
            return tuple(-i for i in range(len(legal)))
    monkeypatch.setattr(p, "_RAW", FirstLegal())
    spec = importlib.util.spec_from_file_location("audit149rollout", Path("scripts/analysis/verify_afterstate_advantage_149.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for state in (positions()[0], positions()[2]):
        actions = [next(a for a in SplendorEnv().legal_actions(state) if a.action_type == kind)
                   for kind in (ActionType.TAKE_TOKENS, ActionType.RESERVE_DECK)]
        for action in actions:
            record = p.trajectory(state, action)
            valid = record["reason"] in p.VALID_NATURAL_TERMINATIONS
            record["wdl"] = p.outcome_value(record["winner"], state.current_player) if valid else None
            assert module.replay_branch(state, action, state.current_player) == record


def test_saved_fresh_source_survives_json_numeric_key_normalization(monkeypatch):
    monkeypatch.setattr(p, "_RAW", None)
    p.initialize_worker()
    seed = 1_499_000_000
    record = {"seed": seed, **p.trajectory(SplendorEnv(seed=seed).initial_state(), include_trace=True)}
    saved = json.loads(json.dumps(record))
    assert record["final_state"]["deck_counts"] != saved["final_state"]["deck_counts"]
    assert p.digest(record) == p.digest(saved)
    states = p.reconstruct_source(saved, verify_raw=True)
    assert len(states) == record["turns"]


def test_independent_statistics_agree_with_different_seed_choices():
    spec = importlib.util.spec_from_file_location("audit149precision", Path("scripts/analysis/verify_afterstate_advantage_149.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    g = torch.Generator().manual_seed(4)
    q = torch.randint(-128, 129, (192, 4), generator=g) / 128
    y = q - q[:, :1]
    mask = torch.tensor([[False, True, True, True]]*192)
    predictions = {arm: [{"choices": torch.randint(4, (192,), generator=g).tolist(),
                          "advantages": torch.randn(192, 4, generator=g).tolist()} for _ in range(3)] for arm in p.ARMS}
    result = p.statistics(q, y, mask, predictions)
    module.independent_stats(q, y, mask, predictions, result)


def test_balanced_source_matching_reassigns_flexible_games():
    assigned = [[] for _ in range(6)]
    cells = {0: [0, 1], 1: [0], 2: [2], 3: [3], 4: [4], 5: [5]}
    for game in range(6):
        assert p.place_source(game, cells, assigned, 1)
    assert assigned == [[1], [0], [2], [3], [4], [5]]
    assert len({g for group in assigned for g in group}) == 6


def test_balanced_source_matching_reports_true_shortage():
    assigned = [[] for _ in range(6)]
    cells = {i: [0, 1, 2, 3, 4] for i in range(20)}
    for game in cells:
        p.place_source(game, cells, assigned, 2)
    assert [len(group) for group in assigned] == [2, 2, 2, 2, 2, 0]
    assert len({g for group in assigned for g in group}) == 10
