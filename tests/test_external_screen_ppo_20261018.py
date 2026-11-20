"""Cheap observer/action/neural fixtures; no arena or simulated complete games."""
from copy import deepcopy

import numpy as np
import pytest
import torch

from tools.external_strength_screen_20261018 import ppo, rules


def initial(seat=0, seed=38016):
    state = rules.CorrectedEnv(seed=seed).initial_state()
    state.current_player = state.start_player = seat
    return state


def tokens(state, seat, counts):
    for color, count in counts.items():
        state.players[seat].tokens[color] += count
        state.bank_tokens[color] -= count


def remove_card(state, card_id):
    for tier in (1, 2, 3):
        for collection in (state.visible_tier_cards[tier], state.hidden_tier_decks[tier]):
            for index, card in enumerate(collection):
                if card.card_id == card_id:
                    result = collection.pop(index)
                    state.deck_counts[tier] = len(state.hidden_tier_decks[tier])
                    return result
    raise ValueError(card_id)


def buy_fixture_cards(state, seat, color, count):
    available = sorted((c for c in rules.corrected_cards_by_id().values() if c.bonus_color == color),
                       key=lambda c: (c.points, c.tier, c.card_id))
    for card in available[:count]:
        value = remove_card(state, card.card_id)
        state.players[seat].purchased_cards.append(value)
        state.players[seat].bonuses[color] += 1
        state.players[seat].score += value.points


def synthetic_states():
    result = [("initial_seat0", initial()), ("initial_seat1", initial(1))]
    limited = initial()
    tokens(limited, 0, {"black": 3, "red": 3, "white": 2, "blue": 1, "gold": 1})
    result.append(("full_tokens_return_pruning", limited))
    no_gold = initial()
    tokens(no_gold, 0, {"gold": 3})
    tokens(no_gold, 1, {"gold": 2})
    result.append(("reserve_without_bank_gold", no_gold))
    payments = initial()
    tokens(payments, 0, {"white": 1, "blue": 1, "green": 1, "red": 1, "black": 1, "gold": 3})
    result.append(("canonical_coloured_first_payment", payments))
    private = initial()
    for seat, tier in ((0, 1), (1, 2)):
        card = private.hidden_tier_decks[tier].pop()
        private.deck_counts[tier] -= 1
        private.players[seat].reserved_cards.append(card)
        private.players[seat].reserved_card_sources.append("deck")
    result.append(("own_and_opposing_blind_reserves", private))
    for count in (7, 8):
        owned = initial()
        buy_fixture_cards(owned, 0, "blue", count)
        result.append((f"own_blue_bonus_{count}", owned))
    opponent = initial()
    buy_fixture_cards(opponent, 1, "red", 8)
    result.append(("opponent_bonus_above_seven", opponent))
    nobles = initial()
    for color in ppo.NORMAL:
        needed = max(n.requirement.get(color, 0) for n in nobles.nobles[:2])
        buy_fixture_cards(nobles, 0, color, needed)
    result.append(("multiple_eligible_nobles_on_all_action_types", nobles))
    # Structural exhausted-market/forced-pass fixture, not a claimed reachable game.
    exhausted = initial(1)
    exhausted.start_player, exhausted.turn_index, exhausted.pending_round_end = 0, 1, True
    for tier in (1, 2, 3):
        cards = exhausted.visible_tier_cards[tier] + exhausted.hidden_tier_decks[tier]
        for card in cards:
            exhausted.players[0].purchased_cards.append(card)
            exhausted.players[0].bonuses[card.bonus_color] += 1
            exhausted.players[0].score += card.points
        exhausted.visible_tier_cards[tier], exhausted.hidden_tier_decks[tier], exhausted.deck_counts[tier] = [], [], 0
    tokens(exhausted, 0, {"white": 4, "blue": 4, "green": 2})
    tokens(exhausted, 1, {"green": 2, "red": 4, "black": 4})
    result.append(("structural_exhausted_market_forced_pass", exhausted))
    return result


def synthetic_payloads():
    return [(name, rules.observation_payload(state)) for name, state in synthetic_states()]


def seven_card_support_gap_state():
    """Complete legal component allocation with only an eighth-blue purchase.

    This is a constructed qualification decision, not a replayed game trajectory.
    Both players have ten coloured tokens, the bank has no coloured tokens, and
    the acting player has three reserves. Seven zero-point blue cards supply
    seven discounts; the available five-blue-cost tier-two blue card is free.
    Every other market/reserved card is unaffordable. No one has reached 15.
    """
    state = initial()
    catalogue = rules.corrected_cards_by_id()
    owned = sorted((c for c in catalogue.values() if c.bonus_color == "blue" and c.points == 0), key=lambda c: c.card_id)[:7]
    assert len(owned) == 7
    player = state.players[0]
    player.purchased_cards = owned
    player.bonuses["blue"] = 7
    tokens(state, 0, {"white": 4, "green": 4, "red": 2})
    tokens(state, 1, {"black": 4, "blue": 4, "red": 2})
    def affordable(card):
        return all(max(card.cost.get(c, 0) - player.bonuses[c], 0) <= player.tokens[c] for c in ppo.NORMAL)
    used = {c.card_id for c in owned}
    target = next(c for c in catalogue.values() if c.bonus_color == "blue" and c.tier == 2 and c.cost == {"blue": 5})
    used.add(target.card_id)
    reserved = [c for c in catalogue.values() if c.card_id not in used and not affordable(c)][:3]
    player.reserved_cards, player.reserved_card_sources = reserved, ["visible"] * 3
    used.update(c.card_id for c in reserved)
    for tier in (1, 2, 3):
        market = [target] if tier == 2 else []
        candidates = [c for c in catalogue.values() if c.tier == tier and c.card_id not in used and not affordable(c)]
        market += candidates[:4 - len(market)]
        assert len(market) == 4
        state.visible_tier_cards[tier] = market
        used.update(c.card_id for c in market)
    for tier in (1, 2, 3):
        state.hidden_tier_decks[tier] = [c for c in catalogue.values() if c.tier == tier and c.card_id not in used]
        state.deck_counts[tier] = len(state.hidden_tier_decks[tier])
    state.turn_index = 80
    rules.validate_state(state)
    return state


def return_support_gap_state():
    """Valid component allocation whose literal index intersection is empty.

    Reachability is not claimed: this fixture isolates representation support.
    Only RGB tokens remain available, and every official take must return three.
    Returning a taken colour has no literal pinned vocabulary entry.
    """
    state = initial()
    catalogue = rules.corrected_cards_by_id()
    tokens(state, 0, {"red": 3, "green": 3, "blue": 3, "gold": 1})
    tokens(state, 1, {"black": 4, "white": 4, "gold": 2})
    def unaffordable(card):
        return sum(max(card.cost.get(c, 0) - state.players[0].tokens[c], 0) for c in ppo.NORMAL) > 1
    used = set()
    for tier in (1, 2, 3):
        market = [c for c in catalogue.values() if c.tier == tier and unaffordable(c)][:4]
        assert len(market) == 4
        state.visible_tier_cards[tier] = market
        used.update(c.card_id for c in market)
    for seat, count in ((0, 3), (1, 2)):
        cards = [c for c in catalogue.values() if c.card_id not in used and unaffordable(c)][:count]
        state.players[seat].reserved_cards = cards
        state.players[seat].reserved_card_sources = ["visible"] * count
        used.update(c.card_id for c in cards)
    for tier in (1, 2, 3):
        state.hidden_tier_decks[tier] = [c for c in catalogue.values() if c.tier == tier and c.card_id not in used]
        state.deck_counts[tier] = len(state.hidden_tier_decks[tier])
    state.turn_index = 80
    rules.validate_state(state)
    return state


def qualification_payloads():
    """Finite targeted component/accounting fixtures, not played game histories."""
    states = synthetic_states()
    for seat in (0, 1):
        for reserved in (False, True):
            state = seven_card_support_gap_state()
            if reserved:
                state.visible_tier_cards[2][0], state.players[0].reserved_cards[0] = state.players[0].reserved_cards[0], state.visible_tier_cards[2][0]
                # Preserve the market's tier: use the removed tier-one reserved
                # slot as a tier-two reserve instead, reallocating by catalogue.
                if state.visible_tier_cards[2][0].tier != 2:
                    wrong = state.visible_tier_cards[2].pop(0)
                    state.hidden_tier_decks[wrong.tier].append(wrong)
                    state.deck_counts[wrong.tier] += 1
                    replacement = next(c for c in state.hidden_tier_decks[2] if c.cost.get("black", 0) > 0)
                    state.hidden_tier_decks[2].remove(replacement)
                    state.deck_counts[2] -= 1
                    state.visible_tier_cards[2].insert(0, replacement)
            if seat:
                state.players.reverse()
                for index, player in enumerate(state.players):
                    player.player_id = index
                state.current_player = state.start_player = 1
            rules.validate_state(state)
            states.append((f"eighth_blue_{'reserved' if reserved else 'visible'}_seat{seat}", state))
    states.append(("return_gap_literal_empty", return_support_gap_state()))
    for total in (8, 9):
        state = initial()
        tokens(state, 0, {"black": 3, "red": 3, "white": total - 6})
        states.append((f"token_alias_holdings_{total}", state))
    narrow = initial()
    tokens(narrow, 0, {"blue": 4, "black": 4, "red": 1, "gold": 1})
    tokens(narrow, 1, {"white": 4, "red": 2, "green": 3, "gold": 1})
    states.append(("token_alias_only_two_bank_colours", narrow))
    ending = deepcopy(dict(synthetic_states())["multiple_eligible_nobles_on_all_action_types"])
    available = [c for values in list(ending.visible_tier_cards.values()) + list(ending.hidden_tier_decks.values()) for c in values]
    for card in sorted((c for c in available if c.points == 4), key=lambda c: c.card_id)[:3]:
        removed = remove_card(ending, card.card_id)
        ending.players[0].purchased_cards.append(removed)
        ending.players[0].bonuses[removed.bonus_color] += 1
        ending.players[0].score += removed.points
    assert ending.players[0].score == 12
    tokens(ending, 0, {"black": 3, "red": 3, "white": 2, "blue": 1, "gold": 1})
    ending.players.reverse()
    for index, player in enumerate(ending.players):
        player.player_id = index
    ending.start_player, ending.current_player, ending.turn_index = 0, 1, 81
    rules.validate_state(ending)
    states.append(("token_alias_multiple_nobles_round_end_seat1", ending))
    return [(name, rules.observation_payload(state)) for name, state in states]


@pytest.fixture(scope="module")
def runtime():
    return ppo.PPOAdapter()


def test_safe_checkpoint_shapes_and_no_training(runtime):
    assert runtime.model.training is False
    assert all(not p.requires_grad and p.dtype == torch.float64 and p.device.type == "cpu" for p in runtime.model.parameters())
    assert runtime.model.input_norm.running_mean.shape == (265,)
    assert runtime.model.actor.weight.shape == (3510, 128)
    assert len(ppo.ACTION_KEYS) == len(set(ppo.ACTION_KEYS)) == 3510


def test_features_masks_logits_choices_match_independent_pinned_reference():
    receipt = ppo.qualify_payloads(synthetic_payloads())
    assert receipt["fixture_count"] == 11
    assert all(r["exact_features_mask_logits_probabilities_value_and_action"] for r in receipt["fixtures"])


def test_original_profile_remains_exact_on_original_fixtures():
    receipt = ppo.qualify_payloads(synthetic_payloads(), profile=ppo.ORIGINAL_PROFILE)
    assert all(row["original_support"] == row["support"] and not row["added_buy_indices"] for row in receipt["fixtures"])


def test_targeted_profiles_and_full_alias_successors():
    receipt = ppo.qualify_payloads(qualification_payloads())
    assert receipt["fixture_count"] == 20
    assert receipt["token_alias_successor_checks"] > 0
    ending = next(row for row in receipt["fixtures"] if row["fixture"].endswith("round_end_seat1"))
    assert ending["token_alias_successor_checks"] > 0


def test_hidden_state_changes_cannot_change_input_or_deterministic_decision(runtime):
    state = dict(synthetic_states())["own_and_opposing_blind_reserves"]
    changed = state.copy_shallow()
    for tier in (1, 2, 3):
        changed.hidden_tier_decks[tier].reverse()
    card = changed.players[1].reserved_cards[0]
    changed.players[1].reserved_cards[0], changed.hidden_tier_decks[card.tier][0] = changed.hidden_tier_decks[card.tier][0], card
    a, b = rules.observation_payload(state), rules.observation_payload(changed)
    assert a == b
    assert runtime.choose(a) == runtime.choose(b)


def test_coloured_first_payment_is_unambiguous_and_excludes_extra_gold():
    state = dict(synthetic_states())["canonical_coloured_first_payment"]
    mask, mapping, info = ppo.legal_mapping(state)
    assert info["excluded"].get("noncanonical_payment", 0) > 0
    assert int(mask.sum()) == len(mapping)
    for action in mapping.values():
        assert action in [ppo._serialize(a) for a in rules.legal_actions(state)]


def test_literal_seven_bonus_rule_does_not_clip_numeric_features():
    states = dict(synthetic_states())
    seven, eight = states["own_blue_bonus_7"], states["own_blue_bonus_8"]
    assert ppo.extract_features(eight)[9 + ppo.NORMAL.index("blue")] == 8
    assert ppo.legal_mapping(seven, ppo.ORIGINAL_PROFILE)[2]["excluded"].get("pinned_seven_card_pruning", 0) > 0


def test_empty_support_rejects_instead_of_synthesizing_pass(monkeypatch):
    state = initial()
    blind = [a for a in rules.legal_actions(state) if a.action_type.name == "RESERVE_DECK"]
    monkeypatch.setattr(rules.CorrectedEnv, "legal_actions", lambda *args: blind)
    with pytest.raises(ValueError, match="Empty translated"):
        ppo.legal_mapping(state, ppo.ORIGINAL_PROFILE)


def test_buy_profile_repairs_seven_card_witness_without_fallback(runtime):
    state = seven_card_support_gap_state()
    result = runtime.choose(rules.observation_payload(state))
    assert result["diagnostics"]["profile"] == ppo.ALIAS_PROFILE
    assert result["action"]["action_type"] == "BUY_VISIBLE"
    assert result["diagnostics"]["support"] == 1


def test_buy_profile_masks_unchanged_when_cap_inactive():
    for name, state in synthetic_states():
        if 7 not in state.players[state.current_player].bonuses.values():
            original = ppo.legal_mapping(state, ppo.ORIGINAL_PROFILE)
            adapted = ppo.legal_mapping(state, ppo.BUY_PROFILE)
            assert np.array_equal(original[0], adapted[0]), name
            assert original[1] == adapted[1], name


def test_buy_profile_return_gap_is_still_fail_closed():
    state = return_support_gap_state()
    actions = rules.legal_actions(state)
    assert actions and all(a.action_type.name == "TAKE_TOKENS" for a in actions)
    with pytest.raises(ValueError, match="Empty translated"):
        ppo.legal_mapping(state, ppo.BUY_PROFILE)


def test_token_alias_profile_repairs_return_witness_with_pinned_source_indices():
    state = return_support_gap_state()
    receipt = ppo.qualify_payloads([("return_support_gap", rules.observation_payload(state))])
    row = receipt["fixtures"][0]
    assert row["original_support"] == 0
    assert row["support"] == len(row["token_alias_indices"]) == 15
    assert row["token_alias_successor_checks"] >= 15


def test_valid_component_state_exposes_actual_seven_card_support_gap():
    state = seven_card_support_gap_state()
    actions = rules.legal_actions(state)
    assert actions and all(a.action_type.name == "BUY_VISIBLE" for a in actions)
    assert all(state.visible_tier_cards[a.tier][a.market_index].bonus_color == "blue" for a in actions)
    assert state.players[0].bonuses["blue"] == 7 and state.players[0].score == 0
    with pytest.raises(ValueError, match="Empty translated"):
        ppo.legal_mapping(state, ppo.ORIGINAL_PROFILE)


def test_first_argmax_is_first_masked_legal_index(runtime, monkeypatch):
    class EqualLegal:
        def __call__(self, features, mask):
            probabilities = (mask / mask.sum()).unsqueeze(0)
            return probabilities, torch.zeros((1, 1), dtype=torch.float64), torch.zeros((1, 3510), dtype=torch.float64)
    monkeypatch.setattr(runtime, "model", EqualLegal())
    payload = rules.observation_payload(initial())
    result = runtime.evaluate(payload)
    assert result["index"] == min(result["mapping"])


@pytest.mark.parametrize("field", ["environment_seed", "hidden_tier_decks"])
def test_malformed_or_hidden_input_is_rejected(runtime, field):
    payload = rules.observation_payload(initial())
    payload["observation"][field] = 1
    with pytest.raises(ValueError):
        runtime.choose(payload)


def test_terminal_input_rejected(runtime):
    state = initial()
    state.terminal = True
    with pytest.raises(ValueError, match="terminal"):
        runtime.choose(rules.observation_payload(state))


def test_checkpoint_bad_tensor_shape_is_rejected(monkeypatch):
    checkpoint = ppo.safe_checkpoint()
    checkpoint["model_state_dict"]["actor.bias"] = torch.zeros(1, dtype=torch.float64)
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: checkpoint)
    with pytest.raises(ValueError, match="actor.bias"):
        ppo.safe_checkpoint()
