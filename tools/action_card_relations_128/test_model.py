"""Observer legality, actual-transition arithmetic and frozen residual integrity."""
import copy
import hashlib

import pytest
import torch

from splendor_ai.alphazero.strong_policy import encode_position, save_model as save_anchor
from splendor_ai.encoding import encode_action_features, encode_public_observation_tensor
from splendor_ai.engine.actions import Action, ActionType
from splendor_ai.engine.constants import ALL_TOKEN_TYPES, TOKEN_COLORS
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.engine.state import RESERVE_SOURCE_DECK, RESERVE_SOURCE_VISIBLE
from splendor_ai.training.model import LegalActionAttentionScorerConfig, LegalActionAttentionScorerModel

from . import model as h


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def small_base():
    torch.manual_seed(671)
    config = LegalActionAttentionScorerConfig(state_hidden_size=8, action_hidden_size=8,
        scorer_hidden_size=8, state_depth=1, action_depth=1, attention_layers=1,
        attention_heads=2, attention_feedforward_size=16, dropout=.2)
    return LegalActionAttentionScorerModel(config).eval()


def representative_state():
    state = SplendorEnv(seed=414).initial_state()
    actor = state.players[0]
    for color, number in zip(ALL_TOKEN_TYPES, (2, 2, 2, 1, 1, 2), strict=True):
        actor.tokens[color] = number
        state.bank_tokens[color] -= number
    actor.bonuses.update({color: 1 for color in TOKEN_COLORS})
    actor.reserve_card(state.hidden_tier_decks[1].pop(0), RESERVE_SOURCE_DECK)
    # Both public and private opponent reserves must be handled differently.
    opponent = state.players[1]
    opponent.reserve_card(state.hidden_tier_decks[2].pop(0), RESERVE_SOURCE_VISIBLE)
    opponent.reserve_card(state.hidden_tier_decks[2].pop(0), RESERVE_SOURCE_DECK)
    for tier in (1, 2, 3):
        state.deck_counts[tier] = len(state.hidden_tier_decks[tier])
    return state


def tensors(state, actions=None):
    actions = SplendorEnv().legal_actions(state) if actions is None else actions
    observation = torch.tensor(encode_public_observation_tensor(state, state.current_player).vector)[None]
    features = torch.tensor([encode_action_features(state, action).vector for action in actions])[None]
    return observation, features, torch.ones((1, len(actions)), dtype=torch.bool)


@pytest.mark.parametrize("arm", h.ARMS)
def test_scalar_audit_matches_all_legal_actions_and_engine_consequences(arm):
    state = representative_state()
    actions = SplendorEnv().legal_actions(state)
    assert {a.action_type for a in actions} == {ActionType.TAKE_TOKENS, ActionType.RESERVE_VISIBLE,
        ActionType.RESERVE_DECK, ActionType.BUY_VISIBLE, ActionType.BUY_RESERVED}
    assert any(a.return_tokens for a in actions)
    assert any("gold" in a.spend_tokens for a in actions)
    obs, features, _ = tensors(state, actions)
    actual, known = h.pair_features(obs, features, arm)
    independent, independent_known = h.independent_pair_features(obs, features, arm)
    assert torch.equal(known, independent_known)
    torch.testing.assert_close(actual, independent, rtol=0, atol=1e-7)
    assert actual.shape == (1, len(actions), 18, 55)
    assert known[0].tolist() == [True]*13 + [False]*2 + [True, False, False]
    components = h.pair_components(obs, features)
    known_cards = [card for tier in (1, 2, 3) for card in state.visible_tier_cards[tier]]
    known_cards += state.players[0].reserved_cards + [None, None]
    known_cards += [state.players[1].reserved_cards[0], None, None]
    for index, action in enumerate(actions):
        after = SplendorEnv().step(state, action)
        actor = after.players[0]
        assert components["tokens_after"][0, index].tolist() == [actor.tokens[c] for c in ALL_TOKEN_TYPES]
        assert components["bonuses_after"][0, index].tolist() == [actor.bonuses[c] for c in TOKEN_COLORS]
        market_ids = {c.card_id for cards in after.visible_tier_cards.values() for c in cards}
        own_ids = {c.card_id for c in actor.reserved_cards}
        opponent_ids = {c.card_id for i, c in enumerate(after.players[1].reserved_cards)
                        if after.players[1].reserved_card_source(i) == RESERVE_SOURCE_VISIBLE}
        for slot, card in enumerate(known_cards):
            if card is None:
                assert torch.equal(actual[0, index, slot], torch.zeros(55))
                continue
            expected_owner = [int(card.card_id in ids) for ids in (market_ids, own_ids, opponent_ids)]
            assert components["owner_after"][0, index, slot].tolist() == expected_owner
            if arm == "relations":
                deficit = [max(0, card.cost.get(c, 0)-actor.bonuses[c]-actor.tokens[c]) for c in TOKEN_COLORS]
                torch.testing.assert_close(actual[0, index, slot, 48:53], torch.tensor(deficit)/7, rtol=0, atol=1e-7)
                assert actual[0, index, slot, 54].item() == pytest.approx(max(0, sum(deficit)-actor.tokens["gold"])/10)


def test_shared_gold_is_spent_once_and_rounding_restores_encoded_integers():
    obs = torch.zeros(1, 295)
    obs[0, 33] = 1.
    obs[0, 41:46] = torch.tensor([2, 2, 0, 0, 0])/7
    obs[0, 194] = 3/10
    actions = torch.zeros(1, 1, 79)
    actions[0, 0, 0] = 1.
    pair, _ = h.pair_features(obs, actions, "relations")
    torch.testing.assert_close(pair[0, 0, 0, 43:48], torch.tensor([2, 2, 0, 0, 0])/7)
    assert pair[0, 0, 0, 53].item() == pytest.approx(.1)
    # Subtracting the gold pool independently from each color would give zero.
    assert pair[0, 0, 0, 53] > 0


@pytest.mark.parametrize("arm", h.ARMS)
def test_scalar_float32_division_agrees_at_high_legal_resource_capacities(arm):
    observations, feature_rows = [], []
    for bonus in range(16):
        for white in range(5):
            for gold in range(6):
                state = SplendorEnv(seed=717).initial_state()
                state.players[0].bonuses["white"] = bonus
                state.players[0].tokens["white"] = white
                state.players[0].tokens["gold"] = gold
                state.bank_tokens["white"], state.bank_tokens["gold"] = 4-white, 5-gold
                # Tokens total at most9. A legal blind reservation preserves
                # every known-card identity and conditionally gains real gold.
                action = next(a for a in SplendorEnv().legal_actions(state)
                              if a.action_type == ActionType.RESERVE_DECK)
                observations.append(encode_public_observation_tensor(state, 0).vector)
                feature_rows.append([encode_action_features(state, action).vector])
    obs, actions = torch.tensor(observations), torch.tensor(feature_rows)
    actual, mask = h.pair_features(obs, actions, arm)
    independent, other = h.independent_pair_features(obs, actions, arm)
    assert len(observations) == 480 and torch.equal(mask, other)
    # Covers colored capacities0..19 and bonus0..15: a1e-7 audit tolerance
    # must not reject these larger float32 values because of a1ULP difference.
    assert torch.equal(actual, independent)


@pytest.mark.parametrize("arm", h.ARMS)
def test_hidden_deck_and_private_opponent_identity_cannot_change_branch(arm):
    state = representative_state()
    changed = copy.deepcopy(state)
    for tier in (1, 2, 3):
        changed.hidden_tier_decks[tier].reverse()
    old_private = changed.players[1].reserved_cards[1]
    changed.players[1].reserved_cards[1] = changed.hidden_tier_decks[2][0]
    changed.hidden_tier_decks[2][0] = old_private
    first, second = tensors(state), tensors(changed)
    assert all(torch.equal(a, b) for a, b in zip(first, second, strict=True))
    p0, mask0 = h.pair_features(*first[:2], arm)
    p1, mask1 = h.pair_features(*second[:2], arm)
    assert torch.equal(p0, p1) and torch.equal(mask0, mask1)
    model = h.RelationModel(small_base(), arm, 954)
    with torch.no_grad():
        model.residual_head[-1].weight.fill_(.1)
    assert torch.equal(model.residual(*first), model.residual(*second))


@pytest.mark.parametrize("arm", h.ARMS)
def test_initial_equivalence_frozen_base_eval_and_gradient_integrity(arm):
    args = tensors(representative_state())
    base = small_base()
    with torch.no_grad():
        initial = base(*args)
    before = {name: value.clone() for name, value in base.state_dict().items()}
    model = h.RelationModel(base, arm, 676)
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 9441
    assert all(not p.requires_grad for p in model.base.parameters())
    model.train()
    assert not model.base.training and all(not m.training for m in model.base.modules())
    actual = model(*args)
    assert all(torch.equal(a, b) for a, b in zip(initial, actual, strict=True))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.01)
    for _ in range(2):
        optimizer.zero_grad()
        logits, value = model(*args)
        loss = -torch.log_softmax(logits, -1)[0, 0]
        loss.backward()
        assert all(p.grad is None for p in model.base.parameters())
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        optimizer.step()
    assert all(torch.equal(before[name], value) for name, value in model.base.state_dict().items())
    assert not torch.equal(initial[0], model(*args)[0])
    assert torch.equal(initial[1], model(*args)[1])
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.pair_encoder.parameters())


def test_matched_arms_have_identical_parameters_rng_and_raw_information():
    base = small_base()
    rng = torch.random.get_rng_state()
    raw = h.RelationModel(copy.deepcopy(base), "raw", 171)
    relation = h.RelationModel(copy.deepcopy(base), "relations", 171)
    assert torch.equal(rng, torch.random.get_rng_state())
    assert raw.state_dict().keys() == relation.state_dict().keys()
    assert all(torch.equal(raw.state_dict()[key], relation.state_dict()[key]) for key in raw.state_dict())
    obs, actions, _ = tensors(representative_state())
    a, mask = h.pair_features(obs, actions, "raw")
    b, other_mask = h.pair_features(obs, actions, "relations")
    assert torch.equal(a[..., :43], b[..., :43]) and torch.equal(mask, other_mask)
    assert not torch.equal(a[..., 43:], b[..., 43:])


@pytest.mark.parametrize("arm", h.ARMS)
def test_action_permutation_padding_known_card_permutation_and_empty_masks(arm):
    state = SplendorEnv(seed=91).initial_state()
    actions = [a for a in SplendorEnv().legal_actions(state) if a.action_type == ActionType.TAKE_TOKENS][:5]
    obs, features, mask = tensors(state, actions)
    model = h.RelationModel(small_base(), arm, 513).eval()
    with torch.no_grad():
        model.residual_head[-1].weight.normal_()
        original = model(*[obs, features, mask])[0]
        order = torch.tensor([3, 0, 4, 1, 2])
        permuted = model(obs, features[:, order], mask[:, order])[0]
        torch.testing.assert_close(permuted, original[:, order], rtol=1e-5, atol=1e-6)
        padded_features = torch.cat((features, torch.zeros(1, 3, 79)), dim=1)
        padded_mask = torch.cat((mask, torch.zeros(1, 3, dtype=torch.bool)), dim=1)
        padded = model(obs, padded_features, padded_mask)[0]
        torch.testing.assert_close(padded[:, :5], original, rtol=1e-5, atol=1e-6)
        assert torch.equal(model.residual(obs, padded_features, padded_mask)[:, 5:], torch.zeros(1, 3))
        swapped = obs.clone()
        swapped[:, 33:46], swapped[:, 46:59] = obs[:, 46:59], obs[:, 33:46]
        torch.testing.assert_close(model.residual(swapped, features, mask), model.residual(obs, features, mask), rtol=1e-5, atol=1e-6)
        empty = obs.clone()
        empty[:, 33:189] = 0
        empty[:, 203:242] = 0
        empty[:, 256:295] = 0
        assert torch.isfinite(model.residual(empty, features, mask)).all()


@pytest.mark.parametrize("arm", h.ARMS)
def test_checkpoint_roundtrip_immutable_and_inference_api(tmp_path, arm):
    anchor = tmp_path/"anchor.pt"
    save_anchor(anchor, small_base(), 0, {"fixture": True})
    model = h.create_model(anchor, arm, 3)
    with torch.no_grad():
        model.residual_head[-1].weight.fill_(.2)
    path = tmp_path/"residual.pt"
    h.save_model(model, path, {"update": 2, "fixture": True})
    expected_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    inference = h.RelationInference.from_path(path, expected_sha)
    state = representative_state()
    args = tensors(state)
    assert all(torch.equal(a, b) for a, b in zip(model(*args), inference.model(*args), strict=True))
    logits, value, row, keys, actions = inference.full_prediction(state)
    assert logits.numel() == len(actions) and len(keys) == row["group_count"]
    prediction = inference.predict_one(state, actions[:3])
    assert len(prediction.logits) == 3
    assert prediction.value == value
    assert inference.checkpoint.generation == 2
    with pytest.raises(FileExistsError):
        h.save_model(model, path, {})
    with pytest.raises(ValueError, match="hash"):
        h.RelationInference.from_path(path, "wrong")
    payload = torch.load(path, weights_only=True)
    payload["residual_config"]["pair_hidden"] += 1
    changed = tmp_path/"changed.pt"
    torch.save(payload, changed)
    with pytest.raises(ValueError, match="changed"):
        h.RelationInference(changed)


@pytest.mark.parametrize("arm", h.ARMS)
def test_pass_and_own_private_reserve_visibility(arm):
    state = representative_state()
    observation, actions, _ = tensors(state, [Action(ActionType.PASS)])
    actual, mask = h.pair_features(observation, actions, arm)
    audit, audit_mask = h.independent_pair_features(observation, actions, arm)
    torch.testing.assert_close(actual, audit, rtol=0, atol=1e-7)
    assert mask[0, 12] and not mask[0, 16] and torch.equal(mask, audit_mask)
    assert torch.equal(actual[..., 21:27], actual[..., 27:33])
    assert torch.equal(actual[..., 33:38], actual[..., 38:43])


def test_rejects_wrong_flat_layout_or_arm():
    obs, actions, _ = tensors(representative_state())
    with pytest.raises(ValueError):
        h.pair_features(obs[:, :294], actions, "raw")
    with pytest.raises(ValueError):
        h.pair_features(obs, actions[..., :78], "raw")
    with pytest.raises(ValueError):
        h.pair_features(obs, actions, "other")
