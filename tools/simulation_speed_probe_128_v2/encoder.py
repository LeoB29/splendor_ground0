"""One bounded prototype: per-state reuse and dense action input construction.

The reference rules, observations and model are unchanged. No inter-state cache
exists, so private deck/reserve identities cannot become cache keys or features.
All divisions occur in float64 before the same final float32 conversion as the
Python reference. Acceptance requires exact tensor equality, not tolerance.
"""
from __future__ import annotations

import numpy as np

from splendor_ai.engine.actions import ActionType
from splendor_ai.engine.constants import ALL_TOKEN_TYPES, TOKEN_COLORS
from splendor_ai.encoding import action_features as reference
from splendor_ai.encoding.observation import encode_public_observation_tensor

TOKENS = ALL_TOKEN_TYPES
COLORS = TOKEN_COLORS
TOKEN_INDEX = {color: i for i, color in enumerate(TOKENS)}
TYPE_INDEX = {kind: i for i, kind in enumerate(reference.ACTION_TYPE_ORDER)}
NORMALIZERS = np.asarray([4., 4., 4., 4., 4., 5.], dtype=np.float64)
MAX_ACTIONS = 4096


def _counts(values):
    out = [0] * 6
    for color in values:
        out[TOKEN_INDEX[color]] += 1
    return out


def action_array(state, actions, player_id=None):
    """Exactly reproduce 79 supplied-order features without per-action metadata."""
    actor = state.current_player if player_id is None else player_id
    player, opponent = state.players[actor], state.players[1 - actor]
    size = len(actions)
    if size > MAX_ACTIONS:
        raise ValueError("Probe action-memory bound exceeded")
    matrix = np.zeros((size, 79), dtype=np.float64)
    if not size:
        return matrix.astype(np.float32)
    own_tokens = np.asarray([int(player.tokens.get(c, 0)) for c in TOKENS], dtype=np.int64)
    opp_tokens = np.asarray([int(opponent.tokens.get(c, 0)) for c in TOKENS], dtype=np.int64)
    own_bonuses = np.asarray([player.bonuses.get(c, 0) for c in COLORS], dtype=np.int64)
    opp_bonuses = np.asarray([opponent.bonuses.get(c, 0) for c in COLORS], dtype=np.int64)
    visible = [card for cards in state.visible_tier_cards.values() for card in cards]
    costs = np.asarray([[card.cost.get(c, 0) for c in COLORS] for card in visible],
                       dtype=np.int64).reshape(-1, 5)
    own_required = np.maximum(costs - own_bonuses, 0)
    opp_required = np.maximum(costs - opp_bonuses, 0)
    own_before = int((np.maximum(own_required - own_tokens[:5], 0).sum(1) <= own_tokens[5]).sum())
    opp_before = int((np.maximum(opp_required - opp_tokens[:5], 0).sum(1) <= opp_tokens[5]).sum())
    takes = np.asarray([_counts(a.take_tokens) for a in actions], dtype=np.int64)
    returns = np.asarray([_counts(a.return_tokens) for a in actions], dtype=np.int64)
    spends = np.asarray([_counts(a.spend_tokens) for a in actions], dtype=np.int64)
    net = takes - returns - spends
    after = own_tokens + net
    matrix[:, 36:42] = takes / NORMALIZERS
    matrix[:, 42:48] = returns / NORMALIZERS
    matrix[:, 48:54] = spends / NORMALIZERS
    matrix[:, 54:60] = net / NORMALIZERS
    # Deduplicate token inventories, as the reference context does, then compute
    # affordability for all visible cards with integer arithmetic in one array.
    unique_after, inverse = np.unique(after, axis=0, return_inverse=True)
    affordable = (np.maximum(own_required[None, :, :] - unique_after[:, None, :5], 0)
                  .sum(2) <= unique_after[:, None, 5]).sum(1)[inverse]
    totals = after.sum(1)
    matrix[:, 64] = float(player.score - opponent.score) / 20.
    matrix[:, 67] = float(opponent.score >= 12)
    matrix[:, 74] = float(len(player.reserved_cards)) / 3.
    matrix[:, 75] = float(len(opponent.reserved_cards)) / 3.
    matrix[:, 76] = totals / 10.
    matrix[:, 77] = np.maximum(totals - 10, 0) / 5.
    matrix[:, 78] = (affordable - own_before - opp_before) / 12.
    # Key only by cards visible to the actor or in that actor's own reserve.
    card_cache, noble_cache = {}, {}
    nobles = {n.noble_id: n for n in state.nobles}
    for row, action in enumerate(actions):
        matrix[row, TYPE_INDEX[action.action_type]] = 1.
        if action.tier in (1, 2, 3):
            matrix[row, 6 + action.tier - 1] = 1.
        if action.market_index in (0, 1, 2, 3):
            matrix[row, 9 + action.market_index] = 1.
        if action.reserved_index in (0, 1, 2):
            matrix[row, 13 + action.reserved_index] = 1.
        card = reference._target_card_for_action(state, action, actor)
        noble = nobles.get(action.noble_id)
        missing_self = missing_opp = 10
        gained = discounted = 0
        buying = action.action_type in (ActionType.BUY_VISIBLE, ActionType.BUY_RESERVED)
        if card is not None:
            key = id(card)
            if key not in card_cache:
                card_costs = np.asarray([card.cost.get(c, 0) for c in COLORS], dtype=np.int64)
                required = np.maximum(card_costs - own_bonuses, 0)
                other = np.maximum(card_costs - opp_bonuses, 0)
                card_cache[key] = (
                    reference._encode_card(card), int(required.sum()),
                    max(int(np.maximum(required - own_tokens[:5], 0).sum()) - int(own_tokens[5]), 0),
                    max(int(np.maximum(other - opp_tokens[:5], 0).sum()) - int(opp_tokens[5]), 0))
            encoded, required_total, missing_self, missing_opp = card_cache[key]
            matrix[row, 16:29] = encoded
            matrix[row, 68] = float(card.points) / 5.
            matrix[row, 71] = float(missing_opp == 0)
            matrix[row, 72] = float(missing_opp <= 2)
            matrix[row, 73] = float(action.action_type == ActionType.RESERVE_VISIBLE and missing_opp <= 2)
            if buying:
                discounted, gained = required_total, card.points
        if noble is not None:
            if noble.noble_id not in noble_cache:
                noble_cache[noble.noble_id] = reference._encode_noble(noble)
            matrix[row, 29:36] = noble_cache[noble.noble_id]
            gained += noble.points
        matrix[row, 60] = float(discounted) / 7.
        matrix[row, 61] = matrix[row, 65] = float(gained) / 8.
        if action.action_type == ActionType.RESERVE_DECK and action.tier is not None:
            matrix[row, 62] = float(state.deck_counts.get(action.tier, 0)) / reference._DECK_NORMALIZERS[action.tier]
        matrix[row, 63] = float(len(action.take_tokens) - len(action.return_tokens) - len(action.spend_tokens)) / 10.
        matrix[row, 66] = float(player.score + gained >= 15)
        matrix[row, 69] = float(missing_self) / 10.
        matrix[row, 70] = float(missing_opp) / 10.
    return matrix.astype(np.float32)


def arrays(state, actions):
    """Observation is deliberately unchanged; optimize the action-heavy path."""
    observation = np.asarray(encode_public_observation_tensor(state, state.current_player).vector,
                             dtype=np.float32)
    return observation, action_array(state, actions)


def tensors(state, actions):
    import torch
    observation, features = arrays(state, actions)
    return (torch.from_numpy(observation).unsqueeze(0), torch.from_numpy(features).unsqueeze(0),
            torch.ones((1, len(actions)), dtype=torch.bool))


def reference_tensors(state, actions):
    import torch
    from splendor_ai.encoding.feature_utils import fit_feature_vector
    observation = encode_public_observation_tensor(state, state.current_player)
    encoded = reference.encode_legal_action_features(state, actions, state.current_player)
    return (torch.tensor(fit_feature_vector(observation.vector, 295), dtype=torch.float32).unsqueeze(0),
            torch.tensor([fit_feature_vector(e.vector, 79) for e in encoded], dtype=torch.float32).reshape(1, len(actions), 79),
            torch.ones((1, len(actions)), dtype=torch.bool))


class Fast035:
    """Probe-only batch-one input replacement; same model and first-maximum tie."""
    def __init__(self, original, boundary=lambda: None, on_position=None):
        if original._device != 'cpu' or original._model.training:
            raise ValueError('Unchanged CPU evaluation model required')
        config = original._model.config
        if config.observation_size != 295 or config.action_feature_size != 79:
            raise ValueError('Unexpected035 encoder layout')
        self.model, self.boundary, self.on_position = original._model, boundary, on_position

    def choose_action(self, env, state, legal_actions=None):
        import torch
        from splendor_ai.training.model import masked_legal_action_scores
        self.boundary()
        actions = legal_actions if legal_actions is not None else env.legal_actions(state)
        if not actions:
            return None
        inputs = tensors(state, actions)
        if self.on_position is not None:
            self.on_position(state, actions, inputs)
        with torch.no_grad():
            scores, _ = self.model(*inputs)
            best = int(torch.argmax(masked_legal_action_scores(scores, inputs[2]), dim=1).item())
        return actions[best]
