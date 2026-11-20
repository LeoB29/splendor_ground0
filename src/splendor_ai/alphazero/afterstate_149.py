"""Isolated deterministic/chance decomposition; frozen engine remains unchanged.

The private partial state is an implementation detail, never a policy input.
Only resolve() returns an ordinary, decision-ready SplendorState. Feature
projection explicitly accounts for a pending private reservation.
"""
from __future__ import annotations

from dataclasses import dataclass
import torch

from splendor_ai.engine.actions import Action, ActionType
from splendor_ai.engine.constants import ALL_TOKEN_TYPES, TOKEN_COLORS
from splendor_ai.engine.data import build_base_deck_by_tier, build_base_nobles
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.engine.state import RESERVE_SOURCE_DECK, RESERVE_SOURCE_VISIBLE, SplendorState
from splendor_ai.encoding import encode_public_observation_tensor

CARDS = tuple(sorted((c for deck in build_base_deck_by_tier().values() for c in deck), key=lambda c: c.card_id))
NOBLES = tuple(sorted(build_base_nobles(), key=lambda n: n.noble_id))


@dataclass(frozen=True)
class Afterstate:
    _partial: SplendorState
    actor: int
    chance_kind: str | None
    chance_tier: int | None

    def resolve(self, card_id: str | None = None) -> SplendorState:
        state = self._partial.copy_shallow()
        if self.chance_kind is None:
            if card_id is not None:
                raise ValueError("No pending chance event")
            return state
        if card_id is None:
            raise ValueError("Pending event requires an explicit card")
        deck = state.hidden_tier_decks[self.chance_tier]
        card = next((c for c in deck if c.card_id == card_id), None)
        if card is None:
            raise ValueError("Card outside the specified world's draw support")
        deck.remove(card)
        if self.chance_kind == "public":
            state.visible_tier_cards[self.chance_tier].append(card)
        else:
            state.players[self.actor].reserve_card(card, RESERVE_SOURCE_DECK)
        state.deck_counts[self.chance_tier] = len(deck)
        return state


class _DeferredEnv(SplendorEnv):
    def __init__(self):
        super().__init__()
        self.kind = self.tier = None

    def _draw_replacement_card(self, state, tier):
        if state.hidden_tier_decks[tier]:
            self.kind, self.tier = "public", tier
            state.deck_counts[tier] -= 1

    def _apply_reserve_deck_action(self, state, player, action):
        self.kind, self.tier = "private", action.tier
        state.deck_counts[action.tier] -= 1
        self._apply_take_action(state, player, action)


def deterministic_part(state: SplendorState, action: Action) -> Afterstate:
    if not isinstance(state, SplendorState):
        raise TypeError("Resolve pending chance before another decision")
    env = _DeferredEnv()
    partial = env.step(state, action)
    return Afterstate(partial, state.current_player, env.kind, env.tier)


def public_card_status(state, observer):
    """Six locations per catalog card, using only legally observed identities.

    Unknown is the complement of known identities, never an inspection of deck
    contents. Opponent private reservations remain in the unknown pool.
    """
    known = {}
    for cards in state.visible_tier_cards.values():
        known.update((c.card_id, 1) for c in cards)
    for seat, player in enumerate(state.players):
        own = seat == observer
        for i, card in enumerate(player.reserved_cards):
            if own or player.reserved_card_source(i) == RESERVE_SOURCE_VISIBLE:
                known[card.card_id] = 2 if own else 3
        known.update((c.card_id, 4 if own else 5) for c in player.purchased_cards)
    return [float(known.get(card.card_id, 0) == location) for card in CARDS for location in range(6)]


def action_vector(state, action):
    if action.metadata:
        raise ValueError("Unspecified metadata cannot enter the fixed feature schema")
    target = None
    if action.action_type in (ActionType.RESERVE_VISIBLE, ActionType.BUY_VISIBLE):
        target = state.visible_tier_cards[action.tier][action.market_index]
    elif action.action_type == ActionType.BUY_RESERVED:
        target = state.players[state.current_player].reserved_cards[action.reserved_index]
    return (
        [float(action.action_type == t) for t in ActionType]
        + [float(action.tier == i) for i in (None, 1, 2, 3)]
        + [float(action.market_index == i) for i in (None, 0, 1, 2, 3)]
        + [float(action.reserved_index == i) for i in (None, 0, 1, 2)]
        + [getattr(action, field).count(c) / 10 for field in ("take_tokens", "return_tokens", "spend_tokens") for c in ALL_TOKEN_TYPES]
        + [float(action.noble_id == n) for n in (None, *(n.noble_id for n in NOBLES))]
        + [float(target is not None and target.card_id == c.card_id) for c in CARDS]
        + ([target.points / 5, *[float(target.bonus_color == c) for c in TOKEN_COLORS],
            *[target.cost.get(c, 0) / 7 for c in TOKEN_COLORS]] if target else [0.] * 11)
    )


def features(state, action, arm):
    if arm not in ("control", "afterstate"):
        raise ValueError(arm)
    after = deterministic_part(state, action)
    board = state if arm == "control" else after._partial
    actor = state.current_player
    observation = encode_public_observation_tensor(board, actor)
    vector = list(observation.vector)
    if arm == "afterstate" and after.chance_kind == "private":
        # 6 token + 5 bonus + score precede reserved count in each summary.
        vector[observation.sections["self_summary"][0] + 12] += 1 / 3
    status = public_card_status(board, actor)
    # Same pre-action legally reconstructed refill pool in both arms.
    pool = public_card_status(state, actor)[::6]
    extras = [float(board.start_player == actor), float(board.terminal),
              float(board.terminal and board.winner == actor),
              float(board.terminal and board.winner == 1 - actor)]
    # Event descriptors are knowable from the prestate/action and shared.
    event = [float(after.chance_kind == kind) for kind in (None, "public", "private")]
    event += [float(after.chance_tier == tier) for tier in (None, 1, 2, 3)]
    return torch.tensor(vector + status + pool + extras + event + action_vector(state, action), dtype=torch.float32)


class AdvantageModel(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(width, 256), torch.nn.LayerNorm(256), torch.nn.GELU(),
            torch.nn.Linear(256, 128), torch.nn.LayerNorm(128), torch.nn.GELU(),
            torch.nn.Linear(128, 64), torch.nn.GELU(), torch.nn.Linear(64, 1), torch.nn.Tanh())

    def forward(self, x):
        scores = self.network(x).squeeze(-1)
        return scores - scores[..., :1]
