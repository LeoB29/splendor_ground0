"""Canonical state objects for base Splendor."""

from __future__ import annotations

from dataclasses import dataclass, field

from .constants import ALL_TOKEN_TYPES, TOKEN_COLORS


def empty_token_bank() -> dict[str, int]:
    return {color: 0 for color in ALL_TOKEN_TYPES}


def empty_bonus_bank() -> dict[str, int]:
    return {color: 0 for color in TOKEN_COLORS}


@dataclass(frozen=True, slots=True)
class Card:
    card_id: str
    tier: int
    bonus_color: str
    points: int
    cost: dict[str, int]


@dataclass(frozen=True, slots=True)
class Noble:
    noble_id: str
    points: int
    requirement: dict[str, int]


@dataclass(slots=True)
class PlayerState:
    player_id: int
    tokens: dict[str, int] = field(default_factory=empty_token_bank)
    bonuses: dict[str, int] = field(default_factory=empty_bonus_bank)
    score: int = 0
    reserved_cards: list[Card] = field(default_factory=list)
    purchased_cards: list[Card] = field(default_factory=list)
    nobles: list[Noble] = field(default_factory=list)

    @property
    def token_count(self) -> int:
        return sum(self.tokens.values())


@dataclass(slots=True)
class SplendorState:
    current_player: int
    bank_tokens: dict[str, int]
    players: list[PlayerState]
    visible_tier_cards: dict[int, list[Card]]
    hidden_tier_decks: dict[int, list[Card]]
    deck_counts: dict[int, int]
    nobles: list[Noble]
    turn_index: int = 0
    start_player: int = 0
    pending_round_end: bool = False
    terminal: bool = False
    winner: int | None = None

    def copy_shallow(self) -> "SplendorState":
        """Return a shallow structural copy suitable for early tests.

        Full optimized cloning will be implemented later once transitions exist.
        """

        return SplendorState(
            current_player=self.current_player,
            bank_tokens=dict(self.bank_tokens),
            players=[
                PlayerState(
                    player_id=player.player_id,
                    tokens=dict(player.tokens),
                    bonuses=dict(player.bonuses),
                    score=player.score,
                    reserved_cards=list(player.reserved_cards),
                    purchased_cards=list(player.purchased_cards),
                    nobles=list(player.nobles),
                )
                for player in self.players
            ],
            visible_tier_cards={
                tier: list(cards) for tier, cards in self.visible_tier_cards.items()
            },
            hidden_tier_decks={
                tier: list(cards) for tier, cards in self.hidden_tier_decks.items()
            },
            deck_counts=dict(self.deck_counts),
            nobles=list(self.nobles),
            turn_index=self.turn_index,
            start_player=self.start_player,
            pending_round_end=self.pending_round_end,
            terminal=self.terminal,
            winner=self.winner,
        )
