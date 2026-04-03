"""Action structures for the Splendor engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto


class ActionType(Enum):
    PASS = auto()
    TAKE_TOKENS = auto()
    RESERVE_VISIBLE = auto()
    RESERVE_DECK = auto()
    BUY_VISIBLE = auto()
    BUY_RESERVED = auto()


@dataclass(frozen=True, slots=True)
class Action:
    """Structured action object used by the engine.

    The fixed model-side action index will be defined by a codec layer later.
    Noble choice, when needed after a buy, is represented by `noble_id`.
    """

    action_type: ActionType
    tier: int | None = None
    market_index: int | None = None
    reserved_index: int | None = None
    take_tokens: tuple[str, ...] = ()
    return_tokens: tuple[str, ...] = ()
    spend_tokens: tuple[str, ...] = ()
    noble_id: str | None = None
    metadata: tuple[tuple[str, str], ...] = field(default_factory=tuple)
