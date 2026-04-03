"""Base-game component data for Splendor.

Rules are anchored to the official rulebook PDF supplied by the user.
The exact 90-card / 10-noble component list is decoded from the user's
complete local legacy dataset at `c:\\Users\\leoba\\Desktop\\splendor_ai\\utils.py`,
because the rulebook itself does not enumerate every component face in text.
"""

from __future__ import annotations

from functools import lru_cache

from .state import Card, Noble

_LEGACY_GEM_INDEX_TO_COLOR = {
    0: "white",
    1: "blue",
    2: "black",
    3: "red",
    4: "green",
}

# value order: tier(0-based), points, bonus color index (DSORE), then costs DSORE
_RAW_CARD_ROWS: tuple[tuple[int, int, int, int, int, int, int, int], ...] = (
    (0, 0, 0, 0, 2, 1, 0, 2),
    (0, 0, 0, 0, 0, 1, 2, 0),
    (0, 0, 0, 0, 1, 1, 1, 1),
    (0, 0, 0, 0, 3, 0, 0, 0),
    (0, 0, 0, 0, 2, 0, 0, 2),
    (0, 0, 0, 0, 1, 1, 1, 2),
    (0, 0, 0, 0, 1, 1, 0, 0),
    (0, 1, 0, 0, 0, 0, 0, 4),
    (1, 1, 0, 0, 0, 2, 2, 3),
    (1, 1, 0, 2, 3, 0, 3, 4),
    (1, 2, 0, 0, 0, 2, 4, 1),
    (1, 2, 0, 0, 0, 0, 5, 4),
    (1, 2, 0, 0, 0, 3, 5, 4),
    (1, 3, 0, 6, 0, 0, 0, 4),
    (2, 3, 0, 0, 3, 3, 5, 3),
    (2, 4, 0, 0, 0, 7, 0, 0),
    (2, 4, 0, 3, 0, 6, 3, 0),
    (2, 5, 0, 3, 0, 7, 0, 0),
    (0, 0, 1, 1, 0, 2, 0, 0),
    (0, 0, 1, 1, 0, 1, 2, 1),
    (0, 0, 1, 1, 0, 1, 1, 1),
    (0, 0, 1, 0, 1, 0, 1, 3),
    (0, 0, 1, 0, 0, 3, 0, 0),
    (0, 0, 1, 1, 0, 0, 2, 2),
    (0, 0, 1, 0, 0, 2, 0, 2),
    (0, 1, 1, 0, 0, 0, 4, 0),
    (1, 1, 1, 0, 2, 0, 3, 2),
    (1, 1, 1, 0, 2, 3, 0, 3),
    (1, 2, 1, 5, 3, 0, 0, 0),
    (1, 2, 1, 0, 5, 0, 0, 0),
    (1, 2, 1, 2, 0, 4, 1, 0),
    (1, 3, 1, 0, 6, 0, 0, 0),
    (2, 3, 1, 3, 0, 5, 3, 3),
    (2, 4, 1, 7, 0, 0, 0, 0),
    (2, 4, 1, 6, 3, 3, 0, 0),
    (2, 5, 1, 7, 3, 0, 0, 0),
    (0, 0, 2, 1, 1, 0, 1, 1),
    (0, 0, 2, 0, 0, 0, 1, 2),
    (0, 0, 2, 2, 0, 0, 0, 2),
    (0, 0, 2, 0, 0, 1, 3, 1),
    (0, 0, 2, 0, 0, 0, 0, 3),
    (0, 0, 2, 1, 2, 0, 1, 1),
    (0, 0, 2, 2, 2, 0, 1, 0),
    (0, 1, 2, 0, 4, 0, 0, 0),
    (1, 1, 2, 3, 2, 0, 0, 2),
    (1, 1, 2, 3, 0, 2, 0, 3),
    (1, 2, 2, 0, 1, 0, 2, 4),
    (1, 2, 2, 5, 0, 0, 0, 0),
    (1, 2, 2, 0, 0, 0, 3, 5),
    (1, 3, 2, 0, 0, 6, 0, 0),
    (2, 3, 2, 3, 3, 0, 3, 5),
    (2, 4, 2, 0, 0, 0, 7, 0),
    (2, 4, 2, 0, 0, 3, 6, 3),
    (2, 5, 2, 0, 0, 3, 7, 0),
    (0, 0, 3, 3, 0, 0, 0, 0),
    (0, 0, 3, 1, 0, 3, 1, 0),
    (0, 0, 3, 0, 2, 0, 0, 1),
    (0, 0, 3, 2, 0, 2, 0, 1),
    (0, 0, 3, 2, 1, 1, 0, 1),
    (0, 0, 3, 1, 1, 1, 0, 1),
    (0, 0, 3, 2, 0, 0, 2, 0),
    (0, 1, 3, 4, 0, 0, 0, 0),
    (1, 1, 3, 0, 3, 3, 2, 0),
    (1, 1, 3, 2, 0, 3, 2, 0),
    (1, 2, 3, 1, 4, 0, 0, 2),
    (1, 2, 3, 3, 0, 5, 0, 0),
    (1, 2, 3, 0, 0, 5, 0, 0),
    (1, 3, 3, 0, 0, 0, 6, 0),
    (2, 3, 3, 3, 5, 3, 0, 3),
    (2, 4, 3, 0, 0, 0, 0, 7),
    (2, 4, 3, 0, 3, 0, 3, 6),
    (2, 5, 3, 0, 0, 0, 3, 7),
    (0, 0, 4, 2, 1, 0, 0, 0),
    (0, 0, 4, 0, 2, 0, 2, 0),
    (0, 0, 4, 1, 3, 0, 0, 1),
    (0, 0, 4, 1, 1, 1, 1, 0),
    (0, 0, 4, 1, 1, 2, 1, 0),
    (0, 0, 4, 0, 1, 2, 2, 0),
    (0, 0, 4, 0, 0, 0, 3, 0),
    (0, 1, 4, 0, 0, 4, 0, 0),
    (1, 1, 4, 3, 0, 0, 3, 2),
    (1, 1, 4, 2, 3, 2, 0, 0),
    (1, 2, 4, 4, 2, 1, 0, 0),
    (1, 2, 4, 0, 0, 0, 0, 5),
    (1, 2, 4, 0, 5, 0, 0, 3),
    (1, 3, 4, 0, 0, 0, 0, 6),
    (2, 3, 4, 5, 3, 3, 3, 0),
    (2, 4, 4, 3, 6, 0, 0, 3),
    (2, 4, 4, 0, 7, 0, 0, 0),
    (2, 5, 4, 0, 7, 0, 0, 3),
)

_RAW_NOBLE_ROWS: tuple[tuple[int, int, int, int, int], ...] = (
    (0, 3, 0, 3, 3),
    (3, 3, 3, 0, 0),
    (4, 0, 4, 0, 0),
    (4, 4, 0, 0, 0),
    (0, 4, 0, 0, 4),
    (3, 3, 0, 0, 3),
    (3, 0, 3, 3, 0),
    (0, 0, 3, 3, 3),
    (0, 0, 4, 4, 0),
    (0, 0, 0, 4, 4),
)


def _decode_cost_tuple(cost_values: tuple[int, int, int, int, int]) -> dict[str, int]:
    return {
        color: amount
        for color, amount in (
            (_LEGACY_GEM_INDEX_TO_COLOR[index], value)
            for index, value in enumerate(cost_values)
        )
        if amount > 0
    }


@lru_cache(maxsize=1)
def build_base_deck_by_tier() -> dict[int, tuple[Card, ...]]:
    deck_by_tier: dict[int, list[Card]] = {1: [], 2: [], 3: []}

    for source_index, row in enumerate(_RAW_CARD_ROWS, start=1):
        tier_zero_based, points, bonus_index, *cost_values = row
        tier = tier_zero_based + 1
        card = Card(
            card_id=f"T{tier}_{source_index:02d}",
            tier=tier,
            bonus_color=_LEGACY_GEM_INDEX_TO_COLOR[bonus_index],
            points=points,
            cost=_decode_cost_tuple(tuple(cost_values)),
        )
        deck_by_tier[tier].append(card)

    return {tier: tuple(cards) for tier, cards in deck_by_tier.items()}


@lru_cache(maxsize=1)
def build_base_nobles() -> tuple[Noble, ...]:
    nobles: list[Noble] = []
    for source_index, requirement_values in enumerate(_RAW_NOBLE_ROWS, start=1):
        nobles.append(
            Noble(
                noble_id=f"N{source_index:02d}",
                points=3,
                requirement=_decode_cost_tuple(requirement_values),
            )
        )
    return tuple(nobles)
