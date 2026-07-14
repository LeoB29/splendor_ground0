"""Observer-relative hidden-card determinization."""

from __future__ import annotations

import random

from splendor_ai.engine.state import (
    RESERVE_SOURCE_VISIBLE,
    SplendorState,
)


def determinize_for_observer(
    state: SplendorState,
    observer_id: int,
    rng: random.Random,
) -> SplendorState:
    """Resample cards unknown to an observer without using true deck order."""

    determinized = state.copy_shallow()
    opponent = determinized.players[1 - observer_id]
    for tier in (1, 2, 3):
        private_indices = [
            index
            for index, card in enumerate(opponent.reserved_cards)
            if card.tier == tier
            and opponent.reserved_card_source(index)
            != RESERVE_SOURCE_VISIBLE
        ]
        unseen_pool = list(determinized.hidden_tier_decks[tier])
        unseen_pool.extend(
            opponent.reserved_cards[index]
            for index in private_indices
        )
        unseen_pool.sort(key=lambda card: card.card_id)
        rng.shuffle(unseen_pool)
        for index in private_indices:
            opponent.reserved_cards[index] = unseen_pool.pop()
        determinized.hidden_tier_decks[tier] = unseen_pool
        determinized.deck_counts[tier] = len(unseen_pool)
    return determinized
