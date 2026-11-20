"""Shared finite-game termination rules used by search, play, and replay."""

from __future__ import annotations

from typing import Protocol, Sequence

from splendor_ai.engine.actions import Action, ActionType
from splendor_ai.engine.state import SplendorState


FORCED_PASS_DEADLOCK = "forced_pass_deadlock"
VALID_NATURAL_TERMINATIONS = ("completed", FORCED_PASS_DEADLOCK)


class _Environment(Protocol):
    def legal_actions(self, state: SplendorState) -> list[Action]: ...

    def step(self, state: SplendorState, action: Action) -> SplendorState: ...


def is_forced_pass_turn(legal_actions: Sequence[Action]) -> bool:
    """Return whether the complete legal set is exactly one forced pass."""

    return len(legal_actions) == 1 and legal_actions[0].action_type is ActionType.PASS


def adjudicate_scores(state: SplendorState) -> int | None:
    """Apply the established score then fewest-purchased-cards tie-break."""

    scores = [player.score for player in state.players]
    contenders = [index for index, score in enumerate(scores) if score == max(scores)]
    if len(contenders) == 1:
        return contenders[0]
    fewest = min(len(state.players[index].purchased_cards) for index in contenders)
    finalists = [
        index
        for index in contenders
        if len(state.players[index].purchased_cards) == fewest
    ]
    return finalists[0] if len(finalists) == 1 else None


def forced_pass_deadlock_successor(
    env: _Environment,
    state_after_first_pass: SplendorState,
) -> SplendorState | None:
    """Apply the second sole PASS when it proves a two-player deadlock."""

    legal_actions = env.legal_actions(state_after_first_pass)
    if not is_forced_pass_turn(legal_actions):
        return None
    return env.step(state_after_first_pass, legal_actions[0])


def outcome_value(winner: int | None, player_id: int) -> float:
    """Convert an adjudicated winner to a value from one player's view."""

    if winner is None:
        return 0.0
    return 1.0 if winner == player_id else -1.0
