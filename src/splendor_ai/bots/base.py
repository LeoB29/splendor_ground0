"""Bot interfaces for Splendor agents."""

from __future__ import annotations

from typing import Protocol

from splendor_ai.engine.actions import Action
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.engine.state import SplendorState


class Bot(Protocol):
    """Protocol for agents that can act in the Splendor environment."""

    def choose_action(
        self,
        env: SplendorEnv,
        state: SplendorState,
        legal_actions: list[Action] | None = None,
    ) -> Action | None:
        ...
