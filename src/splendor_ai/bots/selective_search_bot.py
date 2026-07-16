"""Selective shallow-search wrapper for the promoted canonical entity policy."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from splendor_ai.engine.actions import Action, ActionType
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.engine.state import SplendorState

from .model_bot import CanonicalEntityActionScorerBot
from .search_bot import ShallowSearchBot


@dataclass(frozen=True, slots=True)
class SelectiveSearchConfig:
    """Frozen routing rule for the first selective-search experiment."""

    uncertainty_logit_gap: float = 0.5
    tactical_score_threshold: int = 12
    winning_score: int = 15

    def __post_init__(self) -> None:
        if self.uncertainty_logit_gap < 0.0:
            raise ValueError("uncertainty_logit_gap must be non-negative.")
        if self.tactical_score_threshold < 0:
            raise ValueError("tactical_score_threshold must be non-negative.")
        if self.winning_score <= 0:
            raise ValueError("winning_score must be positive.")


@dataclass(frozen=True, slots=True)
class SelectiveSearchDecision:
    logit_gap: float
    uncertainty_triggered: bool
    tactical_triggered: bool
    used_search: bool


class SelectiveSearchHybridBot:
    """Use the fast canonical policy except on uncertain or tactical turns."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        device: str = "cpu",
        routing: SelectiveSearchConfig | None = None,
        shallow_kwargs: dict[str, object] | None = None,
    ) -> None:
        self._policy = CanonicalEntityActionScorerBot(checkpoint_path, device=device)
        self._routing = routing or SelectiveSearchConfig()
        self._search = ShallowSearchBot(**(shallow_kwargs or {}))
        self._turns = 0
        self._search_turns = 0
        self._uncertainty_turns = 0
        self._tactical_turns = 0
        self._both_trigger_turns = 0
        self._last_decision: SelectiveSearchDecision | None = None

    @property
    def checkpoint_path(self) -> Path:
        return self._policy.checkpoint_path

    @property
    def loop_fallback_triggers(self) -> int:
        return self._policy.loop_fallback_triggers

    @property
    def routing_config(self) -> SelectiveSearchConfig:
        return self._routing

    @property
    def last_decision(self) -> SelectiveSearchDecision | None:
        return self._last_decision

    @property
    def search_fraction(self) -> float:
        return self._search_turns / self._turns if self._turns else 0.0

    @property
    def telemetry(self) -> dict[str, int | float | dict[str, object]]:
        return {
            "turns": self._turns,
            "neural_turns": self._turns - self._search_turns,
            "search_turns": self._search_turns,
            "search_fraction": self.search_fraction,
            "uncertainty_trigger_turns": self._uncertainty_turns,
            "tactical_trigger_turns": self._tactical_turns,
            "both_trigger_turns": self._both_trigger_turns,
            "routing_config": asdict(self._routing),
        }

    def choose_action(
        self,
        env: SplendorEnv,
        state: SplendorState,
        legal_actions: list[Action] | None = None,
    ) -> Action | None:
        actions = legal_actions if legal_actions is not None else env.legal_actions(state)
        if not actions:
            return None

        action_scores = self._policy.score_actions(state, actions)
        ordered_scores = sorted(action_scores.values(), reverse=True)
        logit_gap = (
            ordered_scores[0] - ordered_scores[1]
            if len(ordered_scores) > 1
            else float("inf")
        )
        uncertainty = logit_gap <= self._routing.uncertainty_logit_gap
        tactical = self._is_tactical(env, state, actions)
        use_search = uncertainty or tactical
        self._last_decision = SelectiveSearchDecision(
            logit_gap=logit_gap,
            uncertainty_triggered=uncertainty,
            tactical_triggered=tactical,
            used_search=use_search,
        )
        self._turns += 1
        self._uncertainty_turns += int(uncertainty)
        self._tactical_turns += int(tactical)
        self._both_trigger_turns += int(uncertainty and tactical)

        if use_search:
            self._search_turns += 1
            action = self._search.choose_action(env, state, actions)
            if action is not None:
                self._policy.record_external_action(env, state, action)
            return action
        return self._policy.choose_action_from_scores(env, state, action_scores)

    def _is_tactical(
        self,
        env: SplendorEnv,
        state: SplendorState,
        actions: list[Action],
    ) -> bool:
        actor = state.current_player
        opponent = 1 - actor
        if (
            state.players[actor].score >= self._routing.tactical_score_threshold
            or state.players[opponent].score >= self._routing.tactical_score_threshold
        ):
            return True
        for action in actions:
            if action.action_type not in (ActionType.BUY_VISIBLE, ActionType.BUY_RESERVED):
                continue
            next_state = env.step(state, action)
            if (
                next_state.terminal
                or next_state.players[actor].score >= self._routing.winning_score
            ):
                return True
        return False
