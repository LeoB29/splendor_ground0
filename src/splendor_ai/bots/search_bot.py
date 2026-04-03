"""Shallow adversarial search bot for stronger corpus generation."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from splendor_ai.engine.actions import Action, ActionType
from splendor_ai.engine.constants import ALL_TOKEN_TYPES, TOKEN_COLORS
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.engine.state import SplendorState

from .heuristic_bot import GreedyHeuristicBot


@dataclass(frozen=True, slots=True)
class _RankedCandidate:
    action: Action
    next_state: SplendorState
    heuristic_score: float
    strategic_signature: tuple[object, ...]


class ShallowSearchBot:
    """Depth-limited minimax bot with heuristic leaf evaluation.

    This bot is intentionally modest:
    - deterministic exact lookahead using the current engine
    - heuristic leaf evaluation reused from `GreedyHeuristicBot`
    - action ordering plus root/node pruning for tractability
    """

    def __init__(
        self,
        depth: int = 2,
        max_branching: int = 10,
        max_buy_actions: int = 6,
        max_reserve_actions: int = 3,
        max_take_actions: int = 3,
        max_pass_actions: int = 1,
        repetition_penalty: float = 4_000.0,
        seed: int | None = None,
    ) -> None:
        if depth < 1:
            raise ValueError("Search depth must be at least 1.")
        if max_branching < 1:
            raise ValueError("max_branching must be at least 1.")
        if min(max_buy_actions, max_reserve_actions, max_take_actions, max_pass_actions) < 0:
            raise ValueError("Per-action-type branching caps must be non-negative.")
        self._depth = depth
        self._max_branching = max_branching
        self._max_buy_actions = max_buy_actions
        self._max_reserve_actions = max_reserve_actions
        self._max_take_actions = max_take_actions
        self._max_pass_actions = max_pass_actions
        self._repetition_penalty = repetition_penalty
        self._rng = random.Random(seed)
        self._heuristic = GreedyHeuristicBot(seed=seed)
        self._transposition_cache: dict[tuple[int, int, tuple[object, ...]], float] = {}

    def choose_action(
        self,
        env: SplendorEnv,
        state: SplendorState,
        legal_actions: list[Action] | None = None,
    ) -> Action | None:
        actions = legal_actions if legal_actions is not None else env.legal_actions(state)
        if not actions:
            return None

        root_player = state.current_player
        self._transposition_cache = {}
        root_signature = self._heuristic.state_signature(state)
        candidates = self._ordered_candidates(
            env=env,
            state=state,
            actions=actions,
            root_player=root_player,
        )

        best_value = -math.inf
        best_actions: list[Action] = []
        alpha = -math.inf
        beta = math.inf

        for candidate in candidates:
            value = self._minimax(
                env=env,
                state=candidate.next_state,
                depth=self._depth - 1,
                root_player=root_player,
                alpha=alpha,
                beta=beta,
                path_signatures={root_signature},
            )
            if value > best_value:
                best_value = value
                best_actions = [candidate.action]
            elif value == best_value:
                best_actions.append(candidate.action)
            alpha = max(alpha, best_value)

        return self._rng.choice(best_actions) if best_actions else candidates[0].action

    def _minimax(
        self,
        env: SplendorEnv,
        state: SplendorState,
        depth: int,
        root_player: int,
        alpha: float,
        beta: float,
        path_signatures: set[tuple[object, ...]],
    ) -> float:
        state_signature = self._heuristic.state_signature(state)
        if state_signature in path_signatures:
            return self._heuristic.evaluate_state(state, root_player) - self._repetition_penalty

        if depth <= 0 or state.terminal:
            return self._heuristic.evaluate_state(state, root_player)

        cache_key = (depth, root_player, state_signature)
        cached = self._transposition_cache.get(cache_key)
        if cached is not None:
            return cached

        legal_actions = env.legal_actions(state)
        if not legal_actions:
            return self._heuristic.evaluate_state(state, root_player)

        maximizing = state.current_player == root_player
        candidates = self._ordered_candidates(
            env=env,
            state=state,
            actions=legal_actions,
            root_player=root_player,
        )

        path_signatures.add(state_signature)
        if maximizing:
            value = -math.inf
            for candidate in candidates:
                value = max(
                    value,
                    self._minimax(
                        env=env,
                        state=candidate.next_state,
                        depth=depth - 1,
                        root_player=root_player,
                        alpha=alpha,
                        beta=beta,
                        path_signatures=path_signatures,
                    ),
                )
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            path_signatures.remove(state_signature)
            self._transposition_cache[cache_key] = value
            return value

        value = math.inf
        for candidate in candidates:
            value = min(
                value,
                self._minimax(
                    env=env,
                    state=candidate.next_state,
                    depth=depth - 1,
                    root_player=root_player,
                    alpha=alpha,
                    beta=beta,
                    path_signatures=path_signatures,
                ),
            )
            beta = min(beta, value)
            if alpha >= beta:
                break
        path_signatures.remove(state_signature)
        self._transposition_cache[cache_key] = value
        return value

    def _ordered_candidates(
        self,
        env: SplendorEnv,
        state: SplendorState,
        actions: list[Action],
        root_player: int,
    ) -> list[_RankedCandidate]:
        actor = state.current_player
        before_affordable = self._heuristic.affordable_card_count(state, root_player)
        before_best_missing = self._heuristic.best_purchase_missing(state, root_player)
        candidates: list[_RankedCandidate] = []
        for action in actions:
            next_state = env.step(state, action)
            score = self._heuristic.score_transition(
                before=state,
                after=next_state,
                action=action,
                player_id=root_player,
                before_affordable=before_affordable,
                before_best_missing=before_best_missing,
            )
            candidates.append(
                _RankedCandidate(
                    action=action,
                    next_state=next_state,
                    heuristic_score=score,
                    strategic_signature=self._strategic_signature(next_state, actor, action),
                )
            )

        maximizing = state.current_player == root_player
        candidates.sort(
            key=lambda item: (item.heuristic_score, self._rng.random()),
            reverse=maximizing,
        )
        limited: list[_RankedCandidate] = []
        kept_by_type = {
            ActionType.BUY_VISIBLE: 0,
            ActionType.BUY_RESERVED: 0,
            ActionType.RESERVE_VISIBLE: 0,
            ActionType.RESERVE_DECK: 0,
            ActionType.TAKE_TOKENS: 0,
            ActionType.PASS: 0,
        }
        seen_signatures: set[tuple[object, ...]] = set()
        for candidate in candidates:
            action = candidate.action
            if len(limited) >= self._max_branching:
                break
            if candidate.strategic_signature in seen_signatures:
                continue
            if action.action_type in (ActionType.BUY_VISIBLE, ActionType.BUY_RESERVED):
                if kept_by_type[ActionType.BUY_VISIBLE] + kept_by_type[ActionType.BUY_RESERVED] >= self._max_buy_actions:
                    continue
            elif action.action_type in (ActionType.RESERVE_VISIBLE, ActionType.RESERVE_DECK):
                if kept_by_type[ActionType.RESERVE_VISIBLE] + kept_by_type[ActionType.RESERVE_DECK] >= self._max_reserve_actions:
                    continue
            elif action.action_type == ActionType.TAKE_TOKENS:
                if kept_by_type[ActionType.TAKE_TOKENS] >= self._max_take_actions:
                    continue
            elif action.action_type == ActionType.PASS:
                if kept_by_type[ActionType.PASS] >= self._max_pass_actions:
                    continue

            limited.append(candidate)
            kept_by_type[action.action_type] += 1
            seen_signatures.add(candidate.strategic_signature)

        if not limited:
            limited = candidates[: self._max_branching]
        return limited

    def _strategic_signature(
        self,
        state: SplendorState,
        actor_id: int,
        action: Action,
    ) -> tuple[object, ...]:
        actor = state.players[actor_id]
        return (
            action.action_type,
            actor.score,
            len(actor.reserved_cards),
            len(actor.purchased_cards),
            len(actor.nobles),
            tuple(actor.tokens[color] for color in ALL_TOKEN_TYPES),
            tuple(actor.bonuses[color] for color in TOKEN_COLORS),
            tuple(state.bank_tokens[color] for color in ALL_TOKEN_TYPES),
            self._heuristic.affordable_card_count(state, actor_id),
            self._heuristic.best_purchase_missing(state, actor_id),
            self._heuristic.best_scoring_purchase_missing(state, actor_id),
            self._heuristic.best_noble_gap(state, actor_id),
        )
