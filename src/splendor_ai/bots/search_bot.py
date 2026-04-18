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
    loop_penalty: float
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
        game_history_penalty: float = 3_000.0,
        no_progress_take_penalty: float = 900.0,
        return_token_penalty: float = 120.0,
        repeated_color_return_penalty: float = 250.0,
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
        self._game_history_penalty = game_history_penalty
        self._no_progress_take_penalty = no_progress_take_penalty
        self._return_token_penalty = return_token_penalty
        self._repeated_color_return_penalty = repeated_color_return_penalty
        self._rng = random.Random(seed)
        self._heuristic = GreedyHeuristicBot(seed=seed)
        self._transposition_cache: dict[tuple[int, int, tuple[object, ...]], float] = {}
        self._observed_state_counts: dict[tuple[object, ...], int] = {}
        self._post_action_state_counts: dict[tuple[object, ...], int] = {}
        self._last_turn_index: int | None = None

    def choose_action(
        self,
        env: SplendorEnv,
        state: SplendorState,
        legal_actions: list[Action] | None = None,
    ) -> Action | None:
        actions = legal_actions if legal_actions is not None else env.legal_actions(state)
        if not actions:
            return None

        self._reset_game_tracking_if_needed(state)
        root_player = state.current_player
        self._transposition_cache = {}
        root_signature = self._heuristic.state_signature(state)
        self._observed_state_counts[root_signature] = self._observed_state_counts.get(root_signature, 0) + 1
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

        best_candidates: list[_RankedCandidate] = []
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
            value -= candidate.loop_penalty
            if value > best_value:
                best_value = value
                best_actions = [candidate.action]
                best_candidates = [candidate]
            elif value == best_value:
                best_actions.append(candidate.action)
                best_candidates.append(candidate)
            alpha = max(alpha, best_value)

        if not best_candidates:
            chosen_candidate = candidates[0]
        else:
            chosen_candidate = self._rng.choice(best_candidates)
        next_signature = self._heuristic.state_signature(chosen_candidate.next_state)
        self._post_action_state_counts[next_signature] = self._post_action_state_counts.get(next_signature, 0) + 1
        self._last_turn_index = state.turn_index
        return chosen_candidate.action

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
            loop_penalty = self._candidate_loop_penalty(
                before=state,
                after=next_state,
                action=action,
                player_id=root_player,
            )
            adjusted_score = score - loop_penalty
            candidates.append(
                _RankedCandidate(
                    action=action,
                    next_state=next_state,
                    heuristic_score=adjusted_score,
                    loop_penalty=loop_penalty,
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

    def _candidate_loop_penalty(
        self,
        before: SplendorState,
        after: SplendorState,
        action: Action,
        player_id: int,
    ) -> float:
        before_signature = self._heuristic.state_signature(before)
        after_signature = self._heuristic.state_signature(after)
        current_repeat_count = max(self._observed_state_counts.get(before_signature, 0) - 1, 0)
        after_repeat_count = self._post_action_state_counts.get(after_signature, 0)
        penalty = after_repeat_count * self._game_history_penalty

        if action.action_type != ActionType.TAKE_TOKENS:
            return penalty

        if action.return_tokens:
            penalty += self._return_token_penalty * len(action.return_tokens)
            repeated_colors = set(action.take_tokens) & set(action.return_tokens)
            penalty += self._repeated_color_return_penalty * len(repeated_colors)

        before_affordable = self._heuristic.affordable_card_count(before, player_id)
        after_affordable = self._heuristic.affordable_card_count(after, player_id)
        before_best_missing = self._heuristic.best_purchase_missing(before, player_id)
        after_best_missing = self._heuristic.best_purchase_missing(after, player_id)
        before_best_scoring_missing = self._heuristic.best_scoring_purchase_missing(before, player_id)
        after_best_scoring_missing = self._heuristic.best_scoring_purchase_missing(after, player_id)
        before_noble_gap = self._heuristic.best_noble_gap(before, player_id)
        after_noble_gap = self._heuristic.best_noble_gap(after, player_id)

        if not self._is_progress_transition(before, after, action):
            penalty += self._no_progress_take_penalty
            if current_repeat_count > 0:
                penalty += current_repeat_count * self._game_history_penalty
            if (
                after_affordable <= before_affordable
                and after_best_missing >= before_best_missing
                and after_best_scoring_missing >= before_best_scoring_missing
                and after_noble_gap >= before_noble_gap
            ):
                penalty += self._no_progress_take_penalty

        return penalty

    def _reset_game_tracking_if_needed(self, state: SplendorState) -> None:
        if state.turn_index == 0 or (self._last_turn_index is not None and state.turn_index <= self._last_turn_index):
            self._observed_state_counts = {}
            self._post_action_state_counts = {}
            self._last_turn_index = None

    def _is_progress_transition(
        self,
        before: SplendorState,
        after: SplendorState,
        action: Action,
    ) -> bool:
        current_before = before.players[before.current_player]
        current_after = after.players[before.current_player]
        opponent_before = before.players[1 - before.current_player]
        opponent_after = after.players[1 - before.current_player]

        if action.action_type in (
            ActionType.BUY_VISIBLE,
            ActionType.BUY_RESERVED,
            ActionType.RESERVE_VISIBLE,
            ActionType.RESERVE_DECK,
        ):
            return True
        if len(current_after.purchased_cards) != len(current_before.purchased_cards):
            return True
        if len(current_after.reserved_cards) != len(current_before.reserved_cards):
            return True
        if len(current_after.nobles) != len(current_before.nobles):
            return True
        if current_after.score != current_before.score:
            return True
        if current_after.bonuses != current_before.bonuses:
            return True
        if opponent_after.score != opponent_before.score:
            return True
        if opponent_after.bonuses != opponent_before.bonuses:
            return True
        return False

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
