"""Greedy one-ply heuristic bot."""

from __future__ import annotations

import random

from splendor_ai.engine.actions import Action, ActionType
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.engine.state import Card
from splendor_ai.engine.state import SplendorState


class GreedyHeuristicBot:
    """Chooses the legal move with the best immediate heuristic outcome.

    The heuristic is intentionally simple and fast:
    - strongly prefer winning / scoring moves
    - value prestige, permanent bonuses, and nobles
    - prefer more tokens and lower opponent upside
    - break ties randomly with a local RNG
    """

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def choose_action(
        self,
        env: SplendorEnv,
        state: SplendorState,
        legal_actions: list[Action] | None = None,
    ) -> Action | None:
        actions = legal_actions if legal_actions is not None else env.legal_actions(state)
        if not actions:
            return None

        current_player = state.current_player
        before_affordable = self._affordable_card_count(state, current_player)
        before_best_missing = self._best_purchase_missing(state, current_player)
        scored_actions: list[tuple[float, float, Action]] = []
        for action in actions:
            score = self.score_action(
                env=env,
                state=state,
                action=action,
                player_id=current_player,
                before_affordable=before_affordable,
                before_best_missing=before_best_missing,
            )
            scored_actions.append((score, self._rng.random(), action))

        scored_actions.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return scored_actions[0][2]

    def evaluate_state(self, state: SplendorState, player_id: int) -> float:
        return self._score_state(state, player_id)

    def state_signature(self, state: SplendorState) -> tuple[object, ...]:
        return self._state_signature(state)

    def affordable_card_count(self, state: SplendorState, player_id: int) -> int:
        return self._affordable_card_count(state, player_id)

    def best_purchase_missing(self, state: SplendorState, player_id: int) -> int:
        return self._best_purchase_missing(state, player_id)

    def best_scoring_purchase_missing(self, state: SplendorState, player_id: int) -> int:
        return self._best_scoring_purchase_missing(state, player_id)

    def best_noble_gap(self, state: SplendorState, player_id: int) -> int:
        return self._best_noble_gap(state, player_id)

    def score_action(
        self,
        env: SplendorEnv,
        state: SplendorState,
        action: Action,
        player_id: int | None = None,
        before_affordable: int | None = None,
        before_best_missing: int | None = None,
    ) -> float:
        actor = state.current_player if player_id is None else player_id
        prior_affordable = (
            self._affordable_card_count(state, actor)
            if before_affordable is None
            else before_affordable
        )
        prior_best_missing = (
            self._best_purchase_missing(state, actor)
            if before_best_missing is None
            else before_best_missing
        )
        next_state = env.step(state, action)
        return self.score_transition(
            before=state,
            after=next_state,
            action=action,
            player_id=actor,
            before_affordable=prior_affordable,
            before_best_missing=prior_best_missing,
        )

    def score_transition(
        self,
        before: SplendorState,
        after: SplendorState,
        action: Action,
        player_id: int,
        before_affordable: int | None = None,
        before_best_missing: int | None = None,
    ) -> float:
        prior_affordable = (
            self._affordable_card_count(before, player_id)
            if before_affordable is None
            else before_affordable
        )
        prior_best_missing = (
            self._best_purchase_missing(before, player_id)
            if before_best_missing is None
            else before_best_missing
        )
        score = self._score_state(after, player_id)
        score += self._action_progress_bonus(
            before=before,
            after=after,
            action=action,
            player_id=player_id,
            before_affordable=prior_affordable,
            before_best_missing=prior_best_missing,
        )
        return score

    def _score_state(self, state: SplendorState, player_id: int) -> float:
        player = state.players[player_id]
        opponent = state.players[1 - player_id]

        if state.terminal:
            if state.winner == player_id:
                return 1_000_000.0
            if state.winner == 1 - player_id:
                return -1_000_000.0
            return 0.0

        player_bonus_total = sum(player.bonuses.values())
        opponent_bonus_total = sum(opponent.bonuses.values())
        player_affordable_visible = self._affordable_visible_card_count(state, player_id)
        opponent_affordable_visible = self._affordable_visible_card_count(state, 1 - player_id)
        player_affordable_reserved = self._affordable_reserved_card_count(state, player_id)
        opponent_affordable_reserved = self._affordable_reserved_card_count(state, 1 - player_id)
        player_best_missing = self._best_purchase_missing(state, player_id)
        opponent_best_missing = self._best_purchase_missing(state, 1 - player_id)
        player_best_scoring_missing = self._best_scoring_purchase_missing(state, player_id)
        opponent_best_scoring_missing = self._best_scoring_purchase_missing(state, 1 - player_id)
        player_noble_gap = self._best_noble_gap(state, player_id)
        opponent_noble_gap = self._best_noble_gap(state, 1 - player_id)

        score = 0.0
        score += 1_000.0 * (player.score - opponent.score)
        score += 120.0 * (player_bonus_total - opponent_bonus_total)
        score += 42.0 * (len(player.purchased_cards) - len(opponent.purchased_cards))
        score += 60.0 * (len(player.nobles) - len(opponent.nobles))
        score += 20.0 * (player_affordable_reserved - opponent_affordable_reserved)
        score += 9.0 * player_affordable_visible
        score -= 9.0 * opponent_affordable_visible
        score += 28.0 * (opponent_best_missing - player_best_missing)
        score += 48.0 * (opponent_best_scoring_missing - player_best_scoring_missing)
        score += 20.0 * (opponent_noble_gap - player_noble_gap)
        score += 2.5 * (player.token_count - opponent.token_count)
        score += 10.0 * max(player_bonus_total - 3, 0)
        score -= 10.0 * max(opponent_bonus_total - 3, 0)
        score += 25.0 * max(player.score - 11, 0)
        score -= 30.0 * max(opponent.score - 11, 0)
        score -= 25.0 * max(len(player.reserved_cards) - 2, 0)
        score += 20.0 * max(len(opponent.reserved_cards) - 2, 0)
        score -= 8.0 * max(player.token_count - 7, 0)
        score += 8.0 * max(opponent.token_count - 7, 0)
        score += 3.0 * self._action_flexibility(state, player_id)
        score -= 3.0 * self._action_flexibility(state, 1 - player_id)
        return score

    def _affordable_visible_card_count(self, state: SplendorState, player_id: int) -> int:
        affordable = 0
        for cards in state.visible_tier_cards.values():
            for card in cards:
                if self._missing_to_buy(state, player_id, card) == 0:
                    affordable += 1
        return affordable

    def _affordable_reserved_card_count(self, state: SplendorState, player_id: int) -> int:
        affordable = 0
        for card in state.players[player_id].reserved_cards:
            if self._missing_to_buy(state, player_id, card) == 0:
                affordable += 1
        return affordable

    def _affordable_card_count(self, state: SplendorState, player_id: int) -> int:
        return self._affordable_visible_card_count(state, player_id) + self._affordable_reserved_card_count(
            state, player_id
        )

    def _best_purchase_missing(self, state: SplendorState, player_id: int) -> int:
        candidate_cards: list[Card] = []
        for cards in state.visible_tier_cards.values():
            candidate_cards.extend(cards)
        candidate_cards.extend(state.players[player_id].reserved_cards)
        if not candidate_cards:
            return 0
        return min(self._missing_to_buy(state, player_id, card) for card in candidate_cards)

    def _best_scoring_purchase_missing(self, state: SplendorState, player_id: int) -> int:
        candidate_cards: list[Card] = []
        for cards in state.visible_tier_cards.values():
            candidate_cards.extend(card for card in cards if card.points > 0)
        candidate_cards.extend(card for card in state.players[player_id].reserved_cards if card.points > 0)
        if not candidate_cards:
            return self._best_purchase_missing(state, player_id)
        return min(self._missing_to_buy(state, player_id, card) for card in candidate_cards)

    def _best_noble_gap(self, state: SplendorState, player_id: int) -> int:
        player = state.players[player_id]
        if not state.nobles:
            return 0
        best_gap = None
        for noble in state.nobles:
            gap = 0
            for color in ("white", "blue", "green", "red", "black"):
                gap += max(noble.requirement.get(color, 0) - player.bonuses.get(color, 0), 0)
            best_gap = gap if best_gap is None else min(best_gap, gap)
        return best_gap if best_gap is not None else 0

    def _missing_to_buy(self, state: SplendorState, player_id: int, card: Card) -> int:
        player = state.players[player_id]
        missing = 0
        for color in ("white", "blue", "green", "red", "black"):
            discounted_cost = max(card.cost.get(color, 0) - player.bonuses.get(color, 0), 0)
            missing += max(discounted_cost - player.tokens[color], 0)
        return max(missing - player.tokens["gold"], 0)

    def _action_flexibility(self, state: SplendorState, player_id: int) -> int:
        if state.current_player == player_id:
            env_view = state
        else:
            env_view = state.copy_shallow()
            env_view.current_player = player_id
        # This is only used as a light heuristic term, so the exact off-turn legality
        # approximation is acceptable here.
        return len(SplendorEnv().legal_actions(env_view))

    def _is_effective_no_op(self, before: SplendorState, after: SplendorState) -> bool:
        return self._state_signature(before) == self._state_signature(after)

    def _action_progress_bonus(
        self,
        before: SplendorState,
        after: SplendorState,
        action: Action,
        player_id: int,
        before_affordable: int,
        before_best_missing: int,
    ) -> float:
        if action.action_type == ActionType.PASS:
            return -5_000.0

        bonus = 0.0
        after_affordable = self._affordable_card_count(after, player_id)
        after_best_missing = self._best_purchase_missing(after, player_id)
        before_best_scoring_missing = self._best_scoring_purchase_missing(before, player_id)
        after_best_scoring_missing = self._best_scoring_purchase_missing(after, player_id)
        before_noble_gap = self._best_noble_gap(before, player_id)
        after_noble_gap = self._best_noble_gap(after, player_id)
        before_opp_best_scoring_missing = self._best_scoring_purchase_missing(before, 1 - player_id)
        after_opp_best_scoring_missing = self._best_scoring_purchase_missing(after, 1 - player_id)
        bonus += 250.0 * (after_affordable - before_affordable)
        bonus += 35.0 * (before_best_missing - after_best_missing)
        bonus += 65.0 * (before_best_scoring_missing - after_best_scoring_missing)
        bonus += 18.0 * (before_noble_gap - after_noble_gap)
        bonus += 28.0 * (after_opp_best_scoring_missing - before_opp_best_scoring_missing)

        if self._is_effective_no_op(before, after):
            bonus -= 10_000.0

        if action.action_type == ActionType.TAKE_TOKENS:
            if (
                after_affordable <= before_affordable
                and after_best_missing >= before_best_missing
                and after_best_scoring_missing >= before_best_scoring_missing
                and after_noble_gap >= before_noble_gap
            ):
                bonus -= 500.0
            if after.players[player_id].token_count >= before.players[player_id].token_count:
                bonus -= 40.0 * max(after.players[player_id].token_count - 7, 0)
        elif action.action_type in (ActionType.RESERVE_VISIBLE, ActionType.RESERVE_DECK):
            bonus += 40.0
            if len(after.players[player_id].reserved_cards) >= 3:
                bonus -= 60.0
        elif action.action_type in (ActionType.BUY_VISIBLE, ActionType.BUY_RESERVED):
            purchased_delta = len(after.players[player_id].purchased_cards) - len(before.players[player_id].purchased_cards)
            bonus += 140.0 * purchased_delta

        return bonus

    def _state_signature(self, state: SplendorState) -> tuple[object, ...]:
        return (
            tuple(state.bank_tokens[color] for color in ("white", "blue", "green", "red", "black", "gold")),
            tuple(state.deck_counts[tier] for tier in (1, 2, 3)),
            tuple(tuple(card.card_id for card in state.visible_tier_cards[tier]) for tier in (1, 2, 3)),
            tuple(noble.noble_id for noble in state.nobles),
            tuple(self._player_signature(player) for player in state.players),
            state.pending_round_end,
            state.terminal,
            state.winner,
        )

    def _player_signature(self, player) -> tuple[object, ...]:
        return (
            tuple(player.tokens[color] for color in ("white", "blue", "green", "red", "black", "gold")),
            tuple(player.bonuses[color] for color in ("white", "blue", "green", "red", "black")),
            player.score,
            tuple(card.card_id for card in player.reserved_cards),
            tuple(card.card_id for card in player.purchased_cards),
            tuple(noble.noble_id for noble in player.nobles),
        )
