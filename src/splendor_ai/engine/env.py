"""Environment entry point for Splendor.

The transition function is intentionally conservative at this stage: the
environment interface exists, but rule-complete action generation and stepping
still need implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import random

from .actions import Action, ActionType
from .constants import (
    ALL_TOKEN_TYPES,
    BANK_GEMS_PER_COLOR_2P,
    GOLD_TOKENS,
    MAX_RESERVED_CARDS,
    MAX_TOKENS_PER_PLAYER,
    NOBLES_IN_PLAY_2P,
    TOKEN_COLORS,
    VISIBLE_CARDS_PER_TIER,
)
from .data import build_base_deck_by_tier, build_base_nobles
from .state import Card, PlayerState, SplendorState, empty_token_bank


def _token_sort_key(color: str) -> int:
    return ALL_TOKEN_TYPES.index(color)


def _expand_token_counts(token_counts: dict[str, int]) -> tuple[str, ...]:
    expanded: list[str] = []
    for color in ALL_TOKEN_TYPES:
        expanded.extend([color] * token_counts.get(color, 0))
    return tuple(expanded)

def _available_counts_after_gain(
    player: PlayerState,
    gained_tokens: tuple[str, ...],
) -> dict[str, int]:
    counts = dict(player.tokens)
    for color in gained_tokens:
        counts[color] += 1
    return counts


def _generate_multiset_tuples(
    available_counts: dict[str, int],
    choose_n: int,
    colors: tuple[str, ...] = ALL_TOKEN_TYPES,
) -> list[tuple[str, ...]]:
    if choose_n == 0:
        return [()]

    results: list[tuple[str, ...]] = []

    def recurse(color_index: int, remaining: int, prefix: list[str]) -> None:
        if color_index == len(colors):
            if remaining == 0:
                results.append(tuple(prefix))
            return

        color = colors[color_index]
        max_take = min(available_counts.get(color, 0), remaining)
        for amount in range(max_take + 1):
            if amount:
                prefix.extend([color] * amount)
            recurse(color_index + 1, remaining - amount, prefix)
            if amount:
                del prefix[-amount:]

    recurse(0, choose_n, [])
    return results


def _purchase_spend_options(player: PlayerState, card: Card) -> list[tuple[str, ...]]:
    required_by_color = {
        color: max(card.cost.get(color, 0) - player.bonuses.get(color, 0), 0)
        for color in TOKEN_COLORS
    }
    total_required = sum(required_by_color.values())
    if total_required == 0:
        return [()]

    gold_available = player.tokens["gold"]
    spend_options: list[tuple[str, ...]] = []

    def recurse(color_index: int, colored_spent: dict[str, int]) -> None:
        if color_index == len(TOKEN_COLORS):
            colored_total = sum(colored_spent.values())
            gold_needed = total_required - colored_total
            if gold_needed < 0 or gold_needed > gold_available:
                return
            spend_counts = dict(colored_spent)
            spend_counts["gold"] = gold_needed
            spend_options.append(_expand_token_counts(spend_counts))
            return

        color = TOKEN_COLORS[color_index]
        min_spend = 0
        max_spend = min(required_by_color[color], player.tokens[color])
        for amount in range(min_spend, max_spend + 1):
            colored_spent[color] = amount
            recurse(color_index + 1, colored_spent)
        colored_spent.pop(color, None)

    recurse(0, {})
    return spend_options


def _bonuses_with_card(player: PlayerState, card: Card) -> dict[str, int]:
    bonuses = dict(player.bonuses)
    bonuses[card.bonus_color] += 1
    return bonuses


@dataclass(slots=True)
class SplendorEnv:
    """Base Splendor environment for the 2-player legal-observation project."""

    seed: int | None = None

    def initial_state(self) -> SplendorState:
        """Construct the deterministic seeded base-game initial state."""

        bank = empty_token_bank()
        for color in TOKEN_COLORS:
            bank[color] = BANK_GEMS_PER_COLOR_2P
        bank["gold"] = GOLD_TOKENS

        players = [PlayerState(player_id=0), PlayerState(player_id=1)]
        rng = random.Random(self.seed)

        hidden_tier_decks: dict[int, list] = {}
        visible_tier_cards: dict[int, list] = {}
        for tier, cards in build_base_deck_by_tier().items():
            shuffled_cards = list(cards)
            rng.shuffle(shuffled_cards)
            visible_tier_cards[tier] = shuffled_cards[:VISIBLE_CARDS_PER_TIER]
            hidden_tier_decks[tier] = shuffled_cards[VISIBLE_CARDS_PER_TIER:]

        nobles = list(build_base_nobles())
        rng.shuffle(nobles)
        nobles = nobles[: self.nobles_in_play]

        return SplendorState(
            current_player=0,
            bank_tokens=bank,
            players=players,
            visible_tier_cards=visible_tier_cards,
            hidden_tier_decks=hidden_tier_decks,
            deck_counts={tier: len(cards) for tier, cards in hidden_tier_decks.items()},
            nobles=nobles,
            turn_index=0,
            start_player=0,
            pending_round_end=False,
        )

    def legal_actions(self, state: SplendorState) -> list[Action]:
        """Return all legal actions for the current state."""

        if state.terminal:
            return []

        player = state.players[state.current_player]
        legal_actions: list[Action] = []
        legal_actions.extend(self._legal_take_actions(player, state))
        legal_actions.extend(self._legal_reserve_actions(player, state))
        legal_actions.extend(self._legal_buy_visible_actions(player, state))
        legal_actions.extend(self._legal_buy_reserved_actions(player, state))
        if not legal_actions:
            return [Action(action_type=ActionType.PASS)]
        return legal_actions

    def step(self, state: SplendorState, action: Action) -> SplendorState:
        """Apply an action and return the next state.

        Noble choice is encoded on buy actions via `action.noble_id` when needed.
        """
        if state.terminal:
            raise ValueError("Cannot apply an action to a terminal state.")

        legal_actions = self.legal_actions(state)
        if action not in legal_actions:
            raise ValueError(f"Illegal action: {action!r}")

        next_state = state.copy_shallow()
        player = next_state.players[next_state.current_player]

        if action.action_type == ActionType.PASS:
            pass
        elif action.action_type == ActionType.TAKE_TOKENS:
            self._apply_take_action(next_state, player, action)
        elif action.action_type == ActionType.RESERVE_VISIBLE:
            self._apply_reserve_visible_action(next_state, player, action)
        elif action.action_type == ActionType.RESERVE_DECK:
            self._apply_reserve_deck_action(next_state, player, action)
        elif action.action_type == ActionType.BUY_VISIBLE:
            purchased_card = self._apply_buy_visible_action(next_state, player, action)
            self._apply_noble_resolution(next_state, player, purchased_card, action)
        elif action.action_type == ActionType.BUY_RESERVED:
            purchased_card = self._apply_buy_reserved_action(next_state, player, action)
            self._apply_noble_resolution(next_state, player, purchased_card, action)
        else:
            raise ValueError(f"Unsupported action type: {action.action_type}")

        next_state.turn_index += 1
        if not next_state.pending_round_end and player.score >= 15:
            next_state.pending_round_end = True

        next_state.current_player = 1 - state.current_player
        if next_state.pending_round_end and next_state.current_player == next_state.start_player:
            self._finalize_terminal_state(next_state)

        return next_state

    @property
    def nobles_in_play(self) -> int:
        return NOBLES_IN_PLAY_2P

    def _legal_take_actions(
        self,
        player: PlayerState,
        state: SplendorState,
    ) -> list[Action]:
        actions: list[Action] = []

        available_distinct_colors = tuple(
            color for color in TOKEN_COLORS if state.bank_tokens[color] > 0
        )
        distinct_take_count = min(len(available_distinct_colors), 3)
        if distinct_take_count > 0:
            for taken_colors in combinations(available_distinct_colors, distinct_take_count):
                actions.extend(
                    self._token_gain_actions(
                        player=player,
                        action_type=ActionType.TAKE_TOKENS,
                        gained_tokens=taken_colors,
                    )
                )

        for color in TOKEN_COLORS:
            if state.bank_tokens[color] >= 4:
                actions.extend(
                    self._token_gain_actions(
                        player=player,
                        action_type=ActionType.TAKE_TOKENS,
                        gained_tokens=(color, color),
                    )
                )

        return actions

    def _legal_reserve_actions(
        self,
        player: PlayerState,
        state: SplendorState,
    ) -> list[Action]:
        if len(player.reserved_cards) >= MAX_RESERVED_CARDS:
            return []

        gained_tokens = ("gold",) if state.bank_tokens["gold"] > 0 else ()
        actions: list[Action] = []

        for tier in sorted(state.visible_tier_cards):
            for market_index, _card in enumerate(state.visible_tier_cards[tier]):
                actions.extend(
                    self._token_gain_actions(
                        player=player,
                        action_type=ActionType.RESERVE_VISIBLE,
                        gained_tokens=gained_tokens,
                        tier=tier,
                        market_index=market_index,
                    )
                )

        for tier in sorted(state.hidden_tier_decks):
            if not state.hidden_tier_decks[tier]:
                continue
            actions.extend(
                self._token_gain_actions(
                    player=player,
                    action_type=ActionType.RESERVE_DECK,
                    gained_tokens=gained_tokens,
                    tier=tier,
                )
            )

        return actions

    def _legal_buy_visible_actions(
        self,
        player: PlayerState,
        state: SplendorState,
    ) -> list[Action]:
        actions: list[Action] = []
        for tier in sorted(state.visible_tier_cards):
            for market_index, card in enumerate(state.visible_tier_cards[tier]):
                for spend_tokens in _purchase_spend_options(player, card):
                    actions.extend(
                        self._buy_actions_with_nobles(
                            player=player,
                            state=state,
                            action_type=ActionType.BUY_VISIBLE,
                            card=card,
                            spend_tokens=spend_tokens,
                            tier=tier,
                            market_index=market_index,
                        )
                    )
        return actions

    def _legal_buy_reserved_actions(
        self,
        player: PlayerState,
        state: SplendorState | None = None,
    ) -> list[Action]:
        actions: list[Action] = []
        for reserved_index, card in enumerate(player.reserved_cards):
            for spend_tokens in _purchase_spend_options(player, card):
                actions.extend(
                    self._buy_actions_with_nobles(
                        player=player,
                        state=state,
                        action_type=ActionType.BUY_RESERVED,
                        card=card,
                        spend_tokens=spend_tokens,
                        reserved_index=reserved_index,
                    )
                )
        return actions

    def _token_gain_actions(
        self,
        player: PlayerState,
        action_type: ActionType,
        gained_tokens: tuple[str, ...],
        tier: int | None = None,
        market_index: int | None = None,
    ) -> list[Action]:
        post_gain_counts = _available_counts_after_gain(player, gained_tokens)
        excess_tokens = max(
            sum(post_gain_counts.values()) - MAX_TOKENS_PER_PLAYER,
            0,
        )
        return_options = _generate_multiset_tuples(post_gain_counts, excess_tokens)

        actions: list[Action] = []
        for return_tokens in return_options:
            actions.append(
                Action(
                    action_type=action_type,
                    tier=tier,
                    market_index=market_index,
                    take_tokens=tuple(sorted(gained_tokens, key=_token_sort_key)),
                    return_tokens=return_tokens,
                )
            )
        return actions

    def _buy_actions_with_nobles(
        self,
        player: PlayerState,
        action_type: ActionType,
        card: Card,
        spend_tokens: tuple[str, ...],
        state: SplendorState | None,
        tier: int | None = None,
        market_index: int | None = None,
        reserved_index: int | None = None,
    ) -> list[Action]:
        claimable_nobles = self._claimable_nobles_after_purchase(
            player=player,
            available_nobles=state.nobles if state is not None else [],
            purchased_card=card,
        )
        if not claimable_nobles:
            return [
                Action(
                    action_type=action_type,
                    tier=tier,
                    market_index=market_index,
                    reserved_index=reserved_index,
                    spend_tokens=spend_tokens,
                )
            ]

        return [
            Action(
                action_type=action_type,
                tier=tier,
                market_index=market_index,
                reserved_index=reserved_index,
                spend_tokens=spend_tokens,
                noble_id=noble.noble_id,
            )
            for noble in claimable_nobles
        ]

    def _apply_take_action(
        self,
        state: SplendorState,
        player: PlayerState,
        action: Action,
    ) -> None:
        for color in action.take_tokens:
            state.bank_tokens[color] -= 1
            player.tokens[color] += 1
        for color in action.return_tokens:
            player.tokens[color] -= 1
            state.bank_tokens[color] += 1

    def _apply_reserve_visible_action(
        self,
        state: SplendorState,
        player: PlayerState,
        action: Action,
    ) -> None:
        if action.tier is None or action.market_index is None:
            raise ValueError("Reserve visible action requires tier and market index.")

        card = state.visible_tier_cards[action.tier].pop(action.market_index)
        player.reserved_cards.append(card)
        self._draw_replacement_card(state, action.tier)
        self._apply_take_action(state, player, action)

    def _apply_reserve_deck_action(
        self,
        state: SplendorState,
        player: PlayerState,
        action: Action,
    ) -> None:
        if action.tier is None:
            raise ValueError("Reserve deck action requires a tier.")

        card = state.hidden_tier_decks[action.tier].pop(0)
        player.reserved_cards.append(card)
        state.deck_counts[action.tier] = len(state.hidden_tier_decks[action.tier])
        self._apply_take_action(state, player, action)

    def _apply_buy_visible_action(
        self,
        state: SplendorState,
        player: PlayerState,
        action: Action,
    ) -> Card:
        if action.tier is None or action.market_index is None:
            raise ValueError("Buy visible action requires tier and market index.")

        card = state.visible_tier_cards[action.tier].pop(action.market_index)
        self._pay_tokens(state, player, action.spend_tokens)
        self._award_card(player, card)
        self._draw_replacement_card(state, action.tier)
        return card

    def _apply_buy_reserved_action(
        self,
        state: SplendorState,
        player: PlayerState,
        action: Action,
    ) -> Card:
        if action.reserved_index is None:
            raise ValueError("Buy reserved action requires reserved index.")

        card = player.reserved_cards.pop(action.reserved_index)
        self._pay_tokens(state, player, action.spend_tokens)
        self._award_card(player, card)
        return card

    def _pay_tokens(
        self,
        state: SplendorState,
        player: PlayerState,
        spend_tokens: tuple[str, ...],
    ) -> None:
        for color in spend_tokens:
            player.tokens[color] -= 1
            state.bank_tokens[color] += 1

    def _award_card(self, player: PlayerState, card: Card) -> None:
        player.purchased_cards.append(card)
        player.bonuses[card.bonus_color] += 1
        player.score += card.points

    def _draw_replacement_card(self, state: SplendorState, tier: int) -> None:
        if state.hidden_tier_decks[tier]:
            state.visible_tier_cards[tier].append(state.hidden_tier_decks[tier].pop(0))
        state.deck_counts[tier] = len(state.hidden_tier_decks[tier])

    def _claimable_nobles_after_purchase(
        self,
        player: PlayerState,
        available_nobles: list,
        purchased_card: Card,
    ) -> list:
        bonuses = _bonuses_with_card(player, purchased_card)
        claimable: list = []
        for noble in available_nobles:
            if all(bonuses.get(color, 0) >= required for color, required in noble.requirement.items()):
                claimable.append(noble)
        return claimable

    def _apply_noble_resolution(
        self,
        state: SplendorState,
        player: PlayerState,
        purchased_card: Card,
        action: Action,
    ) -> None:
        if action.noble_id is None:
            return

        chosen_noble = next((noble for noble in state.nobles if noble.noble_id == action.noble_id), None)
        if chosen_noble is None:
            raise ValueError(f"Chosen noble is not claimable: {action.noble_id}")
        if not all(
            player.bonuses.get(color, 0) >= required
            for color, required in chosen_noble.requirement.items()
        ):
            raise ValueError(f"Chosen noble is not claimable: {action.noble_id}")

        state.nobles.remove(chosen_noble)
        player.nobles.append(chosen_noble)
        player.score += chosen_noble.points

    def _finalize_terminal_state(self, state: SplendorState) -> None:
        state.terminal = True
        scores = [player.score for player in state.players]
        best_score = max(scores)
        contenders = [idx for idx, score in enumerate(scores) if score == best_score]
        if len(contenders) == 1:
            state.winner = contenders[0]
            return

        fewest_cards = min(len(state.players[idx].purchased_cards) for idx in contenders)
        fewest_card_contenders = [
            idx for idx in contenders if len(state.players[idx].purchased_cards) == fewest_cards
        ]
        state.winner = fewest_card_contenders[0] if len(fewest_card_contenders) == 1 else None
