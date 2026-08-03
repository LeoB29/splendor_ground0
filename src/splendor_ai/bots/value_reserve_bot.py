"""Deterministic scripted teacher encoding an expert's stated Splendor strategy.

Experiment `072`. Protocol: `docs/experiments/scripted_value_teacher_072.md`.

The rules were elicited verbatim from an experienced player who beat champion
`035` 8-2, and are frozen in that protocol. Nothing here is learned, searched, or
random. The point is to test whether the *strategy* wins in this engine, since no
member of the local opponent league plays it and the public-belief resolver
inherits `035`'s blind spot by construction.

Two quantities are kept deliberately distinct:

* **effective cost** -- card cost minus the bonuses I already own. Discounts only.
* **deficit** -- effective cost minus the tokens and gold I hold.

Targets are ranked by ``points / effective cost``, never by deficit. Ranking by
deficit would chase whatever the bot happens to hold tokens for, which inverts the
elicited strategy.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from splendor_ai.engine.actions import Action, ActionType
from splendor_ai.engine.constants import MAX_RESERVED_CARDS, TARGET_PRESTIGE, TOKEN_COLORS
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.engine.state import Card, PlayerState, SplendorState


#: Effective value at or above which a card is worth reserving. At zero bonuses
#: this admits exactly 18 of the 90 base cards; it widens as bonuses accumulate.
EFFICIENCY_FILTER = 0.40

#: Own turns a hand target may stagnate before it stops steering token takes.
STALE_TARGET_TURNS = 10


@dataclass(frozen=True, slots=True)
class ValueReserveRules:
    """The two elicited changes introduced by `073`.

    Both default to disabled so that `072` remains bit-reproducible from this
    same code. `073` constructs the bot with both enabled and changes nothing
    else. See `docs/experiments/scripted_value_teacher_073.md`.
    """

    #: Spend gold only to win, to buy the current best hand target, or when the
    #: gold still held afterwards covers that target's colour shortfall.
    protect_gold: bool = False
    #: Reserve only when no card in hand passes the efficiency filter, so the bot
    #: carries one live target and must convert before reserving again.
    reserve_and_clear: bool = False


def effective_cost(card: Card, bonuses: dict[str, int]) -> int:
    """Tokens still required for `card` after existing bonuses, ignoring tokens."""
    return sum(
        max(0, card.cost.get(color, 0) - bonuses.get(color, 0))
        for color in TOKEN_COLORS
    )


def effective_value(card: Card, bonuses: dict[str, int]) -> float:
    """Points per still-required token. Infinite when the card is already free."""
    remaining = effective_cost(card, bonuses)
    if remaining <= 0:
        return float("inf")
    return card.points / remaining


def colour_deficit(card: Card, player: PlayerState) -> dict[str, int]:
    """Per-colour tokens still needed after bonuses and held colour tokens."""
    return {
        color: max(
            0,
            card.cost.get(color, 0)
            - player.bonuses.get(color, 0)
            - player.tokens.get(color, 0),
        )
        for color in TOKEN_COLORS
    }


def deficit(card: Card, player: PlayerState) -> int:
    """Total tokens still needed after bonuses, colour tokens and gold."""
    return max(
        0,
        sum(colour_deficit(card, player).values()) - player.tokens.get("gold", 0),
    )


def can_afford(card: Card, player: PlayerState) -> bool:
    return deficit(card, player) == 0


def _sort_key(card: Card, bonuses: dict[str, int]) -> tuple[float, int, int, str]:
    """Descending effective value, then points, then fewest cost colours."""
    colours = sum(1 for color in TOKEN_COLORS if card.cost.get(color, 0) > 0)
    return (-effective_value(card, bonuses), -card.points, colours, card.card_id)


@dataclass(slots=True)
class ValueReserveTelemetry:
    """Faithfulness diagnostics. Gated on nothing; never a tuning target."""

    decisions: int = 0
    rule_counts: dict[str, int] = field(default_factory=dict)
    action_counts: dict[str, int] = field(default_factory=dict)
    reserves_by_tier: dict[int, int] = field(default_factory=dict)
    purchases_by_tier: dict[int, int] = field(default_factory=dict)
    reserves: int = 0
    reserved_purchases: int = 0
    deck_reserves: int = 0
    nobles_claimed: int = 0
    gold_spent: int = 0
    double_takes: int = 0
    triple_takes: int = 0
    gold_discipline_skips: int = 0
    multi_target_hand_decisions: int = 0

    def reset(self) -> None:
        self.decisions = 0
        self.rule_counts = {}
        self.action_counts = {}
        self.reserves_by_tier = {}
        self.purchases_by_tier = {}
        self.reserves = 0
        self.reserved_purchases = 0
        self.deck_reserves = 0
        self.nobles_claimed = 0
        self.gold_spent = 0
        self.double_takes = 0
        self.triple_takes = 0
        self.gold_discipline_skips = 0
        self.multi_target_hand_decisions = 0

    def as_dict(self) -> dict[str, object]:
        return {
            "decisions": self.decisions,
            "rule_counts": dict(sorted(self.rule_counts.items())),
            "action_counts": dict(sorted(self.action_counts.items())),
            "reserves_by_tier": {str(k): v for k, v in sorted(self.reserves_by_tier.items())},
            "purchases_by_tier": {
                str(k): v for k, v in sorted(self.purchases_by_tier.items())
            },
            "reserves": self.reserves,
            "reserved_purchases": self.reserved_purchases,
            "deck_reserves": self.deck_reserves,
            "nobles_claimed": self.nobles_claimed,
            "gold_spent": self.gold_spent,
            "double_takes": self.double_takes,
            "triple_takes": self.triple_takes,
            "gold_discipline_skips": self.gold_discipline_skips,
            "multi_target_hand_decisions": self.multi_target_hand_decisions,
        }

    def merge(self, other: dict[str, object]) -> None:
        for scalar in (
            "decisions",
            "reserves",
            "reserved_purchases",
            "deck_reserves",
            "nobles_claimed",
            "gold_spent",
            "double_takes",
            "triple_takes",
            "gold_discipline_skips",
            "multi_target_hand_decisions",
        ):
            setattr(self, scalar, getattr(self, scalar) + int(other[scalar]))
        for name in ("rule_counts", "action_counts"):
            target = getattr(self, name)
            for key, count in dict(other[name]).items():
                target[key] = target.get(key, 0) + int(count)
        for name in ("reserves_by_tier", "purchases_by_tier"):
            target = getattr(self, name)
            for key, count in dict(other[name]).items():
                tier = int(key)
                target[tier] = target.get(tier, 0) + int(count)


class ValueReserveTeacherBot:
    """Rule-based value-and-reserve Splendor strategy. Deterministic."""

    def __init__(self, rules: ValueReserveRules | None = None) -> None:
        self.rules = rules or ValueReserveRules()
        self.telemetry = ValueReserveTelemetry()
        self._target_history: dict[str, tuple[int, int]] = {}
        self._own_turns = 0

    @property
    def loop_fallback_triggers(self) -> int:
        return 0

    def start_game(self, seed: int) -> None:  # noqa: ARG002 - lifecycle hook
        self._target_history = {}
        self._own_turns = 0

    # ---------------------------------------------------------------- helpers

    @staticmethod
    def _group(actions: list[Action]) -> dict[ActionType, list[Action]]:
        grouped: dict[ActionType, list[Action]] = {}
        for action in actions:
            grouped.setdefault(action.action_type, []).append(action)
        return grouped

    @staticmethod
    def _variant_key(action: Action) -> tuple[int, int, str]:
        """Rule 10: prefer claiming a noble, then least gold, then fewest tokens."""
        spend = list(action.spend_tokens or ())
        gold = sum(1 for token in spend if token == "gold")
        return (0 if action.noble_id is not None else 1, gold, len(spend))

    def _best_variant(self, variants: list[Action]) -> Action:
        return min(variants, key=self._variant_key)

    @staticmethod
    def _colour_shortfall(card: Card, player: PlayerState) -> int:
        """Gold that would be needed to buy `card` right now."""
        return sum(colour_deficit(card, player).values())

    def _gold_allowed(
        self,
        action: Action,
        card: Card,
        player: PlayerState,
        hand_target: Card | None,
    ) -> bool:
        """`073` change 1. Gold is protected for the best hand target."""
        if not self.rules.protect_gold:
            return True
        gold = sum(1 for token in (action.spend_tokens or ()) if token == "gold")
        if gold == 0:
            return True
        if player.score + card.points >= TARGET_PRESTIGE:
            return True
        if hand_target is not None and card.card_id == hand_target.card_id:
            return True
        if hand_target is None:
            return True
        remaining = player.tokens.get("gold", 0) - gold
        return remaining >= self._colour_shortfall(hand_target, player)

    def _affordable_under_gold_discipline(
        self,
        grouped_buys: dict[Any, list[Action]],
        card_of: Any,
        player: PlayerState,
        hand_target: Card | None,
    ) -> dict[Any, list[Action]]:
        """Drop purchases whose cheapest variant would break gold discipline."""
        if not self.rules.protect_gold:
            return grouped_buys
        kept: dict[Any, list[Action]] = {}
        for key, variants in grouped_buys.items():
            card = card_of(key)
            allowed = [
                action
                for action in variants
                if self._gold_allowed(action, card, player, hand_target)
            ]
            if allowed:
                kept[key] = allowed
            else:
                self.telemetry.gold_discipline_skips += 1
        return kept

    def _visible(self, state: SplendorState, action: Action) -> Card:
        return state.visible_tier_cards[action.tier][action.market_index]

    def _hand(self, player: PlayerState, action: Action) -> Card:
        return player.reserved_cards[action.reserved_index]

    def _stale(
        self,
        card: Card,
        player: PlayerState,
        best_visible_value: float,
    ) -> bool:
        """Rule 9: no progress in STALE_TARGET_TURNS own turns *and* a strictly
        better visible card exists. Both conditions are required by the protocol."""
        record = self._target_history.get(card.card_id)
        if record is None:
            return False
        first_turn, first_deficit = record
        if self._own_turns - first_turn < STALE_TARGET_TURNS:
            return False
        if deficit(card, player) < first_deficit:
            return False
        return best_visible_value > effective_value(card, player.bonuses)

    def _track(self, player: PlayerState) -> None:
        live = set()
        for card in player.reserved_cards:
            live.add(card.card_id)
            if card.card_id not in self._target_history:
                self._target_history[card.card_id] = (
                    self._own_turns,
                    deficit(card, player),
                )
        for card_id in list(self._target_history):
            if card_id not in live:
                del self._target_history[card_id]

    def _visible_cards(
        self,
        state: SplendorState,
        player: PlayerState,
    ) -> list[Card]:
        return [
            card
            for cards in state.visible_tier_cards.values()
            for card in cards
            if card is not None
            and effective_value(card, player.bonuses) >= EFFICIENCY_FILTER
        ]

    def _best_hand_target(
        self,
        state: SplendorState,
        player: PlayerState,
    ) -> Card | None:
        if not player.reserved_cards:
            return None
        visible = self._visible_cards(state, player)
        best_visible_value = max(
            (effective_value(card, player.bonuses) for card in visible),
            default=0.0,
        )
        fresh = [
            card
            for card in player.reserved_cards
            if not self._stale(card, player, best_visible_value)
        ]
        pool = fresh or list(player.reserved_cards)
        return sorted(pool, key=lambda card: _sort_key(card, player.bonuses))[0]

    def _best_target(self, state: SplendorState, player: PlayerState) -> Card | None:
        """Best hand target, else the best filter-passing visible card."""
        hand = self._best_hand_target(state, player)
        if hand is not None:
            return hand
        visible = self._visible_cards(state, player)
        if not visible:
            return None
        return sorted(visible, key=lambda card: _sort_key(card, player.bonuses))[0]

    def _record(self, rule: str, action: Action, state: SplendorState) -> Action:
        telemetry = self.telemetry
        telemetry.decisions += 1
        telemetry.rule_counts[rule] = telemetry.rule_counts.get(rule, 0) + 1
        name = action.action_type.name
        telemetry.action_counts[name] = telemetry.action_counts.get(name, 0) + 1
        if action.noble_id is not None:
            telemetry.nobles_claimed += 1
        telemetry.gold_spent += sum(
            1 for token in (action.spend_tokens or ()) if token == "gold"
        )
        if action.action_type is ActionType.RESERVE_VISIBLE:
            telemetry.reserves += 1
            tier = int(action.tier)
            telemetry.reserves_by_tier[tier] = telemetry.reserves_by_tier.get(tier, 0) + 1
        elif action.action_type is ActionType.RESERVE_DECK:
            telemetry.deck_reserves += 1
        elif action.action_type is ActionType.BUY_VISIBLE:
            tier = int(action.tier)
            telemetry.purchases_by_tier[tier] = (
                telemetry.purchases_by_tier.get(tier, 0) + 1
            )
        elif action.action_type is ActionType.BUY_RESERVED:
            telemetry.reserved_purchases += 1
            card = state.players[state.current_player].reserved_cards[
                action.reserved_index
            ]
            telemetry.purchases_by_tier[card.tier] = (
                telemetry.purchases_by_tier.get(card.tier, 0) + 1
            )
        elif action.action_type is ActionType.TAKE_TOKENS:
            taken = list(action.take_tokens or ())
            if len(taken) == 2 and len(set(taken)) == 1:
                telemetry.double_takes += 1
            elif len(taken) >= 3:
                telemetry.triple_takes += 1
        return action

    # ------------------------------------------------------------------ rules

    def choose_action(
        self,
        env: SplendorEnv,
        state: SplendorState,
        legal_actions: list[Action] | None = None,
    ) -> Action | None:
        actions = legal_actions if legal_actions is not None else env.legal_actions(state)
        if not actions:
            return None
        player = state.players[state.current_player]
        opponent = state.players[1 - state.current_player]
        self._own_turns += 1
        self._track(player)
        grouped = self._group(actions)
        bonuses = player.bonuses

        buy_hand: dict[int, list[Action]] = {}
        for action in grouped.get(ActionType.BUY_RESERVED, []):
            buy_hand.setdefault(action.reserved_index, []).append(action)
        buy_visible: dict[tuple[int, int], list[Action]] = {}
        for action in grouped.get(ActionType.BUY_VISIBLE, []):
            buy_visible.setdefault((action.tier, action.market_index), []).append(action)
        reserve_visible: dict[tuple[int, int], list[Action]] = {}
        for action in grouped.get(ActionType.RESERVE_VISIBLE, []):
            reserve_visible.setdefault((action.tier, action.market_index), []).append(
                action
            )

        # Rule 2: win outright.
        for key, variants in buy_hand.items():
            card = player.reserved_cards[key]
            if player.score + card.points >= TARGET_PRESTIGE:
                return self._record("r2_win_now", self._best_variant(variants), state)
        for key, variants in buy_visible.items():
            card = state.visible_tier_cards[key[0]][key[1]]
            if player.score + card.points >= TARGET_PRESTIGE:
                return self._record("r2_win_now", self._best_variant(variants), state)

        # Rule 3: endgame block. Deny a visible card that lets the opponent reach 15.
        if len(player.reserved_cards) < MAX_RESERVED_CARDS:
            threats = [
                (key, variants)
                for key, variants in reserve_visible.items()
                if (
                    lambda card: opponent.score + card.points >= TARGET_PRESTIGE
                    and can_afford(card, opponent)
                )(state.visible_tier_cards[key[0]][key[1]])
            ]
            if threats:
                threats.sort(
                    key=lambda item: _sort_key(
                        state.visible_tier_cards[item[0][0]][item[0][1]], bonuses
                    )
                )
                return self._record(
                    "r3_endgame_block",
                    self._best_variant(threats[0][1]),
                    state,
                )

        # Rule 4: convert from hand.
        if buy_hand:
            best = min(
                buy_hand.items(),
                key=lambda item: (
                    -player.reserved_cards[item[0]].points,
                    _sort_key(player.reserved_cards[item[0]], bonuses),
                ),
            )
            return self._record("r4_buy_hand", self._best_variant(best[1]), state)

        target = self._best_target(state, player)
        hand_target = self._best_hand_target(state, player)

        # `073` change 1: gold is protected for the best hand target. Rules 5 and 6
        # may no longer burn it on cheap cards, which is what paralysed `072`.
        buy_visible = self._affordable_under_gold_discipline(
            buy_visible,
            lambda key: state.visible_tier_cards[key[0]][key[1]],
            player,
            hand_target,
        )

        # `073` change 2 diagnostic: how often the hand carries more than one live
        # target. Must be zero when reserve_and_clear is enabled.
        live_targets = sum(
            1
            for card in player.reserved_cards
            if effective_value(card, bonuses) >= EFFICIENCY_FILTER
        )
        if live_targets > 1:
            self.telemetry.multi_target_hand_decisions += 1

        # Rule 5: buy an affordable visible card at least as efficient as my best
        # *hand* target. With an empty hand the efficiency filter is the floor.
        if buy_visible:
            floor = (
                effective_value(hand_target, bonuses)
                if hand_target is not None
                else EFFICIENCY_FILTER
            )
            candidates = [
                (key, variants)
                for key, variants in buy_visible.items()
                if effective_value(state.visible_tier_cards[key[0]][key[1]], bonuses)
                >= floor
            ]
            if candidates:
                candidates.sort(
                    key=lambda item: _sort_key(
                        state.visible_tier_cards[item[0][0]][item[0][1]], bonuses
                    )
                )
                return self._record(
                    "r5_buy_visible_efficient",
                    self._best_variant(candidates[0][1]),
                    state,
                )

            # Rule 6: buy an affordable card whose bonus cuts my target's
            # *effective cost*, i.e. its cost after bonuses, not after tokens held.
            if target is not None:
                residual = {
                    color: max(0, target.cost.get(color, 0) - bonuses.get(color, 0))
                    for color in TOKEN_COLORS
                }
                helpful = [
                    (key, variants)
                    for key, variants in buy_visible.items()
                    if residual.get(
                        state.visible_tier_cards[key[0]][key[1]].bonus_color, 0
                    )
                    > 0
                ]
                if helpful:
                    helpful.sort(
                        key=lambda item: _sort_key(
                            state.visible_tier_cards[item[0][0]][item[0][1]], bonuses
                        )
                    )
                    return self._record(
                        "r6_buy_discount_builder",
                        self._best_variant(helpful[0][1]),
                        state,
                    )

        # Rule 7: reserve the best filter-passing card, even if unaffordable.
        # `073` change 2: only when no card in hand is already a live target, so the
        # bot converts before reserving again instead of stockpiling slots.
        may_reserve = len(player.reserved_cards) < MAX_RESERVED_CARDS and (
            not self.rules.reserve_and_clear or live_targets == 0
        )
        if reserve_visible and may_reserve:
            candidates = [
                (key, variants)
                for key, variants in reserve_visible.items()
                if effective_value(state.visible_tier_cards[key[0]][key[1]], bonuses)
                >= EFFICIENCY_FILTER
            ]
            if candidates:
                candidates.sort(
                    key=lambda item: _sort_key(
                        state.visible_tier_cards[item[0][0]][item[0][1]], bonuses
                    )
                )
                return self._record(
                    "r7_reserve_high_value",
                    self._best_variant(candidates[0][1]),
                    state,
                )

        # Rule 8: take tokens toward the target's deficit.
        takes = grouped.get(ActionType.TAKE_TOKENS, [])
        if takes:
            need = (
                colour_deficit(target, player)
                if target is not None
                else {color: 1 for color in TOKEN_COLORS}
            )

            def score(action: Action) -> tuple[float, int, int, str]:
                gained = 0
                remaining = dict(need)
                for token in action.take_tokens or ():
                    if remaining.get(token, 0) > 0:
                        gained += 1
                        remaining[token] -= 1
                returned = sum(
                    need.get(token, 0) for token in (action.return_tokens or ())
                )
                return (
                    -(gained - returned),
                    len(action.return_tokens or ()),
                    -len(action.take_tokens or ()),
                    repr(action),
                )

            return self._record("r8_take_tokens", min(takes, key=score), state)

        # Rules 10-11: anything else legal, deterministically; pass last.
        for kind in (
            ActionType.BUY_VISIBLE,
            ActionType.BUY_RESERVED,
            ActionType.RESERVE_VISIBLE,
            ActionType.PASS,
            ActionType.RESERVE_DECK,
        ):
            if kind in grouped:
                rule = "r11_pass" if kind is ActionType.PASS else "r10_fallback"
                return self._record(
                    rule,
                    self._best_variant(grouped[kind]),
                    state,
                )
        return self._record("r10_fallback", actions[0], state)
