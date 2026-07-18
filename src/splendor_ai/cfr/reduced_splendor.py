"""Finite imperfect-information Splendor duel for CFR correctness tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

from splendor_ai.engine.actions import Action
from splendor_ai.encoding.information_state import (
    PublicActionEvent,
    canonical_information_state_history_key,
    public_action_event,
)
from splendor_ai.encoding.public_belief import (
    actions_by_canonical_descriptor,
    canonical_legal_action_descriptor,
)
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.engine.state import Card, PlayerState, SplendorState, empty_token_bank


ActionKey: TypeAlias = tuple[object, ...]
InformationKey: TypeAlias = tuple[object, ...]

V_RED = Card("V_RED", 1, "red", 1, {"red": 1})
V_BLUE = Card("V_BLUE", 1, "blue", 1, {"blue": 1})
H_MIX = Card("H_MIX", 1, "green", 2, {"red": 1, "blue": 1})
H_FREE = Card("H_FREE", 1, "black", 1, {})
HIDDEN_ORDERS = ((H_MIX, H_FREE), (H_FREE, H_MIX))


@dataclass(frozen=True, slots=True)
class ReducedSplendorCFRState:
    state: SplendorState | None
    public_history: tuple[PublicActionEvent, ...] = ()
    chance_pending: bool = True


class ReducedSplendorCFRGame:
    """Four-action real-engine duel with a hidden two-card root chance node."""

    max_player_actions = 4

    def __init__(self) -> None:
        self.env = SplendorEnv(seed=0)
        self._root = ReducedSplendorCFRState(state=None, chance_pending=True)
        self._chance_cache: dict[int, ReducedSplendorCFRState] = {}
        self._action_cache: dict[int, dict[ActionKey, Action]] = {}
        self._successor_cache: dict[tuple[int, ActionKey], ReducedSplendorCFRState] = {}
        self._information_key_cache: dict[tuple[int, int], InformationKey] = {}

    def root(self) -> ReducedSplendorCFRState:
        return self._root

    def is_chance_node(self, node: ReducedSplendorCFRState) -> bool:
        return node.chance_pending

    def chance_outcomes(self, node: ReducedSplendorCFRState) -> tuple[tuple[int, float], ...]:
        if not self.is_chance_node(node):
            return ()
        return ((0, 0.5), (1, 0.5))

    def apply_chance(self, node: ReducedSplendorCFRState, outcome: int) -> ReducedSplendorCFRState:
        if not self.is_chance_node(node) or outcome not in (0, 1):
            raise ValueError(f"Illegal reduced-game chance outcome: {outcome}")
        child = self._chance_cache.get(outcome)
        if child is None:
            child = ReducedSplendorCFRState(
                state=self._initial_state(HIDDEN_ORDERS[outcome]),
                public_history=(),
                chance_pending=False,
            )
            self._chance_cache[outcome] = child
        return child

    def is_terminal(self, node: ReducedSplendorCFRState) -> bool:
        return node.state is not None and node.state.terminal

    def current_player(self, node: ReducedSplendorCFRState) -> int:
        if node.state is None or node.chance_pending or node.state.terminal:
            raise ValueError("Node has no acting player.")
        return node.state.current_player

    def utility(self, node: ReducedSplendorCFRState, player_id: int) -> float:
        if not self.is_terminal(node) or node.state is None:
            raise ValueError("Utility is available only at terminal nodes.")
        if node.state.winner is None:
            return 0.0
        return 1.0 if node.state.winner == player_id else -1.0

    def information_state_key(
        self,
        node: ReducedSplendorCFRState,
        player_id: int | None = None,
    ) -> InformationKey:
        if node.state is None or node.chance_pending:
            raise ValueError("Chance root has no player information state.")
        observer = node.state.current_player if player_id is None else player_id
        cache_key = (id(node), observer)
        key = self._information_key_cache.get(cache_key)
        if key is None:
            key = canonical_information_state_history_key(
                node.state,
                observer,
                node.public_history,
            )
            self._information_key_cache[cache_key] = key
        return key

    def legal_action_keys(self, node: ReducedSplendorCFRState) -> tuple[ActionKey, ...]:
        if node.state is None or node.chance_pending or node.state.terminal:
            return ()
        return tuple(sorted(self._concrete_actions(node).keys(), key=repr))

    def apply_action(
        self,
        node: ReducedSplendorCFRState,
        action_key: ActionKey,
    ) -> ReducedSplendorCFRState:
        if node.state is None or node.chance_pending or node.state.terminal:
            raise ValueError("Cannot apply a player action at this node.")
        cache_key = (id(node), action_key)
        cached = self._successor_cache.get(cache_key)
        if cached is not None:
            return cached
        actions = self._concrete_actions(node)
        if action_key not in actions:
            raise ValueError(f"Illegal reduced-game action key: {action_key!r}")
        action = actions[action_key]
        event = public_action_event(node.state, action)
        next_state = self.env.step(node.state, action)
        if not next_state.terminal and next_state.turn_index >= self.max_player_actions:
            _adjudicate(next_state)
        successor = ReducedSplendorCFRState(
            state=next_state,
            public_history=node.public_history + (event,),
            chance_pending=False,
        )
        self._successor_cache[cache_key] = successor
        return successor

    def _concrete_actions(self, node: ReducedSplendorCFRState) -> dict[ActionKey, Action]:
        cached = self._action_cache.get(id(node))
        if cached is None:
            assert node.state is not None
            cached = actions_by_canonical_descriptor(
                node.state,
                self.env.legal_actions(node.state),
            )
            self._action_cache[id(node)] = cached
        return cached

    def enumerate_tree_contract(self) -> dict[str, int]:
        """Exhaustively verify finite paths, legality, and stable info actions."""

        action_sets: dict[InformationKey, tuple[ActionKey, ...]] = {}
        terminals = 0
        nodes = 0
        max_depth = 0

        def visit(node: ReducedSplendorCFRState, depth: int) -> None:
            nonlocal terminals, nodes, max_depth
            nodes += 1
            max_depth = max(max_depth, depth)
            if self.is_terminal(node):
                terminals += 1
                if depth != self.max_player_actions:
                    raise ValueError(f"Unexpected terminal depth {depth}.")
                return
            if self.is_chance_node(node):
                outcomes = self.chance_outcomes(node)
                if abs(sum(probability for _outcome, probability in outcomes) - 1.0) > 1e-12:
                    raise ValueError("Chance probabilities do not sum to one.")
                for outcome, _probability in outcomes:
                    visit(self.apply_chance(node, outcome), depth)
                return
            key = self.information_state_key(node)
            actions = self.legal_action_keys(node)
            if not actions:
                raise ValueError("Nonterminal player node has no legal action.")
            previous = action_sets.setdefault(key, actions)
            if previous != actions:
                raise ValueError("Information set exposes inconsistent action sets.")
            for action in actions:
                visit(self.apply_action(node, action), depth + 1)

        visit(self.root(), 0)
        return {
            "nodes": nodes,
            "terminal_paths": terminals,
            "information_sets": len(action_sets),
            "max_player_actions": max_depth,
        }

    @staticmethod
    def _initial_state(hidden_order: tuple[Card, Card]) -> SplendorState:
        bank = empty_token_bank()
        bank["red"] = 2
        bank["blue"] = 2
        bank["gold"] = 1
        return SplendorState(
            current_player=0,
            bank_tokens=bank,
            players=[PlayerState(0), PlayerState(1)],
            visible_tier_cards={1: [V_RED, V_BLUE], 2: [], 3: []},
            hidden_tier_decks={1: list(hidden_order), 2: [], 3: []},
            deck_counts={1: 2, 2: 0, 3: 0},
            nobles=[],
            turn_index=0,
            start_player=0,
        )


def _adjudicate(state: SplendorState) -> None:
    state.terminal = True
    scores = [player.score for player in state.players]
    best_score = max(scores)
    contenders = [index for index, score in enumerate(scores) if score == best_score]
    if len(contenders) == 1:
        state.winner = contenders[0]
        return
    fewest = min(len(state.players[index].purchased_cards) for index in contenders)
    finalists = [
        index for index in contenders if len(state.players[index].purchased_cards) == fewest
    ]
    state.winner = finalists[0] if len(finalists) == 1 else None
