"""Exact strategic analysis for finite imperfect-information game adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypeAlias

from .reduced_splendor import (
    ActionKey,
    InformationKey,
    ReducedSplendorCFRGame,
    ReducedSplendorCFRState,
)
from .tabular import expected_utility, strategy_total_variation


Policy: TypeAlias = dict[InformationKey, tuple[float, ...]]


@dataclass(slots=True)
class BestResponseResult:
    player: int
    value: float
    policy: Policy
    payload: dict[str, Any]


def exact_best_response(
    game: ReducedSplendorCFRGame,
    *,
    player: int,
    opponent_policy: Policy,
) -> BestResponseResult:
    """Compute one information-set-consistent pure best response."""

    if player not in (0, 1):
        raise ValueError("player must be 0 or 1")
    graph = _enumerate_graph(game)
    counterfactual_reach = _counterfactual_reaches(
        game,
        excluded_player=player,
        fixed_policy=opponent_policy,
    )
    grouped: dict[InformationKey, list[tuple[ReducedSplendorCFRState, int]]] = {}
    action_sets: dict[InformationKey, tuple[ActionKey, ...]] = {}
    for state, depth in graph:
        if game.is_chance_node(state) or game.is_terminal(state):
            continue
        if game.current_player(state) != player:
            continue
        key = game.information_state_key(state)
        actions = game.legal_action_keys(state)
        previous = action_sets.setdefault(key, actions)
        if previous != actions:
            raise ValueError("Unstable action set in exact best response.")
        grouped.setdefault(key, []).append((state, depth))

    depths: dict[InformationKey, int] = {}
    for key, histories in grouped.items():
        observed_depths = {depth for _state, depth in histories}
        if len(observed_depths) != 1:
            raise ValueError("One information set spans multiple decision depths.")
        depths[key] = observed_depths.pop()

    selected: dict[InformationKey, ActionKey] = {}
    action_values: dict[InformationKey, tuple[float, ...]] = {}
    counterfactual_masses: dict[InformationKey, float] = {}
    for key in sorted(grouped, key=lambda item: (-depths[item], repr(item))):
        histories = grouped[key]
        actions = action_sets[key]
        mass = sum(counterfactual_reach[id(state)] for state, _depth in histories)
        numerators: list[float] = []
        for action in actions:
            numerator = 0.0
            memo: dict[int, float] = {}
            for state, _depth in histories:
                numerator += counterfactual_reach[id(state)] * _response_continuation_value(
                    game,
                    game.apply_action(state, action),
                    responding_player=player,
                    opponent_policy=opponent_policy,
                    selected_actions=selected,
                    memo=memo,
                )
            numerators.append(numerator)
        best_index = max(range(len(actions)), key=lambda index: (numerators[index], -index))
        selected[key] = actions[best_index]
        counterfactual_masses[key] = mass
        action_values[key] = tuple(
            numerator / mass if mass > 0.0 else 0.0 for numerator in numerators
        )

    pure_policy: Policy = {}
    for key, actions in action_sets.items():
        chosen = selected[key]
        pure_policy[key] = tuple(1.0 if action == chosen else 0.0 for action in actions)

    p0_value = (
        expected_utility(game, pure_policy, opponent_policy)
        if player == 0
        else expected_utility(game, opponent_policy, pure_policy)
    )
    response_value = p0_value if player == 0 else -p0_value
    selections = []
    for key in sorted(grouped, key=repr):
        actions = action_sets[key]
        selections.append(
            {
                "information_key": repr(key),
                "depth": depths[key],
                "history_count": len(grouped[key]),
                "counterfactual_mass": counterfactual_masses[key],
                "actions": [repr(action) for action in actions],
                "conditional_action_values": list(action_values[key]),
                "selected_action": repr(selected[key]),
            }
        )
    payload = {
        "player": player,
        "response_value_for_player": response_value,
        "response_value_p0": p0_value,
        "selected_information_sets": len(selected),
        "expected_information_sets": len(grouped),
        "missing_selections": len(grouped.keys() - selected.keys()),
        "unstable_action_sets": 0,
        "selections": selections,
    }
    return BestResponseResult(
        player=player,
        value=response_value,
        policy=pure_policy,
        payload=payload,
    )


def profile_exploitability(
    game: ReducedSplendorCFRGame,
    policy0: Policy,
    policy1: Policy,
) -> dict[str, Any]:
    profile_value_p0 = expected_utility(game, policy0, policy1)
    response0 = exact_best_response(game, player=0, opponent_policy=policy1)
    response1 = exact_best_response(game, player=1, opponent_policy=policy0)
    incentive0 = response0.value - profile_value_p0
    incentive1 = response1.value + profile_value_p0
    nash_conv = incentive0 + incentive1
    return {
        "profile_value_p0": profile_value_p0,
        "profile_value_p1": -profile_value_p0,
        "best_response_value_p0": response0.value,
        "best_response_value_p1": response1.value,
        "deviation_incentive_p0": incentive0,
        "deviation_incentive_p1": incentive1,
        "nash_conv": nash_conv,
        "exploitability": nash_conv / 2.0,
        "best_responses": {
            "0": response0.payload,
            "1": response1.payload,
        },
    }


def discrepant_action_value_analysis(
    game: ReducedSplendorCFRGame,
    *,
    exact_policy: Policy,
    external_policy: Policy,
    exact_nodes: dict[InformationKey, Any],
    external_nodes: dict[InformationKey, Any],
    observation_floor: int = 100,
    tv_floor: float = 0.30,
    unreachable_floor: float = 0.0001,
    loss_ceiling: float = 0.02,
) -> dict[str, Any]:
    graph = _enumerate_graph(game)
    grouped: dict[InformationKey, list[ReducedSplendorCFRState]] = {}
    for state, _depth in graph:
        if game.is_chance_node(state) or game.is_terminal(state):
            continue
        grouped.setdefault(game.information_state_key(state), []).append(state)

    exact_reach = _profile_reaches(game, exact_policy, exact_policy)
    external_reach = _profile_reaches(game, external_policy, external_policy)
    fixed_value_memos = {
        player: _fixed_policy_value_function(game, player, exact_policy, exact_policy)
        for player in (0, 1)
    }
    counterfactual_reaches = {
        player: _counterfactual_reaches(
            game,
            excluded_player=player,
            fixed_policy=exact_policy,
        )
        for player in (0, 1)
    }

    discrepancy_keys = sorted(
        (
            key
            for key in exact_nodes.keys() & external_nodes.keys()
            if external_nodes[key].average_observations >= observation_floor
            and strategy_total_variation(
                exact_nodes[key].average_strategy(),
                external_nodes[key].average_strategy(),
            )
            > tv_floor
        ),
        key=repr,
    )
    records: list[dict[str, Any]] = []
    weighted_loss = {"0": 0.0, "1": 0.0}
    for key in discrepancy_keys:
        histories = grouped[key]
        actor = game.current_player(histories[0])
        actions = game.legal_action_keys(histories[0])
        mass = sum(counterfactual_reaches[actor][id(state)] for state in histories)
        q_values = []
        for action in actions:
            numerator = sum(
                counterfactual_reaches[actor][id(state)]
                * fixed_value_memos[actor](game.apply_action(state, action))
                for state in histories
            )
            q_values.append(numerator / mass if mass > 0.0 else 0.0)
        exact_strategy = exact_nodes[key].average_strategy()
        external_strategy = external_nodes[key].average_strategy()
        exact_value = sum(p * q for p, q in zip(exact_strategy, q_values, strict=True))
        external_value = sum(p * q for p, q in zip(external_strategy, q_values, strict=True))
        positive_loss = max(exact_value - external_value, 0.0)
        exact_probability = sum(exact_reach[id(state)] for state in histories)
        external_probability = sum(external_reach[id(state)] for state in histories)
        strategic_weight = max(exact_probability, external_probability)
        weighted = strategic_weight * positive_loss
        weighted_loss[str(actor)] += weighted
        records.append(
            {
                "information_key": repr(key),
                "player": actor,
                "external_observations": external_nodes[key].average_observations,
                "raw_total_variation": strategy_total_variation(
                    exact_strategy,
                    external_strategy,
                ),
                "counterfactual_mass": mass,
                "actions": [repr(action) for action in actions],
                "conditional_action_values": q_values,
                "action_value_range": max(q_values) - min(q_values),
                "exact_mixture_value": exact_value,
                "external_mixture_value": external_value,
                "positive_external_loss": positive_loss,
                "exact_profile_reach": exact_probability,
                "external_profile_reach": external_probability,
                "conservative_strategic_weight": strategic_weight,
                "reach_weighted_positive_loss": weighted,
                "resolved": strategic_weight < unreachable_floor
                or positive_loss <= loss_ceiling,
            }
        )
    return {
        "observation_floor": observation_floor,
        "raw_tv_floor": tv_floor,
        "unreachable_floor": unreachable_floor,
        "loss_ceiling": loss_ceiling,
        "discrepancy_count": len(records),
        "resolved_count": sum(record["resolved"] for record in records),
        "all_resolved": all(record["resolved"] for record in records),
        "reach_weighted_positive_loss_by_player": weighted_loss,
        "records": records,
    }


def _enumerate_graph(
    game: ReducedSplendorCFRGame,
) -> list[tuple[ReducedSplendorCFRState, int]]:
    graph: list[tuple[ReducedSplendorCFRState, int]] = []

    def visit(state: ReducedSplendorCFRState, depth: int) -> None:
        graph.append((state, depth))
        if game.is_terminal(state):
            return
        if game.is_chance_node(state):
            for outcome, _probability in game.chance_outcomes(state):
                visit(game.apply_chance(state, outcome), depth)
            return
        for action in game.legal_action_keys(state):
            visit(game.apply_action(state, action), depth + 1)

    visit(game.root(), 0)
    return graph


def _strategy(
    game: ReducedSplendorCFRGame,
    state: ReducedSplendorCFRState,
    policy: Policy,
) -> tuple[float, ...]:
    actions = game.legal_action_keys(state)
    strategy = policy.get(game.information_state_key(state))
    if strategy is None:
        return tuple([1.0 / len(actions)] * len(actions))
    if len(strategy) != len(actions):
        raise ValueError("Policy strategy does not match the legal action set.")
    if abs(sum(strategy) - 1.0) > 1e-9 or any(probability < 0.0 for probability in strategy):
        raise ValueError("Policy strategy is not a probability distribution.")
    return strategy


def _counterfactual_reaches(
    game: ReducedSplendorCFRGame,
    *,
    excluded_player: int,
    fixed_policy: Policy,
) -> dict[int, float]:
    reaches: dict[int, float] = {}

    def visit(state: ReducedSplendorCFRState, reach: float) -> None:
        reaches[id(state)] = reach
        if game.is_terminal(state):
            return
        if game.is_chance_node(state):
            for outcome, probability in game.chance_outcomes(state):
                visit(game.apply_chance(state, outcome), reach * probability)
            return
        actor = game.current_player(state)
        actions = game.legal_action_keys(state)
        if actor == excluded_player:
            for action in actions:
                visit(game.apply_action(state, action), reach)
            return
        for action, probability in zip(actions, _strategy(game, state, fixed_policy), strict=True):
            visit(game.apply_action(state, action), reach * probability)

    visit(game.root(), 1.0)
    return reaches


def _profile_reaches(
    game: ReducedSplendorCFRGame,
    policy0: Policy,
    policy1: Policy,
) -> dict[int, float]:
    reaches: dict[int, float] = {}

    def visit(state: ReducedSplendorCFRState, reach: float) -> None:
        reaches[id(state)] = reach
        if game.is_terminal(state):
            return
        if game.is_chance_node(state):
            for outcome, probability in game.chance_outcomes(state):
                visit(game.apply_chance(state, outcome), reach * probability)
            return
        actor = game.current_player(state)
        policy = policy0 if actor == 0 else policy1
        for action, probability in zip(
            game.legal_action_keys(state),
            _strategy(game, state, policy),
            strict=True,
        ):
            visit(game.apply_action(state, action), reach * probability)

    visit(game.root(), 1.0)
    return reaches


def _response_continuation_value(
    game: ReducedSplendorCFRGame,
    state: ReducedSplendorCFRState,
    *,
    responding_player: int,
    opponent_policy: Policy,
    selected_actions: dict[InformationKey, ActionKey],
    memo: dict[int, float],
) -> float:
    cached = memo.get(id(state))
    if cached is not None:
        return cached
    if game.is_terminal(state):
        value = game.utility(state, responding_player)
    elif game.is_chance_node(state):
        value = sum(
            probability
            * _response_continuation_value(
                game,
                game.apply_chance(state, outcome),
                responding_player=responding_player,
                opponent_policy=opponent_policy,
                selected_actions=selected_actions,
                memo=memo,
            )
            for outcome, probability in game.chance_outcomes(state)
        )
    elif game.current_player(state) == responding_player:
        key = game.information_state_key(state)
        if key not in selected_actions:
            raise ValueError("A downstream best-response information set is unresolved.")
        value = _response_continuation_value(
            game,
            game.apply_action(state, selected_actions[key]),
            responding_player=responding_player,
            opponent_policy=opponent_policy,
            selected_actions=selected_actions,
            memo=memo,
        )
    else:
        value = sum(
            probability
            * _response_continuation_value(
                game,
                game.apply_action(state, action),
                responding_player=responding_player,
                opponent_policy=opponent_policy,
                selected_actions=selected_actions,
                memo=memo,
            )
            for action, probability in zip(
                game.legal_action_keys(state),
                _strategy(game, state, opponent_policy),
                strict=True,
            )
        )
    memo[id(state)] = value
    return value


def _fixed_policy_value_function(
    game: ReducedSplendorCFRGame,
    player: int,
    policy0: Policy,
    policy1: Policy,
):
    memo: dict[int, float] = {}

    def value(state: ReducedSplendorCFRState) -> float:
        cached = memo.get(id(state))
        if cached is not None:
            return cached
        if game.is_terminal(state):
            result = game.utility(state, player)
        elif game.is_chance_node(state):
            result = sum(
                probability * value(game.apply_chance(state, outcome))
                for outcome, probability in game.chance_outcomes(state)
            )
        else:
            actor = game.current_player(state)
            policy = policy0 if actor == 0 else policy1
            result = sum(
                probability * value(game.apply_action(state, action))
                for action, probability in zip(
                    game.legal_action_keys(state),
                    _strategy(game, state, policy),
                    strict=True,
                )
            )
        memo[id(state)] = result
        return result

    return value
