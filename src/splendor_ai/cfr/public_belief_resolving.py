"""Information-set-consistent two-ply public-belief resolving tools."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any

from splendor_ai.bots.hidden_information import determinize_for_observer
from splendor_ai.engine.state import SplendorState


@dataclass(frozen=True, slots=True)
class ResolvedLeaf:
    determinization: int
    root_action: tuple[object, ...]
    opponent_information_set: tuple[object, ...]
    opponent_action: tuple[object, ...]
    discovery_payoff: float
    evaluation_payoff: float


@dataclass(frozen=True, slots=True)
class TwoPlyResolution:
    root_actions: tuple[tuple[object, ...], ...]
    action_values: tuple[float, ...]
    regrets_from_best: tuple[float, ...]
    best_index: int
    opponent_choices: tuple[dict[str, Any], ...]

    def to_payload(self) -> dict[str, Any]:
        return {
            "root_actions": [repr(action) for action in self.root_actions],
            "action_values": list(self.action_values),
            "regrets_from_best": list(self.regrets_from_best),
            "best_index": self.best_index,
            "best_action": repr(self.root_actions[self.best_index]),
            "opponent_choices": list(self.opponent_choices),
        }


def deterministic_belief_range(
    state: SplendorState,
    observer: int,
    *,
    root_index: int,
    count: int = 3,
    seed_start: int = 49_000_000,
) -> tuple[SplendorState, ...]:
    if count <= 0:
        raise ValueError("count must be positive.")
    return tuple(
        determinize_for_observer(
            state,
            observer,
            random.Random(seed_start + root_index * 10_003 + index),
        )
        for index in range(count)
    )


def resolve_two_ply(
    leaves: list[ResolvedLeaf],
    root_actions: tuple[tuple[object, ...], ...],
    determinization_count: int,
    *,
    payoff: str,
) -> TwoPlyResolution:
    """Solve root values with one shared minimizing response per opponent infoset."""

    if payoff not in ("discovery_payoff", "evaluation_payoff"):
        raise ValueError("Unknown payoff table.")
    if determinization_count <= 0 or not root_actions:
        raise ValueError("Resolver requires roots and determinizations.")
    values: list[float] = []
    choices: list[dict[str, Any]] = []
    for root_action in root_actions:
        root_leaves = [leaf for leaf in leaves if leaf.root_action == root_action]
        present = {leaf.determinization for leaf in root_leaves}
        if present != set(range(determinization_count)):
            raise ValueError("Every root action must exist in every determinization.")
        groups: dict[tuple[object, ...], list[ResolvedLeaf]] = {}
        for leaf in root_leaves:
            groups.setdefault(leaf.opponent_information_set, []).append(leaf)
        root_total = 0.0
        for information_set, group in sorted(groups.items(), key=lambda item: repr(item[0])):
            determinizations = sorted({leaf.determinization for leaf in group})
            action_sets = {
                leaf.determinization: tuple(
                    sorted(
                        {
                            item.opponent_action
                            for item in group
                            if item.determinization == leaf.determinization
                        },
                        key=repr,
                    )
                )
                for leaf in group
            }
            first_actions = action_sets[determinizations[0]]
            if any(actions != first_actions for actions in action_sets.values()):
                raise ValueError("Opponent information set has unstable semantic actions.")
            action_means = []
            for action in first_actions:
                records = [leaf for leaf in group if leaf.opponent_action == action]
                if len(records) != len(determinizations):
                    raise ValueError("Opponent action is missing from an information history.")
                action_means.append(
                    sum(float(getattr(leaf, payoff)) for leaf in records) / len(records)
                )
            selected_index = min(
                range(len(first_actions)),
                key=lambda index: (action_means[index], repr(first_actions[index])),
            )
            selected_action = first_actions[selected_index]
            selected_records = [leaf for leaf in group if leaf.opponent_action == selected_action]
            contribution = sum(float(getattr(leaf, payoff)) for leaf in selected_records)
            root_total += contribution / determinization_count
            choices.append(
                {
                    "root_action": repr(root_action),
                    "opponent_information_set": repr(information_set),
                    "determinizations": determinizations,
                    "opponent_actions": [repr(action) for action in first_actions],
                    "action_means": action_means,
                    "selected_action": repr(selected_action),
                }
            )
        values.append(root_total)
    best_index = max(range(len(values)), key=lambda index: (values[index], -index))
    best_value = values[best_index]
    return TwoPlyResolution(
        root_actions=root_actions,
        action_values=tuple(values),
        regrets_from_best=tuple(value - best_value for value in values),
        best_index=best_index,
        opponent_choices=tuple(choices),
    )
