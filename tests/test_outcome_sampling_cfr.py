from __future__ import annotations

from dataclasses import replace
import random

import pytest

from splendor_ai.cfr import (
    ExactCFRSolver,
    OutcomeSamplingAdvantageRow,
    OutcomeSamplingMCCFRSolver,
    ReducedSplendorCFRGame,
    behavior_mixture,
    exact_fixed_policy_regret_increments,
    finite_learning_terminal_reason,
    sample_reduced_outcome_updates,
)
from splendor_ai.cfr.full_game import build_schema_row
from splendor_ai.encoding.entity_encoding import ENTITY_HIDDEN_DECK_CARD
from splendor_ai.engine.env import SplendorEnv


def test_behavior_mixture_has_full_support_even_for_zero_target_actions() -> None:
    behavior = behavior_mixture((1.0, 0.0, 0.0, 0.0), exploration=0.1)

    assert behavior == pytest.approx((0.925, 0.025, 0.025, 0.025))
    assert min(behavior) > 0.0
    assert sum(behavior) == pytest.approx(1.0)


def test_outcome_estimator_matches_exact_fixed_uniform_root() -> None:
    game = ReducedSplendorCFRGame()
    exact = exact_fixed_policy_regret_increments(game, traverser=0)
    lookup = ExactCFRSolver(game)
    rng = random.Random(4800480)
    samples = 20_000
    sums: dict[object, list[float]] = {}
    for _ in range(samples):
        updates, _utility, _logs = sample_reduced_outcome_updates(
            game,
            0,
            rng,
            lookup._node,
            lambda _state, node: tuple([1.0 / len(node.actions)] * len(node.actions)),
            exploration=0.1,
        )
        for update in updates:
            values = sums.setdefault(update.information_key, [0.0] * len(update.actions))
            for index, value in enumerate(update.regret_estimate):
                values[index] += value

    root_key = min(exact, key=lambda key: len(key[-1]))
    estimated = [value / samples for value in sums[root_key]]

    assert estimated == pytest.approx(exact[root_key], abs=0.01)


def test_adaptive_outcome_solver_is_deterministic_and_legal() -> None:
    first = OutcomeSamplingMCCFRSolver(ReducedSplendorCFRGame(), seed=480046).run(250)
    second = OutcomeSamplingMCCFRSolver(ReducedSplendorCFRGame(), seed=480046).run(250)

    assert first == second
    assert first["illegal_actions"] == 0
    assert first["unstable_action_sets"] == 0
    assert first["sampled_trajectories"] == 500


def test_finite_learning_terminal_contract_has_frozen_priority() -> None:
    state = SplendorEnv(seed=48).initial_state()
    state.turn_index = 200

    assert finite_learning_terminal_reason(
        state,
        forced_pass_streak=2,
        repetition_count=4,
        no_progress_streak=60,
    ) == "forced_pass_deadlock"
    assert finite_learning_terminal_reason(
        state,
        forced_pass_streak=0,
        repetition_count=4,
        no_progress_streak=60,
    ) == "repetition_adjudication"
    assert finite_learning_terminal_reason(
        state,
        forced_pass_streak=0,
        repetition_count=1,
        no_progress_streak=60,
    ) == "no_progress_adjudication"
    assert finite_learning_terminal_reason(
        state,
        forced_pass_streak=0,
        repetition_count=1,
        no_progress_streak=0,
    ) == "turn_limit_adjudication"


def test_outcome_advantage_row_rejects_privileged_state() -> None:
    env = SplendorEnv(seed=48)
    state = env.initial_state()
    schema = build_schema_row(state, (), env.legal_actions(state))
    count = len(schema.actions.action_type_ids)
    row = OutcomeSamplingAdvantageRow(
        state=schema.state,
        history=schema.history,
        actions=schema.actions,
        target_strategy=tuple([1.0 / count] * count),
        behavior_strategy=tuple([1.0 / count] * count),
        selected_index=0,
        regret_target=tuple([0.0] * count),
        player=0,
        turn=0,
        terminal_reason="completed",
        log_importance_weight=0.0,
    )
    privileged = replace(
        row.state,
        entity_type_ids=(ENTITY_HIDDEN_DECK_CARD,) + row.state.entity_type_ids[1:],
    )

    with pytest.raises(ValueError, match="Privileged entities"):
        replace(row, state=privileged)
