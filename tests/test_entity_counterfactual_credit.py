from __future__ import annotations

from splendor_ai.engine.actions import Action, ActionType
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.training.entity_actor_critic import ActorStep
from splendor_ai.training.run_entity_counterfactual_credit import (
    _select_candidates,
    _select_reference_candidates,
    build_parser,
    select_game_disjoint_phase_states,
    summarize_counterfactual_states,
)
from splendor_ai.training.run_entity_counterfactual_confirmation import (
    _phase_gate,
    build_parser as build_confirmation_parser,
)
from splendor_ai.training.run_policy_search_disagreement import ModelActionScore


def _step(game_index: int, turn_index: int) -> ActorStep:
    state = SplendorEnv(seed=game_index).initial_state()
    state.turn_index = turn_index
    return ActorStep(
        state=state,
        legal_actions=(Action(action_type=ActionType.PASS),),
        action_row=0,
        behavior_scores=(0.0,),
        behavior_probabilities=(1.0,),
        behavior_log_probability=0.0,
        behavior_entropy=0.0,
        actor_seat=0,
        game_index=game_index,
        opponent_label="test",
    )


def test_phase_selection_uses_one_state_per_game() -> None:
    steps = [
        _step(game_index, turn_index)
        for turn_index in (5, 25, 45)
        for game_index in range(9)
    ]

    selected, summary = select_game_disjoint_phase_states(steps, max_states=9, seed=7)

    assert len(selected) == 9
    assert len({step.game_index for step in selected}) == 9
    assert summary["selected_by_phase"] == {"early": 3, "mid": 3, "late": 3}

    selected, summary = select_game_disjoint_phase_states(
        steps,
        max_states=9,
        seed=7,
        phase_targets=(2, 3, 4),
    )

    assert len(selected) == 9
    assert len({step.game_index for step in selected}) == 9
    assert summary["selected_by_phase"] == {"early": 2, "mid": 3, "late": 4}


def test_candidate_selection_retains_sampled_behavior_action() -> None:
    action_types = (
        ActionType.TAKE_TOKENS,
        ActionType.RESERVE_VISIBLE,
        ActionType.BUY_VISIBLE,
        ActionType.BUY_RESERVED,
        ActionType.RESERVE_DECK,
    )
    scores = tuple(
        ModelActionScore(
            action=Action(action_type=action_type),
            action_index=index,
            score=float(10 - index),
        )
        for index, action_type in enumerate(action_types)
    )

    selected = _select_candidates(scores, chosen_index=4, policy_top_k=3, max_candidates=4)

    assert [item.action_index for item in selected[:3]] == [0, 1, 2]
    assert selected[3].action_index == 4

    references = _select_reference_candidates(scores, policy_top_k=3, max_candidates=4)
    assert [item.action_index for item in references] == [0, 1, 2, 3]
    assert 4 not in {item.action_index for item in references}


def test_counterfactual_summary_detects_replicated_advantages() -> None:
    advantages = (-0.20, -0.12, -0.05, 0.06, 0.14, 0.22)
    rows = [
        {
            "phase": ("early", "mid", "late")[index % 3],
            "candidate_count": 5,
            "return_range": 0.20,
            "split_top1_agreement": True,
            "split_spearman": 0.8,
            "chosen_advantage_first": advantage,
            "chosen_advantage_second": advantage * 0.9,
            "chosen_advantage_full": advantage * 0.95,
            "prior_return_spearman": 0.2,
            "prior_regret": 0.05,
            "prior_top_action_index": index,
            "full_top_action_index": index + int(index % 2 == 0),
            "discovery_best_replication_resolved": index < 2,
            "terminal_continuations": 40,
            "total_continuations": 40,
        }
        for index, advantage in enumerate(advantages)
    ]

    summary = summarize_counterfactual_states(
        rows,
        minimum_action_range=0.05,
        minimum_advantage_magnitude=0.01,
        bootstrap_samples=500,
        seed=11,
    )

    assert summary["informative_fraction"] == 1.0
    assert summary["split_top1_agreement"] == 1.0
    assert summary["chosen_advantage_correlation"] > 0.99
    assert summary["chosen_advantage_correlation_ci95"][0] > 0.0
    assert summary["chosen_advantage_sign_agreement"] == 1.0
    assert summary["continuation_cutoff_fraction"] == 0.0


def test_counterfactual_parser_preregisters_cpu_pilot_defaults() -> None:
    args = build_parser().parse_args(
        [
            "--parent-checkpoint",
            "parent.pt",
            "--checkpoint-016",
            "016.pt",
            "--checkpoint-035",
            "035.pt",
            "--output-path",
            "metrics.json",
        ]
    )

    assert args.rollout_games == 36
    assert args.max_states == 18
    assert args.determinizations == 2
    assert args.continuations_per_determinization == 4
    assert args.minimum_split_top1_agreement == 0.50


def test_confirmation_parser_and_phase_gate_are_independent() -> None:
    args = build_confirmation_parser().parse_args(
        [
            "--parent-checkpoint",
            "parent.pt",
            "--checkpoint-016",
            "016.pt",
            "--checkpoint-035",
            "035.pt",
            "--output-path",
            "metrics.json",
        ]
    )
    summary = {
        "informative_fraction": 0.75,
        "split_top1_agreement": 0.50,
        "split_spearman_ci95": [0.1, 0.8],
        "chosen_advantage_correlation_ci95": [0.05, 0.9],
        "chosen_advantage_sign_eligible_states": 8,
        "chosen_advantage_sign_agreement": 0.75,
        "chosen_advantage_std": 0.08,
        "chosen_advantage_positive_fraction": 0.25,
        "chosen_advantage_negative_fraction": 0.75,
        "replicated_discovery_best_states": 2,
    }

    gate = _phase_gate(summary, args)

    assert args.rollout_games == 48
    assert (args.early_control_states, args.mid_states, args.late_states) == (6, 12, 12)
    assert args.determinizations == 3
    assert args.continuations_per_determinization == 8
    assert gate["passed"]

    summary["chosen_advantage_correlation_ci95"] = [-0.01, 0.9]
    assert not _phase_gate(summary, args)["passed"]
