"""Diagnose action-conditional credit with matched entity-policy continuations."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from splendor_ai.bots.hidden_information import determinize_for_observer
from splendor_ai.diagnostics import is_progress_transition, state_signature
from splendor_ai.encoding import (
    ActionCodec,
    actions_by_canonical_successor_key,
    collate_entity_action_batch,
)
from splendor_ai.engine.actions import Action
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.engine.state import SplendorState
from splendor_ai.training.entity_actor_critic import (
    ActorCriticConfig,
    ActorStep,
    collect_league_rollouts,
    load_entity_actor,
    normalized_score_margin,
)
from splendor_ai.training.replay import serialize_action
from splendor_ai.training.run_action_value_training import _spearman
from splendor_ai.training.run_entity_return_ab import _full_league
from splendor_ai.training.run_policy_search_disagreement import ModelActionScore
from splendor_ai.training.run_rollout_action_credit import _sample_scored_action


@dataclass(frozen=True, slots=True)
class EntityCounterfactualConfig:
    parent_checkpoint: Path
    checkpoint_016: Path
    checkpoint_035: Path
    output_path: Path
    rollout_games: int = 36
    seed_start: int = 180_000
    max_states: int = 18
    policy_top_k: int = 3
    max_candidates: int = 5
    determinizations: int = 2
    continuations_per_determinization: int = 4
    continuation_epsilon: float = 0.1
    continuation_temperature: float = 0.5
    continuation_top_k: int = 3
    continuation_max_additional_turns: int = 120
    score_margin_scale: float = 20.0
    minimum_action_range: float = 0.05
    minimum_advantage_magnitude: float = 0.01
    minimum_informative_fraction: float = 0.50
    minimum_split_top1_agreement: float = 0.50
    minimum_split_spearman_lower95: float = 0.0
    minimum_chosen_advantage_correlation_lower95: float = 0.0
    minimum_chosen_advantage_sign_agreement: float = 0.60
    minimum_sign_eligible_states: int = 5
    minimum_chosen_advantage_std: float = 0.03
    minimum_advantage_tail_fraction: float = 0.10
    maximum_continuation_cutoff_fraction: float = 0.01
    bootstrap_samples: int = 5_000
    device: str = "cpu"


@dataclass(frozen=True, slots=True)
class ContinuationResult:
    value: float
    terminal: bool
    turns: int


class EntityPolicyScorer:
    """Score canonical successor actions for stochastic continuation rollouts."""

    def __init__(self, checkpoint: Path, device: str) -> None:
        self.device = device
        self.codec = ActionCodec()
        self.model = load_entity_actor(checkpoint, device)

    def score_actions(
        self,
        state: SplendorState,
        legal_actions: list[Action],
    ) -> tuple[ModelActionScore, ...]:
        actions = list(actions_by_canonical_successor_key(state, legal_actions).values())
        batch = collate_entity_action_batch([(state, actions)], device=self.device)
        with torch.no_grad():
            scores, _ = self.model(**batch)
        values = scores[0, : len(actions)].detach().cpu().tolist()
        ranked = [
            ModelActionScore(
                action=action,
                action_index=self.codec.encode(state, action),
                score=float(value),
            )
            for action, value in zip(actions, values, strict=True)
        ]
        return tuple(sorted(ranked, key=lambda item: (-item.score, item.action_index)))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-checkpoint", required=True)
    parser.add_argument("--checkpoint-016", required=True)
    parser.add_argument("--checkpoint-035", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--rollout-games", type=int, default=36)
    parser.add_argument("--seed-start", type=int, default=180_000)
    parser.add_argument("--max-states", type=int, default=18)
    parser.add_argument("--policy-top-k", type=int, default=3)
    parser.add_argument("--max-candidates", type=int, default=5)
    parser.add_argument("--determinizations", type=int, default=2)
    parser.add_argument("--continuations-per-determinization", type=int, default=4)
    parser.add_argument("--continuation-epsilon", type=float, default=0.1)
    parser.add_argument("--continuation-temperature", type=float, default=0.5)
    parser.add_argument("--continuation-top-k", type=int, default=3)
    parser.add_argument("--continuation-max-additional-turns", type=int, default=120)
    parser.add_argument("--minimum-action-range", type=float, default=0.05)
    parser.add_argument("--minimum-informative-fraction", type=float, default=0.50)
    parser.add_argument("--minimum-split-top1-agreement", type=float, default=0.50)
    parser.add_argument("--minimum-chosen-advantage-sign-agreement", type=float, default=0.60)
    parser.add_argument("--minimum-sign-eligible-states", type=int, default=5)
    parser.add_argument("--minimum-chosen-advantage-std", type=float, default=0.03)
    parser.add_argument("--minimum-advantage-tail-fraction", type=float, default=0.10)
    parser.add_argument("--maximum-continuation-cutoff-fraction", type=float, default=0.01)
    parser.add_argument("--bootstrap-samples", type=int, default=5_000)
    parser.add_argument("--device", default="cpu")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = EntityCounterfactualConfig(
        parent_checkpoint=Path(args.parent_checkpoint),
        checkpoint_016=Path(args.checkpoint_016),
        checkpoint_035=Path(args.checkpoint_035),
        output_path=Path(args.output_path),
        rollout_games=args.rollout_games,
        seed_start=args.seed_start,
        max_states=args.max_states,
        policy_top_k=args.policy_top_k,
        max_candidates=args.max_candidates,
        determinizations=args.determinizations,
        continuations_per_determinization=args.continuations_per_determinization,
        continuation_epsilon=args.continuation_epsilon,
        continuation_temperature=args.continuation_temperature,
        continuation_top_k=args.continuation_top_k,
        continuation_max_additional_turns=args.continuation_max_additional_turns,
        minimum_action_range=args.minimum_action_range,
        minimum_informative_fraction=args.minimum_informative_fraction,
        minimum_split_top1_agreement=args.minimum_split_top1_agreement,
        minimum_chosen_advantage_sign_agreement=(
            args.minimum_chosen_advantage_sign_agreement
        ),
        minimum_sign_eligible_states=args.minimum_sign_eligible_states,
        minimum_chosen_advantage_std=args.minimum_chosen_advantage_std,
        minimum_advantage_tail_fraction=args.minimum_advantage_tail_fraction,
        maximum_continuation_cutoff_fraction=args.maximum_continuation_cutoff_fraction,
        bootstrap_samples=args.bootstrap_samples,
        device=args.device,
    )
    payload = run_counterfactual_credit_diagnostic(config)
    print(
        json.dumps(
            {
                "gates": payload["gates"],
                "all_gates_passed": payload["all_gates_passed"],
            },
            indent=2,
        )
    )


def run_counterfactual_credit_diagnostic(
    config: EntityCounterfactualConfig,
) -> dict[str, Any]:
    _validate_config(config)
    rollout_config = ActorCriticConfig(
        rollout_games=config.rollout_games,
        seed_start=config.seed_start,
        device=config.device,
    )
    league = _full_league(
        config.parent_checkpoint,
        config.checkpoint_016,
        config.checkpoint_035,
        config.device,
    )
    source_steps, rollout = collect_league_rollouts(
        load_entity_actor(config.parent_checkpoint, config.device),
        league,
        rollout_config,
    )
    selected_steps, selection = select_game_disjoint_phase_states(
        source_steps,
        config.max_states,
        config.seed_start + 91,
    )
    scorer = EntityPolicyScorer(config.parent_checkpoint, config.device)
    state_rows = [
        _evaluate_state(step, state_index, scorer, config)
        for state_index, step in enumerate(selected_steps)
    ]
    summary = summarize_counterfactual_states(
        state_rows,
        minimum_action_range=config.minimum_action_range,
        minimum_advantage_magnitude=config.minimum_advantage_magnitude,
        bootstrap_samples=config.bootstrap_samples,
        seed=config.seed_start + 193,
    )
    expected_per_phase = config.max_states // 3
    gates = {
        "clean_balanced_source_rollout": (
            rollout.timed_out_games == 0
            and rollout.stalled_games == 0
            and rollout.minimum_stratum_games == rollout.maximum_stratum_games
        ),
        "game_disjoint_phase_balanced_states": (
            len(state_rows) == config.max_states
            and len({row["game_index"] for row in state_rows}) == len(state_rows)
            and all(
                selection["selected_by_phase"].get(phase, 0) == expected_per_phase
                for phase in ("early", "mid", "late")
            )
        ),
        "continuations_complete": (
            summary["continuation_cutoff_fraction"]
            <= config.maximum_continuation_cutoff_fraction
        ),
        "action_signal_nondegenerate": (
            summary["informative_fraction"] >= config.minimum_informative_fraction
            and summary["chosen_advantage_std"] >= config.minimum_chosen_advantage_std
            and summary["chosen_advantage_positive_fraction"]
            >= config.minimum_advantage_tail_fraction
            and summary["chosen_advantage_negative_fraction"]
            >= config.minimum_advantage_tail_fraction
        ),
        "split_action_rankings_replicate": (
            summary["split_top1_agreement"] >= config.minimum_split_top1_agreement
            and summary["split_spearman_ci95"][0]
            > config.minimum_split_spearman_lower95
        ),
        "chosen_advantages_replicate": (
            summary["chosen_advantage_correlation_ci95"][0]
            > config.minimum_chosen_advantage_correlation_lower95
            and summary["chosen_advantage_sign_eligible_states"]
            >= config.minimum_sign_eligible_states
            and summary["chosen_advantage_sign_agreement"]
            >= config.minimum_chosen_advantage_sign_agreement
        ),
    }
    payload = {
        "diagnostic": "entity_action_conditional_counterfactual_credit",
        "config": _config_payload(config),
        "league_opponents": [label for label, _ in league],
        "rollout": asdict(rollout),
        "selection": selection,
        "summary": summary,
        "gates": gates,
        "all_gates_passed": all(gates.values()),
        "decision": (
            "Counterfactual advantages replicate; a separately gated matched "
            "policy-update experiment is eligible."
            if all(gates.values())
            else "Do not train: action-conditional advantages have not passed the stability gates."
        ),
        "states": state_rows,
    }
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    config.output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def select_game_disjoint_phase_states(
    steps: list[ActorStep] | tuple[ActorStep, ...],
    max_states: int,
    seed: int,
) -> tuple[list[ActorStep], dict[str, Any]]:
    if max_states <= 0 or max_states % 3:
        raise ValueError("max_states must be a positive multiple of three.")
    by_phase = {"early": [], "mid": [], "late": []}
    for step in steps:
        by_phase[_phase(step.state.turn_index)].append(step)
    rng = random.Random(seed)
    used_games: set[int] = set()
    selected: list[ActorStep] = []
    selected_by_phase: dict[str, int] = {}
    target = max_states // 3
    for phase in ("early", "mid", "late"):
        candidates = list(by_phase[phase])
        rng.shuffle(candidates)
        chosen: list[ActorStep] = []
        for step in candidates:
            if step.game_index in used_games:
                continue
            chosen.append(step)
            used_games.add(step.game_index)
            if len(chosen) == target:
                break
        selected.extend(chosen)
        selected_by_phase[phase] = len(chosen)
    rng.shuffle(selected)
    return selected, {
        "eligible_by_phase": {phase: len(rows) for phase, rows in by_phase.items()},
        "selected_by_phase": selected_by_phase,
        "unique_selected_games": len(used_games),
    }


def _evaluate_state(
    step: ActorStep,
    state_index: int,
    scorer: EntityPolicyScorer,
    config: EntityCounterfactualConfig,
) -> dict[str, Any]:
    env = SplendorEnv()
    codec = scorer.codec
    parent_scores = scorer.score_actions(step.state, list(step.legal_actions))
    chosen_index = codec.encode(step.state, step.legal_actions[step.action_row])
    candidates = _select_candidates(
        parent_scores,
        chosen_index,
        config.policy_top_k,
        config.max_candidates,
    )
    action_data = {
        item.action_index: {
            "score": item,
            "first_half": [],
            "second_half": [],
            "terminal": 0,
            "total": 0,
        }
        for item in candidates
    }
    half = config.continuations_per_determinization // 2
    for determinization_index in range(config.determinizations):
        determinized = determinize_for_observer(
            step.state,
            step.actor_seat,
            random.Random(
                config.seed_start + state_index * 1_000_003 + determinization_index
            ),
        )
        actions = list(
            actions_by_canonical_successor_key(
                determinized, env.legal_actions(determinized)
            ).values()
        )
        action_by_index = {codec.encode(determinized, action): action for action in actions}
        for candidate in candidates:
            action = action_by_index.get(candidate.action_index)
            if action is None:
                continue
            data = action_data[candidate.action_index]
            for continuation_index in range(config.continuations_per_determinization):
                result = _rollout_score_margin(
                    env,
                    scorer,
                    determinized,
                    action,
                    step.actor_seat,
                    seed=(
                        config.seed_start
                        + state_index * 10_000_019
                        + determinization_index * 100_003
                        + continuation_index
                    ),
                    epsilon=config.continuation_epsilon,
                    temperature=config.continuation_temperature,
                    top_k=config.continuation_top_k,
                    max_additional_turns=config.continuation_max_additional_turns,
                    score_margin_scale=config.score_margin_scale,
                )
                half_key = "first_half" if continuation_index < half else "second_half"
                data[half_key].append(result.value)
                data["terminal"] += int(result.terminal)
                data["total"] += 1
    actions = []
    for action_index, data in action_data.items():
        if not data["first_half"] or not data["second_half"]:
            continue
        item = data["score"]
        first_mean = statistics.fmean(data["first_half"])
        second_mean = statistics.fmean(data["second_half"])
        actions.append(
            {
                "action_index": action_index,
                "action_type": item.action.action_type.name,
                "action_payload": serialize_action(item.action),
                "prior_score": item.score,
                "chosen": action_index == chosen_index,
                "first_half_returns": data["first_half"],
                "second_half_returns": data["second_half"],
                "first_half_mean": first_mean,
                "second_half_mean": second_mean,
                "full_mean": (first_mean + second_mean) / 2.0,
                "terminal_continuations": data["terminal"],
                "total_continuations": data["total"],
            }
        )
    actions.sort(key=lambda row: int(row["action_index"]))
    return _state_metrics(step, actions, chosen_index)


def _state_metrics(
    step: ActorStep,
    actions: list[dict[str, Any]],
    chosen_index: int,
) -> dict[str, Any]:
    first = [float(row["first_half_mean"]) for row in actions]
    second = [float(row["second_half_mean"]) for row in actions]
    full = [float(row["full_mean"]) for row in actions]
    prior = [float(row["prior_score"]) for row in actions]
    first_top = max(range(len(actions)), key=lambda index: (first[index], -index))
    second_top = max(range(len(actions)), key=lambda index: (second[index], -index))
    full_top = max(range(len(actions)), key=lambda index: (full[index], -index))
    prior_top = max(range(len(actions)), key=lambda index: (prior[index], -index))
    chosen_row = next(
        index for index, row in enumerate(actions) if row["action_index"] == chosen_index
    )
    first_baseline = statistics.fmean(first)
    second_baseline = statistics.fmean(second)
    full_baseline = statistics.fmean(full)
    discovery_order = sorted(range(len(actions)), key=lambda index: (-first[index], index))
    discovery_best, discovery_runner = discovery_order[:2]
    validation_differences = [
        best - runner
        for best, runner in zip(
            actions[discovery_best]["second_half_returns"],
            actions[discovery_runner]["second_half_returns"],
            strict=True,
        )
    ]
    validation_delta = statistics.fmean(validation_differences)
    validation_se = _standard_error(validation_differences)
    total = sum(int(row["total_continuations"]) for row in actions)
    terminal = sum(int(row["terminal_continuations"]) for row in actions)
    return {
        "game_index": step.game_index,
        "turn_index": step.state.turn_index,
        "phase": _phase(step.state.turn_index),
        "actor_seat": step.actor_seat,
        "opponent_label": step.opponent_label,
        "candidate_count": len(actions),
        "chosen_action_index": chosen_index,
        "first_top_action_index": actions[first_top]["action_index"],
        "second_top_action_index": actions[second_top]["action_index"],
        "full_top_action_index": actions[full_top]["action_index"],
        "prior_top_action_index": actions[prior_top]["action_index"],
        "split_top1_agreement": first_top == second_top,
        "split_spearman": _spearman(first, second),
        "prior_return_spearman": _spearman(prior, full),
        "return_range": max(full) - min(full),
        "prior_regret": max(full) - full[prior_top],
        "chosen_advantage_first": first[chosen_row] - first_baseline,
        "chosen_advantage_second": second[chosen_row] - second_baseline,
        "chosen_advantage_full": full[chosen_row] - full_baseline,
        "discovery_best_validation_delta": validation_delta,
        "discovery_best_validation_se": validation_se,
        "discovery_best_replication_resolved": (
            validation_delta >= 0.05 and validation_delta > 1.96 * validation_se
        ),
        "terminal_continuations": terminal,
        "total_continuations": total,
        "actions": actions,
    }


def summarize_counterfactual_states(
    state_rows: list[dict[str, Any]],
    minimum_action_range: float,
    minimum_advantage_magnitude: float,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    if not state_rows:
        raise ValueError("Cannot summarize zero counterfactual states.")
    first_advantages = [float(row["chosen_advantage_first"]) for row in state_rows]
    second_advantages = [float(row["chosen_advantage_second"]) for row in state_rows]
    full_advantages = [float(row["chosen_advantage_full"]) for row in state_rows]
    split_spearmans = [float(row["split_spearman"]) for row in state_rows]
    sign_rows = [
        (first, second)
        for first, second in zip(first_advantages, second_advantages, strict=True)
        if abs(first) >= minimum_advantage_magnitude
        and abs(second) >= minimum_advantage_magnitude
    ]
    total_continuations = sum(int(row["total_continuations"]) for row in state_rows)
    terminal_continuations = sum(int(row["terminal_continuations"]) for row in state_rows)
    correlation = _pearson(first_advantages, second_advantages)
    return {
        "states": len(state_rows),
        "states_by_phase": {
            phase: sum(row["phase"] == phase for row in state_rows)
            for phase in ("early", "mid", "late")
        },
        "mean_candidate_count": statistics.fmean(
            int(row["candidate_count"]) for row in state_rows
        ),
        "informative_states": sum(
            float(row["return_range"]) >= minimum_action_range for row in state_rows
        ),
        "informative_fraction": statistics.fmean(
            float(row["return_range"]) >= minimum_action_range for row in state_rows
        ),
        "mean_return_range": statistics.fmean(
            float(row["return_range"]) for row in state_rows
        ),
        "split_top1_agreement": statistics.fmean(
            bool(row["split_top1_agreement"]) for row in state_rows
        ),
        "mean_split_spearman": statistics.fmean(split_spearmans),
        "split_spearman_ci95": _bootstrap_mean_ci(
            split_spearmans, bootstrap_samples, seed
        ),
        "chosen_advantage_correlation": correlation,
        "chosen_advantage_correlation_ci95": _bootstrap_correlation_ci(
            first_advantages,
            second_advantages,
            bootstrap_samples,
            seed + 1,
        ),
        "chosen_advantage_sign_eligible_states": len(sign_rows),
        "chosen_advantage_sign_agreement": (
            statistics.fmean((first > 0.0) == (second > 0.0) for first, second in sign_rows)
            if sign_rows
            else 0.0
        ),
        "chosen_advantage_mean": statistics.fmean(full_advantages),
        "chosen_advantage_std": (
            statistics.pstdev(full_advantages) if len(full_advantages) > 1 else 0.0
        ),
        "chosen_advantage_positive_fraction": statistics.fmean(
            value >= minimum_advantage_magnitude for value in full_advantages
        ),
        "chosen_advantage_negative_fraction": statistics.fmean(
            value <= -minimum_advantage_magnitude for value in full_advantages
        ),
        "mean_prior_return_spearman": statistics.fmean(
            float(row["prior_return_spearman"]) for row in state_rows
        ),
        "mean_prior_regret": statistics.fmean(
            float(row["prior_regret"]) for row in state_rows
        ),
        "prior_top_disagreement_fraction": statistics.fmean(
            row["prior_top_action_index"] != row["full_top_action_index"]
            for row in state_rows
        ),
        "replicated_discovery_best_states": sum(
            bool(row["discovery_best_replication_resolved"]) for row in state_rows
        ),
        "continuations": total_continuations,
        "continuation_cutoffs": total_continuations - terminal_continuations,
        "continuation_cutoff_fraction": (
            (total_continuations - terminal_continuations) / total_continuations
        ),
    }


def _select_candidates(
    parent_scores: tuple[ModelActionScore, ...],
    chosen_index: int,
    policy_top_k: int,
    max_candidates: int,
) -> tuple[ModelActionScore, ...]:
    selected = list(parent_scores[:policy_top_k])
    selected_indices = {item.action_index for item in selected}
    chosen = next(item for item in parent_scores if item.action_index == chosen_index)
    if chosen_index not in selected_indices and len(selected) < max_candidates:
        selected.append(chosen)
        selected_indices.add(chosen_index)
    for item in parent_scores:
        if len(selected) >= max_candidates:
            break
        if item.action_index in selected_indices:
            continue
        if any(existing.action.action_type == item.action.action_type for existing in selected):
            continue
        selected.append(item)
        selected_indices.add(item.action_index)
    for item in parent_scores:
        if len(selected) >= max_candidates:
            break
        if item.action_index not in selected_indices:
            selected.append(item)
            selected_indices.add(item.action_index)
    return tuple(selected)


def _rollout_score_margin(
    env: SplendorEnv,
    scorer: EntityPolicyScorer,
    state: SplendorState,
    root_action: Action,
    root_player: int,
    seed: int,
    epsilon: float,
    temperature: float,
    top_k: int,
    max_additional_turns: int,
    score_margin_scale: float,
) -> ContinuationResult:
    rng = random.Random(seed)
    rollout_state = env.step(state, root_action)
    seen: dict[tuple[object, ...], int] = {}
    no_progress_turns = 0
    start_turn = rollout_state.turn_index
    end_turn = start_turn + max_additional_turns
    while not rollout_state.terminal and rollout_state.turn_index < end_turn:
        signature = state_signature(rollout_state)
        seen[signature] = seen.get(signature, 0) + 1
        if seen[signature] >= 4 or no_progress_turns >= 60:
            break
        legal_actions = env.legal_actions(rollout_state)
        scores = scorer.score_actions(rollout_state, legal_actions)
        action = _sample_scored_action(
            scores,
            rng=rng,
            epsilon=epsilon,
            temperature=temperature,
            top_k=top_k,
        )
        next_state = env.step(rollout_state, action)
        no_progress_turns = (
            0
            if is_progress_transition(rollout_state, next_state, action)
            else no_progress_turns + 1
        )
        rollout_state = next_state
    margin = float(
        rollout_state.players[root_player].score
        - rollout_state.players[1 - root_player].score
    )
    return ContinuationResult(
        value=normalized_score_margin(margin, score_margin_scale),
        terminal=rollout_state.terminal,
        turns=rollout_state.turn_index - start_turn,
    )


def _phase(turn_index: int) -> str:
    if turn_index < 20:
        return "early"
    if turn_index < 40:
        return "mid"
    return "late"


def _standard_error(values: list[float]) -> float:
    return statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0


def _pearson(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        return 0.0
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    numerator = sum(
        (a - left_mean) * (b - right_mean)
        for a, b in zip(left, right, strict=True)
    )
    denominator = math.sqrt(
        sum((value - left_mean) ** 2 for value in left)
        * sum((value - right_mean) ** 2 for value in right)
    )
    return numerator / denominator if denominator else 0.0


def _bootstrap_mean_ci(
    values: list[float], samples: int, seed: int
) -> list[float]:
    rng = random.Random(seed)
    count = len(values)
    estimates = sorted(
        statistics.fmean(values[rng.randrange(count)] for _ in range(count))
        for _ in range(samples)
    )
    return [estimates[int(0.025 * (samples - 1))], estimates[int(0.975 * (samples - 1))]]


def _bootstrap_correlation_ci(
    left: list[float],
    right: list[float],
    samples: int,
    seed: int,
) -> list[float]:
    rng = random.Random(seed)
    count = len(left)
    estimates = []
    for _ in range(samples):
        indices = [rng.randrange(count) for _ in range(count)]
        estimates.append(
            _pearson([left[index] for index in indices], [right[index] for index in indices])
        )
    estimates.sort()
    return [estimates[int(0.025 * (samples - 1))], estimates[int(0.975 * (samples - 1))]]


def _validate_config(config: EntityCounterfactualConfig) -> None:
    if config.rollout_games <= 0 or config.rollout_games % 12:
        raise ValueError("rollout_games must be a positive multiple of 12.")
    if config.max_states <= 0 or config.max_states % 3:
        raise ValueError("max_states must be a positive multiple of three.")
    if config.policy_top_k <= 0 or config.max_candidates < config.policy_top_k + 1:
        raise ValueError("max_candidates must exceed positive policy_top_k.")
    if config.determinizations <= 0:
        raise ValueError("determinizations must be positive.")
    if (
        config.continuations_per_determinization < 2
        or config.continuations_per_determinization % 2
    ):
        raise ValueError("continuations_per_determinization must be positive and even.")
    if config.bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive.")


def _config_payload(config: EntityCounterfactualConfig) -> dict[str, Any]:
    payload = asdict(config)
    for key in ("parent_checkpoint", "checkpoint_016", "checkpoint_035", "output_path"):
        payload[key] = str(payload[key])
    return payload


if __name__ == "__main__":
    main()
