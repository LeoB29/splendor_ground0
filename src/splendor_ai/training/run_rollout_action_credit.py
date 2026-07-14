"""Estimate action-level credit with complete counterfactual game rollouts."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
import statistics
from typing import Any

from splendor_ai.diagnostics import is_progress_transition, state_signature
from splendor_ai.encoding import ActionCodec
from splendor_ai.engine.actions import Action, ActionType
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.engine.state import SplendorState
from splendor_ai.training.replay import (
    _adjudicate_scores,
    deserialize_state_snapshot,
    serialize_action,
)
from splendor_ai.training.run_policy_search_disagreement import _LegalActionScorer


@dataclass(frozen=True, slots=True)
class RolloutActionCreditConfig:
    checkpoint: Path
    replay_paths: tuple[Path, ...]
    output_path: Path
    summary_path: Path
    model_kind: str = "legal_action_attention_scorer"
    device: str = "cpu"
    max_states: int = 90
    root_policy_top_k: int = 4
    max_root_candidates: int = 6
    rollouts_per_action: int = 6
    changed_refinement_rollouts: int = 0
    rollout_epsilon: float = 0.1
    rollout_temperature: float = 0.5
    rollout_top_k: int = 3
    rollout_max_additional_turns: int = 120
    outcome_weight: float = 0.75
    score_margin_scale: float = 5.0
    min_action_return_range: float = 0.1
    champion_marker: str = "legal_action_scorer_022_attention_distill"
    seed: int = 33
    log_every_states: int = 10


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate root actions with complete stochastic policy rollouts."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--replay-path", action="append", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--summary-path", required=True)
    parser.add_argument("--model-kind", default="legal_action_attention_scorer")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-states", type=int, default=90)
    parser.add_argument("--root-policy-top-k", type=int, default=4)
    parser.add_argument("--max-root-candidates", type=int, default=6)
    parser.add_argument("--rollouts-per-action", type=int, default=6)
    parser.add_argument("--changed-refinement-rollouts", type=int, default=0)
    parser.add_argument("--rollout-epsilon", type=float, default=0.1)
    parser.add_argument("--rollout-temperature", type=float, default=0.5)
    parser.add_argument("--rollout-top-k", type=int, default=3)
    parser.add_argument("--rollout-max-additional-turns", type=int, default=120)
    parser.add_argument("--outcome-weight", type=float, default=0.75)
    parser.add_argument("--score-margin-scale", type=float, default=5.0)
    parser.add_argument("--min-action-return-range", type=float, default=0.1)
    parser.add_argument(
        "--champion-marker",
        default="legal_action_scorer_022_attention_distill",
    )
    parser.add_argument("--seed", type=int, default=33)
    parser.add_argument("--log-every-states", type=int, default=10)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = build_rollout_action_credit_corpus(
        RolloutActionCreditConfig(
            checkpoint=Path(args.checkpoint),
            replay_paths=tuple(Path(path) for path in args.replay_path),
            output_path=Path(args.output_path),
            summary_path=Path(args.summary_path),
            model_kind=args.model_kind,
            device=args.device,
            max_states=args.max_states,
            root_policy_top_k=args.root_policy_top_k,
            max_root_candidates=args.max_root_candidates,
            rollouts_per_action=args.rollouts_per_action,
            changed_refinement_rollouts=args.changed_refinement_rollouts,
            rollout_epsilon=args.rollout_epsilon,
            rollout_temperature=args.rollout_temperature,
            rollout_top_k=args.rollout_top_k,
            rollout_max_additional_turns=args.rollout_max_additional_turns,
            outcome_weight=args.outcome_weight,
            score_margin_scale=args.score_margin_scale,
            min_action_return_range=args.min_action_return_range,
            champion_marker=args.champion_marker,
            seed=args.seed,
            log_every_states=args.log_every_states,
        )
    )
    print(
        "Rollout action credit:"
        f" evaluated={summary['states_evaluated']}"
        f" written={summary['rows_written']}"
        f" changed={summary.get('rollout_best_changed', 0)}"
    )


def build_rollout_action_credit_corpus(
    config: RolloutActionCreditConfig,
) -> dict[str, Any]:
    _validate_config(config)
    scorer = _LegalActionScorer(
        config.checkpoint,
        config.device,
        config.model_kind,
    )
    env = SplendorEnv()
    codec = ActionCodec()
    selected_rows, selection_summary = _select_rows(config)
    counts: Counter[str] = Counter()
    phase_counts: Counter[str] = Counter()
    transition_counts: Counter[str] = Counter()
    examples: list[dict[str, Any]] = []

    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    config.summary_path.parent.mkdir(parents=True, exist_ok=True)
    with config.output_path.open("w", encoding="utf-8") as output:
        for state_index, payload in enumerate(selected_rows):
            snapshot = payload["state_snapshot"]
            state = deserialize_state_snapshot(snapshot)
            legal_actions = env.legal_actions(state)
            parent_scores = scorer.score_actions(state, legal_actions)
            candidates = select_root_candidates(
                parent_scores,
                policy_top_k=config.root_policy_top_k,
                max_candidates=config.max_root_candidates,
            )
            action_results = []
            for candidate in candidates:
                returns = [
                    rollout_action_return(
                        env=env,
                        scorer=scorer,
                        state=state,
                        root_action=candidate.action,
                        root_player=state.current_player,
                        seed=(
                            config.seed
                            + state_index * 100_000
                            + rollout_index
                        ),
                        epsilon=config.rollout_epsilon,
                        temperature=config.rollout_temperature,
                        top_k=config.rollout_top_k,
                        max_additional_turns=config.rollout_max_additional_turns,
                        outcome_weight=config.outcome_weight,
                        score_margin_scale=config.score_margin_scale,
                    )
                    for rollout_index in range(config.rollouts_per_action)
                ]
                action_results.append(
                    {
                        "score": candidate,
                        "returns": returns,
                        "mean": statistics.mean(returns),
                        "standard_error": _standard_error(returns),
                    }
                )
            action_results.sort(key=lambda item: item["mean"], reverse=True)
            parent_top_index = parent_scores[0].action_index
            initially_changed = (
                action_results[0]["score"].action_index != parent_top_index
            )
            if initially_changed and config.changed_refinement_rollouts > 0:
                counts["states_refined"] += 1
                for item in action_results:
                    additional_returns = [
                        rollout_action_return(
                            env=env,
                            scorer=scorer,
                            state=state,
                            root_action=item["score"].action,
                            root_player=state.current_player,
                            seed=(
                                config.seed
                                + state_index * 100_000
                                + config.rollouts_per_action
                                + rollout_offset
                            ),
                            epsilon=config.rollout_epsilon,
                            temperature=config.rollout_temperature,
                            top_k=config.rollout_top_k,
                            max_additional_turns=config.rollout_max_additional_turns,
                            outcome_weight=config.outcome_weight,
                            score_margin_scale=config.score_margin_scale,
                        )
                        for rollout_offset in range(
                            config.changed_refinement_rollouts
                        )
                    ]
                    item["returns"].extend(additional_returns)
                    item["mean"] = statistics.mean(item["returns"])
                    item["standard_error"] = _standard_error(item["returns"])
                action_results.sort(
                    key=lambda item: item["mean"],
                    reverse=True,
                )
            best = action_results[0]
            parent = next(
                item
                for item in action_results
                if item["score"].action_index == parent_top_index
            )
            return_range = float(best["mean"]) - float(
                action_results[-1]["mean"]
            )
            counts["states_evaluated"] += 1
            changed = best["score"].action_index != parent_top_index
            if changed:
                counts["rollout_best_changed"] += 1
            paired_differences = [
                best_return - parent_return
                for best_return, parent_return in zip(
                    best["returns"],
                    parent["returns"],
                    strict=True,
                )
            ]
            difference_mean = statistics.mean(paired_differences)
            difference_se = _standard_error(paired_differences)
            difference_lower95 = difference_mean - 1.96 * difference_se
            if return_range < config.min_action_return_range:
                counts["states_skipped_low_return_range"] += 1
                continue

            row = dict(payload)
            row.update(
                {
                    "action_index": parent_top_index,
                    "action_payload": serialize_action(parent_scores[0].action),
                    "legal_action_indices": list(
                        codec.legal_action_indices(state, legal_actions)
                    ),
                    "training_weight": max(return_range, 0.1),
                    "target_source": "complete_rollout_action_credit",
                    "search_action_scores": [
                        {
                            "rank": rank,
                            "action_index": item["score"].action_index,
                            "action_payload": serialize_action(
                                item["score"].action
                            ),
                            "value": item["mean"],
                            "standard_error": item["standard_error"],
                            "rollout_returns": item["returns"],
                        }
                        for rank, item in enumerate(action_results, start=1)
                    ],
                    "rollout_credit_metadata": {
                        "parent_top_action_index": parent_top_index,
                        "rollout_best_action_index": best["score"].action_index,
                        "rollout_best_changed": changed,
                        "best_minus_parent_mean": difference_mean,
                        "best_minus_parent_standard_error": difference_se,
                        "best_minus_parent_lower95": difference_lower95,
                        "action_return_range": return_range,
                        "rollouts_per_action": config.rollouts_per_action,
                        "refinement_rollouts": (
                            config.changed_refinement_rollouts
                            if initially_changed
                            else 0
                        ),
                    },
                }
            )
            output.write(json.dumps(row) + "\n")
            counts["rows_written"] += 1
            phase = _phase(int(payload.get("turn_index", 0)))
            phase_counts[phase] += 1
            transition = (
                f"{parent_scores[0].action.action_type.name} -> "
                f"{best['score'].action.action_type.name}"
            )
            transition_counts[transition] += 1
            if changed:
                examples.append(
                    {
                        "game_seed": payload.get("game_seed"),
                        "turn_index": payload.get("turn_index"),
                        "transition": transition,
                        "best_minus_parent_mean": difference_mean,
                        "best_minus_parent_lower95": difference_lower95,
                        "action_return_range": return_range,
                    }
                )
                examples.sort(
                    key=lambda item: item["best_minus_parent_lower95"],
                    reverse=True,
                )
                del examples[20:]
            if (
                config.log_every_states > 0
                and counts["states_evaluated"] % config.log_every_states == 0
            ):
                print(
                    f"[rollout-credit] evaluated={counts['states_evaluated']}"
                    f" written={counts['rows_written']}"
                    f" changed={counts['rollout_best_changed']}"
                )

    summary = {
        **selection_summary,
        **dict(counts),
        "states_evaluated": counts["states_evaluated"],
        "rows_written": counts["rows_written"],
        "rollout_best_changed": counts["rollout_best_changed"],
        "states_skipped_low_return_range": counts[
            "states_skipped_low_return_range"
        ],
        "states_refined": counts["states_refined"],
        "written_by_phase": dict(phase_counts),
        "best_transition_counts": dict(transition_counts),
        "examples": examples,
        "config": {
            **asdict(config),
            "checkpoint": str(config.checkpoint),
            "replay_paths": [str(path) for path in config.replay_paths],
            "output_path": str(config.output_path),
            "summary_path": str(config.summary_path),
        },
    }
    config.summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def select_root_candidates(
    parent_scores: tuple[Any, ...],
    policy_top_k: int,
    max_candidates: int,
) -> tuple[Any, ...]:
    selected = list(parent_scores[:policy_top_k])
    selected_indices = {item.action_index for item in selected}
    for action_type in (ActionType.BUY_VISIBLE, ActionType.BUY_RESERVED):
        candidate = next(
            (
                item
                for item in parent_scores
                if item.action.action_type == action_type
            ),
            None,
        )
        if candidate is not None and candidate.action_index not in selected_indices:
            selected.append(candidate)
            selected_indices.add(candidate.action_index)
    return tuple(selected[:max_candidates])


def rollout_action_return(
    env: SplendorEnv,
    scorer: Any,
    state: SplendorState,
    root_action: Action,
    root_player: int,
    seed: int,
    epsilon: float,
    temperature: float,
    top_k: int,
    max_additional_turns: int,
    outcome_weight: float,
    score_margin_scale: float,
) -> float:
    rng = random.Random(seed)
    rollout_state = env.step(state, root_action)
    seen: Counter[tuple[object, ...]] = Counter()
    no_progress_turns = 0
    end_turn = rollout_state.turn_index + max_additional_turns
    while not rollout_state.terminal and rollout_state.turn_index < end_turn:
        signature = state_signature(rollout_state)
        seen[signature] += 1
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
        if is_progress_transition(rollout_state, next_state, action):
            no_progress_turns = 0
        else:
            no_progress_turns += 1
        rollout_state = next_state

    winner = (
        rollout_state.winner
        if rollout_state.terminal
        else _adjudicate_scores(rollout_state)
    )
    outcome = (
        0.0
        if winner is None
        else 1.0
        if winner == root_player
        else -1.0
    )
    root_score = rollout_state.players[root_player].score
    opponent_score = rollout_state.players[1 - root_player].score
    score_signal = math.tanh(
        (root_score - opponent_score) / score_margin_scale
    )
    return outcome_weight * outcome + (1.0 - outcome_weight) * score_signal


def _sample_scored_action(
    scores: tuple[Any, ...],
    rng: random.Random,
    epsilon: float,
    temperature: float,
    top_k: int,
) -> Action:
    if len(scores) == 1 or rng.random() >= epsilon:
        return scores[0].action
    support = scores[: min(top_k, len(scores))]
    maximum = max(item.score for item in support)
    weights = [
        math.exp((item.score - maximum) / temperature)
        for item in support
    ]
    return rng.choices(
        [item.action for item in support],
        weights=weights,
        k=1,
    )[0]


def _select_rows(
    config: RolloutActionCreditConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_phase: dict[str, list[dict[str, Any]]] = {
        "early": [],
        "mid": [],
        "late": [],
    }
    rows_read = 0
    eligible = 0
    for replay_path in config.replay_paths:
        with replay_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                rows_read += 1
                payload = json.loads(line)
                if not isinstance(payload.get("state_snapshot"), dict):
                    continue
                if not _is_champion_turn(payload, config.champion_marker):
                    continue
                eligible += 1
                by_phase[_phase(int(payload.get("turn_index", 0)))].append(
                    payload
                )
    rng = random.Random(config.seed)
    selected = []
    base = config.max_states // 3
    remainder = config.max_states % 3
    selected_by_phase = {}
    for phase_index, phase in enumerate(("early", "mid", "late")):
        rows = by_phase[phase]
        rng.shuffle(rows)
        target = base + (1 if phase_index < remainder else 0)
        chosen = rows[:target]
        selected.extend(chosen)
        selected_by_phase[phase] = len(chosen)
    rng.shuffle(selected)
    return selected, {
        "rows_read": rows_read,
        "eligible_champion_rows": eligible,
        "states_selected": len(selected),
        "states_selected_by_phase": selected_by_phase,
    }


def _is_champion_turn(payload: dict[str, Any], marker: str) -> bool:
    player_id = int(payload.get("player_id", -1))
    metadata = payload.get("game_bot_metadata")
    if not isinstance(metadata, list) or not 0 <= player_id < len(metadata):
        return False
    seat = metadata[player_id]
    if not isinstance(seat, dict):
        return False
    checkpoint = str(seat.get("checkpoint_path", "")).replace("\\", "/")
    return marker.replace("\\", "/").lower() in checkpoint.lower()


def _standard_error(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return statistics.stdev(values) / math.sqrt(len(values))


def _phase(turn_index: int) -> str:
    if turn_index < 20:
        return "early"
    if turn_index < 50:
        return "mid"
    return "late"


def _validate_config(config: RolloutActionCreditConfig) -> None:
    if not config.replay_paths:
        raise ValueError("At least one replay path is required.")
    for name, value in (
        ("max_states", config.max_states),
        ("root_policy_top_k", config.root_policy_top_k),
        ("max_root_candidates", config.max_root_candidates),
        ("rollouts_per_action", config.rollouts_per_action),
        ("rollout_top_k", config.rollout_top_k),
        ("rollout_max_additional_turns", config.rollout_max_additional_turns),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be positive.")
    if config.changed_refinement_rollouts < 0:
        raise ValueError("changed_refinement_rollouts must be non-negative.")
    if config.max_root_candidates < config.root_policy_top_k:
        raise ValueError("max_root_candidates cannot be smaller than root_policy_top_k.")
    if not 0.0 <= config.rollout_epsilon <= 1.0:
        raise ValueError("rollout_epsilon must be in [0, 1].")
    if config.rollout_temperature <= 0.0:
        raise ValueError("rollout_temperature must be positive.")
    if not 0.0 <= config.outcome_weight <= 1.0:
        raise ValueError("outcome_weight must be in [0, 1].")
    if config.score_margin_scale <= 0.0:
        raise ValueError("score_margin_scale must be positive.")
    if config.min_action_return_range < 0.0:
        raise ValueError("min_action_return_range must be non-negative.")


if __name__ == "__main__":
    main()
