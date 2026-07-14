"""Compare PUCT leaf evaluators against matched counterfactual continuations."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from splendor_ai.bots.hidden_information import determinize_for_observer
from splendor_ai.bots.puct_bot import (
    InformationSetPuctBot,
    PuctConfig,
    _combine_n_step_leaf_value,
    legal_heuristic_value,
)
from splendor_ai.bots.search_bot import CheckpointValueEvaluator
from splendor_ai.encoding import ActionCodec
from splendor_ai.engine.actions import Action
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.training.replay import deserialize_state_snapshot, serialize_action
from splendor_ai.training.run_policy_search_disagreement import _LegalActionScorer
from splendor_ai.training.run_rollout_action_credit import (
    rollout_action_return,
    select_root_candidates,
)


SIGNAL_NAMES = (
    "heuristic",
    "clipped_n_step",
    "value009_value",
    "value009_outcome",
    "value010_value",
    "value010_outcome",
)


@dataclass(frozen=True, slots=True)
class LeafQualityAuditConfig:
    replay_paths: tuple[Path, ...]
    policy_checkpoint: Path
    value009_checkpoint: Path
    value010_checkpoint: Path
    output_path: Path
    corpus_output_path: Path | None = None
    max_states: int = 30
    candidates_per_state: int = 6
    determinizations: int = 3
    rollouts_per_determinization: int = 10
    min_turn_index: int = 40
    rollout_epsilon: float = 0.1
    rollout_temperature: float = 0.5
    rollout_top_k: int = 3
    rollout_max_additional_turns: int = 120
    outcome_weight: float = 0.75
    score_margin_scale: float = 5.0
    min_return_range: float = 0.05
    bootstrap_samples: int = 5_000
    seed: int = 63
    device: str = "cpu"
    log_every_states: int = 5


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate PUCT leaf rankings against matched legal determinizations "
            "and continuation rollouts from existing replay snapshots."
        )
    )
    parser.add_argument("--replay-path", action="append", required=True)
    parser.add_argument("--policy-checkpoint", required=True)
    parser.add_argument("--value009-checkpoint", required=True)
    parser.add_argument("--value010-checkpoint", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument(
        "--corpus-output-path",
        default=None,
        help=(
            "Optional JSONL path for informative state rows with matched "
            "counterfactual action-return targets."
        ),
    )
    parser.add_argument("--max-states", type=int, default=30)
    parser.add_argument("--candidates-per-state", type=int, default=6)
    parser.add_argument("--determinizations", type=int, default=3)
    parser.add_argument("--rollouts-per-determinization", type=int, default=10)
    parser.add_argument("--min-turn-index", type=int, default=40)
    parser.add_argument("--rollout-epsilon", type=float, default=0.1)
    parser.add_argument("--rollout-temperature", type=float, default=0.5)
    parser.add_argument("--rollout-top-k", type=int, default=3)
    parser.add_argument("--rollout-max-additional-turns", type=int, default=120)
    parser.add_argument("--outcome-weight", type=float, default=0.75)
    parser.add_argument("--score-margin-scale", type=float, default=5.0)
    parser.add_argument("--min-return-range", type=float, default=0.05)
    parser.add_argument("--bootstrap-samples", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=63)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--log-every-states", type=int, default=5)
    return parser


def _phase(turn_index: int) -> str:
    if turn_index < 50:
        return "late_40_49"
    if turn_index < 60:
        return "endgame_50_59"
    return "endgame_60_plus"


def _select_rows(config: LeafQualityAuditConfig) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_phase: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rows_read = 0
    for path in config.replay_paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                rows_read += 1
                payload = json.loads(line)
                if not isinstance(payload.get("state_snapshot"), dict):
                    continue
                if int(payload.get("turn_index", 0)) < config.min_turn_index:
                    continue
                if not isinstance(payload.get("root_visit_distribution"), list):
                    continue
                by_phase[_phase(int(payload["turn_index"]))].append(payload)
    rng = random.Random(config.seed)
    selected: list[dict[str, Any]] = []
    selected_by_phase: dict[str, int] = {}
    phase_names = ("late_40_49", "endgame_50_59", "endgame_60_plus")
    base, remainder = divmod(config.max_states, len(phase_names))
    for index, phase_name in enumerate(phase_names):
        rows = by_phase[phase_name]
        rng.shuffle(rows)
        count = min(len(rows), base + (1 if index < remainder else 0))
        selected.extend(rows[:count])
        selected_by_phase[phase_name] = count
    rng.shuffle(selected)
    return selected, {
        "rows_read": rows_read,
        "eligible_rows_by_phase": {phase: len(rows) for phase, rows in by_phase.items()},
        "states_selected": len(selected),
        "states_selected_by_phase": selected_by_phase,
    }


def _candidate_action_indices(
    payload: dict[str, Any],
    state: Any,
    legal_actions: list[Action],
    scorer: _LegalActionScorer,
    codec: ActionCodec,
    limit: int,
) -> list[int]:
    scored = scorer.score_actions(state, legal_actions)
    selected = list(select_root_candidates(scored, policy_top_k=4, max_candidates=limit))
    selected_indices = {item.action_index for item in selected}
    root_visits = sorted(
        (
            item
            for item in payload.get("root_visit_distribution", [])
            if isinstance(item, dict)
        ),
        key=lambda item: (-float(item.get("probability", 0.0)), int(item.get("action_index", -1))),
    )
    for visit in root_visits:
        action_index = int(visit.get("action_index", -1))
        if action_index in selected_indices:
            continue
        try:
            action_row = codec.legal_action_indices(state, legal_actions).index(action_index)
        except ValueError:
            continue
        selected.append(scored[action_row])
        selected_indices.add(action_index)
        if len(selected) >= limit:
            break
    return [item.action_index for item in selected[:limit]]


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2:
        return None
    mean_x = statistics.fmean(xs)
    mean_y = statistics.fmean(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
    denominator = math.sqrt(
        sum((x - mean_x) ** 2 for x in xs)
        * sum((y - mean_y) ** 2 for y in ys)
    )
    return numerator / denominator if denominator > 0.0 else None


def _average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index], reverse=True)
    ranks = [0.0] * len(values)
    offset = 0
    while offset < len(order):
        end = offset + 1
        while end < len(order) and values[order[end]] == values[order[offset]]:
            end += 1
        rank = (offset + end - 1) / 2.0
        for order_index in order[offset:end]:
            ranks[order_index] = rank
        offset = end
    return ranks


def _spearman(xs: list[float], ys: list[float]) -> float | None:
    return _pearson(_average_ranks(xs), _average_ranks(ys))


def _mean_ci(values: list[float], seed: int, samples: int) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "ci95_low": 0.0, "ci95_high": 0.0}
    mean = statistics.fmean(values)
    if len(values) == 1 or samples <= 0:
        return {"mean": mean, "ci95_low": mean, "ci95_high": mean}
    rng = random.Random(seed)
    count = len(values)
    bootstrap = sorted(
        statistics.fmean(values[rng.randrange(count)] for _ in range(count))
        for _ in range(samples)
    )
    return {
        "mean": mean,
        "ci95_low": bootstrap[int(0.025 * (samples - 1))],
        "ci95_high": bootstrap[int(0.975 * (samples - 1))],
    }


def _stable_seed(base: int, name: str) -> int:
    for character in name:
        base = ((base * 33) + ord(character)) & 0xFFFFFFFF
    return base


def _summarize_states(
    state_rows: list[dict[str, Any]],
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    informative = [row for row in state_rows if row["informative"]]
    by_phase: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in informative:
        by_phase[row["phase"]].append(row)

    def summarize(rows: list[dict[str, Any]], label: str) -> dict[str, Any]:
        summary: dict[str, Any] = {"states": len(rows), "signals": {}}
        all_returns = [item["rollout_return"] for row in rows for item in row["actions"]]
        for signal in SIGNAL_NAMES:
            signal_values = [item["signals"][signal] for row in rows for item in row["actions"]]
            per_state_top_hit = [row["ranking"][signal]["top1_hit"] for row in rows]
            per_state_regret = [row["ranking"][signal]["regret"] for row in rows]
            per_state_spearman = [
                row["ranking"][signal]["spearman"]
                for row in rows
                if row["ranking"][signal]["spearman"] is not None
            ]
            summary["signals"][signal] = {
                "top1_hit_rate": _mean_ci(per_state_top_hit, _stable_seed(seed, label + signal + "hit"), bootstrap_samples),
                "mean_regret": _mean_ci(per_state_regret, _stable_seed(seed, label + signal + "regret"), bootstrap_samples),
                "mean_spearman": _mean_ci(per_state_spearman, _stable_seed(seed, label + signal + "spearman"), bootstrap_samples),
                "all_action_pearson": _pearson(signal_values, all_returns),
                "all_action_mae": statistics.fmean(
                    abs(value - target) for value, target in zip(signal_values, all_returns, strict=True)
                ) if signal_values else None,
            }
        heuristic = summary["signals"]["heuristic"]
        for signal in SIGNAL_NAMES:
            if signal == "heuristic":
                continue
            summary["signals"][signal]["top1_hit_delta_vs_heuristic"] = _mean_ci(
                [row["ranking"][signal]["top1_hit"] - row["ranking"]["heuristic"]["top1_hit"] for row in rows],
                _stable_seed(seed, label + signal + "hit_delta"),
                bootstrap_samples,
            )
            summary["signals"][signal]["regret_delta_vs_heuristic"] = _mean_ci(
                [row["ranking"][signal]["regret"] - row["ranking"]["heuristic"]["regret"] for row in rows],
                _stable_seed(seed, label + signal + "regret_delta"),
                bootstrap_samples,
            )
        return summary

    return {
        "states_evaluated": len(state_rows),
        "informative_states": len(informative),
        "overall": summarize(informative, "overall"),
        "by_phase": {phase: summarize(rows, phase) for phase, rows in sorted(by_phase.items())},
    }


def run_leaf_quality_audit(config: LeafQualityAuditConfig) -> dict[str, Any]:
    _validate_config(config)
    rows, selection = _select_rows(config)
    env = SplendorEnv()
    codec = ActionCodec()
    continuation_scorer = _LegalActionScorer(
        config.policy_checkpoint,
        config.device,
        "legal_action_attention_scorer",
    )
    n_step_bot = InformationSetPuctBot(
        config.policy_checkpoint,
        device=config.device,
        config=PuctConfig(
            n_step_leaf_depth=2,
            n_step_leaf_width=1,
            n_step_leaf_candidate_limit=4,
            n_step_leaf_weight=1.0,
            n_step_leaf_delta_clip=0.20,
            n_step_leaf_min_turn_index=40,
            heuristic_scale=4_000.0,
        ),
    )
    value_evaluators = {
        "value009_value": CheckpointValueEvaluator(config.value009_checkpoint, config.device, value_scale=1.0, prediction_mode="value"),
        "value009_outcome": CheckpointValueEvaluator(config.value009_checkpoint, config.device, value_scale=1.0, prediction_mode="outcome"),
        "value010_value": CheckpointValueEvaluator(config.value010_checkpoint, config.device, value_scale=1.0, prediction_mode="value"),
        "value010_outcome": CheckpointValueEvaluator(config.value010_checkpoint, config.device, value_scale=1.0, prediction_mode="outcome"),
    }
    state_rows: list[dict[str, Any]] = []
    corpus_rows: list[dict[str, Any]] = []
    for state_index, payload in enumerate(rows):
        root_state = deserialize_state_snapshot(payload["state_snapshot"])
        root_player = root_state.current_player
        root_legal = env.legal_actions(root_state)
        candidate_indices = _candidate_action_indices(
            payload, root_state, root_legal, continuation_scorer, codec, config.candidates_per_state
        )
        action_data: dict[int, dict[str, Any]] = {
            index: {"action_index": index, "signals_by_determinization": defaultdict(list), "rollout_returns": []}
            for index in candidate_indices
        }
        for determinization_index in range(config.determinizations):
            determinized = determinize_for_observer(
                root_state,
                root_player,
                random.Random(config.seed + state_index * 1_000_003 + determinization_index),
            )
            legal_actions = env.legal_actions(determinized)
            actions_by_index = {
                action_index: action
                for action_index, action in zip(codec.legal_action_indices(determinized, legal_actions), legal_actions, strict=True)
            }
            for action_index in candidate_indices:
                action = actions_by_index.get(action_index)
                if action is None:
                    continue
                child = env.step(determinized, action)
                heuristic = legal_heuristic_value(child, root_player, scale=4_000.0)
                n_step = n_step_bot._n_step_leaf_value(env, child, root_player)
                clipped_n_step, _raw_delta, _applied_delta = _combine_n_step_leaf_value(
                    heuristic, n_step, weight=1.0, delta_clip=0.20
                )
                action_data[action_index]["signals_by_determinization"]["heuristic"].append(heuristic)
                action_data[action_index]["signals_by_determinization"]["clipped_n_step"].append(clipped_n_step)
                for name, evaluator in value_evaluators.items():
                    action_data[action_index]["signals_by_determinization"][name].append(
                        evaluator.evaluate_state(child, root_player)
                    )
                for rollout_index in range(config.rollouts_per_determinization):
                    action_data[action_index]["rollout_returns"].append(
                        rollout_action_return(
                            env=env,
                            scorer=continuation_scorer,
                            state=determinized,
                            root_action=action,
                            root_player=root_player,
                            seed=(
                                config.seed
                                + state_index * 10_000_019
                                + determinization_index * 100_003
                                + rollout_index
                            ),
                            epsilon=config.rollout_epsilon,
                            temperature=config.rollout_temperature,
                            top_k=config.rollout_top_k,
                            max_additional_turns=config.rollout_max_additional_turns,
                            outcome_weight=config.outcome_weight,
                            score_margin_scale=config.score_margin_scale,
                        )
                    )
        actions = []
        for action_index, data in action_data.items():
            if not data["rollout_returns"]:
                continue
            action = next(action for action in root_legal if codec.encode(root_state, action) == action_index)
            actions.append(
                {
                    "action_index": action_index,
                    "action_payload": serialize_action(action),
                    "signals": {
                        signal: statistics.fmean(data["signals_by_determinization"][signal])
                        for signal in SIGNAL_NAMES
                    },
                    "rollout_return": statistics.fmean(data["rollout_returns"]),
                    "rollout_standard_error": _standard_error(data["rollout_returns"]),
                    "determinizations": config.determinizations,
                    "rollouts": len(data["rollout_returns"]),
                }
            )
        if len(actions) < 2:
            continue
        rollout_values = [item["rollout_return"] for item in actions]
        best_rollout = max(rollout_values)
        return_range = best_rollout - min(rollout_values)
        informative = return_range >= config.min_return_range
        ranking: dict[str, dict[str, float | int | None]] = {}
        for signal in SIGNAL_NAMES:
            signal_values = [item["signals"][signal] for item in actions]
            selected_row = max(range(len(actions)), key=lambda index: (signal_values[index], -actions[index]["action_index"]))
            ranking[signal] = {
                "selected_action_index": actions[selected_row]["action_index"],
                "top1_hit": float(rollout_values[selected_row] == best_rollout),
                "regret": best_rollout - rollout_values[selected_row],
                "spearman": _spearman(signal_values, rollout_values),
            }
        state_rows.append(
            {
                "game_seed": payload.get("game_seed"),
                "turn_index": payload.get("turn_index"),
                "player_id": root_player,
                "phase": _phase(int(payload["turn_index"])),
                "candidate_count": len(actions),
                "return_range": return_range,
                "informative": informative,
                "actions": actions,
                "ranking": ranking,
            }
        )
        if informative:
            corpus_payload = dict(payload)
            corpus_payload["search_action_scores"] = [
                {
                    "rank": rank,
                    "action_index": action_row["action_index"],
                    "action_payload": action_row["action_payload"],
                    "value": action_row["rollout_return"],
                    "standard_error": action_row["rollout_standard_error"],
                }
                for rank, action_row in enumerate(
                    sorted(
                        actions,
                        key=lambda item: item["rollout_return"],
                        reverse=True,
                    ),
                    start=1,
                )
            ]
            corpus_payload["training_weight"] = return_range
            corpus_payload["target_source"] = "counterfactual_action_value"
            corpus_payload["counterfactual_action_value_metadata"] = {
                "return_range": return_range,
                "candidate_count": len(actions),
                "determinizations": config.determinizations,
                "rollouts_per_determinization": config.rollouts_per_determinization,
                "outcome_weight": config.outcome_weight,
                "score_margin_scale": config.score_margin_scale,
            }
            corpus_rows.append(corpus_payload)
        if config.log_every_states > 0 and (state_index + 1) % config.log_every_states == 0:
            print(f"[leaf-quality-audit] evaluated={state_index + 1}/{len(rows)} kept={len(state_rows)}")

    summary = _summarize_states(state_rows, config.bootstrap_samples, config.seed)
    output = {
        "audit": "counterfactual_leaf_quality",
        "config": {
            **asdict(config),
            "replay_paths": [str(path) for path in config.replay_paths],
            "policy_checkpoint": str(config.policy_checkpoint),
            "value009_checkpoint": str(config.value009_checkpoint),
            "value010_checkpoint": str(config.value010_checkpoint),
            "output_path": str(config.output_path),
            "corpus_output_path": (
                str(config.corpus_output_path)
                if config.corpus_output_path is not None
                else None
            ),
        },
        "selection": selection,
        "summary": summary,
        "states": state_rows,
    }
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    config.output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    if config.corpus_output_path is not None:
        config.corpus_output_path.parent.mkdir(parents=True, exist_ok=True)
        with config.corpus_output_path.open("w", encoding="utf-8") as handle:
            for corpus_row in corpus_rows:
                handle.write(json.dumps(corpus_row) + "\n")
    return output


def _standard_error(values: list[float]) -> float:
    return statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0


def _validate_config(config: LeafQualityAuditConfig) -> None:
    if not config.replay_paths:
        raise ValueError("At least one replay path is required.")
    for name, value in (
        ("max_states", config.max_states),
        ("candidates_per_state", config.candidates_per_state),
        ("determinizations", config.determinizations),
        ("rollouts_per_determinization", config.rollouts_per_determinization),
        ("rollout_top_k", config.rollout_top_k),
        ("rollout_max_additional_turns", config.rollout_max_additional_turns),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be positive.")
    if config.min_turn_index < 0 or config.min_return_range < 0.0:
        raise ValueError("Minimum turn and return range must be non-negative.")
    if not 0.0 <= config.rollout_epsilon <= 1.0:
        raise ValueError("rollout_epsilon must be in [0, 1].")
    if config.rollout_temperature <= 0.0 or config.score_margin_scale <= 0.0:
        raise ValueError("Rollout temperature and score margin scale must be positive.")
    if not 0.0 <= config.outcome_weight <= 1.0 or config.bootstrap_samples < 0:
        raise ValueError("Outcome weight and bootstrap samples are invalid.")


def main() -> None:
    args = build_parser().parse_args()
    output = run_leaf_quality_audit(
        LeafQualityAuditConfig(
            replay_paths=tuple(Path(path) for path in args.replay_path),
            policy_checkpoint=Path(args.policy_checkpoint),
            value009_checkpoint=Path(args.value009_checkpoint),
            value010_checkpoint=Path(args.value010_checkpoint),
            output_path=Path(args.output_path),
            corpus_output_path=(
                Path(args.corpus_output_path)
                if args.corpus_output_path is not None
                else None
            ),
            max_states=args.max_states,
            candidates_per_state=args.candidates_per_state,
            determinizations=args.determinizations,
            rollouts_per_determinization=args.rollouts_per_determinization,
            min_turn_index=args.min_turn_index,
            rollout_epsilon=args.rollout_epsilon,
            rollout_temperature=args.rollout_temperature,
            rollout_top_k=args.rollout_top_k,
            rollout_max_additional_turns=args.rollout_max_additional_turns,
            outcome_weight=args.outcome_weight,
            score_margin_scale=args.score_margin_scale,
            min_return_range=args.min_return_range,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
            device=args.device,
            log_every_states=args.log_every_states,
        )
    )
    overall = output["summary"]["overall"]
    print(
        "[leaf-quality-audit]"
        f" evaluated={output['summary']['states_evaluated']}"
        f" informative={output['summary']['informative_states']}"
        f" heuristic_top1={overall['signals']['heuristic']['top1_hit_rate']['mean']:.3f}"
        f" nstep_top1={overall['signals']['clipped_n_step']['top1_hit_rate']['mean']:.3f}"
        f" output={args.output_path}"
    )


if __name__ == "__main__":
    main()
