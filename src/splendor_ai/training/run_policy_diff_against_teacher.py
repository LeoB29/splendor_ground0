"""Compare two policy checkpoints against stored or live teacher action rankings."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from splendor_ai.bots import CheckpointLegalActionPriorScorer, ShallowSearchBot
from splendor_ai.encoding import ActionCodec
from splendor_ai.engine.actions import ActionType
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.training.replay import deserialize_state_snapshot, serialize_action
from splendor_ai.training.run_policy_search_disagreement import (
    DisagreementConfig,
    ModelActionScore,
    _load_policy_scorer,
    _phase_for_turn,
)


MODEL_BOT_NAMES = {
    "CheckpointPolicyBot",
    "LegalActionAttentionScorerBot",
    "LegalActionScorerBot",
    "LegalActionSetScorerBot",
    "PolicyGuidedSearchBot",
}


@dataclass(frozen=True, slots=True)
class PolicyDiffTeacherConfig:
    before_checkpoint: Path
    before_model_kind: str
    after_checkpoint: Path
    after_model_kind: str
    replay_paths: tuple[Path, ...]
    output_path: Path
    device: str = "cpu"
    max_rows: int | None = None
    log_every: int = 500
    teacher_source: str = "stored"
    search_depth: int = 2
    search_max_branching: int = 10
    search_buy_branching: int = 6
    search_reserve_branching: int = 3
    search_take_branching: int = 3
    teacher_policy_prior_checkpoint_path: Path | None = None
    teacher_policy_prior_device: str = "cpu"
    teacher_policy_prior_weight: float = 0.0
    seed: int = 0
    player_id_filter: tuple[int, ...] | None = None
    phase_filter: tuple[str, ...] | None = None
    game_seed_filter: tuple[int, ...] | None = None
    model_turns_only: bool = False
    examples_limit: int = 20


@dataclass(frozen=True, slots=True)
class _TeacherActionScore:
    action_index: int
    action_type: str
    value: float
    action_payload: dict[str, Any] | None = None
    heuristic_score: float | None = None
    loop_penalty: float | None = None
    policy_score: float | None = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare two policy checkpoints against teacher-ranked replay states."
    )
    parser.add_argument("--before-checkpoint", required=True)
    parser.add_argument(
        "--before-model-kind",
        choices=(
            "flat_policy",
            "legal_action_scorer",
            "legal_action_attention_scorer",
            "legal_action_set_scorer",
        ),
        required=True,
    )
    parser.add_argument("--after-checkpoint", required=True)
    parser.add_argument(
        "--after-model-kind",
        choices=(
            "flat_policy",
            "legal_action_scorer",
            "legal_action_attention_scorer",
            "legal_action_set_scorer",
        ),
        required=True,
    )
    parser.add_argument("--replay-path", action="append", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument(
        "--teacher-source",
        choices=("stored", "search", "policy_guided_search"),
        default="stored",
    )
    parser.add_argument("--search-depth", type=int, default=2)
    parser.add_argument("--search-max-branching", type=int, default=10)
    parser.add_argument("--search-buy-branching", type=int, default=6)
    parser.add_argument("--search-reserve-branching", type=int, default=3)
    parser.add_argument("--search-take-branching", type=int, default=3)
    parser.add_argument("--teacher-policy-prior-checkpoint", default=None)
    parser.add_argument("--teacher-policy-prior-device", default="cpu")
    parser.add_argument("--teacher-policy-prior-weight", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--player-id-filter",
        nargs="*",
        type=int,
        choices=(0, 1),
        default=None,
    )
    parser.add_argument(
        "--phase-filter",
        nargs="*",
        choices=("early", "mid", "late"),
        default=None,
    )
    parser.add_argument("--game-seed-filter", nargs="*", type=int, default=None)
    parser.add_argument(
        "--model-turns-only",
        action="store_true",
        help="Only analyze rows where the acting seat is a model-backed bot.",
    )
    parser.add_argument("--examples-limit", type=int, default=20)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = compare_policies_against_teacher(
        PolicyDiffTeacherConfig(
            before_checkpoint=Path(args.before_checkpoint),
            before_model_kind=args.before_model_kind,
            after_checkpoint=Path(args.after_checkpoint),
            after_model_kind=args.after_model_kind,
            replay_paths=tuple(Path(path) for path in args.replay_path),
            output_path=Path(args.output_path),
            device=args.device,
            max_rows=args.max_rows,
            log_every=args.log_every,
            teacher_source=args.teacher_source,
            search_depth=args.search_depth,
            search_max_branching=args.search_max_branching,
            search_buy_branching=args.search_buy_branching,
            search_reserve_branching=args.search_reserve_branching,
            search_take_branching=args.search_take_branching,
            teacher_policy_prior_checkpoint_path=(
                Path(args.teacher_policy_prior_checkpoint)
                if args.teacher_policy_prior_checkpoint is not None
                else None
            ),
            teacher_policy_prior_device=args.teacher_policy_prior_device,
            teacher_policy_prior_weight=args.teacher_policy_prior_weight,
            seed=args.seed,
            player_id_filter=(
                tuple(args.player_id_filter)
                if args.player_id_filter is not None and len(args.player_id_filter) > 0
                else None
            ),
            phase_filter=(
                tuple(args.phase_filter)
                if args.phase_filter is not None and len(args.phase_filter) > 0
                else None
            ),
            game_seed_filter=(
                tuple(args.game_seed_filter)
                if args.game_seed_filter is not None and len(args.game_seed_filter) > 0
                else None
            ),
            model_turns_only=args.model_turns_only,
            examples_limit=args.examples_limit,
        )
    )
    overall = summary["overall"]
    print(
        "[policy-diff]"
        f" analyzed={summary['rows_analyzed']}"
        f" before_match={overall['before_teacher_match_rate']:.4f}"
        f" after_match={overall['after_teacher_match_rate']:.4f}"
        f" before_after_agreement={overall['before_after_top_action_agreement_rate']:.4f}"
        f" output={args.output_path}"
    )


def compare_policies_against_teacher(config: PolicyDiffTeacherConfig) -> dict[str, Any]:
    if not config.replay_paths:
        raise ValueError("At least one replay path is required.")
    if config.teacher_source == "policy_guided_search":
        if (
            config.teacher_policy_prior_checkpoint_path is None
            or config.teacher_policy_prior_weight <= 0.0
        ):
            raise ValueError(
                "policy_guided_search teacher requires --teacher-policy-prior-checkpoint "
                "and a positive --teacher-policy-prior-weight."
            )
    if config.teacher_policy_prior_weight < 0.0:
        raise ValueError("teacher_policy_prior_weight must be non-negative.")
    unknown_phases = set(config.phase_filter or ()) - {"early", "mid", "late"}
    if unknown_phases:
        raise ValueError(f"Unknown phases: {sorted(unknown_phases)}")

    env = SplendorEnv()
    codec = ActionCodec()
    before_scorer = _load_policy_scorer(
        DisagreementConfig(
            checkpoint=config.before_checkpoint,
            model_kind=config.before_model_kind,
            replay_paths=config.replay_paths,
            output_path=config.output_path,
            device=config.device,
        )
    )
    after_scorer = _load_policy_scorer(
        DisagreementConfig(
            checkpoint=config.after_checkpoint,
            model_kind=config.after_model_kind,
            replay_paths=config.replay_paths,
            output_path=config.output_path,
            device=config.device,
        )
    )
    action_prior = _build_action_prior(config)
    counters = _empty_counters()
    examples: list[dict[str, Any]] = []
    rows_seen = 0
    rows_skipped_by_game_seed = 0
    rows_skipped_by_model_turn = 0
    rows_skipped_by_player_id = 0
    rows_skipped_by_phase = 0
    rows_missing_snapshot = 0
    rows_without_legal_actions = 0
    rows_without_teacher = 0

    for replay_path in config.replay_paths:
        with replay_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if config.max_rows is not None and rows_seen >= config.max_rows:
                    break
                stripped = line.strip()
                if not stripped:
                    continue
                rows_seen += 1
                payload = json.loads(stripped)
                if (
                    config.game_seed_filter is not None
                    and int(payload.get("game_seed", -1)) not in config.game_seed_filter
                ):
                    rows_skipped_by_game_seed += 1
                    continue
                player_id = int(payload.get("player_id", -1))
                if config.player_id_filter is not None and player_id not in config.player_id_filter:
                    rows_skipped_by_player_id += 1
                    continue
                if config.model_turns_only and not _is_model_turn(payload, player_id):
                    rows_skipped_by_model_turn += 1
                    continue
                phase = _phase_for_turn(int(payload.get("turn_index", 0)))
                if config.phase_filter is not None and phase not in config.phase_filter:
                    rows_skipped_by_phase += 1
                    continue
                snapshot = payload.get("state_snapshot")
                if not isinstance(snapshot, dict):
                    rows_missing_snapshot += 1
                    continue
                state = deserialize_state_snapshot(snapshot)
                legal_actions = env.legal_actions(state)
                if not legal_actions:
                    rows_without_legal_actions += 1
                    continue
                teacher_scores = _teacher_scores_for_row(
                    config=config,
                    codec=codec,
                    env=env,
                    state=state,
                    legal_actions=legal_actions,
                    payload=payload,
                    row_offset=rows_seen,
                    action_prior=action_prior,
                )
                if not teacher_scores:
                    rows_without_teacher += 1
                    continue
                before_scores = before_scorer.score_actions(state, legal_actions)
                after_scores = after_scorer.score_actions(state, legal_actions)
                _record_row(
                    counters=counters,
                    examples=examples,
                    examples_limit=config.examples_limit,
                    state=state,
                    payload=payload,
                    replay_path=replay_path,
                    before_top=before_scores[0],
                    after_top=after_scores[0],
                    teacher_scores=teacher_scores,
                )
                if config.log_every > 0 and rows_seen % config.log_every == 0:
                    overall = _finalize_counter(counters["overall"])
                    print(
                        f"[policy-diff] rows_seen={rows_seen}"
                        f" analyzed={counters['overall']['count']}"
                        f" before_match={overall['before_teacher_match_rate']:.4f}"
                        f" after_match={overall['after_teacher_match_rate']:.4f}"
                    )
        if config.max_rows is not None and rows_seen >= config.max_rows:
            break

    summary = _build_summary(
        config=config,
        counters=counters,
        rows_seen=rows_seen,
        rows_skipped_by_game_seed=rows_skipped_by_game_seed,
        rows_skipped_by_model_turn=rows_skipped_by_model_turn,
        rows_skipped_by_player_id=rows_skipped_by_player_id,
        rows_skipped_by_phase=rows_skipped_by_phase,
        rows_missing_snapshot=rows_missing_snapshot,
        rows_without_legal_actions=rows_without_legal_actions,
        rows_without_teacher=rows_without_teacher,
        examples=examples,
    )
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    config.output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _build_action_prior(config: PolicyDiffTeacherConfig) -> CheckpointLegalActionPriorScorer | None:
    if (
        config.teacher_source == "policy_guided_search"
        and config.teacher_policy_prior_checkpoint_path is not None
        and config.teacher_policy_prior_weight > 0.0
    ):
        return CheckpointLegalActionPriorScorer(
            checkpoint_path=config.teacher_policy_prior_checkpoint_path,
            device=config.teacher_policy_prior_device,
        )
    return None


def _is_model_turn(payload: dict[str, Any], player_id: int) -> bool:
    bot_seats = payload.get("game_bot_seats")
    if not isinstance(bot_seats, list | tuple):
        return False
    if player_id < 0 or player_id >= len(bot_seats):
        return False
    return str(bot_seats[player_id]) in MODEL_BOT_NAMES


def _teacher_scores_for_row(
    config: PolicyDiffTeacherConfig,
    codec: ActionCodec,
    env: SplendorEnv,
    state: Any,
    legal_actions: list[Any],
    payload: dict[str, Any],
    row_offset: int,
    action_prior: CheckpointLegalActionPriorScorer | None,
) -> tuple[_TeacherActionScore, ...]:
    if config.teacher_source == "stored":
        return _stored_teacher_scores(payload)

    action_prior_weight = (
        config.teacher_policy_prior_weight
        if config.teacher_source == "policy_guided_search"
        else 0.0
    )
    search_bot = ShallowSearchBot(
        depth=config.search_depth,
        max_branching=config.search_max_branching,
        max_buy_actions=config.search_buy_branching,
        max_reserve_actions=config.search_reserve_branching,
        max_take_actions=config.search_take_branching,
        seed=config.seed + row_offset,
        action_prior=action_prior,
        action_prior_weight=action_prior_weight,
    )
    ranked_actions = search_bot.rank_actions(env, state, legal_actions)
    return tuple(
        _TeacherActionScore(
            action_index=codec.encode(state, score.action),
            action_type=score.action.action_type.name,
            value=float(score.value),
            action_payload=serialize_action(score.action),
            heuristic_score=float(score.heuristic_score),
            loop_penalty=float(score.loop_penalty),
            policy_score=float(score.policy_score),
        )
        for score in ranked_actions
    )


def _stored_teacher_scores(payload: dict[str, Any]) -> tuple[_TeacherActionScore, ...]:
    scores = []
    for score in payload.get("search_action_scores", []):
        action_payload = score.get("action_payload")
        action_type = (
            action_payload.get("action_type")
            if isinstance(action_payload, dict)
            else None
        )
        if action_type is None:
            continue
        scores.append(
            _TeacherActionScore(
                action_index=int(score["action_index"]),
                action_type=str(action_type),
                value=float(score["value"]),
                action_payload=action_payload,
                heuristic_score=(
                    float(score["heuristic_score"])
                    if "heuristic_score" in score
                    else None
                ),
                loop_penalty=(
                    float(score["loop_penalty"])
                    if "loop_penalty" in score
                    else None
                ),
                policy_score=(
                    float(score["policy_score"])
                    if "policy_score" in score
                    else None
                ),
            )
        )
    return tuple(sorted(scores, key=lambda item: item.value, reverse=True))


def _empty_counters() -> dict[str, Any]:
    return {
        "overall": _new_counter(),
        "by_phase": {},
        "by_seat": {},
        "by_teacher_action_type": {},
        "by_before_action_type": {},
        "by_after_action_type": {},
        "by_teacher_phase": {},
        "before_after_action_type_matrix": {},
        "teacher_to_before_action_type_matrix": {},
        "teacher_to_after_action_type_matrix": {},
    }


def _new_counter() -> dict[str, Any]:
    return {
        "count": 0,
        "before_teacher_matches": 0,
        "after_teacher_matches": 0,
        "both_teacher_match": 0,
        "neither_teacher_match": 0,
        "after_teacher_gains": 0,
        "after_teacher_losses": 0,
        "before_after_top_action_agreements": 0,
        "before_missing_teacher_value": 0,
        "after_missing_teacher_value": 0,
        "before_teacher_gaps": [],
        "after_teacher_gaps": [],
    }


def _record_row(
    counters: dict[str, Any],
    examples: list[dict[str, Any]],
    examples_limit: int,
    state: Any,
    payload: dict[str, Any],
    replay_path: Path,
    before_top: ModelActionScore,
    after_top: ModelActionScore,
    teacher_scores: tuple[_TeacherActionScore, ...],
) -> None:
    teacher_top = teacher_scores[0]
    teacher_values = {score.action_index: score.value for score in teacher_scores}
    before_match = before_top.action_index == teacher_top.action_index
    after_match = after_top.action_index == teacher_top.action_index
    top_agreement = before_top.action_index == after_top.action_index
    teacher_value = teacher_top.value
    before_value = teacher_values.get(before_top.action_index)
    after_value = teacher_values.get(after_top.action_index)
    before_gap = None if before_value is None else teacher_value - before_value
    after_gap = None if after_value is None else teacher_value - after_value
    phase = _phase_for_turn(int(payload.get("turn_index", 0)))
    seat = f"seat{int(payload.get('player_id', state.current_player))}"
    teacher_type = teacher_top.action_type
    before_type = before_top.action.action_type.name
    after_type = after_top.action.action_type.name

    for bucket in (
        counters["overall"],
        _bucket(counters["by_phase"], phase),
        _bucket(counters["by_seat"], seat),
        _bucket(counters["by_teacher_action_type"], teacher_type),
        _bucket(counters["by_before_action_type"], before_type),
        _bucket(counters["by_after_action_type"], after_type),
        _bucket(counters["by_teacher_phase"], f"{teacher_type}:{phase}"),
    ):
        _record_counter(
            bucket,
            before_match=before_match,
            after_match=after_match,
            top_agreement=top_agreement,
            before_gap=before_gap,
            after_gap=after_gap,
        )

    _increment_matrix(counters["before_after_action_type_matrix"], before_type, after_type)
    _increment_matrix(counters["teacher_to_before_action_type_matrix"], teacher_type, before_type)
    _increment_matrix(counters["teacher_to_after_action_type_matrix"], teacher_type, after_type)
    _maybe_store_example(
        examples=examples,
        examples_limit=examples_limit,
        replay_path=replay_path,
        payload=payload,
        state=state,
        before_top=before_top,
        after_top=after_top,
        teacher_top=teacher_top,
        before_match=before_match,
        after_match=after_match,
        before_gap=before_gap,
        after_gap=after_gap,
    )


def _bucket(buckets: dict[str, dict[str, Any]], key: str) -> dict[str, Any]:
    return buckets.setdefault(key, _new_counter())


def _record_counter(
    counter: dict[str, Any],
    before_match: bool,
    after_match: bool,
    top_agreement: bool,
    before_gap: float | None,
    after_gap: float | None,
) -> None:
    counter["count"] += 1
    if before_match:
        counter["before_teacher_matches"] += 1
    if after_match:
        counter["after_teacher_matches"] += 1
    if before_match and after_match:
        counter["both_teacher_match"] += 1
    if not before_match and not after_match:
        counter["neither_teacher_match"] += 1
    if after_match and not before_match:
        counter["after_teacher_gains"] += 1
    if before_match and not after_match:
        counter["after_teacher_losses"] += 1
    if top_agreement:
        counter["before_after_top_action_agreements"] += 1
    if before_gap is None:
        counter["before_missing_teacher_value"] += 1
    else:
        counter["before_teacher_gaps"].append(float(before_gap))
    if after_gap is None:
        counter["after_missing_teacher_value"] += 1
    else:
        counter["after_teacher_gaps"].append(float(after_gap))


def _increment_matrix(matrix: dict[str, dict[str, int]], row: str, column: str) -> None:
    matrix.setdefault(row, {})
    matrix[row][column] = matrix[row].get(column, 0) + 1


def _maybe_store_example(
    examples: list[dict[str, Any]],
    examples_limit: int,
    replay_path: Path,
    payload: dict[str, Any],
    state: Any,
    before_top: ModelActionScore,
    after_top: ModelActionScore,
    teacher_top: _TeacherActionScore,
    before_match: bool,
    after_match: bool,
    before_gap: float | None,
    after_gap: float | None,
) -> None:
    if examples_limit <= 0:
        return
    changed = before_top.action_index != after_top.action_index
    gain = after_match and not before_match
    loss = before_match and not after_match
    if not changed and not gain and not loss:
        return
    before_gap_sort = math.inf if before_gap is None else before_gap
    after_gap_sort = math.inf if after_gap is None else after_gap
    examples.append(
        {
            "replay_path": str(replay_path),
            "game_seed": payload.get("game_seed"),
            "turn_index": payload.get("turn_index"),
            "player_id": payload.get("player_id"),
            "phase": _phase_for_turn(int(payload.get("turn_index", 0))),
            "scores": [state.players[0].score, state.players[1].score],
            "before_action_index": before_top.action_index,
            "before_action_type": before_top.action.action_type.name,
            "before_score": before_top.score,
            "before_teacher_match": before_match,
            "before_teacher_gap": before_gap,
            "before_action_payload": serialize_action(before_top.action),
            "after_action_index": after_top.action_index,
            "after_action_type": after_top.action.action_type.name,
            "after_score": after_top.score,
            "after_teacher_match": after_match,
            "after_teacher_gap": after_gap,
            "after_action_payload": serialize_action(after_top.action),
            "teacher_action_index": teacher_top.action_index,
            "teacher_action_type": teacher_top.action_type,
            "teacher_value": teacher_top.value,
            "teacher_action_payload": teacher_top.action_payload,
            "gap_delta_after_minus_before": (
                None
                if before_gap is None or after_gap is None
                else after_gap - before_gap
            ),
            "changed": changed,
            "gain": gain,
            "loss": loss,
        }
    )
    examples.sort(
        key=lambda item: (
            1 if item["loss"] else 0,
            1 if item["gain"] else 0,
            abs(
                (math.inf if item["after_teacher_gap"] is None else item["after_teacher_gap"])
                - (math.inf if item["before_teacher_gap"] is None else item["before_teacher_gap"])
            ),
            before_gap_sort,
            after_gap_sort,
        ),
        reverse=True,
    )
    del examples[examples_limit:]


def _build_summary(
    config: PolicyDiffTeacherConfig,
    counters: dict[str, Any],
    rows_seen: int,
    rows_skipped_by_game_seed: int,
    rows_skipped_by_model_turn: int,
    rows_skipped_by_player_id: int,
    rows_skipped_by_phase: int,
    rows_missing_snapshot: int,
    rows_without_legal_actions: int,
    rows_without_teacher: int,
    examples: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "before_checkpoint": str(config.before_checkpoint),
        "before_model_kind": config.before_model_kind,
        "after_checkpoint": str(config.after_checkpoint),
        "after_model_kind": config.after_model_kind,
        "replay_paths": [str(path) for path in config.replay_paths],
        "teacher_source": config.teacher_source,
        "rows_seen": rows_seen,
        "rows_analyzed": counters["overall"]["count"],
        "rows_skipped_by_game_seed": rows_skipped_by_game_seed,
        "rows_skipped_by_model_turn": rows_skipped_by_model_turn,
        "rows_skipped_by_player_id": rows_skipped_by_player_id,
        "rows_skipped_by_phase": rows_skipped_by_phase,
        "rows_missing_snapshot": rows_missing_snapshot,
        "rows_without_legal_actions": rows_without_legal_actions,
        "rows_without_teacher": rows_without_teacher,
        "overall": _finalize_counter(counters["overall"]),
        "by_phase": _finalize_buckets(counters["by_phase"]),
        "by_seat": _finalize_buckets(counters["by_seat"]),
        "by_teacher_action_type": _finalize_buckets(counters["by_teacher_action_type"]),
        "by_before_action_type": _finalize_buckets(counters["by_before_action_type"]),
        "by_after_action_type": _finalize_buckets(counters["by_after_action_type"]),
        "by_teacher_phase": _finalize_buckets(counters["by_teacher_phase"]),
        "before_after_action_type_matrix": counters["before_after_action_type_matrix"],
        "teacher_to_before_action_type_matrix": counters["teacher_to_before_action_type_matrix"],
        "teacher_to_after_action_type_matrix": counters["teacher_to_after_action_type_matrix"],
        "examples": examples,
        "config": {
            **asdict(config),
            "before_checkpoint": str(config.before_checkpoint),
            "after_checkpoint": str(config.after_checkpoint),
            "replay_paths": [str(path) for path in config.replay_paths],
            "output_path": str(config.output_path),
            "teacher_policy_prior_checkpoint_path": (
                str(config.teacher_policy_prior_checkpoint_path)
                if config.teacher_policy_prior_checkpoint_path is not None
                else None
            ),
        },
    }


def _finalize_buckets(buckets: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        key: _finalize_counter(buckets[key])
        for key in sorted(buckets)
    }


def _finalize_counter(counter: dict[str, Any]) -> dict[str, Any]:
    count = counter["count"]
    return {
        "count": count,
        "before_teacher_matches": counter["before_teacher_matches"],
        "after_teacher_matches": counter["after_teacher_matches"],
        "both_teacher_match": counter["both_teacher_match"],
        "neither_teacher_match": counter["neither_teacher_match"],
        "after_teacher_gains": counter["after_teacher_gains"],
        "after_teacher_losses": counter["after_teacher_losses"],
        "before_after_top_action_agreements": counter["before_after_top_action_agreements"],
        "before_missing_teacher_value": counter["before_missing_teacher_value"],
        "after_missing_teacher_value": counter["after_missing_teacher_value"],
        "before_teacher_match_rate": _rate(counter["before_teacher_matches"], count),
        "after_teacher_match_rate": _rate(counter["after_teacher_matches"], count),
        "before_after_top_action_agreement_rate": _rate(
            counter["before_after_top_action_agreements"],
            count,
        ),
        "before_teacher_gap_mean": _mean(counter["before_teacher_gaps"]),
        "after_teacher_gap_mean": _mean(counter["after_teacher_gaps"]),
        "before_teacher_gap_max": _max_or_none(counter["before_teacher_gaps"]),
        "after_teacher_gap_max": _max_or_none(counter["after_teacher_gaps"]),
    }


def _rate(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else math.nan


def _mean(values: list[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


def _max_or_none(values: list[float]) -> float | None:
    return float(max(values)) if values else None


if __name__ == "__main__":
    main()
