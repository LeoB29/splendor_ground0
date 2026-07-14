"""Audit root decisions changed by an alternative PUCT leaf evaluator."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import json
from pathlib import Path
import statistics
import time
from typing import Any

from splendor_ai.bots import (
    InformationSetPuctBot,
    LegalActionAttentionScorerBot,
    PolicyGuidedSearchBot,
    PuctConfig,
    ShallowSearchBot,
)
from splendor_ai.encoding import ActionCodec
from splendor_ai.engine.actions import Action
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.engine.state import SplendorState
from splendor_ai.eval.match import GameResult, play_game
from splendor_ai.training.replay import serialize_action


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare plain and candidate PUCT decisions on the same roots while "
            "running matched full-game trajectories."
        )
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--opponent",
        choices=("policy", "search", "pgs"),
        default="pgs",
    )
    parser.add_argument("--opponent-checkpoint", default=None)
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--simulations", type=int, default=64)
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--prior-temperature", type=float, default=1.0)
    parser.add_argument("--fpu-reduction", type=float, default=0.0)
    parser.add_argument("--max-depth", type=int, default=80)
    parser.add_argument("--heuristic-scale", type=float, default=4_000.0)
    parser.add_argument("--inference-batch-size", type=int, default=8)
    parser.add_argument("--max-tree-nodes", type=int, default=50_000)
    parser.add_argument("--candidate-n-step-leaf-depth", type=int, default=2)
    parser.add_argument("--candidate-n-step-leaf-width", type=int, default=1)
    parser.add_argument(
        "--candidate-n-step-leaf-candidate-limit", type=int, default=4
    )
    parser.add_argument("--candidate-n-step-leaf-weight", type=float, default=1.0)
    parser.add_argument(
        "--candidate-n-step-leaf-delta-clip", type=float, default=None
    )
    parser.add_argument(
        "--candidate-n-step-leaf-min-turn-index", type=int, default=40
    )
    parser.add_argument(
        "--candidate-n-step-leaf-max-turn-index", type=int, default=None
    )
    parser.add_argument(
        "--candidate-n-step-leaf-min-max-score", type=int, default=None
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-turns", type=int, default=400)
    parser.add_argument("--repetition-limit", type=int, default=4)
    parser.add_argument("--no-progress-limit", type=int, default=60)
    parser.add_argument("--search-depth", type=int, default=2)
    parser.add_argument("--search-determinizations", type=int, default=4)
    parser.add_argument("--search-max-branching", type=int, default=10)
    parser.add_argument("--search-buy-branching", type=int, default=6)
    parser.add_argument("--search-reserve-branching", type=int, default=3)
    parser.add_argument("--search-take-branching", type=int, default=3)
    parser.add_argument("--pgs-prior-weight", type=float, default=3_000.0)
    parser.add_argument("--root-top-k", type=int, default=8)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--log-every", type=int, default=1)
    return parser


class _DecisionDiffBot:
    def __init__(
        self,
        plain: InformationSetPuctBot,
        candidate: InformationSetPuctBot,
        trajectory_policy: str,
        root_top_k: int,
    ) -> None:
        self.plain = plain
        self.candidate = candidate
        self.trajectory_policy = trajectory_policy
        self.root_top_k = root_top_k
        self.rows: list[dict[str, Any]] = []
        self.seed = 0
        self.puct_seat = 0

    def begin_game(self, seed: int, puct_seat: int) -> int:
        self.seed = seed
        self.puct_seat = puct_seat
        return len(self.rows)

    @property
    def loop_fallback_triggers(self) -> int:
        return 0

    def choose_action(
        self,
        env: SplendorEnv,
        state: SplendorState,
        legal_actions: list[Action] | None = None,
    ) -> Action | None:
        actions = legal_actions if legal_actions is not None else env.legal_actions(state)
        if len(actions) <= 1:
            return actions[0] if actions else None

        candidate_action = self.candidate.choose_action(env, state, actions)
        candidate_stats = self.candidate.last_search_stats
        plain_inferred = candidate_stats.n_step_leaf_evaluations == 0
        if plain_inferred:
            plain_action = candidate_action
            plain_stats = candidate_stats
        else:
            plain_action = self.plain.choose_action(env, state, actions)
            plain_stats = self.plain.last_search_stats
        if plain_action is None or candidate_action is None:
            return plain_action if self.trajectory_policy == "plain" else candidate_action

        codec = ActionCodec()
        plain_index = codec.encode(state, plain_action)
        candidate_index = codec.encode(state, candidate_action)
        changed = plain_action != candidate_action
        self.rows.append(
            {
                "seed": self.seed,
                "puct_seat": self.puct_seat,
                "trajectory_policy": self.trajectory_policy,
                "turn_index": state.turn_index,
                "phase": _phase(state.turn_index),
                "current_player": state.current_player,
                "scores": [player.score for player in state.players],
                "purchased_cards": [
                    len(player.purchased_cards) for player in state.players
                ],
                "reserved_cards": [
                    len(player.reserved_cards) for player in state.players
                ],
                "legal_action_type_counts": dict(
                    sorted(Counter(action.action_type.name for action in actions).items())
                ),
                "plain_action_index": plain_index,
                "plain_action_type": plain_action.action_type.name,
                "plain_action_payload": serialize_action(plain_action),
                "plain_action_card": _action_card_payload(state, plain_action),
                "candidate_action_index": candidate_index,
                "candidate_action_type": candidate_action.action_type.name,
                "candidate_action_payload": serialize_action(candidate_action),
                "candidate_action_card": _action_card_payload(
                    state, candidate_action
                ),
                "changed": changed,
                "plain_search_inferred_identical": plain_inferred,
                "plain_search": _search_payload(plain_stats),
                "candidate_search": _search_payload(candidate_stats),
                "transition": (
                    f"{plain_action.action_type.name} -> "
                    f"{candidate_action.action_type.name}"
                ),
                "plain_root": _root_table(
                    state,
                    actions,
                    plain_stats,
                    required_indices={plain_index, candidate_index},
                    top_k=self.root_top_k,
                ) if changed else [],
                "candidate_root": _root_table(
                    state,
                    actions,
                    candidate_stats,
                    required_indices={plain_index, candidate_index},
                    top_k=self.root_top_k,
                ) if changed else [],
            }
        )
        return plain_action if self.trajectory_policy == "plain" else candidate_action


def main() -> None:
    args = build_parser().parse_args()
    if args.games <= 0 or args.games % 2:
        raise ValueError("games must be a positive even number for paired seats.")
    if args.root_top_k <= 0:
        raise ValueError("root-top-k must be positive.")
    if args.opponent in {"policy", "pgs"} and args.opponent_checkpoint is None:
        raise ValueError(f"{args.opponent} requires --opponent-checkpoint.")

    checkpoint = Path(args.checkpoint)
    plain_config = _puct_config(args, candidate=False)
    candidate_config = _puct_config(args, candidate=True)
    plain_trajectory = _build_diff_bot(
        checkpoint, args, plain_config, candidate_config, "plain"
    )
    candidate_trajectory = _build_diff_bot(
        checkpoint, args, plain_config, candidate_config, "candidate"
    )
    plain_opponent = _build_opponent(args, seed_offset=100_000)
    candidate_opponent = _build_opponent(args, seed_offset=100_000)

    conditions: list[dict[str, Any]] = []
    started = time.perf_counter()
    for game_index in range(args.games):
        pair_index = game_index // 2
        seed = args.seed_start + pair_index
        puct_seat = game_index % 2
        plain_start = plain_trajectory.begin_game(seed, puct_seat)
        plain_result = _play_condition(
            plain_trajectory, plain_opponent, puct_seat, seed, args
        )
        candidate_start = candidate_trajectory.begin_game(seed, puct_seat)
        candidate_result = _play_condition(
            candidate_trajectory, candidate_opponent, puct_seat, seed, args
        )
        condition = _condition_summary(
            pair_index=pair_index,
            seed=seed,
            puct_seat=puct_seat,
            plain_result=plain_result,
            candidate_result=candidate_result,
            plain_rows=plain_trajectory.rows[plain_start:],
            candidate_rows=candidate_trajectory.rows[candidate_start:],
        )
        conditions.append(condition)
        for row in plain_trajectory.rows[plain_start:]:
            _attach_condition_result(row, condition)
        for row in candidate_trajectory.rows[candidate_start:]:
            _attach_condition_result(row, condition)

        done = game_index + 1
        if done == 1 or done == args.games or (
            args.log_every > 0 and done % args.log_every == 0
        ):
            changed = sum(item["changed_roots_total"] for item in conditions)
            print(
                f"[puct-decision-diff] {done}/{args.games} conditions"
                f" changed_roots={changed}"
                f" elapsed={time.perf_counter() - started:.1f}s"
            )

    rows = plain_trajectory.rows + candidate_trajectory.rows
    payload = {
        "checkpoint": str(checkpoint),
        "opponent": args.opponent,
        "opponent_checkpoint": args.opponent_checkpoint,
        "games_per_trajectory": args.games,
        "total_full_games": args.games * 2,
        "paired_seeds": args.games // 2,
        "seed_start": args.seed_start,
        "plain_puct_config": _config_payload(plain_config),
        "candidate_puct_config": _config_payload(candidate_config),
        "opponent_config": {
            "depth": args.search_depth,
            "determinizations": args.search_determinizations,
            "information_legal": True,
            "pgs_prior_weight": args.pgs_prior_weight,
        },
        "summary": summarize_audit(conditions, rows),
        "conditions": conditions,
        "decision_rows": rows,
        "elapsed_seconds": time.perf_counter() - started,
    }
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[puct-decision-diff] wrote: {output_path}")


def _build_diff_bot(
    checkpoint: Path,
    args: argparse.Namespace,
    plain_config: PuctConfig,
    candidate_config: PuctConfig,
    trajectory_policy: str,
) -> _DecisionDiffBot:
    return _DecisionDiffBot(
        plain=InformationSetPuctBot(
            checkpoint, device=args.device, config=plain_config
        ),
        candidate=InformationSetPuctBot(
            checkpoint, device=args.device, config=candidate_config
        ),
        trajectory_policy=trajectory_policy,
        root_top_k=args.root_top_k,
    )


def _puct_config(args: argparse.Namespace, candidate: bool) -> PuctConfig:
    return PuctConfig(
        simulations=args.simulations,
        c_puct=args.c_puct,
        prior_temperature=args.prior_temperature,
        fpu_reduction=args.fpu_reduction,
        max_depth=args.max_depth,
        heuristic_scale=args.heuristic_scale,
        seed=args.seed_start,
        inference_batch_size=args.inference_batch_size,
        reuse_tree=False,
        max_tree_nodes=args.max_tree_nodes,
        n_step_leaf_depth=(args.candidate_n_step_leaf_depth if candidate else 0),
        n_step_leaf_width=args.candidate_n_step_leaf_width,
        n_step_leaf_candidate_limit=args.candidate_n_step_leaf_candidate_limit,
        n_step_leaf_weight=args.candidate_n_step_leaf_weight,
        n_step_leaf_delta_clip=(
            args.candidate_n_step_leaf_delta_clip if candidate else None
        ),
        n_step_leaf_min_turn_index=(
            args.candidate_n_step_leaf_min_turn_index if candidate else None
        ),
        n_step_leaf_max_turn_index=(
            args.candidate_n_step_leaf_max_turn_index if candidate else None
        ),
        n_step_leaf_min_max_score=(
            args.candidate_n_step_leaf_min_max_score if candidate else None
        ),
    )


def _build_opponent(args: argparse.Namespace, seed_offset: int):
    checkpoint = (
        Path(args.opponent_checkpoint)
        if args.opponent_checkpoint is not None
        else None
    )
    if args.opponent == "policy":
        assert checkpoint is not None
        return LegalActionAttentionScorerBot(checkpoint, device=args.device)
    search_kwargs = {
        "depth": args.search_depth,
        "max_branching": args.search_max_branching,
        "max_buy_actions": args.search_buy_branching,
        "max_reserve_actions": args.search_reserve_branching,
        "max_take_actions": args.search_take_branching,
        "seed": args.seed_start + seed_offset,
        "information_legal": True,
        "determinizations": args.search_determinizations,
        "determinization_seed": args.seed_start + seed_offset * 2,
    }
    if args.opponent == "search":
        return ShallowSearchBot(**search_kwargs)
    assert checkpoint is not None
    return PolicyGuidedSearchBot(
        checkpoint,
        device=args.device,
        policy_prior_weight=args.pgs_prior_weight,
        **search_kwargs,
    )


def _play_condition(
    puct: _DecisionDiffBot,
    opponent: Any,
    puct_seat: int,
    seed: int,
    args: argparse.Namespace,
) -> GameResult:
    return play_game(
        bot_seat_0=puct if puct_seat == 0 else opponent,
        bot_seat_1=opponent if puct_seat == 0 else puct,
        seed=seed,
        max_turns=args.max_turns,
        repetition_limit=args.repetition_limit,
        no_progress_limit=args.no_progress_limit,
    )


def _root_table(
    state: SplendorState,
    actions: list[Action],
    stats: Any,
    required_indices: set[int],
    top_k: int,
) -> list[dict[str, Any]]:
    codec = ActionCodec()
    action_by_index = {codec.encode(state, action): action for action in actions}
    ranked = sorted(
        stats.root_edge_stats.items(),
        key=lambda item: (
            int(item[1]["visits"]),
            float(item[1]["q"]),
            float(item[1]["prior"]),
        ),
        reverse=True,
    )
    included = {action_index for action_index, _ in ranked[:top_k]}
    included.update(required_indices)
    rank_by_index = {
        action_index: rank for rank, (action_index, _) in enumerate(ranked, start=1)
    }
    rows = []
    for action_index, edge in ranked:
        if action_index not in included or action_index not in action_by_index:
            continue
        values = stats.root_action_values_by_determinization.get(action_index, [])
        action = action_by_index[action_index]
        rows.append(
            {
                "rank": rank_by_index[action_index],
                "action_index": action_index,
                "action_type": action.action_type.name,
                "action_payload": serialize_action(action),
                "action_card": _action_card_payload(state, action),
                "visits": int(edge["visits"]),
                "visit_share": (
                    float(edge["visits"]) / stats.simulations
                    if stats.simulations
                    else 0.0
                ),
                "q": float(edge["q"]),
                "prior": float(edge["prior"]),
                "determinization_value_samples": len(values),
                "determinization_value_std": (
                    statistics.pstdev(values) if len(values) > 1 else 0.0
                ),
            }
        )
    return rows


def _search_payload(stats: Any) -> dict[str, Any]:
    return {
        "simulations": stats.simulations,
        "max_depth_reached": stats.max_depth_reached,
        "n_step_leaf_evaluations": stats.n_step_leaf_evaluations,
        "n_step_leaf_clipped_evaluations": (
            stats.n_step_leaf_clipped_evaluations
        ),
        "n_step_leaf_absolute_delta_sum": (
            stats.n_step_leaf_absolute_delta_sum
        ),
        "n_step_leaf_applied_absolute_delta_sum": (
            stats.n_step_leaf_applied_absolute_delta_sum
        ),
        "elapsed_seconds": stats.elapsed_seconds,
    }


def _condition_summary(
    pair_index: int,
    seed: int,
    puct_seat: int,
    plain_result: GameResult,
    candidate_result: GameResult,
    plain_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    plain_points = _result_points(plain_result, puct_seat)
    candidate_points = _result_points(candidate_result, puct_seat)
    outcome_delta = candidate_points - plain_points
    return {
        "pair_index": pair_index,
        "seed": seed,
        "puct_seat": puct_seat,
        "plain": _result_payload(plain_result, puct_seat),
        "candidate": _result_payload(candidate_result, puct_seat),
        "outcome_delta": outcome_delta,
        "outcome_flip": (
            "improved" if outcome_delta > 0 else "regressed" if outcome_delta < 0 else "unchanged"
        ),
        "score_margin_delta": (
            _score_margin(candidate_result, puct_seat)
            - _score_margin(plain_result, puct_seat)
        ),
        "plain_trajectory_roots": len(plain_rows),
        "candidate_trajectory_roots": len(candidate_rows),
        "plain_trajectory_changed_roots": sum(bool(row["changed"]) for row in plain_rows),
        "candidate_trajectory_changed_roots": sum(
            bool(row["changed"]) for row in candidate_rows
        ),
        "changed_roots_total": sum(bool(row["changed"]) for row in plain_rows + candidate_rows),
    }


def _result_payload(result: GameResult, puct_seat: int) -> dict[str, Any]:
    return {
        "winner": result.winner,
        "puct_points": _result_points(result, puct_seat),
        "final_scores": list(result.final_scores),
        "puct_score_margin": _score_margin(result, puct_seat),
        "turns": result.turns,
        "timed_out": result.timed_out,
        "termination_reason": result.termination_reason,
    }


def _attach_condition_result(row: dict[str, Any], condition: dict[str, Any]) -> None:
    row["outcome_flip"] = condition["outcome_flip"]
    row["outcome_delta"] = condition["outcome_delta"]
    row["score_margin_delta"] = condition["score_margin_delta"]


def summarize_audit(
    conditions: list[dict[str, Any]],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    changed_rows = [row for row in rows if row["changed"]]
    transitions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in changed_rows:
        transitions[str(row["transition"])].append(row)
    return {
        "conditions": len(conditions),
        "candidate_improved_conditions": sum(
            item["outcome_flip"] == "improved" for item in conditions
        ),
        "candidate_regressed_conditions": sum(
            item["outcome_flip"] == "regressed" for item in conditions
        ),
        "candidate_unchanged_conditions": sum(
            item["outcome_flip"] == "unchanged" for item in conditions
        ),
        "mean_candidate_score_margin_delta": (
            statistics.fmean(item["score_margin_delta"] for item in conditions)
            if conditions
            else 0.0
        ),
        "decision_roots": len(rows),
        "changed_roots": len(changed_rows),
        "changed_root_rate": len(changed_rows) / len(rows) if rows else 0.0,
        "changed_roots_by_trajectory": dict(
            sorted(Counter(row["trajectory_policy"] for row in changed_rows).items())
        ),
        "changed_roots_by_phase": dict(
            sorted(Counter(row["phase"] for row in changed_rows).items())
        ),
        "changed_roots_by_outcome_flip": dict(
            sorted(Counter(row["outcome_flip"] for row in changed_rows).items())
        ),
        "changed_action_transitions": {
            transition: {
                "count": len(group),
                "conditions": len(
                    {(row["seed"], row["puct_seat"]) for row in group}
                ),
                "improved_count": sum(
                    row["outcome_flip"] == "improved" for row in group
                ),
                "regressed_count": sum(
                    row["outcome_flip"] == "regressed" for row in group
                ),
                "mean_score_margin_delta": statistics.fmean(
                    row["score_margin_delta"] for row in group
                ),
            }
            for transition, group in sorted(
                transitions.items(), key=lambda item: (-len(item[1]), item[0])
            )
        },
        "changed_root_signal": _changed_root_signal(changed_rows),
    }


def _changed_root_signal(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = []
    for row in rows:
        plain_root = row.get("plain_root", [])
        candidate_root = row.get("candidate_root", [])
        plain_action_index = row.get("plain_action_index")
        candidate_action_index = row.get("candidate_action_index")
        if plain_action_index is None or candidate_action_index is None:
            continue
        plain_selected = _find_root_entry(
            plain_root, plain_action_index
        )
        plain_candidate = _find_root_entry(
            plain_root, candidate_action_index
        )
        candidate_selected = _find_root_entry(
            candidate_root, candidate_action_index
        )
        candidate_plain = _find_root_entry(
            candidate_root, plain_action_index
        )
        if not all(
            (plain_selected, plain_candidate, candidate_selected, candidate_plain)
        ):
            continue
        metrics.append(
            {
                "outcome_flip": row["outcome_flip"],
                "candidate_lower_prior": (
                    candidate_selected["prior"] < plain_selected["prior"]
                ),
                "candidate_plain_rank": plain_candidate["rank"],
                "candidate_q_uplift": (
                    candidate_selected["q"] - plain_candidate["q"]
                ),
                "plain_action_q_shift": (
                    candidate_plain["q"] - plain_selected["q"]
                ),
                "candidate_visit_share": candidate_selected["visit_share"],
            }
        )
    grouped = {
        label: [item for item in metrics if item["outcome_flip"] == label]
        for label in ("improved", "regressed", "unchanged")
    }
    return {
        "roots": len(metrics),
        "candidate_lower_prior_count": sum(
            item["candidate_lower_prior"] for item in metrics
        ),
        "mean_candidate_plain_rank": _mean_metric(
            metrics, "candidate_plain_rank"
        ),
        "mean_candidate_q_uplift": _mean_metric(metrics, "candidate_q_uplift"),
        "mean_plain_action_q_shift": _mean_metric(
            metrics, "plain_action_q_shift"
        ),
        "mean_candidate_visit_share": _mean_metric(
            metrics, "candidate_visit_share"
        ),
        "by_outcome_flip": {
            label: {
                "roots": len(group),
                "candidate_lower_prior_count": sum(
                    item["candidate_lower_prior"] for item in group
                ),
                "mean_candidate_q_uplift": _mean_metric(
                    group, "candidate_q_uplift"
                ),
                "mean_plain_action_q_shift": _mean_metric(
                    group, "plain_action_q_shift"
                ),
                "mean_candidate_visit_share": _mean_metric(
                    group, "candidate_visit_share"
                ),
            }
            for label, group in grouped.items()
        },
    }


def _find_root_entry(
    root: list[dict[str, Any]], action_index: int
) -> dict[str, Any] | None:
    return next(
        (item for item in root if item["action_index"] == action_index),
        None,
    )


def _mean_metric(rows: list[dict[str, Any]], key: str) -> float:
    return statistics.fmean(float(row[key]) for row in rows) if rows else 0.0


def _action_card_payload(
    state: SplendorState, action: Action
) -> dict[str, Any] | None:
    card = None
    if action.action_type.name in {"BUY_VISIBLE", "RESERVE_VISIBLE"}:
        if action.tier is not None and action.market_index is not None:
            cards = state.visible_tier_cards.get(action.tier, [])
            if 0 <= action.market_index < len(cards):
                card = cards[action.market_index]
    elif action.action_type.name == "BUY_RESERVED":
        if action.reserved_index is not None:
            cards = state.players[state.current_player].reserved_cards
            if 0 <= action.reserved_index < len(cards):
                card = cards[action.reserved_index]
    if card is None:
        return None
    return {
        "card_id": card.card_id,
        "tier": card.tier,
        "bonus_color": card.bonus_color,
        "points": card.points,
        "cost": dict(card.cost),
    }


def _config_payload(config: PuctConfig) -> dict[str, Any]:
    return asdict(config)


def _result_points(result: GameResult, puct_seat: int) -> float:
    if result.winner is None:
        return 0.5
    return 1.0 if result.winner == puct_seat else 0.0


def _score_margin(result: GameResult, puct_seat: int) -> int:
    return result.final_scores[puct_seat] - result.final_scores[1 - puct_seat]


def _phase(turn_index: int) -> str:
    if turn_index < 20:
        return "early"
    if turn_index < 40:
        return "middle"
    return "late"


if __name__ == "__main__":
    main()
