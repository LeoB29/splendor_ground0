"""Forensic action-quality audit for held-out human-versus-policy games."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
import statistics
from typing import Any

from splendor_ai.bots.hidden_information import determinize_for_observer
from splendor_ai.encoding import ActionCodec
from splendor_ai.engine.actions import Action, ActionType
from splendor_ai.engine.constants import ALL_TOKEN_TYPES
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.engine.state import SplendorState
from splendor_ai.training.replay import (
    deserialize_state_snapshot,
    serialize_action,
)
from splendor_ai.training.run_policy_search_disagreement import _load_policy_scorer
from splendor_ai.training.run_policy_drift import _ScorerConfig
from splendor_ai.training.run_rollout_action_credit import rollout_action_return


@dataclass(frozen=True, slots=True)
class HumanGameForensicAuditConfig:
    replay_path: Path
    policy_checkpoint: Path
    output_path: Path
    report_path: Path
    progress_path: Path | None = None
    tactical_determinizations: int = 8
    max_rollout_states: int = 40
    max_candidates_per_state: int = 8
    policy_top_k: int = 4
    rollout_determinizations: int = 3
    rollouts_per_determinization: int = 8
    rollout_epsilon: float = 0.1
    rollout_temperature: float = 0.5
    rollout_top_k: int = 3
    rollout_max_additional_turns: int = 120
    outcome_weight: float = 0.75
    score_margin_scale: float = 5.0
    min_resolved_delta: float = 0.10
    confidence_z: float = 1.96
    repeated_nonbuy_turns: int = 4
    seed: int = 91
    device: str = "cpu"
    log_every_states: int = 5
    retain_rollout_returns: bool = False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Classify every policy turn in held-out human games and refine suspicious states."
    )
    parser.add_argument("--replay-path", required=True)
    parser.add_argument("--policy-checkpoint", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--report-path", required=True)
    parser.add_argument("--progress-path", default=None)
    parser.add_argument("--tactical-determinizations", type=int, default=8)
    parser.add_argument("--max-rollout-states", type=int, default=40)
    parser.add_argument("--max-candidates-per-state", type=int, default=8)
    parser.add_argument("--policy-top-k", type=int, default=4)
    parser.add_argument("--rollout-determinizations", type=int, default=3)
    parser.add_argument("--rollouts-per-determinization", type=int, default=8)
    parser.add_argument("--rollout-epsilon", type=float, default=0.1)
    parser.add_argument("--rollout-temperature", type=float, default=0.5)
    parser.add_argument("--rollout-top-k", type=int, default=3)
    parser.add_argument("--rollout-max-additional-turns", type=int, default=120)
    parser.add_argument("--outcome-weight", type=float, default=0.75)
    parser.add_argument("--score-margin-scale", type=float, default=5.0)
    parser.add_argument("--min-resolved-delta", type=float, default=0.10)
    parser.add_argument("--confidence-z", type=float, default=1.96)
    parser.add_argument("--repeated-nonbuy-turns", type=int, default=4)
    parser.add_argument("--seed", type=int, default=91)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--log-every-states", type=int, default=5)
    parser.add_argument("--retain-rollout-returns", action="store_true")
    return parser


def run_human_game_forensic_audit(
    config: HumanGameForensicAuditConfig,
) -> dict[str, Any]:
    _validate(config)
    replay_rows = _load_rows(config.replay_path)
    model_rows = [row for row in replay_rows if _is_model_turn(row)]
    if not model_rows:
        raise ValueError("The replay contains no model turns.")
    scorer = _load_policy_scorer(
        _ScorerConfig(
            checkpoint=config.policy_checkpoint,
            model_kind="legal_action_attention_scorer",
            device=config.device,
        )
    )
    env = SplendorEnv()
    codec = ActionCodec()
    reserve_outcomes = _reserve_outcomes(replay_rows, env, codec)

    screens: list[dict[str, Any]] = []
    nonbuy_streaks: dict[tuple[int, int], int] = defaultdict(int)
    for position, payload in enumerate(model_rows):
        key = (int(payload["game_seed"]), int(payload["player_id"]))
        screen = _screen_turn(
            payload=payload,
            position=position,
            scorer=scorer,
            env=env,
            codec=codec,
            reserve_outcome=reserve_outcomes.get(_row_key(payload)),
            nonbuy_streak_before=nonbuy_streaks[key],
            config=config,
        )
        screens.append(screen)
        if screen["chosen_action_type"] in {"BUY_VISIBLE", "BUY_RESERVED"}:
            nonbuy_streaks[key] = 0
        else:
            nonbuy_streaks[key] += 1

    selected = _select_suspicious_states(screens, config.max_rollout_states)
    progress_path = config.progress_path or config.output_path.with_suffix(".progress.json")
    existing = json.loads(progress_path.read_text(encoding="utf-8")) if progress_path.exists() else {}
    existing_results = {
        _result_key(row): row for row in existing.get("rollout_results", [])
    }
    rollout_results: list[dict[str, Any]] = []
    payload_by_key = {_row_key(row): row for row in model_rows}
    for state_position, screen in enumerate(selected):
        key = _result_key(screen)
        if key in existing_results:
            rollout_results.append(existing_results[key])
            continue
        result = _evaluate_suspicious_state(
            payload=payload_by_key[key],
            screen=screen,
            state_position=state_position,
            scorer=scorer,
            env=env,
            codec=codec,
            config=config,
        )
        rollout_results.append(result)
        _write_progress(progress_path, screens, selected, rollout_results)
        if config.log_every_states > 0 and (state_position + 1) % config.log_every_states == 0:
            print(
                f"[human-forensic] rollout_states={state_position + 1}/{len(selected)}"
                f" resolved={sum(bool(row['resolved_mistake']) for row in rollout_results)}"
            )

    result_by_key = {_result_key(row): row for row in rollout_results}
    for screen in screens:
        screen["rollout_evidence"] = result_by_key.get(_result_key(screen))
        screen["high_confidence_mistake"] = bool(
            screen["exact_tactical_mistake"]
            or (
                screen["rollout_evidence"] is not None
                and screen["rollout_evidence"]["resolved_mistake"]
            )
        )
    summary = _summarize(replay_rows, screens, rollout_results)
    output = {
        "audit": "human_game_forensic_action_quality_v1",
        "config": _config_payload(config, progress_path),
        "summary": summary,
        "games": _game_summary(replay_rows, screens),
        "turns": screens,
        "rollout_results": rollout_results,
    }
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    config.output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    config.report_path.parent.mkdir(parents=True, exist_ok=True)
    config.report_path.write_text(_render_report(output), encoding="utf-8")
    _write_progress(progress_path, screens, selected, rollout_results, complete=True)
    return output


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows.sort(key=lambda row: (int(row["game_seed"]), int(row["turn_index"])))
    return rows


def _is_model_turn(payload: dict[str, Any]) -> bool:
    player = int(payload.get("player_id", -1))
    metadata = payload.get("game_bot_metadata", [])
    return (
        isinstance(metadata, list)
        and 0 <= player < len(metadata)
        and isinstance(metadata[player], dict)
        and metadata[player].get("gui_opponent_kind") == "model"
    )


def _row_key(payload: dict[str, Any]) -> tuple[int, int, int]:
    return int(payload["game_seed"]), int(payload["turn_index"]), int(payload["player_id"])


def _result_key(payload: dict[str, Any]) -> tuple[int, int, int]:
    return _row_key(payload)


def _action_map(state: SplendorState, env: SplendorEnv, codec: ActionCodec) -> dict[int, Action]:
    actions = env.legal_actions(state)
    return dict(zip(codec.legal_action_indices(state, actions), actions, strict=True))


def _score_gain(state: SplendorState, next_state: SplendorState, player: int) -> int:
    return next_state.players[player].score - state.players[player].score


def _screen_turn(
    *,
    payload: dict[str, Any],
    position: int,
    scorer: Any,
    env: SplendorEnv,
    codec: ActionCodec,
    reserve_outcome: dict[str, Any] | None,
    nonbuy_streak_before: int,
    config: HumanGameForensicAuditConfig,
) -> dict[str, Any]:
    state = deserialize_state_snapshot(payload["state_snapshot"])
    player = state.current_player
    actions_by_index = _action_map(state, env, codec)
    chosen_index = int(payload["action_index"])
    chosen = actions_by_index[chosen_index]
    scored = scorer.score_actions(state, list(actions_by_index.values()))
    policy_rank = {
        item.action_index: rank for rank, item in enumerate(scored, start=1)
    }
    facts = {
        action_index: _immediate_action_facts(state, env, action)
        for action_index, action in actions_by_index.items()
    }
    scoring_indices = sorted(
        index for index, fact in facts.items() if int(fact["score_gain"]) > 0
    )
    max_score_gain = max((int(fact["score_gain"]) for fact in facts.values()), default=0)
    chosen_fact = facts[chosen_index]
    dominated_by = _same_target_dominators(chosen_index, actions_by_index, facts)

    tactical = _robust_tactical_screen(
        state=state,
        chosen_index=chosen_index,
        env=env,
        codec=codec,
        determinizations=config.tactical_determinizations,
        seed=config.seed + position * 1_000_003,
    )
    categories: list[str] = []
    if tactical["missed_forced_win"]:
        categories.append("missed_forced_win")
    if tactical["failed_immediate_win_block"]:
        categories.append("failed_immediate_win_block")
    if dominated_by:
        categories.append("same_target_immediate_dominance")
    conversion_delay = (
        chosen.action_type not in {ActionType.BUY_VISIBLE, ActionType.BUY_RESERVED}
        and max_score_gain > 0
    )
    if conversion_delay:
        categories.append("conversion_delay")

    reserve_flags: list[str] = []
    if chosen.action_type in {ActionType.RESERVE_VISIBLE, ActionType.RESERVE_DECK}:
        if chosen.action_type == ActionType.RESERVE_VISIBLE and _visible_target_affordable(chosen, actions_by_index.values()):
            reserve_flags.append("reserved_affordable_visible_card")
        if scoring_indices:
            reserve_flags.append("reserved_while_scoring_buy_available")
        if chosen.action_type == ActionType.RESERVE_DECK and scoring_indices:
            reserve_flags.append("blind_reserve_while_scoring_buy_available")
        if reserve_outcome is not None and not reserve_outcome["converted"]:
            reserve_flags.append("reserved_card_unconverted")
    if reserve_flags:
        categories.append("reserve_misuse_suspicion")

    nonbuy_streak_after = (
        0
        if chosen.action_type in {ActionType.BUY_VISIBLE, ActionType.BUY_RESERVED}
        else nonbuy_streak_before + 1
    )
    if nonbuy_streak_after >= config.repeated_nonbuy_turns:
        categories.append("repeated_low_progress")

    risk = _risk_score(
        categories=categories,
        max_score_gain=max_score_gain,
        score=state.players[player].score,
        nonbuy_streak=nonbuy_streak_after,
        reserve_flags=reserve_flags,
    )
    alternatives = sorted(
        set(tactical["robust_forced_win_action_indices"])
        | set(tactical["robust_blocking_action_indices"])
        | set(dominated_by)
        | set(scoring_indices)
    )
    return {
        "game_seed": int(payload["game_seed"]),
        "turn_index": int(payload["turn_index"]),
        "player_id": player,
        "human_player_id": int(payload.get("human_player_id", 1 - player)),
        "game_winner": payload.get("game_winner"),
        "game_final_scores": payload.get("game_final_scores"),
        "phase": _phase(int(payload["turn_index"])),
        "current_scores": [item.score for item in state.players],
        "chosen_action_index": chosen_index,
        "chosen_action_type": chosen.action_type.name,
        "chosen_action": _describe_action(state, chosen),
        "chosen_policy_rank": policy_rank.get(chosen_index),
        "policy_top_matches_replay": bool(scored and scored[0].action_index == chosen_index),
        "legal_action_count": len(actions_by_index),
        "chosen_score_gain": int(chosen_fact["score_gain"]),
        "max_immediate_score_gain": max_score_gain,
        "scoring_action_indices": scoring_indices,
        "same_target_dominating_action_indices": dominated_by,
        "nonbuy_streak_before": nonbuy_streak_before,
        "nonbuy_streak_after": nonbuy_streak_after,
        "reserve_flags": reserve_flags,
        "reserve_outcome": reserve_outcome,
        "categories": categories,
        "risk_score": risk,
        "suspicious": bool(categories),
        "exact_tactical_mistake": bool(
            tactical["missed_forced_win"] or tactical["failed_immediate_win_block"]
        ),
        "tactical": tactical,
        "priority_alternative_action_indices": alternatives,
    }


def _immediate_action_facts(
    state: SplendorState,
    env: SplendorEnv,
    action: Action,
) -> dict[str, Any]:
    player = state.current_player
    next_state = env.step(state, action)
    before = state.players[player]
    after = next_state.players[player]
    return {
        "score_gain": _score_gain(state, next_state, player),
        "tokens": [after.tokens[color] for color in ALL_TOKEN_TYPES],
        "bonuses": dict(after.bonuses),
        "reserved_card_ids": [card.card_id for card in after.reserved_cards],
        "purchased_card_ids": [card.card_id for card in after.purchased_cards],
        "terminal": next_state.terminal,
        "winner": next_state.winner,
    }


def _same_target_key(action: Action) -> tuple[Any, ...]:
    return (
        action.action_type,
        action.tier,
        action.market_index,
        action.reserved_index,
        action.noble_id,
        action.take_tokens,
    )


def _same_target_dominators(
    chosen_index: int,
    actions_by_index: dict[int, Action],
    facts: dict[int, dict[str, Any]],
) -> list[int]:
    chosen_action = actions_by_index[chosen_index]
    chosen = facts[chosen_index]
    dominators: list[int] = []
    for action_index, action in actions_by_index.items():
        if action_index == chosen_index or _same_target_key(action) != _same_target_key(chosen_action):
            continue
        candidate = facts[action_index]
        token_no_worse = all(
            left >= right
            for left, right in zip(candidate["tokens"], chosen["tokens"], strict=True)
        )
        other_equal = (
            candidate["bonuses"] == chosen["bonuses"]
            and candidate["reserved_card_ids"] == chosen["reserved_card_ids"]
            and candidate["purchased_card_ids"] == chosen["purchased_card_ids"]
        )
        score_no_worse = int(candidate["score_gain"]) >= int(chosen["score_gain"])
        strict = candidate["tokens"] != chosen["tokens"] or int(candidate["score_gain"]) > int(chosen["score_gain"])
        if token_no_worse and other_equal and score_no_worse and strict:
            dominators.append(action_index)
    return sorted(dominators)


def _visible_target_affordable(chosen: Action, legal_actions: Any) -> bool:
    return any(
        action.action_type == ActionType.BUY_VISIBLE
        and action.tier == chosen.tier
        and action.market_index == chosen.market_index
        for action in legal_actions
    )


def _robust_tactical_screen(
    *,
    state: SplendorState,
    chosen_index: int,
    env: SplendorEnv,
    codec: ActionCodec,
    determinizations: int,
    seed: int,
) -> dict[str, Any]:
    forced_sets: list[set[int]] = []
    blocker_sets: list[set[int]] = []
    chosen_threats: list[bool] = []
    for index in range(determinizations):
        determinized = determinize_for_observer(
            state,
            state.current_player,
            random.Random(seed + index),
        )
        action_map = _action_map(determinized, env, codec)
        forced = {
            action_index
            for action_index, action in action_map.items()
            if _guarantees_root_win(env, determinized, action, state.current_player)
        }
        forced_sets.append(forced)
        chosen_action = action_map[chosen_index]
        chosen_child = env.step(determinized, chosen_action)
        threat = bool(_immediate_winning_replies(env, chosen_child, 1 - state.current_player))
        chosen_threats.append(threat)
        blockers: set[int] = set()
        if threat:
            for action_index, action in action_map.items():
                child = env.step(determinized, action)
                if not _immediate_winning_replies(env, child, 1 - state.current_player):
                    blockers.add(action_index)
        blocker_sets.append(blockers)
    robust_forced = set.intersection(*forced_sets) if forced_sets else set()
    robust_blockers = set.intersection(*blocker_sets) if blocker_sets else set()
    missed_forced = bool(robust_forced and chosen_index not in robust_forced)
    failed_block = bool(
        chosen_threats
        and all(chosen_threats)
        and robust_blockers
        and chosen_index not in robust_blockers
    )
    return {
        "determinizations": determinizations,
        "robust_forced_win_action_indices": sorted(robust_forced),
        "forced_win_action_union": sorted(set().union(*forced_sets) if forced_sets else set()),
        "chosen_leaves_immediate_loss_rate": statistics.fmean(chosen_threats) if chosen_threats else 0.0,
        "robust_blocking_action_indices": sorted(robust_blockers),
        "missed_forced_win": missed_forced,
        "failed_immediate_win_block": failed_block,
    }


def _guarantees_root_win(
    env: SplendorEnv,
    state: SplendorState,
    action: Action,
    root_player: int,
) -> bool:
    child = env.step(state, action)
    if child.terminal:
        return child.winner == root_player
    if not child.pending_round_end:
        return False
    replies = env.legal_actions(child)
    if not replies:
        return False
    reply_states = [env.step(child, reply) for reply in replies]
    return all(reply.terminal and reply.winner == root_player for reply in reply_states)


def _immediate_winning_replies(
    env: SplendorEnv,
    state: SplendorState,
    opponent: int,
) -> list[Action]:
    if state.terminal or state.current_player != opponent:
        return []
    return [
        action
        for action in env.legal_actions(state)
        if (lambda child: child.terminal and child.winner == opponent)(env.step(state, action))
    ]


def _risk_score(
    *,
    categories: list[str],
    max_score_gain: int,
    score: int,
    nonbuy_streak: int,
    reserve_flags: list[str],
) -> float:
    weights = {
        "missed_forced_win": 100.0,
        "failed_immediate_win_block": 90.0,
        "same_target_immediate_dominance": 50.0,
        "conversion_delay": 25.0,
        "reserve_misuse_suspicion": 15.0,
        "repeated_low_progress": 10.0,
    }
    return (
        sum(weights[item] for item in categories)
        + 3.0 * max_score_gain
        + max(score - 10, 0)
        + max(nonbuy_streak - 3, 0)
        + 2.0 * len(reserve_flags)
    )


def _select_suspicious_states(
    screens: list[dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    suspicious = [row for row in screens if row["suspicious"]]
    return sorted(
        suspicious,
        key=lambda row: (-float(row["risk_score"]), int(row["game_seed"]), int(row["turn_index"])),
    )[:limit]


def _evaluate_suspicious_state(
    *,
    payload: dict[str, Any],
    screen: dict[str, Any],
    state_position: int,
    scorer: Any,
    env: SplendorEnv,
    codec: ActionCodec,
    config: HumanGameForensicAuditConfig,
) -> dict[str, Any]:
    state = deserialize_state_snapshot(payload["state_snapshot"])
    action_map = _action_map(state, env, codec)
    scored = scorer.score_actions(state, list(action_map.values()))
    score_by_index = {item.action_index: item.score for item in scored}
    candidate_indices = _candidate_indices(screen, state, action_map, scored, config)
    returns: dict[int, list[float]] = {index: [] for index in candidate_indices}
    by_determinization: dict[int, list[list[float]]] = {
        index: [] for index in candidate_indices
    }
    base_seed = config.seed + 9_000_000_000 + state_position * 10_000_019
    for det_index in range(config.rollout_determinizations):
        determinized = determinize_for_observer(
            state,
            state.current_player,
            random.Random(base_seed + det_index),
        )
        det_actions = _action_map(determinized, env, codec)
        for action_index in candidate_indices:
            action = det_actions.get(action_index)
            if action is None:
                continue
            values = [
                rollout_action_return(
                    env=env,
                    scorer=scorer,
                    state=determinized,
                    root_action=action,
                    root_player=state.current_player,
                    seed=base_seed + det_index * 100_003 + rollout_index,
                    epsilon=config.rollout_epsilon,
                    temperature=config.rollout_temperature,
                    top_k=config.rollout_top_k,
                    max_additional_turns=config.rollout_max_additional_turns,
                    outcome_weight=config.outcome_weight,
                    score_margin_scale=config.score_margin_scale,
                )
                for rollout_index in range(config.rollouts_per_determinization)
            ]
            returns[action_index].extend(values)
            by_determinization[action_index].append(values)
    action_rows = []
    for action_index in candidate_indices:
        values = returns[action_index]
        if not values:
            continue
        action_rows.append(
            {
                "action_index": action_index,
                "action": _describe_action(state, action_map[action_index]),
                "prior_score": float(score_by_index[action_index]),
                "rollout_return": statistics.fmean(values),
                "rollout_standard_error": _standard_error(values),
                "rollout_count": len(values),
                "determinization_means": [
                    statistics.fmean(items) for items in by_determinization[action_index]
                ],
                "returns": values,
            }
        )
    chosen_index = int(screen["chosen_action_index"])
    chosen_row = next(row for row in action_rows if row["action_index"] == chosen_index)
    alternatives = [row for row in action_rows if row["action_index"] != chosen_index]
    best = max(
        alternatives,
        key=lambda row: (float(row["rollout_return"]), -int(row["action_index"])),
    )
    delta_values = [
        float(left) - float(right)
        for left, right in zip(best["returns"], chosen_row["returns"], strict=True)
    ]
    delta = statistics.fmean(delta_values)
    delta_se = _standard_error(delta_values)
    det_deltas = [
        float(left) - float(right)
        for left, right in zip(
            best["determinization_means"],
            chosen_row["determinization_means"],
            strict=True,
        )
    ]
    resolved = bool(
        delta >= config.min_resolved_delta
        and delta > config.confidence_z * delta_se
        and det_deltas
        and min(det_deltas) > 0.0
    )
    if not config.retain_rollout_returns:
        for row in action_rows:
            row.pop("returns")
    return {
        "game_seed": int(payload["game_seed"]),
        "turn_index": int(payload["turn_index"]),
        "player_id": state.current_player,
        "categories": list(screen["categories"]),
        "candidate_count": len(action_rows),
        "chosen_action_index": chosen_index,
        "best_alternative_action_index": int(best["action_index"]),
        "paired_return_delta": delta,
        "paired_return_delta_standard_error": delta_se,
        "determinization_deltas": det_deltas,
        "resolved_mistake": resolved,
        "actions": sorted(action_rows, key=lambda row: (-float(row["rollout_return"]), int(row["action_index"]))),
    }


def _candidate_indices(
    screen: dict[str, Any],
    state: SplendorState,
    action_map: dict[int, Action],
    scored: tuple[Any, ...],
    config: HumanGameForensicAuditConfig,
) -> list[int]:
    chosen = int(screen["chosen_action_index"])
    prioritized: list[int] = [chosen]
    for action_type in ActionType:
        type_top = next(
            (item.action_index for item in scored if item.action.action_type == action_type),
            None,
        )
        if type_top is not None:
            prioritized.append(type_top)
    prioritized.extend(int(index) for index in screen["priority_alternative_action_indices"])
    prioritized.extend(item.action_index for item in scored[: config.policy_top_k])
    semantic_seen: set[tuple[Any, ...]] = set()
    selected: list[int] = []
    for action_index in dict.fromkeys(prioritized):
        action = action_map.get(action_index)
        if action is None:
            continue
        semantic = _semantic_action_key(action)
        if action_index != chosen and semantic in semantic_seen:
            continue
        selected.append(action_index)
        semantic_seen.add(semantic)
        if len(selected) >= config.max_candidates_per_state:
            break
    if len(selected) < 2:
        for item in scored:
            if item.action_index not in selected:
                selected.append(item.action_index)
                break
    return selected


def _semantic_action_key(action: Action) -> tuple[Any, ...]:
    return (
        action.action_type,
        action.tier,
        action.market_index,
        action.reserved_index,
        action.noble_id,
        action.take_tokens,
    )


def _reserve_outcomes(
    replay_rows: list[dict[str, Any]],
    env: SplendorEnv,
    codec: ActionCodec,
) -> dict[tuple[int, int, int], dict[str, Any]]:
    by_game: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in replay_rows:
        by_game[int(row["game_seed"])].append(row)
    output: dict[tuple[int, int, int], dict[str, Any]] = {}
    for rows in by_game.values():
        model_buys: list[tuple[int, int, str]] = []
        reserves: list[tuple[dict[str, Any], str]] = []
        for row in rows:
            if not _is_model_turn(row):
                continue
            state = deserialize_state_snapshot(row["state_snapshot"])
            action = _action_map(state, env, codec)[int(row["action_index"])]
            card_id = _target_card_id(state, action)
            if card_id is None:
                continue
            if action.action_type in {ActionType.BUY_VISIBLE, ActionType.BUY_RESERVED}:
                model_buys.append((int(row["turn_index"]), int(row["player_id"]), card_id))
            elif action.action_type in {ActionType.RESERVE_VISIBLE, ActionType.RESERVE_DECK}:
                reserves.append((row, card_id))
        for row, card_id in reserves:
            turn = int(row["turn_index"])
            player = int(row["player_id"])
            later = [buy_turn for buy_turn, buy_player, bought in model_buys if buy_player == player and bought == card_id and buy_turn > turn]
            output[_row_key(row)] = {
                "card_id": card_id,
                "converted": bool(later),
                "turns_to_conversion": min(later) - turn if later else None,
            }
    return output


def _target_card_id(state: SplendorState, action: Action) -> str | None:
    if action.action_type in {ActionType.BUY_VISIBLE, ActionType.RESERVE_VISIBLE}:
        if action.tier is None or action.market_index is None:
            return None
        return state.visible_tier_cards[action.tier][action.market_index].card_id
    if action.action_type == ActionType.RESERVE_DECK:
        if action.tier is None or not state.hidden_tier_decks[action.tier]:
            return None
        return state.hidden_tier_decks[action.tier][0].card_id
    if action.action_type == ActionType.BUY_RESERVED:
        if action.reserved_index is None:
            return None
        return state.players[state.current_player].reserved_cards[action.reserved_index].card_id
    return None


def _describe_action(state: SplendorState, action: Action) -> dict[str, Any]:
    card_id = _target_card_id(state, action)
    card = None
    if action.action_type in {ActionType.BUY_VISIBLE, ActionType.RESERVE_VISIBLE} and action.tier is not None and action.market_index is not None:
        card = state.visible_tier_cards[action.tier][action.market_index]
    elif action.action_type == ActionType.BUY_RESERVED and action.reserved_index is not None:
        card = state.players[state.current_player].reserved_cards[action.reserved_index]
    return {
        **serialize_action(action),
        "card_id": card_id,
        "card_points": card.points if card is not None else None,
        "card_bonus": card.bonus_color if card is not None else None,
        "card_cost": card.cost if card is not None else None,
    }


def _standard_error(values: list[float]) -> float:
    return statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0


def _phase(turn_index: int) -> str:
    if turn_index < 20:
        return "early"
    if turn_index < 50:
        return "mid"
    return "late"


def _summarize(
    replay_rows: list[dict[str, Any]],
    screens: list[dict[str, Any]],
    rollout_results: list[dict[str, Any]],
) -> dict[str, Any]:
    categories = Counter(category for row in screens for category in row["categories"])
    exact = [row for row in screens if row["exact_tactical_mistake"]]
    resolved_rollouts = [row for row in rollout_results if row["resolved_mistake"]]
    resolved_categories = Counter(
        category for row in resolved_rollouts for category in row["categories"]
    )
    high_confidence_keys = {
        _result_key(row) for row in exact
    } | {
        _result_key(row) for row in resolved_rollouts
    }
    screen_by_key = {_result_key(row): row for row in screens}
    transitions: Counter[str] = Counter()
    resolved_phases: Counter[str] = Counter()
    resolved_reserve_flags: Counter[str] = Counter()
    resolved_in_human_wins = 0
    prior_gaps: list[float] = []
    for row in resolved_rollouts:
        screen = screen_by_key[_result_key(row)]
        chosen = next(
            item for item in row["actions"]
            if item["action_index"] == row["chosen_action_index"]
        )
        best = next(
            item for item in row["actions"]
            if item["action_index"] == row["best_alternative_action_index"]
        )
        transitions[
            f"{chosen['action']['action_type']} -> {best['action']['action_type']}"
        ] += 1
        resolved_phases[str(screen["phase"])] += 1
        resolved_reserve_flags.update(screen["reserve_flags"])
        resolved_in_human_wins += int(
            int(screen["game_winner"]) == int(screen["human_player_id"])
        )
        prior_gaps.append(float(chosen["prior_score"]) - float(best["prior_score"]))
    diagnosis = _diagnosis(exact, resolved_rollouts, resolved_categories)
    return {
        "games": len({int(row["game_seed"]) for row in replay_rows}),
        "replay_rows": len(replay_rows),
        "model_turns": len(screens),
        "policy_top_replay_match_rate": statistics.fmean(
            bool(row["policy_top_matches_replay"]) for row in screens
        ),
        "suspicious_turns": sum(bool(row["suspicious"]) for row in screens),
        "category_counts": dict(sorted(categories.items())),
        "exact_tactical_mistakes": len(exact),
        "rollout_states": len(rollout_results),
        "resolved_rollout_mistakes": len(resolved_rollouts),
        "resolved_rollout_category_counts": dict(sorted(resolved_categories.items())),
        "resolved_rollout_by_phase": dict(sorted(resolved_phases.items())),
        "resolved_rollout_action_transitions": dict(sorted(transitions.items())),
        "resolved_rollout_reserve_flags": dict(sorted(resolved_reserve_flags.items())),
        "resolved_rollout_in_human_wins": resolved_in_human_wins,
        "resolved_rollout_mean_paired_delta": (
            statistics.fmean(float(row["paired_return_delta"]) for row in resolved_rollouts)
            if resolved_rollouts else 0.0
        ),
        "resolved_rollout_mean_chosen_minus_best_prior_score": (
            statistics.fmean(prior_gaps) if prior_gaps else 0.0
        ),
        "high_confidence_mistakes": len(high_confidence_keys),
        "diagnosis": diagnosis,
    }


def _diagnosis(
    exact: list[dict[str, Any]],
    resolved: list[dict[str, Any]],
    resolved_categories: Counter[str],
) -> dict[str, Any]:
    tactical = len(exact)
    conversion = resolved_categories["conversion_delay"]
    reserve = resolved_categories["reserve_misuse_suspicion"]
    low_progress = resolved_categories["repeated_low_progress"]
    if tactical >= 3:
        primary = "tactical"
        recommendation = "Build and gate a phase-limited forced-win/immediate-threat solver around policy035."
    elif conversion >= 3 and conversion >= max(reserve, low_progress):
        primary = "conversion"
        recommendation = "Target conversion planning with a new trajectory-level or multi-generation game-outcome method; do not add a tactical wrapper alone."
    elif len(resolved) >= 3:
        primary = "broader_planning"
        recommendation = "The errors are not concentrated in provable tactics; design a genuinely iterative league-based game-level learning loop."
    else:
        primary = "insufficient_resolved_evidence"
        recommendation = "Do not choose a new learner from these games alone; collect more held-out human games or increase only continuation confidence on the flagged states."
    return {
        "primary": primary,
        "exact_tactical_count": tactical,
        "resolved_conversion_count": conversion,
        "resolved_reserve_count": reserve,
        "resolved_low_progress_count": low_progress,
        "recommendation": recommendation,
    }


def _game_summary(
    replay_rows: list[dict[str, Any]],
    screens: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_game_rows: dict[int, list[dict[str, Any]]] = defaultdict(list)
    by_game_screens: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in replay_rows:
        by_game_rows[int(row["game_seed"])].append(row)
    for row in screens:
        by_game_screens[int(row["game_seed"])].append(row)
    output = []
    for seed, rows in sorted(by_game_rows.items()):
        first = rows[0]
        human = int(first.get("human_player_id", 0))
        output.append(
            {
                "game_seed": seed,
                "human_player_id": human,
                "winner": first.get("game_winner"),
                "human_won": first.get("game_winner") == human,
                "final_scores": first.get("game_final_scores"),
                "model_turns": len(by_game_screens[seed]),
                "suspicious_turns": sum(bool(row["suspicious"]) for row in by_game_screens[seed]),
                "high_confidence_mistakes": sum(bool(row["high_confidence_mistake"]) for row in by_game_screens[seed]),
            }
        )
    return output


def _render_report(output: dict[str, Any]) -> str:
    summary = output["summary"]
    lines = [
        "# Human Game Forensic Audit",
        "",
        "## Scope",
        "",
        f"- Games: {summary['games']}",
        f"- Policy turns classified: {summary['model_turns']}",
        f"- Suspicious turns screened: {summary['suspicious_turns']}",
        f"- Suspicious states with matched continuations: {summary['rollout_states']}",
        f"- Policy/replay top-action consistency: {summary['policy_top_replay_match_rate']:.1%}",
        "",
        "Human rows were held out from training. Tactical claims are invariant across the configured legal determinizations. Continuation claims use common rollout seeds across candidate actions.",
        "",
        "## Classification",
        "",
    ]
    for category, count in summary["category_counts"].items():
        lines.append(f"- {category}: {count}")
    lines.extend(
        [
            "",
            "## High-Confidence Findings",
            "",
            f"- Exact tactical mistakes: {summary['exact_tactical_mistakes']}",
            f"- Resolved continuation mistakes: {summary['resolved_rollout_mistakes']}",
            f"- Distinct high-confidence mistakes: {summary['high_confidence_mistakes']}",
            f"- Resolved mistakes in human wins: {summary['resolved_rollout_in_human_wins']}/{summary['resolved_rollout_mistakes']}",
            f"- Mean paired continuation improvement: {summary['resolved_rollout_mean_paired_delta']:+.3f}",
            f"- Mean chosen-minus-better prior score: {summary['resolved_rollout_mean_chosen_minus_best_prior_score']:+.3f}",
            "",
            "Resolved phases:",
            *[f"- {phase}: {count}" for phase, count in summary["resolved_rollout_by_phase"].items()],
            "",
            "Resolved action transitions:",
            *[f"- {transition}: {count}" for transition, count in summary["resolved_rollout_action_transitions"].items()],
            "",
        ]
    )
    exact = [row for row in output["turns"] if row["exact_tactical_mistake"]]
    for row in exact:
        lines.append(
            f"- Seed {row['game_seed']} turn {row['turn_index']}: exact {', '.join(row['categories'])}; chosen `{row['chosen_action_type']}`."
        )
    resolved = [row for row in output["rollout_results"] if row["resolved_mistake"]]
    for row in sorted(resolved, key=lambda item: -float(item["paired_return_delta"])):
        chosen = next(item for item in row["actions"] if item["action_index"] == row["chosen_action_index"])
        best = next(item for item in row["actions"] if item["action_index"] == row["best_alternative_action_index"])
        lines.append(
            f"- Seed {row['game_seed']} turn {row['turn_index']}: paired delta {row['paired_return_delta']:+.3f} +/- {row['paired_return_delta_standard_error']:.3f}; chosen `{chosen['action']['action_type']}` vs `{best['action']['action_type']}` ({best['action'].get('card_id') or 'no card'})."
        )
    diagnosis = summary["diagnosis"]
    lines.extend(
        [
            "",
            "## Diagnosis",
            "",
            f"Primary classification: **{diagnosis['primary']}**.",
            "",
            diagnosis["recommendation"],
            "",
            "## Limitations",
            "",
            "- Ten games establish a practical weakness signal but not a population estimate of human strength.",
            "- Continuations use policy035 for both players, not a model of the human opponent.",
            "- Conversion and reserve labels are screening heuristics; only exact tactical proofs and statistically resolved continuation deltas are called high-confidence mistakes.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_progress(
    path: Path,
    screens: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    rollout_results: list[dict[str, Any]],
    complete: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "audit": "human_game_forensic_progress_v1",
                "complete": complete,
                "screened_turns": len(screens),
                "selected_keys": [list(_result_key(row)) for row in selected],
                "rollout_results": rollout_results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _config_payload(
    config: HumanGameForensicAuditConfig,
    progress_path: Path,
) -> dict[str, Any]:
    payload = asdict(config)
    for key in ("replay_path", "policy_checkpoint", "output_path", "report_path"):
        payload[key] = str(payload[key])
    payload["progress_path"] = str(progress_path)
    return payload


def _validate(config: HumanGameForensicAuditConfig) -> None:
    for name, value in (
        ("tactical_determinizations", config.tactical_determinizations),
        ("max_rollout_states", config.max_rollout_states),
        ("max_candidates_per_state", config.max_candidates_per_state),
        ("policy_top_k", config.policy_top_k),
        ("rollout_determinizations", config.rollout_determinizations),
        ("rollouts_per_determinization", config.rollouts_per_determinization),
        ("rollout_top_k", config.rollout_top_k),
        ("rollout_max_additional_turns", config.rollout_max_additional_turns),
        ("repeated_nonbuy_turns", config.repeated_nonbuy_turns),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be positive.")
    if config.max_candidates_per_state < 2:
        raise ValueError("At least two rollout candidates are required.")
    if not 0.0 <= config.rollout_epsilon <= 1.0:
        raise ValueError("rollout_epsilon must be in [0, 1].")
    if config.rollout_temperature <= 0.0 or config.score_margin_scale <= 0.0:
        raise ValueError("Rollout temperature and score scale must be positive.")
    if not 0.0 <= config.outcome_weight <= 1.0:
        raise ValueError("outcome_weight must be in [0, 1].")
    if config.min_resolved_delta < 0.0 or config.confidence_z < 0.0:
        raise ValueError("Resolution thresholds must be non-negative.")


def main() -> None:
    args = build_parser().parse_args()
    output = run_human_game_forensic_audit(
        HumanGameForensicAuditConfig(
            replay_path=Path(args.replay_path),
            policy_checkpoint=Path(args.policy_checkpoint),
            output_path=Path(args.output_path),
            report_path=Path(args.report_path),
            progress_path=Path(args.progress_path) if args.progress_path else None,
            tactical_determinizations=args.tactical_determinizations,
            max_rollout_states=args.max_rollout_states,
            max_candidates_per_state=args.max_candidates_per_state,
            policy_top_k=args.policy_top_k,
            rollout_determinizations=args.rollout_determinizations,
            rollouts_per_determinization=args.rollouts_per_determinization,
            rollout_epsilon=args.rollout_epsilon,
            rollout_temperature=args.rollout_temperature,
            rollout_top_k=args.rollout_top_k,
            rollout_max_additional_turns=args.rollout_max_additional_turns,
            outcome_weight=args.outcome_weight,
            score_margin_scale=args.score_margin_scale,
            min_resolved_delta=args.min_resolved_delta,
            confidence_z=args.confidence_z,
            repeated_nonbuy_turns=args.repeated_nonbuy_turns,
            seed=args.seed,
            device=args.device,
            log_every_states=args.log_every_states,
            retain_rollout_returns=args.retain_rollout_returns,
        )
    )
    summary = output["summary"]
    print(
        "[human-forensic]"
        f" turns={summary['model_turns']}"
        f" suspicious={summary['suspicious_turns']}"
        f" exact={summary['exact_tactical_mistakes']}"
        f" rollout_resolved={summary['resolved_rollout_mistakes']}"
        f" diagnosis={summary['diagnosis']['primary']}"
    )


if __name__ == "__main__":
    main()
