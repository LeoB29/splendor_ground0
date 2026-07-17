"""Outcome-blind diagnostic for persistent card-directed resource plans."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
from typing import Any, Iterable

import torch

from splendor_ai.bots import (
    CanonicalEntityActionScorerBot,
    LegalActionAttentionScorerBot,
    LegalActionScorerBot,
    ShallowSearchBot,
)
from splendor_ai.engine.actions import Action, ActionType
from splendor_ai.engine.constants import TOKEN_COLORS
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.engine.state import Card, SplendorState
from splendor_ai.eval.run_shallow_operational_gate import SHALLOW_CONTRACT
from splendor_ai.training.replay import (
    collect_game_replay,
    deserialize_state_snapshot,
    serialize_state_snapshot,
)
from splendor_ai.training.run_entity_public_value import phase_for_turn


OPPONENTS = ("035", "016", "shallow", "023")
SPLITS = ("discovery", "confirmation")


@dataclass(frozen=True, slots=True)
class PlanPersistenceConfig:
    output_dir: Path
    data_dir: Path
    policy_checkpoint: Path
    checkpoint_016: Path
    checkpoint_023: Path
    games: int = 256
    seed_start: int = 390_000
    horizon_own_turns: int = 4
    workers: int = 4
    max_turns: int = 400
    repetition_limit: int = 4
    no_progress_limit: int = 60
    bootstrap_samples: int = 10_000
    bootstrap_seed: int = 37_0717
    minimum_anchors_per_split: int = 200
    minimum_multi_take_episodes_per_split: int = 40
    minimum_cell_anchors: int = 30
    minimum_phase_anchors: int = 50
    minimum_progress_advantage: float = 0.10
    minimum_progress_rank_percentile: float = 0.60
    minimum_progress_rank_ci_low: float = 0.55
    minimum_multi_take_advantage: float = 0.08
    device: str = "cpu"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", default="outputs/plan_persistence_diagnostic_038"
    )
    parser.add_argument(
        "--data-dir", default="data/plan_persistence_diagnostic_038"
    )
    parser.add_argument(
        "--policy-checkpoint",
        default=(
            "outputs/legal_action_scorer_035_legal_observation/"
            "legal_action_attention_scorer_best.pt"
        ),
    )
    parser.add_argument(
        "--checkpoint-016",
        default=(
            "outputs/legal_action_scorer_016_policy_guided_mix/"
            "legal_action_scorer_best.pt"
        ),
    )
    parser.add_argument(
        "--checkpoint-023",
        default=(
            "outputs/canonical_entity_coherent_023/"
            "canonical_entity_action_scorer_coherent_width64_best.pt"
        ),
    )
    parser.add_argument("--games", type=int, default=256)
    parser.add_argument("--seed-start", type=int, default=390_000)
    parser.add_argument("--horizon-own-turns", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--device", default="cpu")
    return parser


def config_from_args(args: argparse.Namespace) -> PlanPersistenceConfig:
    return PlanPersistenceConfig(
        output_dir=Path(args.output_dir),
        data_dir=Path(args.data_dir),
        policy_checkpoint=Path(args.policy_checkpoint),
        checkpoint_016=Path(args.checkpoint_016),
        checkpoint_023=Path(args.checkpoint_023),
        games=args.games,
        seed_start=args.seed_start,
        horizon_own_turns=args.horizon_own_turns,
        workers=args.workers,
        bootstrap_samples=args.bootstrap_samples,
        device=args.device,
    )


def run_plan_persistence_diagnostic(
    config: PlanPersistenceConfig,
) -> dict[str, Any]:
    _validate(config)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.data_dir.mkdir(parents=True, exist_ok=True)
    protocol = _protocol(config)
    protocol_path = config.output_dir / "protocol.json"
    if protocol_path.exists():
        if json.loads(protocol_path.read_text(encoding="utf-8")) != protocol:
            raise ValueError("Existing protocol differs from the frozen contract.")
    else:
        _atomic_json(protocol_path, protocol)

    collection = collect_outcome_blind_games(config)
    if not collection["passed"]:
        return _finalize(
            config,
            decision="collection_integrity_failed_no_hierarchy",
            collection=collection,
        )
    anchors = extract_plan_anchors(config)
    analysis = summarize_plan_anchors(anchors, config)
    gates = plan_persistence_gates(analysis, config)
    decision = (
        "authorize_hierarchical_option_design"
        if all(gates.values())
        else "plan_persistence_not_established_no_hierarchy"
    )
    _atomic_json(config.output_dir / "analysis.json", analysis)
    return _finalize(
        config,
        decision=decision,
        collection=collection,
        analysis=analysis,
        gates=gates,
    )


def collect_outcome_blind_games(config: PlanPersistenceConfig) -> dict[str, Any]:
    completed: dict[int, dict[str, Any]] = {}
    pending = []
    for game_index in range(config.games):
        summary_path = _summary_path(config, game_index)
        replay_path = _replay_path(config, game_index)
        if (
            summary_path.exists()
            and replay_path.exists()
            and _stored_replay_contract_valid(replay_path)
        ):
            completed[game_index] = json.loads(summary_path.read_text(encoding="utf-8"))
        else:
            pending.append(game_index)
    print(
        f"[plan-collect] games={config.games} complete={len(completed)}"
        f" pending={len(pending)} workers={config.workers}",
        flush=True,
    )
    if pending and config.workers == 1:
        for position, game_index in enumerate(pending, 1):
            completed[game_index] = _collect_one_game(config, game_index)
            if position == 1 or position % 16 == 0 or position == len(pending):
                print(f"[plan-collect] complete={len(completed)}/{config.games}", flush=True)
    elif pending:
        with ProcessPoolExecutor(max_workers=config.workers) as executor:
            futures = {
                executor.submit(_collect_one_game, config, game_index): game_index
                for game_index in pending
            }
            for position, future in enumerate(as_completed(futures), 1):
                game_index = futures[future]
                completed[game_index] = future.result()
                if position == 1 or position % 16 == 0 or position == len(pending):
                    print(f"[plan-collect] complete={len(completed)}/{config.games}", flush=True)

    ordered = [completed[index] for index in range(config.games)]
    by_cell_split = Counter(
        f"{row['split']}:{row['opponent']}:seat{row['actor_seat']}"
        for row in ordered
    )
    expected = config.games // (len(SPLITS) * len(OPPONENTS) * 2)
    payload = {
        "format": "outcome_blind_plan_collection_v1",
        "games": len(ordered),
        "actor_turns": sum(int(row["actor_turns"]) for row in ordered),
        "by_split_opponent_seat": dict(sorted(by_cell_split.items())),
        "stalled_games": sum(bool(row["stalled"]) for row in ordered),
        "timed_out_games": sum(bool(row["timed_out"]) for row in ordered),
        "termination_reasons": dict(
            sorted(Counter(str(row["termination_reason"]) for row in ordered).items())
        ),
    }
    payload["gates"] = {
        "complete": len(ordered) == config.games,
        "exact_split_opponent_seat_balance": (
            len(by_cell_split) == len(SPLITS) * len(OPPONENTS) * 2
            and set(by_cell_split.values()) == {expected}
        ),
        "zero_stalls": payload["stalled_games"] == 0,
        "zero_timeouts": payload["timed_out_games"] == 0,
    }
    payload["passed"] = all(payload["gates"].values())
    _atomic_json(config.output_dir / "collection.json", payload)
    return payload


def _collect_one_game(
    config: PlanPersistenceConfig, game_index: int
) -> dict[str, Any]:
    torch.set_num_threads(1)
    opponent, actor_seat, split = group_assignment(config.games, game_index)
    actor = LegalActionAttentionScorerBot(
        config.policy_checkpoint, device=config.device
    )
    opponent_bot = _build_opponent(config, opponent)
    bots = (actor, opponent_bot) if actor_seat == 0 else (opponent_bot, actor)
    replay = collect_game_replay(
        bots[0],
        bots[1],
        seed=config.seed_start + game_index,
        max_turns=config.max_turns,
        repetition_limit=config.repetition_limit,
        no_progress_limit=config.no_progress_limit,
        include_state_snapshots=True,
    )
    rows = []
    for step in replay.steps:
        if step.player_id != actor_seat:
            continue
        if step.state_snapshot is None:
            raise RuntimeError("Plan diagnostic requires state snapshots.")
        rows.append(
            {
                "game_index": game_index,
                "environment_seed": config.seed_start + game_index,
                "split": split,
                "opponent": opponent,
                "actor_seat": actor_seat,
                "turn_index": step.turn_index,
                "action_index": step.action_index,
                "action_payload": step.action_payload,
                "public_state_snapshot": sanitize_snapshot(
                    step.state_snapshot, actor_seat
                ),
            }
        )
    replay_path = _replay_path(config, game_index)
    replay_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = replay_path.with_suffix(replay_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row))
            handle.write("\n")
    os.replace(temporary, replay_path)
    summary = {
        "game_index": game_index,
        "environment_seed": config.seed_start + game_index,
        "split": split,
        "opponent": opponent,
        "actor_seat": actor_seat,
        "actor_turns": len(rows),
        "turns": replay.turns,
        "stalled": replay.stalled,
        "timed_out": replay.timed_out,
        "termination_reason": replay.termination_reason,
    }
    _atomic_json(_summary_path(config, game_index), summary)
    return summary


def sanitize_snapshot(snapshot: dict[str, Any], observer: int) -> dict[str, Any]:
    """Remove true deck order and opponent-private reserved identities."""
    clean = json.loads(json.dumps(snapshot))
    clean["hidden_tier_decks"] = {
        str(tier): [] for tier in clean.get("hidden_tier_decks", {})
    }
    clean.pop("winner", None)
    for player in clean["players"]:
        if int(player["player_id"]) != observer:
            player["reserved_card_ids"] = []
            player["reserved_card_sources"] = []
    return clean


def extract_plan_anchors(config: PlanPersistenceConfig) -> list[dict[str, Any]]:
    anchors = []
    for game_index in range(config.games):
        rows = _read_jsonl(_replay_path(config, game_index))
        anchors.extend(_anchors_for_game(rows, config.horizon_own_turns))
    _atomic_json(
        config.output_dir / "anchors.json",
        {
            "format": "outcome_blind_plan_anchors_v1",
            "anchors": anchors,
        },
    )
    return anchors


def _anchors_for_game(
    rows: list[dict[str, Any]], horizon_own_turns: int
) -> list[dict[str, Any]]:
    env = SplendorEnv()
    anchors = []
    for position, row in enumerate(rows):
        if row["action_payload"]["action_type"] != ActionType.TAKE_TOKENS.name:
            continue
        state = deserialize_state_snapshot(row["public_state_snapshot"])
        target = _future_target(rows, position, horizon_own_turns, state)
        if target is None:
            continue
        target_card, target_row, own_turn_distance = target
        player = state.players[int(row["actor_seat"])]
        legal_takes = env._legal_take_actions(player, state)
        if len(legal_takes) < 2:
            continue
        actual_signature = _payload_take_signature(row["action_payload"])
        matching = [
            action for action in legal_takes if _take_signature(action) == actual_signature
        ]
        if len(matching) != 1:
            raise RuntimeError("Recorded take does not match exactly one legal take.")
        progresses = [
            card_progress_for_take(state, target_card, action) for action in legal_takes
        ]
        if max(progresses) <= 0:
            continue
        actual_index = legal_takes.index(matching[0])
        actual_progress = progresses[actual_index]
        alternatives = [
            progress for index, progress in enumerate(progresses) if index != actual_index
        ]
        less = sum(progress < actual_progress for progress in alternatives)
        equal = sum(progress == actual_progress for progress in alternatives)
        before_rank, after_rank = target_card_rank_change(
            state, target_card, matching[0]
        )
        anchors.append(
            {
                "game_index": int(row["game_index"]),
                "environment_seed": int(row["environment_seed"]),
                "split": row["split"],
                "opponent": row["opponent"],
                "actor_seat": int(row["actor_seat"]),
                "anchor_turn": int(row["turn_index"]),
                "phase": phase_for_turn(int(row["turn_index"])),
                "target_turn": int(target_row["turn_index"]),
                "own_turn_distance": own_turn_distance,
                "target_card_id": target_card.card_id,
                "target_event": target_row["action_payload"]["action_type"],
                "legal_take_count": len(legal_takes),
                "actual_progress": actual_progress,
                "alternative_mean_progress": statistics.fmean(alternatives),
                "progress_advantage": (
                    actual_progress - statistics.fmean(alternatives)
                ),
                "progress_rank_percentile": (
                    less + 0.5 * equal
                ) / len(alternatives),
                "actual_is_top": actual_progress == max(progresses),
                "actual_supports_target": actual_progress > 0,
                "target_card_rank_before": before_rank,
                "target_card_rank_after": after_rank,
                "target_card_rank_gain": after_rank - before_rank,
            }
        )
    return anchors


def _future_target(
    rows: list[dict[str, Any]],
    anchor_position: int,
    horizon_own_turns: int,
    anchor_state: SplendorState,
) -> tuple[Card, dict[str, Any], int] | None:
    candidate_ids = {
        card.card_id
        for cards in anchor_state.visible_tier_cards.values()
        for card in cards
    }
    candidate_ids.update(
        card.card_id
        for card in anchor_state.players[anchor_state.current_player].reserved_cards
    )
    direct_types = {
        ActionType.BUY_VISIBLE.name,
        ActionType.BUY_RESERVED.name,
        ActionType.RESERVE_VISIBLE.name,
        ActionType.RESERVE_DECK.name,
    }
    for distance, future_row in enumerate(
        rows[anchor_position + 1 : anchor_position + 1 + horizon_own_turns], 1
    ):
        action_type = future_row["action_payload"]["action_type"]
        if action_type not in direct_types:
            continue
        if action_type == ActionType.RESERVE_DECK.name:
            return None
        future_state = deserialize_state_snapshot(
            future_row["public_state_snapshot"]
        )
        card = _action_target_card(future_state, future_row["action_payload"])
        if card is None or card.card_id not in candidate_ids:
            return None
        return card, future_row, distance
    return None


def _action_target_card(
    state: SplendorState, action_payload: dict[str, Any]
) -> Card | None:
    action_type = action_payload["action_type"]
    if action_type in {
        ActionType.BUY_VISIBLE.name,
        ActionType.RESERVE_VISIBLE.name,
    }:
        tier = action_payload.get("tier")
        market_index = action_payload.get("market_index")
        if tier is None or market_index is None:
            return None
        cards = state.visible_tier_cards.get(int(tier), [])
        return cards[int(market_index)] if int(market_index) < len(cards) else None
    if action_type == ActionType.BUY_RESERVED.name:
        reserved_index = action_payload.get("reserved_index")
        cards = state.players[state.current_player].reserved_cards
        if reserved_index is None or int(reserved_index) >= len(cards):
            return None
        return cards[int(reserved_index)]
    return None


def remaining_card_cost(
    state: SplendorState,
    card: Card,
    tokens: dict[str, int] | None = None,
) -> int:
    player = state.players[state.current_player]
    available = tokens if tokens is not None else player.tokens
    colored_deficit = sum(
        max(
            int(card.cost.get(color, 0))
            - int(player.bonuses.get(color, 0))
            - int(available.get(color, 0)),
            0,
        )
        for color in TOKEN_COLORS
    )
    return max(colored_deficit - int(available.get("gold", 0)), 0)


def card_progress_for_take(
    state: SplendorState, card: Card, action: Action
) -> int:
    player = state.players[state.current_player]
    tokens = dict(player.tokens)
    for color in action.take_tokens:
        tokens[color] = tokens.get(color, 0) + 1
    for color in action.return_tokens:
        tokens[color] = tokens.get(color, 0) - 1
    return remaining_card_cost(state, card) - remaining_card_cost(
        state, card, tokens
    )


def target_card_rank_change(
    state: SplendorState, target: Card, action: Action
) -> tuple[float, float]:
    cards = [
        card for tier_cards in state.visible_tier_cards.values() for card in tier_cards
    ] + list(state.players[state.current_player].reserved_cards)
    unique = {card.card_id: card for card in cards}
    cards = list(unique.values())
    player = state.players[state.current_player]
    tokens_after = dict(player.tokens)
    for color in action.take_tokens:
        tokens_after[color] = tokens_after.get(color, 0) + 1
    for color in action.return_tokens:
        tokens_after[color] = tokens_after.get(color, 0) - 1
    before = [remaining_card_cost(state, card) for card in cards]
    after = [remaining_card_cost(state, card, tokens_after) for card in cards]
    target_index = next(
        index for index, card in enumerate(cards) if card.card_id == target.card_id
    )
    return (
        _lower_is_better_percentile(before, target_index),
        _lower_is_better_percentile(after, target_index),
    )


def _lower_is_better_percentile(values: list[int], target_index: int) -> float:
    target = values[target_index]
    others = [value for index, value in enumerate(values) if index != target_index]
    if not others:
        return 1.0
    worse = sum(value > target for value in others)
    equal = sum(value == target for value in others)
    return (worse + 0.5 * equal) / len(others)


def summarize_plan_anchors(
    anchors: list[dict[str, Any]], config: PlanPersistenceConfig
) -> dict[str, Any]:
    splits = {
        split: _anchor_summary(
            [row for row in anchors if row["split"] == split], config, offset
        )
        for offset, split in enumerate(SPLITS)
    }
    episodes = _multi_take_episodes(anchors)
    multi_splits = {
        split: _episode_summary(
            [row for row in episodes if row["split"] == split], config, offset
        )
        for offset, split in enumerate(SPLITS)
    }
    breakdowns = {}
    dimensions = {
        "opponent_seat": lambda row: f"{row['opponent']}:seat{row['actor_seat']}",
        "phase": lambda row: str(row["phase"]),
        "target_event": lambda row: str(row["target_event"]),
        "horizon": lambda row: str(row["own_turn_distance"]),
    }
    for name, key_fn in dimensions.items():
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in anchors:
            grouped[key_fn(row)].append(row)
        breakdowns[name] = {
            key: _simple_anchor_metrics(rows)
            for key, rows in sorted(grouped.items())
        }
    return {
        "format": "outcome_blind_plan_persistence_analysis_v1",
        "outcome_fields_read": False,
        "statistical_unit": "game-cluster mean",
        "anchors": len(anchors),
        "multi_take_episodes": len(episodes),
        "splits": splits,
        "multi_take_splits": multi_splits,
        "breakdowns": breakdowns,
    }


def _anchor_summary(
    rows: list[dict[str, Any]], config: PlanPersistenceConfig, offset: int
) -> dict[str, Any]:
    return {
        **_simple_anchor_metrics(rows),
        "progress_advantage_ci": _cluster_mean_ci(
            rows,
            "progress_advantage",
            config.bootstrap_samples,
            config.bootstrap_seed + offset * 10,
        ),
        "progress_rank_percentile_ci": _cluster_mean_ci(
            rows,
            "progress_rank_percentile",
            config.bootstrap_samples,
            config.bootstrap_seed + offset * 10 + 1,
        ),
        "target_card_rank_gain_ci": _cluster_mean_ci(
            rows,
            "target_card_rank_gain",
            config.bootstrap_samples,
            config.bootstrap_seed + offset * 10 + 2,
        ),
    }


def _simple_anchor_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "anchors": len(rows),
        "games": len({int(row["game_index"]) for row in rows}),
        "mean_progress_advantage": _mean(rows, "progress_advantage"),
        "mean_progress_rank_percentile": _mean(
            rows, "progress_rank_percentile"
        ),
        "target_support_rate": _mean(rows, "actual_supports_target"),
        "target_top_progress_rate": _mean(rows, "actual_is_top"),
        "mean_target_card_rank_gain": _mean(rows, "target_card_rank_gain"),
    }


def _multi_take_episodes(
    anchors: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in anchors:
        grouped[
            (
                int(row["game_index"]),
                int(row["target_turn"]),
                str(row["target_card_id"]),
            )
        ].append(row)
    episodes = []
    for rows in grouped.values():
        if len(rows) < 2:
            continue
        first = rows[0]
        episodes.append(
            {
                "game_index": first["game_index"],
                "split": first["split"],
                "opponent": first["opponent"],
                "actor_seat": first["actor_seat"],
                "target_turn": first["target_turn"],
                "target_card_id": first["target_card_id"],
                "takes": len(rows),
                "mean_progress_advantage": statistics.fmean(
                    float(row["progress_advantage"]) for row in rows
                ),
                "all_takes_support_target": all(
                    bool(row["actual_supports_target"]) for row in rows
                ),
            }
        )
    return episodes


def _episode_summary(
    rows: list[dict[str, Any]], config: PlanPersistenceConfig, offset: int
) -> dict[str, Any]:
    return {
        "episodes": len(rows),
        "games": len({int(row["game_index"]) for row in rows}),
        "mean_progress_advantage": _mean(rows, "mean_progress_advantage"),
        "all_takes_support_rate": _mean(rows, "all_takes_support_target"),
        "progress_advantage_ci": _cluster_mean_ci(
            rows,
            "mean_progress_advantage",
            config.bootstrap_samples,
            config.bootstrap_seed + 100 + offset,
        ),
    }


def _cluster_mean_ci(
    rows: list[dict[str, Any]], key: str, samples: int, seed: int
) -> dict[str, Any]:
    grouped: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        grouped[int(row["game_index"])].append(float(row[key]))
    game_means = [statistics.fmean(values) for values in grouped.values()]
    if not game_means:
        return {"games": 0, "mean": 0.0, "ci95_low": 0.0, "ci95_high": 0.0}
    mean = statistics.fmean(game_means)
    if samples <= 0 or len(game_means) == 1:
        return {
            "games": len(game_means),
            "mean": mean,
            "ci95_low": mean,
            "ci95_high": mean,
        }
    rng = random.Random(seed)
    draws = sorted(
        statistics.fmean(rng.choice(game_means) for _ in game_means)
        for _ in range(samples)
    )
    return {
        "games": len(game_means),
        "mean": mean,
        "ci95_low": draws[int(0.025 * (samples - 1))],
        "ci95_high": draws[int(0.975 * (samples - 1))],
    }


def plan_persistence_gates(
    analysis: dict[str, Any], config: PlanPersistenceConfig
) -> dict[str, bool]:
    gates = {}
    for split in SPLITS:
        summary = analysis["splits"][split]
        multi = analysis["multi_take_splits"][split]
        gates[f"{split}_anchor_coverage"] = (
            int(summary["anchors"]) >= config.minimum_anchors_per_split
        )
        gates[f"{split}_progress_advantage"] = (
            float(summary["progress_advantage_ci"]["mean"])
            >= config.minimum_progress_advantage
            and float(summary["progress_advantage_ci"]["ci95_low"]) > 0.0
        )
        gates[f"{split}_progress_rank"] = (
            float(summary["progress_rank_percentile_ci"]["mean"])
            >= config.minimum_progress_rank_percentile
            and float(summary["progress_rank_percentile_ci"]["ci95_low"])
            >= config.minimum_progress_rank_ci_low
        )
        gates[f"{split}_multi_take_coverage"] = (
            int(multi["episodes"])
            >= config.minimum_multi_take_episodes_per_split
        )
        gates[f"{split}_multi_take_advantage"] = (
            float(multi["progress_advantage_ci"]["mean"])
            >= config.minimum_multi_take_advantage
            and float(multi["progress_advantage_ci"]["ci95_low"]) > 0.0
        )
    cell_rows = analysis["breakdowns"]["opponent_seat"]
    gates["all_opponent_seat_cells_covered"] = (
        len(cell_rows) == len(OPPONENTS) * 2
        and all(
            int(row["anchors"]) >= config.minimum_cell_anchors
            for row in cell_rows.values()
        )
    )
    gates["no_opponent_seat_collapse"] = all(
        float(row["mean_progress_advantage"]) > 0.0
        for row in cell_rows.values()
    )
    phase_rows = [
        row
        for row in analysis["breakdowns"]["phase"].values()
        if int(row["anchors"]) >= config.minimum_phase_anchors
    ]
    gates["multiple_phases_covered"] = len(phase_rows) >= 3
    gates["no_covered_phase_collapse"] = all(
        float(row["mean_progress_advantage"]) > 0.0 for row in phase_rows
    )
    return gates


def group_assignment(total_games: int, game_index: int) -> tuple[str, int, str]:
    if not 0 <= game_index < total_games:
        raise ValueError("game_index is outside the corpus.")
    split = SPLITS[0] if game_index < total_games // 2 else SPLITS[1]
    within_split = game_index % (total_games // 2)
    opponent = OPPONENTS[within_split % len(OPPONENTS)]
    actor_seat = (within_split // len(OPPONENTS)) % 2
    return opponent, actor_seat, split


def _build_opponent(config: PlanPersistenceConfig, kind: str):
    if kind == "035":
        return LegalActionAttentionScorerBot(
            config.policy_checkpoint, device=config.device
        )
    if kind == "016":
        return LegalActionScorerBot(config.checkpoint_016, device=config.device)
    if kind == "023":
        return CanonicalEntityActionScorerBot(
            config.checkpoint_023, device=config.device
        )
    if kind == "shallow":
        return ShallowSearchBot(**SHALLOW_CONTRACT)
    raise ValueError(f"Unsupported opponent: {kind}")


def _take_signature(action: Action) -> tuple[tuple[str, ...], tuple[str, ...]]:
    return tuple(action.take_tokens), tuple(action.return_tokens)


def _payload_take_signature(
    payload: dict[str, Any]
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    return tuple(payload.get("take_tokens", ())), tuple(payload.get("return_tokens", ()))


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    return (
        statistics.fmean(float(row[key]) for row in rows) if rows else 0.0
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _stored_replay_contract_valid(path: Path) -> bool:
    forbidden = {"game_winner", "game_final_scores", "final_value", "winner"}
    try:
        for row in _read_jsonl(path):
            if forbidden.intersection(row):
                return False
            snapshot = row.get("public_state_snapshot")
            if not isinstance(snapshot, dict) or "winner" in snapshot:
                return False
            if any(snapshot.get("hidden_tier_decks", {}).values()):
                return False
            observer = int(row["actor_seat"])
            for player in snapshot.get("players", []):
                if (
                    int(player["player_id"]) != observer
                    and player.get("reserved_card_ids")
                ):
                    return False
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False
    return True


def _protocol(config: PlanPersistenceConfig) -> dict[str, Any]:
    return {
        "experiment": "outcome_blind_plan_persistence_038",
        "frozen_at": "before fresh collection or future-action inspection",
        "diagnostic_only": True,
        "training_allowed": False,
        "promotion_allowed": False,
        "observation_contract": (
            "acting-player public snapshots only; true deck order and opponent "
            "reserved-card identities are removed before storage"
        ),
        "outcome_contract": (
            "winner, final score, final margin, and replay final_value are never "
            "stored or read"
        ),
        "collection": {
            "games": config.games,
            "seed_range": [config.seed_start, config.seed_start + config.games - 1],
            "policy": "greedy frozen 035",
            "opponents": list(OPPONENTS),
            "balance": "two game-disjoint halves, each exact opponent/actor-seat balance",
        },
        "anchor": (
            "035 TAKE_TOKENS with at least two legal take alternatives; first "
            "visible/reserved card bought or visibly reserved within four own turns "
            "must already be visible or self-reserved at the anchor; at least one "
            "legal take must reduce its remaining colored cost"
        ),
        "control": (
            "compare the chosen take with every other legal take from the exact "
            "same public state; bootstrap equal-weight per-game means"
        ),
        "primary_metrics": [
            "chosen-minus-alternative target-cost progress",
            "chosen progress rank percentile among legal takes",
            "same-target multi-take episode progress advantage",
        ],
        "gates": {
            "replication": "all primary gates pass independently on discovery and confirmation",
            "anchors_per_split": config.minimum_anchors_per_split,
            "multi_take_episodes_per_split": config.minimum_multi_take_episodes_per_split,
            "progress_advantage": (
                f">={config.minimum_progress_advantage} and cluster CI95 low >0"
            ),
            "rank_percentile": (
                f">={config.minimum_progress_rank_percentile} and cluster CI95 low "
                f">={config.minimum_progress_rank_ci_low}"
            ),
            "multi_take_advantage": (
                f">={config.minimum_multi_take_advantage} and cluster CI95 low >0"
            ),
            "strata": "all eight opponent-seat cells and at least three phases covered; no positive-coverage collapse",
        },
        "decision": (
            "a full pass authorizes design of a separate hierarchical option "
            "representation only; failure closes the proposed hierarchy"
        ),
        "checkpoint_hashes": {
            "035": _sha256(config.policy_checkpoint),
            "016": _sha256(config.checkpoint_016),
            "023": _sha256(config.checkpoint_023),
        },
        "config": _config_payload(config),
    }


def _config_payload(config: PlanPersistenceConfig) -> dict[str, Any]:
    payload = asdict(config)
    for key, value in tuple(payload.items()):
        if isinstance(value, Path):
            payload[key] = str(value)
    return payload


def _validate(config: PlanPersistenceConfig) -> None:
    if config.games <= 0 or config.games % (len(SPLITS) * len(OPPONENTS) * 2):
        raise ValueError("games must exactly balance split/opponent/seat cells.")
    if min(
        config.horizon_own_turns,
        config.workers,
        config.bootstrap_samples,
        config.minimum_anchors_per_split,
        config.minimum_multi_take_episodes_per_split,
    ) <= 0:
        raise ValueError("Diagnostic counts and horizon must be positive.")
    for path in (
        config.policy_checkpoint,
        config.checkpoint_016,
        config.checkpoint_023,
    ):
        if not path.exists():
            raise FileNotFoundError(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _replay_path(config: PlanPersistenceConfig, game_index: int) -> Path:
    return config.data_dir / "raw" / f"game_{game_index:04d}.jsonl"


def _summary_path(config: PlanPersistenceConfig, game_index: int) -> Path:
    return config.data_dir / "raw" / f"game_{game_index:04d}.summary.json"


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _finalize(
    config: PlanPersistenceConfig, *, decision: str, **stages: Any
) -> dict[str, Any]:
    payload = {
        "gate": "outcome_blind_plan_persistence_038",
        "decision": decision,
        "diagnostic_only": True,
        "training_allowed": False,
        "promotion_allowed": False,
        **stages,
    }
    _atomic_json(config.output_dir / "gate_final.json", payload)
    print(
        json.dumps(
            {
                "decision": decision,
                "training_allowed": False,
                "promotion_allowed": False,
            },
            indent=2,
        ),
        flush=True,
    )
    return payload


def main() -> None:
    run_plan_persistence_diagnostic(config_from_args(build_parser().parse_args()))


if __name__ == "__main__":
    main()
