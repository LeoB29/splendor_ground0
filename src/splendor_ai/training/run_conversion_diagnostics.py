"""Conversion-efficiency diagnostics for snapshot-enabled replay corpora."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from splendor_ai.engine.actions import Action, ActionType
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.engine.state import Card, SplendorState
from splendor_ai.training.replay import deserialize_state_snapshot
from splendor_ai.training.run_policy_diff_against_teacher import MODEL_BOT_NAMES
from splendor_ai.training.run_policy_search_disagreement import _phase_for_turn


BUY_ACTION_TYPES = {"BUY_VISIBLE", "BUY_RESERVED"}
RESERVE_ACTION_TYPES = {"RESERVE_VISIBLE", "RESERVE_DECK"}


@dataclass(frozen=True, slots=True)
class ConversionDiagnosticsConfig:
    replay_paths: tuple[Path, ...]
    labels: tuple[str, ...]
    output_path: Path
    player_id_filter: tuple[int, ...] | None = None
    game_seed_filter: tuple[int, ...] | None = None
    model_turns_only: bool = False
    early_turn_max: int = 20
    window_turns: tuple[int, ...] = (20, 35, 50)
    max_rows: int | None = None
    examples_limit: int = 20
    log_every: int = 1_000


@dataclass(slots=True)
class _PurchaseEvent:
    turn_index: int
    action_type: str
    card_id: str | None
    card_points: int
    noble_points: int


@dataclass(slots=True)
class _ReserveEvent:
    turn_index: int
    action_type: str
    card_id: str | None
    converted: bool = False


@dataclass(slots=True)
class _MissedPointEvent:
    turn_index: int
    affordable_point_cards: int
    missed_point_cards: int


@dataclass(slots=True)
class _EarlyActionEvent:
    turn_index: int
    phase: str
    action_type: str
    action_signature: str
    action_bucket: str
    card_id: str | None
    card_points: int | None


@dataclass(slots=True)
class _PlayerGameTrace:
    label: str
    game_seed: int
    player_id: int
    final_score: int
    opponent_final_score: int
    winner: int | None
    game_turns: int
    game_bot_seat: str | None
    purchases: list[_PurchaseEvent]
    reserves: list[_ReserveEvent]
    missed_point_events: list[_MissedPointEvent]
    early_actions: list[_EarlyActionEvent]

    @property
    def won(self) -> bool:
        return self.winner == self.player_id


class _NumericAccumulator:
    def __init__(self) -> None:
        self.count = 0
        self.sums: Counter[str] = Counter()
        self.value_counts: Counter[str] = Counter()

    def add(self, values: dict[str, float | int | None]) -> None:
        self.count += 1
        for key, value in values.items():
            if value is not None:
                self.sums[key] += float(value)
                self.value_counts[key] += 1

    def means(self) -> dict[str, float]:
        if self.count == 0:
            return {}
        return {
            f"{key}_mean": float(value / self.value_counts[key])
            for key, value in sorted(self.sums.items())
            if self.value_counts[key] > 0
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize conversion efficiency and attribute later conversion "
            "outcomes to early replay actions."
        )
    )
    parser.add_argument(
        "--replay-path",
        action="append",
        required=True,
        help="Snapshot-enabled replay JSONL path. Repeat once per candidate/baseline.",
    )
    parser.add_argument(
        "--label",
        action="append",
        default=None,
        help="Label for the corresponding --replay-path. Defaults to path stem.",
    )
    parser.add_argument("--output-path", required=True)
    parser.add_argument(
        "--player-id-filter",
        nargs="*",
        type=int,
        choices=(0, 1),
        default=None,
    )
    parser.add_argument("--game-seed-filter", nargs="*", type=int, default=None)
    parser.add_argument(
        "--model-turns-only",
        action="store_true",
        help="Only include turns where the acting seat is model-backed.",
    )
    parser.add_argument("--early-turn-max", type=int, default=20)
    parser.add_argument("--window-turns", nargs="*", type=int, default=[20, 35, 50])
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--examples-limit", type=int, default=20)
    parser.add_argument("--log-every", type=int, default=1_000)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    replay_paths = tuple(Path(path) for path in args.replay_path)
    labels = (
        tuple(args.label)
        if args.label is not None and len(args.label) > 0
        else tuple(path.parent.name if path.name == "replays.jsonl" else path.stem for path in replay_paths)
    )
    summary = run_conversion_diagnostics(
        ConversionDiagnosticsConfig(
            replay_paths=replay_paths,
            labels=labels,
            output_path=Path(args.output_path),
            player_id_filter=(
                tuple(args.player_id_filter)
                if args.player_id_filter is not None and len(args.player_id_filter) > 0
                else None
            ),
            game_seed_filter=(
                tuple(args.game_seed_filter)
                if args.game_seed_filter is not None and len(args.game_seed_filter) > 0
                else None
            ),
            model_turns_only=args.model_turns_only,
            early_turn_max=args.early_turn_max,
            window_turns=tuple(args.window_turns),
            max_rows=args.max_rows,
            examples_limit=args.examples_limit,
            log_every=args.log_every,
        )
    )
    print(
        "[conversion]"
        f" rows_seen={summary['rows_seen']}"
        f" player_games={summary['player_games']}"
        f" labels={','.join(summary['labels'])}"
        f" output={args.output_path}"
    )


def run_conversion_diagnostics(config: ConversionDiagnosticsConfig) -> dict[str, Any]:
    _validate_config(config)
    env = SplendorEnv()
    traces_by_label_key: dict[str, dict[tuple[int, int], _PlayerGameTrace]] = {}
    rows_seen = 0
    rows_missing_snapshot = 0
    rows_skipped_by_game_seed = 0
    rows_skipped_by_player_id = 0
    rows_skipped_by_model_turn = 0
    rows_without_legal_actions = 0

    for label, replay_path in zip(config.labels, config.replay_paths, strict=True):
        traces: dict[tuple[int, int], _PlayerGameTrace] = {}
        with replay_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if config.max_rows is not None and rows_seen >= config.max_rows:
                    break
                stripped = line.strip()
                if not stripped:
                    continue
                rows_seen += 1
                payload = json.loads(stripped)
                game_seed = int(payload.get("game_seed", payload.get("seed", -1)))
                if (
                    config.game_seed_filter is not None
                    and game_seed not in config.game_seed_filter
                ):
                    rows_skipped_by_game_seed += 1
                    continue
                snapshot = payload.get("state_snapshot")
                if not isinstance(snapshot, dict):
                    rows_missing_snapshot += 1
                    continue
                player_id = int(payload.get("player_id", -1))
                if config.player_id_filter is not None and player_id not in config.player_id_filter:
                    rows_skipped_by_player_id += 1
                    continue
                if config.model_turns_only and not _is_model_turn(payload, player_id):
                    rows_skipped_by_model_turn += 1
                    continue

                state = deserialize_state_snapshot(snapshot)
                legal_actions = env.legal_actions(state)
                if not legal_actions:
                    rows_without_legal_actions += 1
                    continue
                _record_payload_row(
                    traces=traces,
                    label=label,
                    state=state,
                    legal_actions=legal_actions,
                    payload=payload,
                    early_turn_max=config.early_turn_max,
                )
                if config.log_every > 0 and rows_seen % config.log_every == 0:
                    print(
                        f"[conversion] rows_seen={rows_seen}"
                        f" label={label}"
                        f" player_games={len(traces)}"
                    )
        traces_by_label_key[label] = traces
        if config.max_rows is not None and rows_seen >= config.max_rows:
            break

    summary = _build_summary(
        config=config,
        traces_by_label_key=traces_by_label_key,
        rows_seen=rows_seen,
        rows_missing_snapshot=rows_missing_snapshot,
        rows_skipped_by_game_seed=rows_skipped_by_game_seed,
        rows_skipped_by_player_id=rows_skipped_by_player_id,
        rows_skipped_by_model_turn=rows_skipped_by_model_turn,
        rows_without_legal_actions=rows_without_legal_actions,
    )
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    config.output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _validate_config(config: ConversionDiagnosticsConfig) -> None:
    if not config.replay_paths:
        raise ValueError("At least one replay path is required.")
    if len(config.labels) != len(config.replay_paths):
        raise ValueError("--label count must match --replay-path count.")
    if len(set(config.labels)) != len(config.labels):
        raise ValueError("Labels must be unique.")
    if config.early_turn_max < 0:
        raise ValueError("early_turn_max must be non-negative.")
    if any(turn < 0 for turn in config.window_turns):
        raise ValueError("window_turns must be non-negative.")
    if config.examples_limit < 0:
        raise ValueError("examples_limit must be non-negative.")


def _record_payload_row(
    traces: dict[tuple[int, int], _PlayerGameTrace],
    label: str,
    state: SplendorState,
    legal_actions: list[Action],
    payload: dict[str, Any],
    early_turn_max: int,
) -> None:
    player_id = int(payload.get("player_id", state.current_player))
    game_seed = int(payload.get("game_seed", payload.get("seed", -1)))
    trace = traces.setdefault(
        (game_seed, player_id),
        _new_trace(label=label, game_seed=game_seed, player_id=player_id, payload=payload),
    )
    action_payload = payload.get("action_payload", {})
    if not isinstance(action_payload, dict):
        action_payload = {}
    action_type = str(action_payload.get("action_type", "UNKNOWN"))
    turn_index = int(payload.get("turn_index", state.turn_index))
    card = _card_for_payload_action(state, player_id, action_payload)

    if action_type in BUY_ACTION_TYPES:
        noble_points = _noble_points_for_action(state, action_payload)
        trace.purchases.append(
            _PurchaseEvent(
                turn_index=turn_index,
                action_type=action_type,
                card_id=card.card_id if card is not None else None,
                card_points=card.points if card is not None else 0,
                noble_points=noble_points,
            )
        )
        if action_type == "BUY_RESERVED" and card is not None:
            _mark_reserved_card_converted(trace.reserves, card.card_id)

    if action_type in RESERVE_ACTION_TYPES:
        trace.reserves.append(
            _ReserveEvent(
                turn_index=turn_index,
                action_type=action_type,
                card_id=card.card_id if card is not None else None,
            )
        )

    missed = _missed_affordable_point_cards(
        state=state,
        player_id=player_id,
        legal_actions=legal_actions,
        chosen_card=card,
        chosen_action_type=action_type,
    )
    if missed["affordable_point_cards"] > 0:
        trace.missed_point_events.append(
            _MissedPointEvent(
                turn_index=turn_index,
                affordable_point_cards=missed["affordable_point_cards"],
                missed_point_cards=missed["missed_point_cards"],
            )
        )

    if turn_index < early_turn_max:
        trace.early_actions.append(
            _EarlyActionEvent(
                turn_index=turn_index,
                phase=_phase_for_turn(turn_index),
                action_type=action_type,
                action_signature=_action_signature(action_payload, card),
                action_bucket=_action_bucket(action_payload, card),
                card_id=card.card_id if card is not None else None,
                card_points=card.points if card is not None else None,
            )
        )


def _new_trace(
    label: str,
    game_seed: int,
    player_id: int,
    payload: dict[str, Any],
) -> _PlayerGameTrace:
    final_scores = payload.get("game_final_scores", [0, 0])
    bot_seats = payload.get("game_bot_seats", [])
    return _PlayerGameTrace(
        label=label,
        game_seed=game_seed,
        player_id=player_id,
        final_score=int(final_scores[player_id]) if len(final_scores) > player_id else 0,
        opponent_final_score=(
            int(final_scores[1 - player_id]) if len(final_scores) > 1 - player_id else 0
        ),
        winner=payload.get("game_winner"),
        game_turns=int(payload.get("game_turns", 0)),
        game_bot_seat=(
            str(bot_seats[player_id])
            if isinstance(bot_seats, list | tuple) and len(bot_seats) > player_id
            else None
        ),
        purchases=[],
        reserves=[],
        missed_point_events=[],
        early_actions=[],
    )


def _build_summary(
    config: ConversionDiagnosticsConfig,
    traces_by_label_key: dict[str, dict[tuple[int, int], _PlayerGameTrace]],
    rows_seen: int,
    rows_missing_snapshot: int,
    rows_skipped_by_game_seed: int,
    rows_skipped_by_player_id: int,
    rows_skipped_by_model_turn: int,
    rows_without_legal_actions: int,
) -> dict[str, Any]:
    by_label = {
        label: _summarize_label(traces, config.window_turns)
        for label, traces in traces_by_label_key.items()
    }
    paired = _summarize_paired_labels(
        labels=config.labels,
        traces_by_label_key=traces_by_label_key,
        examples_limit=config.examples_limit,
    )
    return {
        "labels": list(config.labels),
        "replay_paths": [str(path) for path in config.replay_paths],
        "rows_seen": rows_seen,
        "rows_missing_snapshot": rows_missing_snapshot,
        "rows_skipped_by_game_seed": rows_skipped_by_game_seed,
        "rows_skipped_by_player_id": rows_skipped_by_player_id,
        "rows_skipped_by_model_turn": rows_skipped_by_model_turn,
        "rows_without_legal_actions": rows_without_legal_actions,
        "player_games": sum(len(traces) for traces in traces_by_label_key.values()),
        "by_label": by_label,
        "paired_first_two_labels": paired,
        "config": {
            **asdict(config),
            "replay_paths": [str(path) for path in config.replay_paths],
            "output_path": str(config.output_path),
        },
    }


def _summarize_label(
    traces: dict[tuple[int, int], _PlayerGameTrace],
    window_turns: tuple[int, ...],
) -> dict[str, Any]:
    accumulator = _NumericAccumulator()
    by_bot_seat: dict[str, _NumericAccumulator] = defaultdict(_NumericAccumulator)
    by_window: dict[str, _NumericAccumulator] = {
        f"turns_0_{turn - 1}": _NumericAccumulator()
        for turn in sorted(set(window_turns))
    }
    by_window["full_game"] = _NumericAccumulator()
    early_buckets: dict[str, dict[str, Any]] = defaultdict(_new_early_bucket_counter)

    for trace in traces.values():
        end_metrics = _trace_metrics_after(trace, min_turn_exclusive=-1)
        accumulator.add(end_metrics)
        by_bot_seat[trace.game_bot_seat or "unknown"].add(end_metrics)
        for turn in sorted(set(window_turns)):
            by_window[f"turns_0_{turn - 1}"].add(
                _trace_metrics_before(trace, max_turn_exclusive=turn)
            )
        by_window["full_game"].add(end_metrics)
        for early_action in trace.early_actions:
            _record_early_bucket(early_buckets[early_action.action_bucket], trace, early_action)

    return {
        "player_games": len(traces),
        "overall": _finalize_accumulator(accumulator),
        "by_bot_seat": {
            key: _finalize_accumulator(value)
            for key, value in sorted(by_bot_seat.items())
        },
        "by_window": {
            key: _finalize_accumulator(value)
            for key, value in sorted(by_window.items())
        },
        "early_action_buckets": _finalize_early_buckets(early_buckets),
    }


def _summarize_paired_labels(
    labels: tuple[str, ...],
    traces_by_label_key: dict[str, dict[tuple[int, int], _PlayerGameTrace]],
    examples_limit: int,
) -> dict[str, Any] | None:
    if len(labels) < 2:
        return None
    before_label, after_label = labels[0], labels[1]
    before_traces = traces_by_label_key.get(before_label, {})
    after_traces = traces_by_label_key.get(after_label, {})
    common_keys = sorted(set(before_traces) & set(after_traces))
    delta_accumulator = _NumericAccumulator()
    changed_action_buckets: dict[str, _NumericAccumulator] = defaultdict(_NumericAccumulator)
    changed_action_type_buckets: dict[str, _NumericAccumulator] = defaultdict(_NumericAccumulator)
    examples: list[dict[str, Any]] = []

    for key in common_keys:
        before_trace = before_traces[key]
        after_trace = after_traces[key]
        delta = _metric_delta(before_trace, after_trace)
        delta_accumulator.add(delta)
        _record_changed_early_actions(
            before_label=before_label,
            after_label=after_label,
            before_trace=before_trace,
            after_trace=after_trace,
            delta=delta,
            changed_action_buckets=changed_action_buckets,
            changed_action_type_buckets=changed_action_type_buckets,
            examples=examples,
            examples_limit=examples_limit,
        )

    return {
        "before_label": before_label,
        "after_label": after_label,
        "paired_player_games": len(common_keys),
        "end_metric_delta_after_minus_before": _finalize_accumulator(delta_accumulator),
        "changed_early_action_buckets": _finalize_accumulator_buckets(changed_action_buckets),
        "changed_early_action_type_buckets": _finalize_accumulator_buckets(
            changed_action_type_buckets
        ),
        "examples": examples,
    }


def _record_changed_early_actions(
    before_label: str,
    after_label: str,
    before_trace: _PlayerGameTrace,
    after_trace: _PlayerGameTrace,
    delta: dict[str, float | int | None],
    changed_action_buckets: dict[str, _NumericAccumulator],
    changed_action_type_buckets: dict[str, _NumericAccumulator],
    examples: list[dict[str, Any]],
    examples_limit: int,
) -> None:
    before_by_turn = {event.turn_index: event for event in before_trace.early_actions}
    after_by_turn = {event.turn_index: event for event in after_trace.early_actions}
    for turn_index in sorted(set(before_by_turn) & set(after_by_turn)):
        before_event = before_by_turn[turn_index]
        after_event = after_by_turn[turn_index]
        if before_event.action_signature == after_event.action_signature:
            continue
        action_key = f"{before_event.action_bucket} -> {after_event.action_bucket}"
        type_key = f"{before_event.action_type} -> {after_event.action_type}"
        changed_action_buckets[action_key].add(delta)
        changed_action_type_buckets[type_key].add(delta)
        if examples_limit <= 0:
            continue
        example = {
            "game_seed": before_trace.game_seed,
            "player_id": before_trace.player_id,
            "turn_index": turn_index,
            "before_label": before_label,
            "after_label": after_label,
            "before_action_signature": before_event.action_signature,
            "after_action_signature": after_event.action_signature,
            "before_action_bucket": before_event.action_bucket,
            "after_action_bucket": after_event.action_bucket,
            "delta_after_minus_before": delta,
        }
        examples.append(example)
        examples.sort(
            key=lambda item: (
                float(item["delta_after_minus_before"].get("bought_card_points", 0.0)),
                float(item["delta_after_minus_before"].get("scoring_buys", 0.0)),
                float(item["delta_after_minus_before"].get("total_buys", 0.0)),
            )
        )
        del examples[examples_limit:]


def _trace_metrics_before(
    trace: _PlayerGameTrace,
    max_turn_exclusive: int,
) -> dict[str, float | int | None]:
    purchases = [event for event in trace.purchases if event.turn_index < max_turn_exclusive]
    reserves = [event for event in trace.reserves if event.turn_index < max_turn_exclusive]
    missed_events = [
        event for event in trace.missed_point_events if event.turn_index < max_turn_exclusive
    ]
    return _metrics_from_events(trace, purchases, reserves, missed_events)


def _trace_metrics_after(
    trace: _PlayerGameTrace,
    min_turn_exclusive: int,
) -> dict[str, float | int | None]:
    purchases = [event for event in trace.purchases if event.turn_index > min_turn_exclusive]
    reserves = [event for event in trace.reserves if event.turn_index > min_turn_exclusive]
    missed_events = [
        event for event in trace.missed_point_events if event.turn_index > min_turn_exclusive
    ]
    return _metrics_from_events(trace, purchases, reserves, missed_events)


def _metrics_from_events(
    trace: _PlayerGameTrace,
    purchases: list[_PurchaseEvent],
    reserves: list[_ReserveEvent],
    missed_events: list[_MissedPointEvent],
) -> dict[str, float | int | None]:
    scoring_purchases = [event for event in purchases if event.card_points > 0]
    return {
        "wins": 1 if trace.won else 0,
        "final_score": trace.final_score,
        "score_margin": trace.final_score - trace.opponent_final_score,
        "total_buys": len(purchases),
        "visible_buys": sum(1 for event in purchases if event.action_type == "BUY_VISIBLE"),
        "reserved_buys": sum(1 for event in purchases if event.action_type == "BUY_RESERVED"),
        "scoring_buys": len(scoring_purchases),
        "bought_card_points": sum(event.card_points for event in purchases),
        "noble_points_on_buys": sum(event.noble_points for event in purchases),
        "reserves": len(reserves),
        "nonconverting_reserves": sum(1 for event in reserves if not event.converted),
        "missed_affordable_point_turns": sum(
            1 for event in missed_events if event.missed_point_cards > 0
        ),
        "missed_affordable_point_cards": sum(
            event.missed_point_cards for event in missed_events
        ),
        "first_buy_turn": _min_turn(purchases),
        "first_scoring_buy_turn": _min_turn(scoring_purchases),
    }


def _metric_delta(
    before_trace: _PlayerGameTrace,
    after_trace: _PlayerGameTrace,
) -> dict[str, float | int | None]:
    before_metrics = _trace_metrics_after(before_trace, min_turn_exclusive=-1)
    after_metrics = _trace_metrics_after(after_trace, min_turn_exclusive=-1)
    return {
        key: (
            None
            if before_metrics.get(key) is None or after_metrics.get(key) is None
            else float(after_metrics[key]) - float(before_metrics[key])
        )
        for key in sorted(set(before_metrics) | set(after_metrics))
    }


def _new_early_bucket_counter() -> dict[str, Any]:
    return {
        "count": 0,
        "game_keys": set(),
        "reserve_actions": 0,
        "converted_reserve_actions": 0,
        "later": _NumericAccumulator(),
    }


def _record_early_bucket(
    bucket: dict[str, Any],
    trace: _PlayerGameTrace,
    early_action: _EarlyActionEvent,
) -> None:
    later_metrics = _trace_metrics_after(trace, min_turn_exclusive=early_action.turn_index)
    bucket["count"] += 1
    bucket["game_keys"].add((trace.game_seed, trace.player_id))
    if early_action.action_type in RESERVE_ACTION_TYPES:
        bucket["reserve_actions"] += 1
        if early_action.card_id is not None and _card_was_converted(trace, early_action.card_id):
            bucket["converted_reserve_actions"] += 1
    bucket["later"].add(later_metrics)


def _finalize_early_buckets(buckets: dict[str, dict[str, Any]]) -> dict[str, Any]:
    finalized = {}
    for key, bucket in sorted(
        buckets.items(),
        key=lambda item: (-item[1]["count"], item[0]),
    ):
        reserve_actions = bucket["reserve_actions"]
        finalized[key] = {
            "count": bucket["count"],
            "player_games": len(bucket["game_keys"]),
            "reserve_actions": reserve_actions,
            "converted_reserve_actions": bucket["converted_reserve_actions"],
            "reserve_conversion_rate": _rate(
                bucket["converted_reserve_actions"],
                reserve_actions,
            ),
            "later_metrics": _finalize_accumulator(bucket["later"]),
        }
    return finalized


def _finalize_accumulator(accumulator: _NumericAccumulator) -> dict[str, Any]:
    return {
        "count": accumulator.count,
        **accumulator.means(),
    }


def _finalize_accumulator_buckets(
    buckets: dict[str, _NumericAccumulator],
) -> dict[str, dict[str, Any]]:
    return {
        key: _finalize_accumulator(bucket)
        for key, bucket in sorted(
            buckets.items(),
            key=lambda item: (-item[1].count, item[0]),
        )
    }


def _is_model_turn(payload: dict[str, Any], player_id: int) -> bool:
    bot_seats = payload.get("game_bot_seats")
    if not isinstance(bot_seats, list | tuple):
        return False
    if player_id < 0 or player_id >= len(bot_seats):
        return False
    return str(bot_seats[player_id]) in MODEL_BOT_NAMES


def _card_for_payload_action(
    state: SplendorState,
    player_id: int,
    action_payload: dict[str, Any],
) -> Card | None:
    action_type = str(action_payload.get("action_type", ""))
    tier = action_payload.get("tier")
    market_index = action_payload.get("market_index")
    reserved_index = action_payload.get("reserved_index")
    if action_type in {"BUY_VISIBLE", "RESERVE_VISIBLE"}:
        if tier is None or market_index is None:
            return None
        return _visible_card(state, int(tier), int(market_index))
    if action_type == "RESERVE_DECK":
        if tier is None:
            return None
        deck = state.hidden_tier_decks.get(int(tier), [])
        return deck[0] if deck else None
    if action_type == "BUY_RESERVED":
        if reserved_index is None:
            return None
        reserved_cards = state.players[player_id].reserved_cards
        index = int(reserved_index)
        return reserved_cards[index] if 0 <= index < len(reserved_cards) else None
    return None


def _card_for_action(state: SplendorState, player_id: int, action: Action) -> Card | None:
    if action.action_type == ActionType.BUY_VISIBLE:
        if action.tier is None or action.market_index is None:
            return None
        return _visible_card(state, action.tier, action.market_index)
    if action.action_type == ActionType.BUY_RESERVED:
        if action.reserved_index is None:
            return None
        reserved_cards = state.players[player_id].reserved_cards
        return (
            reserved_cards[action.reserved_index]
            if 0 <= action.reserved_index < len(reserved_cards)
            else None
        )
    return None


def _visible_card(state: SplendorState, tier: int, market_index: int) -> Card | None:
    cards = state.visible_tier_cards.get(tier, [])
    return cards[market_index] if 0 <= market_index < len(cards) else None


def _missed_affordable_point_cards(
    state: SplendorState,
    player_id: int,
    legal_actions: list[Action],
    chosen_card: Card | None,
    chosen_action_type: str,
) -> dict[str, int]:
    affordable_card_ids = {
        card.card_id
        for action in legal_actions
        if action.action_type in {ActionType.BUY_VISIBLE, ActionType.BUY_RESERVED}
        for card in [_card_for_action(state, player_id, action)]
        if card is not None and card.points > 0
    }
    if not affordable_card_ids:
        return {"affordable_point_cards": 0, "missed_point_cards": 0}
    bought_point_card_id = (
        chosen_card.card_id
        if chosen_action_type in BUY_ACTION_TYPES
        and chosen_card is not None
        and chosen_card.points > 0
        else None
    )
    missed = len(affordable_card_ids - ({bought_point_card_id} if bought_point_card_id else set()))
    return {
        "affordable_point_cards": len(affordable_card_ids),
        "missed_point_cards": missed,
    }


def _noble_points_for_action(state: SplendorState, action_payload: dict[str, Any]) -> int:
    noble_id = action_payload.get("noble_id")
    if noble_id is None:
        return 0
    for noble in state.nobles:
        if noble.noble_id == noble_id:
            return noble.points
    return 0


def _mark_reserved_card_converted(reserves: list[_ReserveEvent], card_id: str) -> None:
    for reserve in reversed(reserves):
        if reserve.card_id == card_id and not reserve.converted:
            reserve.converted = True
            return


def _card_was_converted(trace: _PlayerGameTrace, card_id: str) -> bool:
    return any(
        event.card_id == card_id and event.action_type == "BUY_RESERVED"
        for event in trace.purchases
    )


def _action_signature(action_payload: dict[str, Any], card: Card | None) -> str:
    action_type = str(action_payload.get("action_type", "UNKNOWN"))
    if action_type == "TAKE_TOKENS":
        take = ",".join(action_payload.get("take_tokens", []))
        returned = ",".join(action_payload.get("return_tokens", []))
        return f"{action_type}:take={take}:return={returned}"
    if action_type in {"RESERVE_VISIBLE", "BUY_VISIBLE"}:
        return (
            f"{action_type}:tier={action_payload.get('tier')}"
            f":market={action_payload.get('market_index')}"
            f":card={card.card_id if card is not None else 'unknown'}"
        )
    if action_type == "RESERVE_DECK":
        return (
            f"{action_type}:tier={action_payload.get('tier')}"
            f":card={card.card_id if card is not None else 'unknown'}"
        )
    if action_type == "BUY_RESERVED":
        return (
            f"{action_type}:slot={action_payload.get('reserved_index')}"
            f":card={card.card_id if card is not None else 'unknown'}"
        )
    return action_type


def _action_bucket(action_payload: dict[str, Any], card: Card | None) -> str:
    action_type = str(action_payload.get("action_type", "UNKNOWN"))
    if action_type == "TAKE_TOKENS":
        take = ",".join(action_payload.get("take_tokens", []))
        return f"{action_type}:take={take}"
    if action_type in BUY_ACTION_TYPES | RESERVE_ACTION_TYPES:
        if card is None:
            return f"{action_type}:card=unknown"
        return (
            f"{action_type}:tier={card.tier}"
            f":points={card.points}"
            f":bonus={card.bonus_color}"
        )
    return action_type


def _min_turn(events: list[_PurchaseEvent]) -> int | None:
    return min((event.turn_index for event in events), default=None)


def _rate(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator else None


if __name__ == "__main__":
    main()
