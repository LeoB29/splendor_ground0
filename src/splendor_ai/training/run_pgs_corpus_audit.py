"""Audit and conservatively filter policy-guided-search relabel corpora."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from splendor_ai.engine.state import Card, SplendorState
from splendor_ai.training.replay import deserialize_state_snapshot


RESERVE_TYPES = {"RESERVE_VISIBLE", "RESERVE_DECK"}
BUY_TYPES = {"BUY_VISIBLE", "BUY_RESERVED"}


@dataclass(frozen=True, slots=True)
class PGSCorpusAuditConfig:
    input_paths: tuple[Path, ...]
    output_path: Path
    filtered_output_path: Path | None = None
    filtered_summary_path: Path | None = None
    filter_profile: str = "none"
    early_turn_max: int = 20
    examples_limit: int = 20
    log_every: int = 1_000


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit PGS relabel rows and optionally write a conservative filtered corpus."
    )
    parser.add_argument("--input-path", action="append", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--filtered-output-path", default=None)
    parser.add_argument("--filtered-summary-path", default=None)
    parser.add_argument(
        "--filter-profile",
        choices=("none", "conservative_029"),
        default="none",
    )
    parser.add_argument("--early-turn-max", type=int, default=20)
    parser.add_argument("--examples-limit", type=int, default=20)
    parser.add_argument("--log-every", type=int, default=1_000)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = run_pgs_corpus_audit(
        PGSCorpusAuditConfig(
            input_paths=tuple(Path(path) for path in args.input_path),
            output_path=Path(args.output_path),
            filtered_output_path=(
                Path(args.filtered_output_path)
                if args.filtered_output_path is not None
                else None
            ),
            filtered_summary_path=(
                Path(args.filtered_summary_path)
                if args.filtered_summary_path is not None
                else None
            ),
            filter_profile=args.filter_profile,
            early_turn_max=args.early_turn_max,
            examples_limit=args.examples_limit,
            log_every=args.log_every,
        )
    )
    print(
        "[pgs-audit]"
        f" rows_read={summary['rows_read']}"
        f" changed={summary['rows_changed']}"
        f" risky_early_changed={summary['risky_early_changed_rows']}"
        f" output={args.output_path}"
    )
    if summary.get("filtered") is not None:
        filtered = summary["filtered"]
        print(
            "[pgs-filter]"
            f" rows_written={filtered['rows_written']}"
            f" rows_dropped={filtered['rows_dropped']}"
            f" rows_downweighted={filtered['rows_downweighted']}"
            f" output={filtered['output_path']}"
        )


def run_pgs_corpus_audit(config: PGSCorpusAuditConfig) -> dict[str, Any]:
    _validate_config(config)
    counters = _new_counters()
    examples: list[dict[str, Any]] = []
    filtered_rows: list[str] = []
    rows_read = 0

    for input_path in config.input_paths:
        with input_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                rows_read += 1
                payload = json.loads(stripped)
                row = _analyze_row(payload, input_path, config.early_turn_max)
                _record_row(counters, row)
                _maybe_store_example(examples, row, config.examples_limit)
                if config.filtered_output_path is not None:
                    filtered_payload = _filtered_payload_for_row(
                        payload,
                        row,
                        config.filter_profile,
                    )
                    if filtered_payload is not None:
                        filtered_rows.append(json.dumps(filtered_payload))
                if config.log_every > 0 and rows_read % config.log_every == 0:
                    print(
                        f"[pgs-audit] rows_read={rows_read}"
                        f" changed={counters['rows_changed']}"
                        f" risky={counters['risky_early_changed_rows']}"
                    )

    summary = _build_summary(
        config=config,
        counters=counters,
        rows_read=rows_read,
        examples=examples,
        filtered_rows=filtered_rows,
    )
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    config.output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if config.filtered_output_path is not None:
        config.filtered_output_path.parent.mkdir(parents=True, exist_ok=True)
        config.filtered_output_path.write_text(
            "\n".join(filtered_rows) + ("\n" if filtered_rows else ""),
            encoding="utf-8",
        )
        filtered_summary_path = config.filtered_summary_path or config.filtered_output_path.with_suffix(
            ".summary.json"
        )
        filtered_summary_path.parent.mkdir(parents=True, exist_ok=True)
        filtered_summary_path.write_text(
            json.dumps(summary["filtered"], indent=2),
            encoding="utf-8",
        )
    return summary


def _validate_config(config: PGSCorpusAuditConfig) -> None:
    if not config.input_paths:
        raise ValueError("At least one input path is required.")
    if config.early_turn_max < 0:
        raise ValueError("early_turn_max must be non-negative.")
    if config.examples_limit < 0:
        raise ValueError("examples_limit must be non-negative.")
    if config.filter_profile != "none" and config.filtered_output_path is None:
        raise ValueError("A filter profile requires --filtered-output-path.")


def _new_counters() -> dict[str, Any]:
    return {
        "rows_changed": 0,
        "rows_unchanged": 0,
        "rows_missing_snapshot": 0,
        "risky_early_changed_rows": 0,
        "search_margins": [],
        "changed_search_margins": [],
        "risky_search_margins": [],
        "by_transition": Counter(),
        "by_transition_early_changed": Counter(),
        "by_selected_bucket": Counter(),
        "by_original_bucket": Counter(),
        "by_risk_reason": Counter(),
        "selected_card_points": Counter(),
        "original_card_points": Counter(),
        "filtered": {
            "rows_written": 0,
            "rows_dropped": 0,
            "rows_downweighted": 0,
            "rows_unchanged_weight": 0,
            "dropped_by_reason": Counter(),
            "downweighted_by_reason": Counter(),
            "original_weight_sum": 0.0,
            "filtered_weight_sum": 0.0,
        },
    }


def _analyze_row(
    payload: dict[str, Any],
    input_path: Path,
    early_turn_max: int,
) -> dict[str, Any]:
    selected_payload = _action_payload(payload.get("action_payload"))
    original_payload = _action_payload(
        payload.get("original_action_payload", payload.get("action_payload"))
    )
    selected_type = _action_type(selected_payload)
    original_type = _action_type(original_payload)
    changed = bool(
        payload.get("improvement_metadata", {}).get(
            "changed_action",
            payload.get("action_index") != payload.get("original_action_index"),
        )
    )
    turn_index = int(payload.get("turn_index", 0))
    state = _state_from_payload(payload)
    player_id = int(payload.get("player_id", state.current_player if state is not None else 0))
    selected_card = (
        _card_for_payload_action(state, player_id, selected_payload)
        if state is not None
        else None
    )
    original_card = (
        _card_for_payload_action(state, player_id, original_payload)
        if state is not None
        else None
    )
    search_margin = _search_margin(payload)
    risk_reasons = _risk_reasons(
        changed=changed,
        turn_index=turn_index,
        early_turn_max=early_turn_max,
        original_type=original_type,
        selected_type=selected_type,
        selected_card=selected_card,
    )
    filter_decision = _conservative_029_decision(
        changed=changed,
        turn_index=turn_index,
        early_turn_max=early_turn_max,
        original_type=original_type,
        selected_type=selected_type,
        selected_card=selected_card,
    )
    return {
        "input_path": str(input_path),
        "game_seed": payload.get("game_seed"),
        "turn_index": turn_index,
        "early": turn_index < early_turn_max,
        "player_id": player_id,
        "changed": changed,
        "original_type": original_type,
        "selected_type": selected_type,
        "transition": f"{original_type} -> {selected_type}",
        "original_bucket": _action_bucket(original_type, original_card, original_payload),
        "selected_bucket": _action_bucket(selected_type, selected_card, selected_payload),
        "original_card": _card_payload(original_card),
        "selected_card": _card_payload(selected_card),
        "search_margin": search_margin,
        "training_weight": float(payload.get("training_weight", 1.0)),
        "risk_reasons": risk_reasons,
        "filter_decision": filter_decision,
        "selected_action_payload": selected_payload,
        "original_action_payload": original_payload,
    }


def _record_row(counters: dict[str, Any], row: dict[str, Any]) -> None:
    changed = bool(row["changed"])
    if changed:
        counters["rows_changed"] += 1
    else:
        counters["rows_unchanged"] += 1
    if row["selected_card"] is None and row["selected_type"] in RESERVE_TYPES | BUY_TYPES:
        counters["rows_missing_snapshot"] += 1
    search_margin = row["search_margin"]
    if search_margin is not None:
        counters["search_margins"].append(search_margin)
        if changed:
            counters["changed_search_margins"].append(search_margin)
    counters["by_transition"][row["transition"]] += 1
    counters["by_selected_bucket"][row["selected_bucket"]] += 1
    counters["by_original_bucket"][row["original_bucket"]] += 1
    if changed and bool(row["early"]):
        counters["by_transition_early_changed"][row["transition"]] += 1
    for reason in row["risk_reasons"]:
        counters["by_risk_reason"][reason] += 1
    if row["risk_reasons"]:
        counters["risky_early_changed_rows"] += 1
        if search_margin is not None:
            counters["risky_search_margins"].append(search_margin)
    if row["selected_card"] is not None:
        counters["selected_card_points"][str(row["selected_card"]["points"])] += 1
    if row["original_card"] is not None:
        counters["original_card_points"][str(row["original_card"]["points"])] += 1
    decision = row["filter_decision"]
    filtered = counters["filtered"]
    original_weight = float(row["training_weight"])
    filtered["original_weight_sum"] += original_weight
    if decision["action"] == "drop":
        filtered["rows_dropped"] += 1
        filtered["dropped_by_reason"][decision["reason"]] += 1
    else:
        filtered["rows_written"] += 1
        new_weight = original_weight * float(decision["weight_multiplier"])
        filtered["filtered_weight_sum"] += new_weight
        if new_weight != original_weight:
            filtered["rows_downweighted"] += 1
            filtered["downweighted_by_reason"][decision["reason"]] += 1
        else:
            filtered["rows_unchanged_weight"] += 1


def _maybe_store_example(
    examples: list[dict[str, Any]],
    row: dict[str, Any],
    examples_limit: int,
) -> None:
    if examples_limit <= 0 or not row["risk_reasons"]:
        return
    examples.append(
        {
            "game_seed": row["game_seed"],
            "turn_index": row["turn_index"],
            "player_id": row["player_id"],
            "transition": row["transition"],
            "original_bucket": row["original_bucket"],
            "selected_bucket": row["selected_bucket"],
            "original_card": row["original_card"],
            "selected_card": row["selected_card"],
            "search_margin": row["search_margin"],
            "training_weight": row["training_weight"],
            "risk_reasons": row["risk_reasons"],
            "filter_decision": row["filter_decision"],
            "original_action_payload": row["original_action_payload"],
            "selected_action_payload": row["selected_action_payload"],
        }
    )
    examples.sort(
        key=lambda item: (
            len(item["risk_reasons"]),
            float(item["search_margin"] if item["search_margin"] is not None else -math.inf),
        ),
        reverse=True,
    )
    del examples[examples_limit:]


def _filtered_payload_for_row(
    payload: dict[str, Any],
    row: dict[str, Any],
    filter_profile: str,
) -> dict[str, Any] | None:
    if filter_profile == "none":
        return dict(payload)
    if filter_profile != "conservative_029":
        raise ValueError(f"Unsupported filter profile: {filter_profile}")
    decision = row["filter_decision"]
    if decision["action"] == "drop":
        return None
    filtered = dict(payload)
    original_weight = float(filtered.get("training_weight", 1.0))
    filtered["training_weight"] = original_weight * float(decision["weight_multiplier"])
    metadata = dict(filtered.get("corpus_filter_metadata", {}))
    metadata["pgs_corpus_audit"] = {
        "filter_profile": filter_profile,
        "decision": decision,
        "risk_reasons": row["risk_reasons"],
        "original_training_weight": original_weight,
    }
    filtered["corpus_filter_metadata"] = metadata
    return filtered


def _build_summary(
    config: PGSCorpusAuditConfig,
    counters: dict[str, Any],
    rows_read: int,
    examples: list[dict[str, Any]],
    filtered_rows: list[str],
) -> dict[str, Any]:
    filtered = counters["filtered"]
    filtered_summary = {
        "filter_profile": config.filter_profile,
        "output_path": str(config.filtered_output_path) if config.filtered_output_path else None,
        "summary_path": (
            str(config.filtered_summary_path)
            if config.filtered_summary_path
            else (
                str(config.filtered_output_path.with_suffix(".summary.json"))
                if config.filtered_output_path
                else None
            )
        ),
        "rows_written": filtered["rows_written"],
        "rows_dropped": filtered["rows_dropped"],
        "rows_downweighted": filtered["rows_downweighted"],
        "rows_unchanged_weight": filtered["rows_unchanged_weight"],
        "dropped_by_reason": dict(filtered["dropped_by_reason"]),
        "downweighted_by_reason": dict(filtered["downweighted_by_reason"]),
        "original_weight_sum": filtered["original_weight_sum"],
        "filtered_weight_sum": filtered["filtered_weight_sum"],
    }
    return {
        "input_paths": [str(path) for path in config.input_paths],
        "rows_read": rows_read,
        "rows_changed": counters["rows_changed"],
        "rows_unchanged": counters["rows_unchanged"],
        "rows_missing_snapshot": counters["rows_missing_snapshot"],
        "risky_early_changed_rows": counters["risky_early_changed_rows"],
        "search_margin": _stats(counters["search_margins"]),
        "changed_search_margin": _stats(counters["changed_search_margins"]),
        "risky_search_margin": _stats(counters["risky_search_margins"]),
        "by_transition": _counter_payload(counters["by_transition"]),
        "by_transition_early_changed": _counter_payload(
            counters["by_transition_early_changed"]
        ),
        "by_selected_bucket": _counter_payload(counters["by_selected_bucket"]),
        "by_original_bucket": _counter_payload(counters["by_original_bucket"]),
        "by_risk_reason": _counter_payload(counters["by_risk_reason"]),
        "selected_card_points": dict(sorted(counters["selected_card_points"].items())),
        "original_card_points": dict(sorted(counters["original_card_points"].items())),
        "examples": examples,
        "filtered": filtered_summary if config.filtered_output_path is not None else None,
        "config": {
            **asdict(config),
            "input_paths": [str(path) for path in config.input_paths],
            "output_path": str(config.output_path),
            "filtered_output_path": (
                str(config.filtered_output_path)
                if config.filtered_output_path is not None
                else None
            ),
            "filtered_summary_path": (
                str(config.filtered_summary_path)
                if config.filtered_summary_path is not None
                else None
            ),
        },
        "filtered_row_count_check": len(filtered_rows),
    }


def _risk_reasons(
    changed: bool,
    turn_index: int,
    early_turn_max: int,
    original_type: str,
    selected_type: str,
    selected_card: Card | None,
) -> list[str]:
    if not changed or turn_index >= early_turn_max:
        return []
    reasons = []
    if original_type == "RESERVE_VISIBLE" and selected_type == "RESERVE_DECK":
        reasons.append("early_reserve_visible_to_deck")
    if original_type in RESERVE_TYPES and selected_type in RESERVE_TYPES:
        reasons.append("early_reserve_to_reserve")
    if selected_type == "RESERVE_DECK":
        reasons.append("early_selected_blind_deck_reserve")
    if selected_type == "BUY_RESERVED" and (selected_card is None or selected_card.points == 0):
        reasons.append("early_zero_point_reserved_buy")
    if original_type == "BUY_RESERVED" and selected_type == "BUY_RESERVED":
        reasons.append("early_reserved_buy_swap")
    return reasons


def _conservative_029_decision(
    changed: bool,
    turn_index: int,
    early_turn_max: int,
    original_type: str,
    selected_type: str,
    selected_card: Card | None,
) -> dict[str, Any]:
    if not changed or turn_index >= early_turn_max:
        return {"action": "keep", "weight_multiplier": 1.0, "reason": "keep"}
    if original_type == "RESERVE_VISIBLE" and selected_type == "RESERVE_DECK":
        return {
            "action": "drop",
            "weight_multiplier": 0.0,
            "reason": "drop_early_reserve_visible_to_deck",
        }
    if selected_type == "RESERVE_DECK":
        return {
            "action": "drop",
            "weight_multiplier": 0.0,
            "reason": "drop_early_selected_blind_deck_reserve",
        }
    if original_type in RESERVE_TYPES and selected_type in RESERVE_TYPES:
        return {
            "action": "keep",
            "weight_multiplier": 0.3,
            "reason": "downweight_early_reserve_to_reserve",
        }
    if selected_type == "BUY_RESERVED" and (selected_card is None or selected_card.points == 0):
        return {
            "action": "keep",
            "weight_multiplier": 0.3,
            "reason": "downweight_early_zero_point_reserved_buy",
        }
    if original_type == "BUY_RESERVED" and selected_type == "BUY_RESERVED":
        return {
            "action": "keep",
            "weight_multiplier": 0.3,
            "reason": "downweight_early_reserved_buy_swap",
        }
    return {"action": "keep", "weight_multiplier": 1.0, "reason": "keep"}


def _state_from_payload(payload: dict[str, Any]) -> SplendorState | None:
    snapshot = payload.get("state_snapshot")
    if not isinstance(snapshot, dict):
        return None
    return deserialize_state_snapshot(snapshot)


def _action_payload(payload: Any) -> dict[str, Any]:
    return dict(payload) if isinstance(payload, dict) else {}


def _action_type(payload: dict[str, Any]) -> str:
    action_type = payload.get("action_type")
    return str(action_type) if action_type is not None else "UNKNOWN"


def _search_margin(payload: dict[str, Any]) -> float | None:
    metadata = payload.get("improvement_metadata")
    if not isinstance(metadata, dict):
        return None
    value = metadata.get("search_margin")
    return float(value) if value is not None else None


def _card_for_payload_action(
    state: SplendorState,
    player_id: int,
    action_payload: dict[str, Any],
) -> Card | None:
    action_type = _action_type(action_payload)
    tier = action_payload.get("tier")
    market_index = action_payload.get("market_index")
    reserved_index = action_payload.get("reserved_index")
    if action_type in {"BUY_VISIBLE", "RESERVE_VISIBLE"}:
        if tier is None or market_index is None:
            return None
        cards = state.visible_tier_cards.get(int(tier), [])
        index = int(market_index)
        return cards[index] if 0 <= index < len(cards) else None
    if action_type == "RESERVE_DECK":
        if tier is None:
            return None
        deck = state.hidden_tier_decks.get(int(tier), [])
        return deck[0] if deck else None
    if action_type == "BUY_RESERVED":
        if reserved_index is None:
            return None
        cards = state.players[player_id].reserved_cards
        index = int(reserved_index)
        return cards[index] if 0 <= index < len(cards) else None
    return None


def _action_bucket(
    action_type: str,
    card: Card | None,
    action_payload: dict[str, Any],
) -> str:
    if action_type == "TAKE_TOKENS":
        return f"{action_type}:take={','.join(action_payload.get('take_tokens', []))}"
    if action_type in RESERVE_TYPES | BUY_TYPES:
        if card is None:
            return f"{action_type}:card=unknown"
        return (
            f"{action_type}:tier={card.tier}"
            f":points={card.points}"
            f":bonus={card.bonus_color}"
        )
    return action_type


def _card_payload(card: Card | None) -> dict[str, Any] | None:
    if card is None:
        return None
    return {
        "card_id": card.card_id,
        "tier": card.tier,
        "points": card.points,
        "bonus_color": card.bonus_color,
        "cost": dict(card.cost),
    }


def _stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "min": None, "max": None}
    sorted_values = sorted(values)
    return {
        "count": len(values),
        "mean": sum(values) / len(values),
        "min": sorted_values[0],
        "p25": _percentile(sorted_values, 0.25),
        "median": _percentile(sorted_values, 0.5),
        "p75": _percentile(sorted_values, 0.75),
        "max": sorted_values[-1],
    }


def _percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return math.nan
    position = (len(sorted_values) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def _counter_payload(counter: Counter[str]) -> dict[str, int]:
    return {
        key: counter[key]
        for key in sorted(counter, key=lambda item: (-counter[item], item))
    }


if __name__ == "__main__":
    main()
