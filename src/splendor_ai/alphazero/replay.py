"""Information-legal AlphaZero replay rows and JSONL persistence."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
from typing import Any, Iterable

from splendor_ai.encoding import (
    encode_canonical_action_entities,
    encode_canonical_state_entities,
)
from splendor_ai.engine.state import SplendorState
from splendor_ai.training.replay import serialize_action

from .search import SearchResult


@dataclass(frozen=True, slots=True)
class AlphaZeroReplayRow:
    generation: int
    game_seed: int
    turn_index: int
    player_id: int
    entity_type_ids: tuple[int, ...]
    entity_card_ids: tuple[int, ...]
    entity_noble_ids: tuple[int, ...]
    entity_numeric_features: tuple[tuple[float, ...], ...]
    action_type_ids: tuple[int, ...]
    action_card_ids: tuple[int, ...]
    action_noble_ids: tuple[int, ...]
    action_numeric_features: tuple[tuple[float, ...], ...]
    action_payloads: tuple[dict[str, Any], ...]
    root_visits: tuple[int, ...]
    policy_target: tuple[float, ...]
    root_network_priors: tuple[float, ...]
    root_search_priors: tuple[float, ...]
    root_q_values: tuple[float, ...]
    selected_action_row: int
    raw_policy_top_row: int
    search_changed_raw_action: bool
    visit_argmax_row: int
    visit_argmax_changed_raw_action: bool
    raw_network_value: float
    search_root_value: float
    value_target: float | None
    winner: int | None
    final_scores: tuple[int, int] | None
    termination_reason: str | None
    checkpoint_generation: int
    checkpoint_path: str
    search_config: dict[str, Any]
    search_telemetry: dict[str, float | int | str]

    def with_outcome(
        self,
        *,
        winner: int | None,
        final_scores: tuple[int, int],
        termination_reason: str,
    ) -> "AlphaZeroReplayRow":
        value = 0.0 if winner is None else (1.0 if winner == self.player_id else -1.0)
        return replace(
            self,
            value_target=value,
            winner=winner,
            final_scores=final_scores,
            termination_reason=termination_reason,
        )


def replay_row_from_search(
    *,
    generation: int,
    game_seed: int,
    state: SplendorState,
    result: SearchResult,
    checkpoint_path: str,
    checkpoint_generation: int,
    search_config: dict[str, Any],
) -> AlphaZeroReplayRow:
    actions = [edge.action for edge in result.root_edges]
    state_encoding = encode_canonical_state_entities(state, state.current_player)
    action_encoding = encode_canonical_action_entities(
        state,
        actions,
        state.current_player,
    )
    probabilities = tuple(float(edge.probability) for edge in result.root_edges)
    if abs(sum(probabilities) - 1.0) > 1e-8:
        raise ValueError("Search probabilities must sum to one.")
    if not 0 <= result.selected_action_row < len(actions):
        raise ValueError("Selected action row is outside the legal action set.")
    return AlphaZeroReplayRow(
        generation=int(generation),
        game_seed=int(game_seed),
        turn_index=int(state.turn_index),
        player_id=int(state.current_player),
        entity_type_ids=state_encoding.entity_type_ids,
        entity_card_ids=state_encoding.card_ids,
        entity_noble_ids=state_encoding.noble_ids,
        entity_numeric_features=state_encoding.numeric_features,
        action_type_ids=action_encoding.action_type_ids,
        action_card_ids=action_encoding.card_ids,
        action_noble_ids=action_encoding.noble_ids,
        action_numeric_features=action_encoding.numeric_features,
        action_payloads=tuple(serialize_action(action) for action in actions),
        root_visits=tuple(int(edge.visits) for edge in result.root_edges),
        policy_target=probabilities,
        root_network_priors=tuple(
            float(edge.network_prior) for edge in result.root_edges
        ),
        root_search_priors=tuple(
            float(edge.search_prior) for edge in result.root_edges
        ),
        root_q_values=tuple(float(edge.q) for edge in result.root_edges),
        selected_action_row=int(result.selected_action_row),
        raw_policy_top_row=int(result.raw_policy_top_row),
        search_changed_raw_action=(
            result.selected_action_row != result.raw_policy_top_row
        ),
        visit_argmax_row=int(result.visit_argmax_row),
        visit_argmax_changed_raw_action=(
            result.visit_argmax_row != result.raw_policy_top_row
        ),
        raw_network_value=float(result.raw_network_value),
        search_root_value=float(result.search_root_value),
        value_target=None,
        winner=None,
        final_scores=None,
        termination_reason=None,
        checkpoint_generation=int(checkpoint_generation),
        checkpoint_path=str(checkpoint_path),
        search_config=dict(search_config),
        search_telemetry={
            "simulations": int(result.simulations),
            "expansions": int(result.expansions),
            "inference_calls": int(result.inference_calls),
            "max_depth_reached": int(result.max_depth_reached),
            "elapsed_seconds": float(result.elapsed_seconds),
            "search_mode": result.search_mode,
        },
    )


def write_replay_jsonl(
    path: str | Path,
    rows: Iterable[AlphaZeroReplayRow],
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(asdict(row), sort_keys=True))
            handle.write("\n")
    return output_path


def append_replay_jsonl(
    path: str | Path,
    rows: Iterable[AlphaZeroReplayRow],
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(asdict(row), sort_keys=True))
            handle.write("\n")
        handle.flush()
    return output_path


def read_replay_jsonl(path: str | Path) -> list[AlphaZeroReplayRow]:
    rows: list[AlphaZeroReplayRow] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            rows.append(_row_from_payload(payload))
    return rows


def _row_from_payload(payload: dict[str, Any]) -> AlphaZeroReplayRow:
    return AlphaZeroReplayRow(
        generation=int(payload["generation"]),
        game_seed=int(payload["game_seed"]),
        turn_index=int(payload["turn_index"]),
        player_id=int(payload["player_id"]),
        entity_type_ids=tuple(int(value) for value in payload["entity_type_ids"]),
        entity_card_ids=tuple(int(value) for value in payload["entity_card_ids"]),
        entity_noble_ids=tuple(int(value) for value in payload["entity_noble_ids"]),
        entity_numeric_features=tuple(
            tuple(float(value) for value in row)
            for row in payload["entity_numeric_features"]
        ),
        action_type_ids=tuple(int(value) for value in payload["action_type_ids"]),
        action_card_ids=tuple(int(value) for value in payload["action_card_ids"]),
        action_noble_ids=tuple(int(value) for value in payload["action_noble_ids"]),
        action_numeric_features=tuple(
            tuple(float(value) for value in row)
            for row in payload["action_numeric_features"]
        ),
        action_payloads=tuple(dict(value) for value in payload["action_payloads"]),
        root_visits=tuple(int(value) for value in payload["root_visits"]),
        policy_target=tuple(float(value) for value in payload["policy_target"]),
        root_network_priors=tuple(
            float(value)
            for value in payload.get("root_network_priors", payload["policy_target"])
        ),
        root_search_priors=tuple(
            float(value)
            for value in payload.get("root_search_priors", payload["policy_target"])
        ),
        root_q_values=tuple(
            float(value)
            for value in payload.get(
                "root_q_values",
                [0.0 for _ in payload["policy_target"]],
            )
        ),
        selected_action_row=int(payload["selected_action_row"]),
        raw_policy_top_row=int(
            payload.get("raw_policy_top_row", payload["selected_action_row"])
        ),
        search_changed_raw_action=bool(
            payload.get("search_changed_raw_action", False)
        ),
        visit_argmax_row=int(
            payload.get("visit_argmax_row", payload["selected_action_row"])
        ),
        visit_argmax_changed_raw_action=bool(
            payload.get("visit_argmax_changed_raw_action", False)
        ),
        raw_network_value=float(payload["raw_network_value"]),
        search_root_value=float(payload["search_root_value"]),
        value_target=(
            float(payload["value_target"])
            if payload.get("value_target") is not None
            else None
        ),
        winner=(int(payload["winner"]) if payload.get("winner") is not None else None),
        final_scores=(
            tuple(int(value) for value in payload["final_scores"])
            if payload.get("final_scores") is not None
            else None
        ),
        termination_reason=payload.get("termination_reason"),
        checkpoint_generation=int(payload["checkpoint_generation"]),
        checkpoint_path=str(payload["checkpoint_path"]),
        search_config=dict(payload["search_config"]),
        search_telemetry=dict(payload["search_telemetry"]),
    )
