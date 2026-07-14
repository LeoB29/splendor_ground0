"""Build selective preference corrections and champion anti-regression anchors."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

from splendor_ai.encoding import ActionCodec
from splendor_ai.engine.actions import Action, ActionType
from splendor_ai.engine.constants import TOKEN_COLORS
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.training.replay import deserialize_state_snapshot, serialize_action
from splendor_ai.training.run_policy_search_disagreement import _load_policy_scorer


@dataclass(frozen=True, slots=True)
class ConstrainedPreferenceCorpusConfig:
    checkpoint: Path
    pgs_replay_paths: tuple[Path, ...]
    regression_replay_paths: tuple[Path, ...]
    preference_output_path: Path
    anchor_output_path: Path
    summary_path: Path
    model_kind: str = "legal_action_attention_scorer"
    device: str = "cpu"
    min_search_margin: float = 50.0
    preference_training_weight: float = 1.0
    anchor_training_weight: float = 2.0
    regression_game_seeds: tuple[int, ...] | None = None
    champion_marker: str = "legal_action_scorer_022_attention_distill"
    log_every: int = 500


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build conversion-filtered preference rows and champion regression anchors."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--pgs-replay-path", action="append", required=True)
    parser.add_argument("--regression-replay-path", action="append", required=True)
    parser.add_argument("--preference-output-path", required=True)
    parser.add_argument("--anchor-output-path", required=True)
    parser.add_argument("--summary-path", required=True)
    parser.add_argument("--model-kind", default="legal_action_attention_scorer")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--min-search-margin", type=float, default=50.0)
    parser.add_argument("--preference-training-weight", type=float, default=1.0)
    parser.add_argument("--anchor-training-weight", type=float, default=2.0)
    parser.add_argument("--regression-game-seeds", nargs="*", type=int, default=None)
    parser.add_argument(
        "--champion-marker",
        default="legal_action_scorer_022_attention_distill",
    )
    parser.add_argument("--log-every", type=int, default=500)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = build_constrained_preference_corpus(
        ConstrainedPreferenceCorpusConfig(
            checkpoint=Path(args.checkpoint),
            pgs_replay_paths=tuple(Path(path) for path in args.pgs_replay_path),
            regression_replay_paths=tuple(Path(path) for path in args.regression_replay_path),
            preference_output_path=Path(args.preference_output_path),
            anchor_output_path=Path(args.anchor_output_path),
            summary_path=Path(args.summary_path),
            model_kind=args.model_kind,
            device=args.device,
            min_search_margin=args.min_search_margin,
            preference_training_weight=args.preference_training_weight,
            anchor_training_weight=args.anchor_training_weight,
            regression_game_seeds=(
                tuple(args.regression_game_seeds)
                if args.regression_game_seeds
                else None
            ),
            champion_marker=args.champion_marker,
            log_every=args.log_every,
        )
    )
    print(
        "Constrained preference corpus:"
        f" preferences={summary['preferences']['rows_written']}"
        f" anchors={summary['anchors']['rows_written']}"
    )


def build_constrained_preference_corpus(
    config: ConstrainedPreferenceCorpusConfig,
) -> dict[str, Any]:
    if config.min_search_margin < 0:
        raise ValueError("min_search_margin must be non-negative.")
    if config.preference_training_weight <= 0 or config.anchor_training_weight <= 0:
        raise ValueError("Training weights must be positive.")

    for path in (
        config.preference_output_path,
        config.anchor_output_path,
        config.summary_path,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)

    env = SplendorEnv()
    codec = ActionCodec()
    scorer = _load_policy_scorer(config)
    preference_counts: Counter[str] = Counter()
    preference_transitions: Counter[str] = Counter()
    anchor_counts: Counter[str] = Counter()

    with config.preference_output_path.open("w", encoding="utf-8") as output:
        for payload in _iter_rows(config.pgs_replay_paths):
            preference_counts["rows_seen"] += 1
            row, reason = _select_preference_row(
                payload=payload,
                config=config,
                env=env,
                codec=codec,
                scorer=scorer,
            )
            if row is None:
                preference_counts[f"skipped_{reason}"] += 1
            else:
                output.write(json.dumps(row) + "\n")
                preference_counts["rows_written"] += 1
                selected_type = str(row["action_payload"]["action_type"])
                model_type = str(row["model_top_action_payload"]["action_type"])
                preference_counts[f"written_{selected_type}"] += 1
                preference_transitions[f"{model_type} -> {selected_type}"] += 1
            if (
                config.log_every > 0
                and preference_counts["rows_seen"] % config.log_every == 0
            ):
                print(
                    f"[constrained-corpus] pgs_seen={preference_counts['rows_seen']}"
                    f" preferences={preference_counts['rows_written']}"
                )

    allowed_seeds = set(config.regression_game_seeds or ())
    with config.anchor_output_path.open("w", encoding="utf-8") as output:
        for payload in _iter_rows(config.regression_replay_paths):
            anchor_counts["rows_seen"] += 1
            seed = int(payload.get("game_seed", payload.get("seed", -1)))
            if allowed_seeds and seed not in allowed_seeds:
                anchor_counts["skipped_seed"] += 1
                continue
            if not _is_champion_turn(payload, config.champion_marker):
                anchor_counts["skipped_non_champion_turn"] += 1
                continue
            row = _sanitized_anchor(payload, config.anchor_training_weight)
            output.write(json.dumps(row) + "\n")
            anchor_counts["rows_written"] += 1
            anchor_counts[f"written_{row['action_payload']['action_type']}"] += 1

    summary = {
        "config": {
            **asdict(config),
            "checkpoint": str(config.checkpoint),
            "pgs_replay_paths": [str(path) for path in config.pgs_replay_paths],
            "regression_replay_paths": [
                str(path) for path in config.regression_replay_paths
            ],
            "preference_output_path": str(config.preference_output_path),
            "anchor_output_path": str(config.anchor_output_path),
            "summary_path": str(config.summary_path),
        },
        "preferences": {
            **dict(preference_counts),
            "transitions": dict(preference_transitions),
        },
        "anchors": dict(anchor_counts),
    }
    config.summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _select_preference_row(
    payload: dict[str, Any],
    config: ConstrainedPreferenceCorpusConfig,
    env: SplendorEnv,
    codec: ActionCodec,
    scorer: Any,
) -> tuple[dict[str, Any] | None, str]:
    snapshot = payload.get("state_snapshot")
    if not isinstance(snapshot, dict):
        return None, "missing_snapshot"
    original_index = payload.get("original_action_index")
    selected_index = payload.get("action_index")
    if original_index is None or selected_index is None:
        return None, "missing_action"
    if int(original_index) == int(selected_index):
        return None, "unchanged"

    state = deserialize_state_snapshot(snapshot)
    legal_actions = env.legal_actions(state)
    actions_by_index = {
        codec.encode(state, action): action
        for action in legal_actions
    }
    selected_action = actions_by_index.get(int(selected_index))
    if selected_action is None:
        return None, "illegal_selected_action"
    model_scores = scorer.score_actions(state, legal_actions)
    if not model_scores:
        return None, "missing_model_scores"
    model_top = model_scores[0]
    if model_top.action_index == int(selected_index):
        return None, "already_champion_top"

    search_values = {
        int(item["action_index"]): float(item["value"])
        for item in payload.get("search_action_scores", ())
        if isinstance(item, dict)
        and item.get("action_index") is not None
        and item.get("value") is not None
    }
    selected_value = search_values.get(int(selected_index))
    model_value = search_values.get(model_top.action_index)
    if selected_value is None or model_value is None:
        return None, "missing_comparable_search_value"
    search_margin = selected_value - model_value
    if search_margin < config.min_search_margin:
        return None, "search_margin"

    quality_reason = _conversion_quality_reason(
        env=env,
        state=state,
        selected_action=selected_action,
        model_action=model_top.action,
    )
    if quality_reason is None:
        return None, "conversion_quality"

    row = dict(payload)
    row.update(
        {
            "model_top_action_index": model_top.action_index,
            "model_top_action_payload": serialize_action(model_top.action),
            "model_top_score": model_top.score,
            "model_top_search_value": model_value,
            "search_margin": search_margin,
            "training_weight": config.preference_training_weight,
            "target_source": "constrained_pgs_preference",
            "constrained_preference_metadata": {
                "champion_checkpoint": str(config.checkpoint),
                "min_search_margin": config.min_search_margin,
                "conversion_quality_rule": quality_reason,
            },
        }
    )
    return row, "written"


def _conversion_quality_reason(
    env: SplendorEnv,
    state: Any,
    selected_action: Action,
    model_action: Action,
) -> str | None:
    selected_type = selected_action.action_type
    if selected_type in {ActionType.BUY_VISIBLE, ActionType.BUY_RESERVED}:
        card = _card_for_action(state, selected_action)
        return "scoring_buy" if card is not None and card.points > 0 else None
    if selected_type == ActionType.RESERVE_VISIBLE:
        card = _card_for_action(state, selected_action)
        if (
            model_action.action_type == ActionType.RESERVE_DECK
            and card is not None
            and card.points >= 2
        ):
            return "replace_blind_reserve_with_visible_scoring_card"
        return None
    if selected_type != ActionType.TAKE_TOKENS:
        return None
    if model_action.action_type in {ActionType.BUY_VISIBLE, ActionType.BUY_RESERVED}:
        return None

    actor = state.current_player
    selected_after = env.step(state, selected_action)
    model_after = env.step(state, model_action)
    selected_deficit = _minimum_visible_scoring_deficit(selected_after, actor)
    model_deficit = _minimum_visible_scoring_deficit(model_after, actor)
    if selected_deficit is not None and (
        model_deficit is None or selected_deficit < model_deficit
    ):
        return "improves_visible_scoring_card_deficit"
    return None


def _minimum_visible_scoring_deficit(state: Any, player_id: int) -> int | None:
    player = state.players[player_id]
    deficits = []
    for cards in state.visible_tier_cards.values():
        for card in cards:
            if card.points <= 0:
                continue
            colored_deficit = sum(
                max(
                    int(card.cost.get(color, 0))
                    - int(player.bonuses.get(color, 0))
                    - int(player.tokens.get(color, 0)),
                    0,
                )
                for color in TOKEN_COLORS
            )
            deficits.append(max(colored_deficit - int(player.tokens.get("gold", 0)), 0))
    return min(deficits) if deficits else None


def _card_for_action(state: Any, action: Action) -> Any | None:
    if action.action_type in {ActionType.RESERVE_VISIBLE, ActionType.BUY_VISIBLE}:
        if action.tier is None or action.market_index is None:
            return None
        cards = state.visible_tier_cards.get(action.tier, ())
        return cards[action.market_index] if action.market_index < len(cards) else None
    if action.action_type == ActionType.BUY_RESERVED:
        if action.reserved_index is None:
            return None
        cards = state.players[state.current_player].reserved_cards
        return cards[action.reserved_index] if action.reserved_index < len(cards) else None
    return None


def _is_champion_turn(payload: dict[str, Any], marker: str) -> bool:
    player_id = int(payload.get("player_id", -1))
    metadata = payload.get("game_bot_metadata")
    if not isinstance(metadata, list) or player_id < 0 or player_id >= len(metadata):
        return False
    seat_metadata = metadata[player_id]
    if not isinstance(seat_metadata, dict):
        return False
    checkpoint_path = str(seat_metadata.get("checkpoint_path", "")).replace("\\", "/")
    return marker.replace("\\", "/").lower() in checkpoint_path.lower()


def _sanitized_anchor(payload: dict[str, Any], training_weight: float) -> dict[str, Any]:
    row = dict(payload)
    for key in (
        "model_top_action_index",
        "model_top_action_payload",
        "model_top_score",
        "model_top_search_value",
    ):
        row.pop(key, None)
    row["training_weight"] = training_weight
    row["target_source"] = "champion_anti_regression_anchor"
    return row


def _iter_rows(paths: tuple[Path, ...]):
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)


if __name__ == "__main__":
    main()
