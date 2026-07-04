"""Build targeted search-correction replay rows from policy/search disagreements."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from splendor_ai.bots import ShallowSearchBot
from splendor_ai.encoding import ActionCodec, encode_public_observation_tensor
from splendor_ai.engine.actions import ActionType
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.training.replay import deserialize_state_snapshot, serialize_action
from splendor_ai.training.run_policy_search_disagreement import _load_policy_scorer


_DEFAULT_TARGET_ACTION_TYPES = (
    "TAKE_TOKENS",
    "RESERVE_VISIBLE",
    "RESERVE_DECK",
    "BUY_VISIBLE",
    "BUY_RESERVED",
)


@dataclass(frozen=True, slots=True)
class TargetedSearchCorpusConfig:
    checkpoint: Path
    model_kind: str
    replay_paths: tuple[Path, ...]
    output_path: Path
    summary_path: Path
    device: str = "cpu"
    max_rows: int | None = None
    log_every: int = 500
    search_depth: int = 2
    search_max_branching: int = 10
    search_buy_branching: int = 6
    search_reserve_branching: int = 3
    search_take_branching: int = 3
    seed: int = 0
    min_search_margin: float = 100.0
    include_missing_model_action: bool = True
    target_action_types: tuple[str, ...] = _DEFAULT_TARGET_ACTION_TYPES
    late_buy_turn_min: int = 50
    max_examples: int = 20


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Write targeted replay rows where shallow search disagrees with a model."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--model-kind",
        choices=("flat_policy", "legal_action_scorer"),
        default="legal_action_scorer",
    )
    parser.add_argument("--replay-path", action="append", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--summary-path", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--search-depth", type=int, default=2)
    parser.add_argument("--search-max-branching", type=int, default=10)
    parser.add_argument("--search-buy-branching", type=int, default=6)
    parser.add_argument("--search-reserve-branching", type=int, default=3)
    parser.add_argument("--search-take-branching", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-search-margin", type=float, default=100.0)
    parser.add_argument("--include-missing-model-action", action="store_true", default=False)
    parser.add_argument(
        "--target-action-types",
        nargs="+",
        default=list(_DEFAULT_TARGET_ACTION_TYPES),
        choices=tuple(action_type.name for action_type in ActionType),
    )
    parser.add_argument(
        "--late-buy-turn-min",
        type=int,
        default=50,
        help="Only keep BUY_* search corrections from this turn onward.",
    )
    parser.add_argument("--max-examples", type=int, default=20)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = build_targeted_search_corpus(
        TargetedSearchCorpusConfig(
            checkpoint=Path(args.checkpoint),
            model_kind=args.model_kind,
            replay_paths=tuple(Path(path) for path in args.replay_path),
            output_path=Path(args.output_path),
            summary_path=Path(args.summary_path),
            device=args.device,
            max_rows=args.max_rows,
            log_every=args.log_every,
            search_depth=args.search_depth,
            search_max_branching=args.search_max_branching,
            search_buy_branching=args.search_buy_branching,
            search_reserve_branching=args.search_reserve_branching,
            search_take_branching=args.search_take_branching,
            seed=args.seed,
            min_search_margin=args.min_search_margin,
            include_missing_model_action=args.include_missing_model_action,
            target_action_types=tuple(args.target_action_types),
            late_buy_turn_min=args.late_buy_turn_min,
            max_examples=args.max_examples,
        )
    )
    print(
        "Targeted search corpus:"
        f" analyzed={summary['rows_analyzed']}"
        f" written={summary['rows_written']}"
        f" output={args.output_path}"
    )


def build_targeted_search_corpus(config: TargetedSearchCorpusConfig) -> dict[str, Any]:
    if not config.replay_paths:
        raise ValueError("At least one replay path is required.")
    if config.min_search_margin < 0.0:
        raise ValueError("min_search_margin must be non-negative.")
    unknown_action_types = set(config.target_action_types) - {action_type.name for action_type in ActionType}
    if unknown_action_types:
        raise ValueError(f"Unknown target action types: {sorted(unknown_action_types)}")

    env = SplendorEnv()
    codec = ActionCodec()
    model_scorer = _load_policy_scorer(config)
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    config.summary_path.parent.mkdir(parents=True, exist_ok=True)

    counters = {
        "rows_seen": 0,
        "rows_analyzed": 0,
        "rows_written": 0,
        "rows_missing_snapshot": 0,
        "rows_without_legal_actions": 0,
        "rows_without_search_candidate": 0,
        "rows_disagreed": 0,
        "rows_skipped_by_target_filter": 0,
        "rows_skipped_by_margin": 0,
        "rows_missing_model_search_value": 0,
        "written_by_search_action_type": {},
        "written_by_phase": {},
    }
    examples: list[dict[str, Any]] = []

    with config.output_path.open("w", encoding="utf-8") as output:
        for replay_path in config.replay_paths:
            with replay_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if config.max_rows is not None and counters["rows_seen"] >= config.max_rows:
                        break
                    stripped = line.strip()
                    if not stripped:
                        continue
                    counters["rows_seen"] += 1
                    payload = json.loads(stripped)
                    snapshot = payload.get("state_snapshot")
                    if not isinstance(snapshot, dict):
                        counters["rows_missing_snapshot"] += 1
                        continue
                    state = deserialize_state_snapshot(snapshot)
                    legal_actions = env.legal_actions(state)
                    if not legal_actions:
                        counters["rows_without_legal_actions"] += 1
                        continue
                    search_bot = _build_search_bot(config, counters["rows_seen"])
                    search_scores = search_bot.rank_actions(env, state, legal_actions)
                    if not search_scores:
                        counters["rows_without_search_candidate"] += 1
                        continue
                    counters["rows_analyzed"] += 1

                    model_scores = model_scorer.score_actions(state, legal_actions)
                    target_payload = _targeted_payload_for_row(
                        config=config,
                        codec=codec,
                        state=state,
                        payload=payload,
                        replay_path=replay_path,
                        legal_actions=legal_actions,
                        model_scores=model_scores,
                        search_scores=search_scores,
                        counters=counters,
                    )
                    if target_payload is not None:
                        output.write(json.dumps(target_payload))
                        output.write("\n")
                        counters["rows_written"] += 1
                        search_type = target_payload["action_payload"]["action_type"]
                        phase = _phase_for_turn(int(target_payload.get("turn_index", 0)))
                        _increment(counters["written_by_search_action_type"], search_type)
                        _increment(counters["written_by_phase"], phase)
                        _store_example(examples, target_payload, config.max_examples)

                    if config.log_every > 0 and counters["rows_seen"] % config.log_every == 0:
                        print(
                            f"[targeted-corpus] rows_seen={counters['rows_seen']}"
                            f" analyzed={counters['rows_analyzed']}"
                            f" written={counters['rows_written']}"
                        )
            if config.max_rows is not None and counters["rows_seen"] >= config.max_rows:
                break

    summary = {
        **counters,
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


def _targeted_payload_for_row(
    config: TargetedSearchCorpusConfig,
    codec: ActionCodec,
    state: Any,
    payload: dict[str, Any],
    replay_path: Path,
    legal_actions: list[Any],
    model_scores: tuple[Any, ...],
    search_scores: tuple[Any, ...],
    counters: dict[str, Any],
) -> dict[str, Any] | None:
    model_top = model_scores[0]
    search_top = search_scores[0]
    search_top_index = codec.encode(state, search_top.action)
    if model_top.action_index == search_top_index:
        return None

    counters["rows_disagreed"] += 1
    search_action_type = search_top.action.action_type.name
    turn_index = int(payload.get("turn_index", 0))
    if search_action_type not in config.target_action_types or (
        search_action_type in {"BUY_VISIBLE", "BUY_RESERVED"}
        and turn_index < config.late_buy_turn_min
    ):
        counters["rows_skipped_by_target_filter"] += 1
        return None

    search_value_by_index = {
        codec.encode(state, score.action): float(score.value)
        for score in search_scores
    }
    model_search_value = search_value_by_index.get(model_top.action_index)
    search_margin = None if model_search_value is None else float(search_top.value) - model_search_value
    if model_search_value is None:
        counters["rows_missing_model_search_value"] += 1
        if not config.include_missing_model_action:
            counters["rows_skipped_by_margin"] += 1
            return None
    elif search_margin is None or search_margin < config.min_search_margin:
        counters["rows_skipped_by_margin"] += 1
        return None

    player_id = int(payload.get("player_id", state.current_player))
    observation = encode_public_observation_tensor(state, player_id)
    legal_action_indices = list(codec.legal_action_indices(state, legal_actions))
    target_payload = dict(payload)
    target_payload.update(
        {
            "action_index": search_top_index,
            "action_payload": serialize_action(search_top.action),
            "legal_action_indices": legal_action_indices,
            "observation_vector": list(observation.vector),
            "target_source": "targeted_search_disagreement",
            "target_source_replay_path": str(replay_path),
            "original_action_index": payload.get("action_index"),
            "original_action_payload": payload.get("action_payload"),
            "model_top_action_index": model_top.action_index,
            "model_top_action_payload": serialize_action(model_top.action),
            "model_top_score": model_top.score,
            "model_top_search_value": model_search_value,
            "search_margin": search_margin,
            "search_action_scores": [
                {
                    "action_index": codec.encode(state, score.action),
                    "action_payload": serialize_action(score.action),
                    "value": float(score.value),
                    "heuristic_score": float(score.heuristic_score),
                    "loop_penalty": float(score.loop_penalty),
                }
                for score in search_scores
            ],
        }
    )
    return target_payload


def _build_search_bot(config: TargetedSearchCorpusConfig, row_offset: int) -> ShallowSearchBot:
    return ShallowSearchBot(
        depth=config.search_depth,
        max_branching=config.search_max_branching,
        max_buy_actions=config.search_buy_branching,
        max_reserve_actions=config.search_reserve_branching,
        max_take_actions=config.search_take_branching,
        seed=config.seed + row_offset,
    )


def _phase_for_turn(turn_index: int) -> str:
    if turn_index < 20:
        return "early"
    if turn_index < 50:
        return "mid"
    return "late"


def _increment(bucket: dict[str, int], key: str) -> None:
    bucket[key] = bucket.get(key, 0) + 1


def _store_example(examples: list[dict[str, Any]], payload: dict[str, Any], max_examples: int) -> None:
    if max_examples <= 0:
        return
    examples.append(
        {
            "game_seed": payload.get("game_seed"),
            "turn_index": payload.get("turn_index"),
            "player_id": payload.get("player_id"),
            "search_margin": payload.get("search_margin"),
            "search_action_type": payload["action_payload"]["action_type"],
            "model_action_type": payload["model_top_action_payload"]["action_type"],
            "target_action_index": payload.get("action_index"),
            "original_action_index": payload.get("original_action_index"),
        }
    )
    examples.sort(
        key=lambda item: -math.inf if item["search_margin"] is None else float(item["search_margin"]),
        reverse=True,
    )
    del examples[max_examples:]


if __name__ == "__main__":
    main()
