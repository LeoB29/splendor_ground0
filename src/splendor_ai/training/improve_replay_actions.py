"""Relabel snapshot-enabled replay rows with shallow-search policy targets."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from splendor_ai.bots import ShallowSearchBot
from splendor_ai.encoding import ActionCodec
from splendor_ai.engine.actions import ActionType
from splendor_ai.engine.env import SplendorEnv

from .replay import deserialize_state_snapshot, serialize_action


@dataclass(frozen=True, slots=True)
class ImproveReplayConfig:
    input_paths: tuple[Path, ...]
    output_path: Path
    summary_path: Path
    search_depth: int = 2
    search_max_branching: int = 10
    search_buy_branching: int = 6
    search_reserve_branching: int = 3
    search_take_branching: int = 3
    seed: int = 0
    write_unchanged: bool = False
    min_search_margin: float | None = None
    exclude_changed_action_types: tuple[str, ...] = ()
    stateful_search_history: bool = False
    max_rows: int | None = None
    log_every: int = 1_000


@dataclass(frozen=True, slots=True)
class ImproveReplaySummary:
    rows_read: int
    rows_written: int
    rows_changed: int
    rows_unchanged: int
    rows_changed_written: int
    rows_unchanged_written: int
    rows_filtered_by_margin: int
    rows_filtered_by_missing_original_score: int
    rows_filtered_by_action_type: int
    rows_missing_snapshot: int
    rows_without_search_candidate: int
    input_paths: tuple[str, ...]
    output_path: str
    summary_path: str
    config: dict[str, Any]


def improve_replay_actions(config: ImproveReplayConfig) -> ImproveReplaySummary:
    if not config.input_paths:
        raise ValueError("At least one input path is required.")
    output_resolved = config.output_path.resolve()
    for input_path in config.input_paths:
        if input_path.resolve() == output_resolved:
            raise ValueError("Output path must be different from every input path.")

    env = SplendorEnv()
    codec = ActionCodec()
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    config.summary_path.parent.mkdir(parents=True, exist_ok=True)

    rows_read = 0
    rows_written = 0
    rows_changed = 0
    rows_unchanged = 0
    rows_changed_written = 0
    rows_unchanged_written = 0
    rows_filtered_by_margin = 0
    rows_filtered_by_missing_original_score = 0
    rows_filtered_by_action_type = 0
    rows_missing_snapshot = 0
    rows_without_search_candidate = 0
    stateful_bot = _build_search_bot(config, row_offset=0) if config.stateful_search_history else None

    with config.output_path.open("w", encoding="utf-8") as output_handle:
        for input_path in config.input_paths:
            with input_path.open("r", encoding="utf-8") as input_handle:
                for line in input_handle:
                    if config.max_rows is not None and rows_read >= config.max_rows:
                        break
                    rows_read += 1
                    payload = json.loads(line)
                    snapshot = payload.get("state_snapshot")
                    if snapshot is None:
                        rows_missing_snapshot += 1
                        continue

                    state = deserialize_state_snapshot(snapshot)
                    legal_actions = env.legal_actions(state)
                    search_bot = stateful_bot or _build_search_bot(config, row_offset=rows_read)
                    ranked_actions = search_bot.rank_actions(env, state, legal_actions)
                    if not ranked_actions:
                        rows_without_search_candidate += 1
                        continue

                    selected = ranked_actions[0]
                    selected_index = codec.encode(state, selected.action)
                    original_index = int(payload["action_index"])
                    original_value = _value_for_action_index(
                        ranked_actions=ranked_actions,
                        codec=codec,
                        state=state,
                        action_index=original_index,
                    )
                    search_margin = (
                        None
                        if original_value is None
                        else selected.value - original_value
                    )
                    changed = selected_index != original_index
                    if changed:
                        rows_changed += 1
                    else:
                        rows_unchanged += 1
                    if changed and selected.action.action_type.name in config.exclude_changed_action_types:
                        rows_filtered_by_action_type += 1
                        continue
                    if changed and config.min_search_margin is not None:
                        if search_margin is None:
                            rows_filtered_by_missing_original_score += 1
                            continue
                        if search_margin < config.min_search_margin:
                            rows_filtered_by_margin += 1
                            continue
                    if not changed and not config.write_unchanged:
                        continue

                    legal_action_indices = codec.legal_action_indices(state, legal_actions)
                    improved_payload = dict(payload)
                    improved_payload["original_action_index"] = original_index
                    improved_payload["original_action_payload"] = payload.get("action_payload")
                    improved_payload["action_index"] = selected_index
                    improved_payload["action_payload"] = serialize_action(selected.action)
                    improved_payload["legal_action_indices"] = legal_action_indices
                    improved_payload["improvement_metadata"] = {
                        "source": "shallow_search",
                        "target_semantics": "policy_relabel_only_original_final_value_retained",
                        "changed_action": changed,
                        "search_depth": config.search_depth,
                        "search_max_branching": config.search_max_branching,
                        "search_buy_branching": config.search_buy_branching,
                        "search_reserve_branching": config.search_reserve_branching,
                        "search_take_branching": config.search_take_branching,
                        "searched_candidate_count": len(ranked_actions),
                        "selected_search_value": selected.value,
                        "selected_heuristic_score": selected.heuristic_score,
                        "selected_loop_penalty": selected.loop_penalty,
                        "original_search_value": original_value,
                        "search_margin": search_margin,
                        "min_search_margin": config.min_search_margin,
                        "exclude_changed_action_types": list(config.exclude_changed_action_types),
                    }
                    output_handle.write(json.dumps(improved_payload))
                    output_handle.write("\n")
                    rows_written += 1
                    if changed:
                        rows_changed_written += 1
                    else:
                        rows_unchanged_written += 1

                    if config.log_every > 0 and rows_read % config.log_every == 0:
                        print(
                            f"[improve] rows_read={rows_read} rows_written={rows_written} "
                            f"changed={rows_changed} changed_written={rows_changed_written} "
                            f"filtered_margin={rows_filtered_by_margin} "
                            f"filtered_missing_original={rows_filtered_by_missing_original_score} "
                            f"filtered_type={rows_filtered_by_action_type} "
                            f"missing_snapshot={rows_missing_snapshot}"
                        )
                if config.max_rows is not None and rows_read >= config.max_rows:
                    break

    summary = ImproveReplaySummary(
        rows_read=rows_read,
        rows_written=rows_written,
        rows_changed=rows_changed,
        rows_unchanged=rows_unchanged,
        rows_changed_written=rows_changed_written,
        rows_unchanged_written=rows_unchanged_written,
        rows_filtered_by_margin=rows_filtered_by_margin,
        rows_filtered_by_missing_original_score=rows_filtered_by_missing_original_score,
        rows_filtered_by_action_type=rows_filtered_by_action_type,
        rows_missing_snapshot=rows_missing_snapshot,
        rows_without_search_candidate=rows_without_search_candidate,
        input_paths=tuple(str(path) for path in config.input_paths),
        output_path=str(config.output_path),
        summary_path=str(config.summary_path),
        config={
            "search_depth": config.search_depth,
            "search_max_branching": config.search_max_branching,
            "search_buy_branching": config.search_buy_branching,
            "search_reserve_branching": config.search_reserve_branching,
            "search_take_branching": config.search_take_branching,
            "seed": config.seed,
            "write_unchanged": config.write_unchanged,
            "min_search_margin": config.min_search_margin,
            "exclude_changed_action_types": config.exclude_changed_action_types,
            "stateful_search_history": config.stateful_search_history,
            "max_rows": config.max_rows,
        },
    )
    config.summary_path.write_text(json.dumps(asdict(summary), indent=2), encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Relabel snapshot-enabled replay JSONL rows with shallow-search actions."
    )
    parser.add_argument(
        "--input-path",
        action="append",
        required=True,
        help="Input replay JSONL file. Repeat to concatenate multiple files.",
    )
    parser.add_argument("--output-path", required=True, help="Output JSONL path for relabeled rows.")
    parser.add_argument(
        "--summary-path",
        default=None,
        help="Output summary JSON path. Defaults to OUTPUT_PATH with .summary.json suffix.",
    )
    parser.add_argument("--search-depth", type=int, default=2)
    parser.add_argument("--search-max-branching", type=int, default=10)
    parser.add_argument("--search-buy-branching", type=int, default=6)
    parser.add_argument("--search-reserve-branching", type=int, default=3)
    parser.add_argument("--search-take-branching", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--write-unchanged",
        action="store_true",
        help="Write rows even when shallow search keeps the original action.",
    )
    parser.add_argument(
        "--min-search-margin",
        type=float,
        default=None,
        help=(
            "Only write changed rows when the selected search value exceeds the "
            "original action's searched value by at least this margin. Changed "
            "rows whose original action was pruned from the search candidates are skipped."
        ),
    )
    parser.add_argument(
        "--exclude-changed-action-types",
        nargs="*",
        choices=tuple(action_type.name for action_type in ActionType),
        default=[],
        help="Skip changed relabel rows whose selected action type is in this list.",
    )
    parser.add_argument(
        "--stateful-search-history",
        action="store_true",
        help="Keep shallow-search loop history across sequential rows.",
    )
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=1_000)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_path = Path(args.output_path)
    summary_path = (
        Path(args.summary_path)
        if args.summary_path is not None
        else output_path.with_suffix(".summary.json")
    )
    summary = improve_replay_actions(
        ImproveReplayConfig(
            input_paths=tuple(Path(path) for path in args.input_path),
            output_path=output_path,
            summary_path=summary_path,
            search_depth=args.search_depth,
            search_max_branching=args.search_max_branching,
            search_buy_branching=args.search_buy_branching,
            search_reserve_branching=args.search_reserve_branching,
            search_take_branching=args.search_take_branching,
            seed=args.seed,
            write_unchanged=args.write_unchanged,
            min_search_margin=args.min_search_margin,
            exclude_changed_action_types=tuple(args.exclude_changed_action_types),
            stateful_search_history=args.stateful_search_history,
            max_rows=args.max_rows,
            log_every=args.log_every,
        )
    )
    print(
        f"[improve] complete rows_read={summary.rows_read} "
        f"rows_written={summary.rows_written} rows_changed={summary.rows_changed} "
        f"rows_changed_written={summary.rows_changed_written} "
        f"missing_snapshot={summary.rows_missing_snapshot}"
    )
    print(f"wrote improved replay rows: {summary.output_path}")
    print(f"wrote summary: {summary.summary_path}")


def _build_search_bot(config: ImproveReplayConfig, row_offset: int) -> ShallowSearchBot:
    return ShallowSearchBot(
        depth=config.search_depth,
        max_branching=config.search_max_branching,
        max_buy_actions=config.search_buy_branching,
        max_reserve_actions=config.search_reserve_branching,
        max_take_actions=config.search_take_branching,
        seed=config.seed + row_offset,
    )


def _value_for_action_index(
    ranked_actions: tuple[Any, ...],
    codec: ActionCodec,
    state: Any,
    action_index: int,
) -> float | None:
    for score in ranked_actions:
        if codec.encode(state, score.action) == action_index:
            return score.value
    return None


if __name__ == "__main__":
    main()
