"""Measure top-action drift between two checkpoints on replay states."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

from splendor_ai.engine.env import SplendorEnv
from splendor_ai.training.replay import deserialize_state_snapshot
from splendor_ai.training.run_policy_search_disagreement import _load_policy_scorer


@dataclass(frozen=True, slots=True)
class PolicyDriftConfig:
    before_checkpoint: Path
    after_checkpoint: Path
    replay_paths: tuple[Path, ...]
    output_path: Path
    before_model_kind: str = "legal_action_attention_scorer"
    after_model_kind: str = "legal_action_attention_scorer"
    device: str = "cpu"
    player_id_filter: tuple[int, ...] | None = None
    game_seed_filter: tuple[int, ...] | None = None
    max_rows: int | None = None
    log_every: int = 500


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Measure checkpoint top-action drift.")
    parser.add_argument("--before-checkpoint", required=True)
    parser.add_argument("--after-checkpoint", required=True)
    parser.add_argument("--replay-path", action="append", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--before-model-kind", default="legal_action_attention_scorer")
    parser.add_argument("--after-model-kind", default="legal_action_attention_scorer")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--player-id-filter", nargs="*", type=int, default=None)
    parser.add_argument("--game-seed-filter", nargs="*", type=int, default=None)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=500)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = measure_policy_drift(
        PolicyDriftConfig(
            before_checkpoint=Path(args.before_checkpoint),
            after_checkpoint=Path(args.after_checkpoint),
            replay_paths=tuple(Path(path) for path in args.replay_path),
            output_path=Path(args.output_path),
            before_model_kind=args.before_model_kind,
            after_model_kind=args.after_model_kind,
            device=args.device,
            player_id_filter=tuple(args.player_id_filter) if args.player_id_filter else None,
            game_seed_filter=tuple(args.game_seed_filter) if args.game_seed_filter else None,
            max_rows=args.max_rows,
            log_every=args.log_every,
        )
    )
    print(
        "Policy drift:"
        f" rows={summary['overall']['rows_compared']}"
        f" agreement={summary['overall']['top_action_agreement_rate']:.4f}"
    )


def measure_policy_drift(config: PolicyDriftConfig) -> dict[str, Any]:
    before = _load_policy_scorer(
        _ScorerConfig(config.before_checkpoint, config.before_model_kind, config.device)
    )
    after = _load_policy_scorer(
        _ScorerConfig(config.after_checkpoint, config.after_model_kind, config.device)
    )
    env = SplendorEnv()
    player_filter = set(config.player_id_filter or ())
    seed_filter = set(config.game_seed_filter or ())
    counts: Counter[str] = Counter()
    phases: dict[str, Counter[str]] = {}
    before_types: dict[str, Counter[str]] = {}
    transitions: Counter[str] = Counter()

    stop = False
    for path in config.replay_paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                counts["rows_seen"] += 1
                payload = json.loads(line)
                if player_filter and int(payload.get("player_id", -1)) not in player_filter:
                    counts["rows_filtered"] += 1
                    continue
                seed = int(payload.get("game_seed", payload.get("seed", -1)))
                if seed_filter and seed not in seed_filter:
                    counts["rows_filtered"] += 1
                    continue
                snapshot = payload.get("state_snapshot")
                if not isinstance(snapshot, dict):
                    counts["rows_missing_snapshot"] += 1
                    continue
                state = deserialize_state_snapshot(snapshot)
                legal_actions = env.legal_actions(state)
                before_scores = before.score_actions(state, legal_actions)
                after_scores = after.score_actions(state, legal_actions)
                if not before_scores or not after_scores:
                    counts["rows_without_scores"] += 1
                    continue
                before_top = before_scores[0]
                after_top = after_scores[0]
                same = before_top.action_index == after_top.action_index
                counts["rows_compared"] += 1
                counts["agreements" if same else "changes"] += 1
                phase = _phase(int(payload.get("turn_index", 0)))
                _record(phases.setdefault(phase, Counter()), same)
                before_type = before_top.action.action_type.name
                after_type = after_top.action.action_type.name
                _record(before_types.setdefault(before_type, Counter()), same)
                transitions[f"{before_type} -> {after_type}"] += 1
                if (
                    config.log_every > 0
                    and counts["rows_compared"] % config.log_every == 0
                ):
                    print(
                        f"[policy-drift] compared={counts['rows_compared']}"
                        f" agreement={_rate(counts['agreements'], counts['rows_compared']):.4f}"
                    )
                if config.max_rows is not None and counts["rows_compared"] >= config.max_rows:
                    stop = True
                    break
        if stop:
            break

    summary = {
        "config": {
            **asdict(config),
            "before_checkpoint": str(config.before_checkpoint),
            "after_checkpoint": str(config.after_checkpoint),
            "replay_paths": [str(path) for path in config.replay_paths],
            "output_path": str(config.output_path),
        },
        "overall": _finalize(counts),
        "by_phase": {key: _finalize(value) for key, value in phases.items()},
        "by_before_action_type": {
            key: _finalize(value) for key, value in before_types.items()
        },
        "transitions": dict(transitions),
    }
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    config.output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


@dataclass(frozen=True, slots=True)
class _ScorerConfig:
    checkpoint: Path
    model_kind: str
    device: str


def _record(counter: Counter[str], same: bool) -> None:
    counter["rows_compared"] += 1
    counter["agreements" if same else "changes"] += 1


def _finalize(counter: Counter[str]) -> dict[str, Any]:
    rows = counter["rows_compared"]
    return {
        **dict(counter),
        "top_action_agreement_rate": _rate(counter["agreements"], rows),
        "top_action_change_rate": _rate(counter["changes"], rows),
    }


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _phase(turn_index: int) -> str:
    if turn_index < 20:
        return "early"
    if turn_index < 50:
        return "mid"
    return "late"


if __name__ == "__main__":
    main()
