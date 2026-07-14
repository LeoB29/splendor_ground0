"""Compile matched reserve-return audits into game-disjoint learning corpora."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
import random
from typing import Any

from .run_reserve_generalization_audit import (
    _is_nonhuman_policy_row as _is_nonhuman_policy035_row,
)


@dataclass(frozen=True, slots=True)
class ReserveCounterfactualCorpusConfig:
    source_path: Path
    external_source_path: Path
    anchor_replay_path: Path
    output_dir: Path
    heldout_games_per_band: int = 25
    split_seed: int = 142
    anchor_rows: int = 5_000
    anchor_seed: int = 143
    policy_marker: str = "legal_action_scorer_035_legal_observation"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compile reserve counterfactual outcomes into train, held-out, and anchor streams."
    )
    parser.add_argument("--source-path", required=True)
    parser.add_argument("--external-source-path", required=True)
    parser.add_argument("--anchor-replay-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--heldout-games-per-band", type=int, default=25)
    parser.add_argument("--split-seed", type=int, default=142)
    parser.add_argument("--anchor-rows", type=int, default=5_000)
    parser.add_argument("--anchor-seed", type=int, default=143)
    parser.add_argument("--policy-marker", default="legal_action_scorer_035_legal_observation")
    return parser


def build_reserve_counterfactual_corpus(
    config: ReserveCounterfactualCorpusConfig,
) -> dict[str, Any]:
    _validate(config)
    source_rows = _counterfactual_rows(config.source_path)
    external_rows = _counterfactual_rows(config.external_source_path)
    train_rows, heldout_rows, split = _split_rows(
        source_rows,
        heldout_games_per_band=config.heldout_games_per_band,
        seed=config.split_seed,
    )
    anchor_rows = _sample_anchor_rows(
        config.anchor_replay_path,
        count=config.anchor_rows,
        seed=config.anchor_seed,
        policy_marker=config.policy_marker,
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "train": config.output_dir / "train_counterfactual.jsonl",
        "heldout": config.output_dir / "heldout_counterfactual.jsonl",
        "external": config.output_dir / "external_counterfactual.jsonl",
        "anchor": config.output_dir / "anchor_035_legal.jsonl",
    }
    for name, rows in (
        ("train", train_rows),
        ("heldout", heldout_rows),
        ("external", external_rows),
        ("anchor", anchor_rows),
    ):
        _write_jsonl(paths[name], rows)
    manifest = {
        "format": "reserve_counterfactual_corpus_v1",
        "config": {
            **asdict(config),
            "source_path": str(config.source_path),
            "external_source_path": str(config.external_source_path),
            "anchor_replay_path": str(config.anchor_replay_path),
            "output_dir": str(config.output_dir),
        },
        "paths": {name: str(path) for name, path in paths.items()},
        "split": split,
        "rows": {
            "train": len(train_rows),
            "heldout": len(heldout_rows),
            "external": len(external_rows),
            "anchor": len(anchor_rows),
        },
        "signal": {
            "train_resolved": sum(bool(row["resolved_mistake"]) for row in train_rows),
            "heldout_resolved": sum(bool(row["resolved_mistake"]) for row in heldout_rows),
            "external_resolved": sum(bool(row["resolved_mistake"]) for row in external_rows),
        },
    }
    manifest_path = config.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _counterfactual_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    selected = payload.get("selected_payloads")
    if not isinstance(selected, list):
        selected = payload.get("selected_payloads", [])
    if not selected:
        raise ValueError(
            f"{path} has no selected payloads. Use the resumable progress artifact when necessary."
        )
    results = payload.get("rollout_results", [])
    result_by_key = {
        _row_key(row): row
        for row in results
        if isinstance(row, dict)
    }
    rows = []
    for source in selected:
        if not isinstance(source, dict):
            continue
        result = result_by_key.get(_row_key(source))
        if result is None:
            raise ValueError(f"Missing rollout result for selected state {_row_key(source)}.")
        snapshot = source.get("state_snapshot")
        actions = result.get("actions")
        if not isinstance(snapshot, dict) or not isinstance(actions, list) or len(actions) < 2:
            raise ValueError(f"Counterfactual state {_row_key(source)} is incomplete.")
        rows.append(
            {
                "format": "reserve_counterfactual_row_v1",
                "game_seed": int(source["game_seed"]),
                "turn_index": int(source["turn_index"]),
                "player_id": int(source["player_id"]),
                "state_snapshot": snapshot,
                "turn_band": str(result["turn_band"]),
                "chosen_action_index": int(result["chosen_action_index"]),
                "best_alternative_action_index": int(result["best_alternative_action_index"]),
                "resolved_mistake": bool(result["resolved_mistake"]),
                "paired_return_delta": float(result["paired_return_delta"]),
                "paired_return_delta_standard_error": float(
                    result["paired_return_delta_standard_error"]
                ),
                "counterfactual_actions": [
                    {
                        "action_index": int(action["action_index"]),
                        "rollout_return": float(action["rollout_return"]),
                        "rollout_standard_error": float(action["rollout_standard_error"]),
                        "rollout_count": int(action["rollout_count"]),
                    }
                    for action in actions
                ],
            }
        )
    rows.sort(key=lambda row: (int(row["game_seed"]), int(row["turn_index"])))
    return rows


def _split_rows(
    rows: list[dict[str, Any]],
    *,
    heldout_games_per_band: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    by_band: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_band.setdefault(str(row["turn_band"]), []).append(row)
    heldout_games: set[int] = set()
    selected_by_band: dict[str, list[int]] = {}
    for band, band_rows in sorted(by_band.items()):
        seeds = sorted({int(row["game_seed"]) for row in band_rows})
        if len(seeds) < heldout_games_per_band:
            raise ValueError(f"Turn band {band} has too few games for held-out split.")
        rng = random.Random(seed + sum(ord(char) for char in band))
        rng.shuffle(seeds)
        selected = sorted(seeds[:heldout_games_per_band])
        heldout_games.update(selected)
        selected_by_band[band] = selected
    heldout = [row for row in rows if int(row["game_seed"]) in heldout_games]
    train = [row for row in rows if int(row["game_seed"]) not in heldout_games]
    if len(heldout) != heldout_games_per_band * len(by_band):
        raise ValueError("Held-out split leaked multiple rows from a game or lacks band balance.")
    if len({int(row["game_seed"]) for row in train}) != len(train):
        raise ValueError("Training states must remain one per game.")
    return train, heldout, {
        "split_unit": "game",
        "split_seed": seed,
        "heldout_games_per_band": heldout_games_per_band,
        "heldout_games_by_band": selected_by_band,
        "train_games": len(train),
        "heldout_games": len(heldout),
        "train_by_band": dict(Counter(str(row["turn_band"]) for row in train)),
        "heldout_by_band": dict(Counter(str(row["turn_band"]) for row in heldout)),
    }


def _sample_anchor_rows(
    path: Path,
    *,
    count: int,
    seed: int,
    policy_marker: str,
) -> list[dict[str, Any]]:
    pool: list[dict[str, Any]] = []
    seen = 0
    rng = random.Random(seed)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            if not _is_nonhuman_policy_row(payload, policy_marker):
                continue
            seen += 1
            if len(pool) < count:
                pool.append(payload)
            else:
                replacement = rng.randrange(seen)
                if replacement < count:
                    pool[replacement] = payload
    if len(pool) != count:
        raise ValueError(f"Anchor source has only {len(pool)} eligible policy035 rows; need {count}.")
    return pool


def _is_nonhuman_policy_row(payload: dict[str, Any], policy_marker: str) -> bool:
    return _is_nonhuman_policy035_row(payload, policy_marker)


def _row_key(row: dict[str, Any]) -> tuple[int, int, int]:
    return int(row["game_seed"]), int(row["turn_index"]), int(row["player_id"])


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row))
            handle.write("\n")


def _validate(config: ReserveCounterfactualCorpusConfig) -> None:
    if config.heldout_games_per_band <= 0 or config.anchor_rows <= 0:
        raise ValueError("Held-out and anchor row counts must be positive.")
    for path in (config.source_path, config.external_source_path, config.anchor_replay_path):
        if not path.exists():
            raise FileNotFoundError(path)


def main() -> None:
    args = build_parser().parse_args()
    manifest = build_reserve_counterfactual_corpus(
        ReserveCounterfactualCorpusConfig(
            source_path=Path(args.source_path),
            external_source_path=Path(args.external_source_path),
            anchor_replay_path=Path(args.anchor_replay_path),
            output_dir=Path(args.output_dir),
            heldout_games_per_band=args.heldout_games_per_band,
            split_seed=args.split_seed,
            anchor_rows=args.anchor_rows,
            anchor_seed=args.anchor_seed,
            policy_marker=args.policy_marker,
        )
    )
    print(
        "[reserve-counterfactual-corpus]"
        f" train={manifest['rows']['train']}"
        f" heldout={manifest['rows']['heldout']}"
        f" external={manifest['rows']['external']}"
        f" anchor={manifest['rows']['anchor']}"
    )


if __name__ == "__main__":
    main()
