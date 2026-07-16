"""Summarize preregistered broad promotion matches for a clean entity candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


OPPONENTS = ("greedy", "shallow", "016", "035")
ABSOLUTE_FLOORS = {"greedy": 0.80, "shallow": 0.30, "016": 0.20, "035": 0.20}
CONTROL_DELTAS = {"greedy": -0.05, "shallow": 0.10, "016": 0.05, "035": 0.05}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--match-dir", required=True)
    parser.add_argument("--output-path", required=True)
    return parser


def checkpoint_score_share(payload: dict[str, Any]) -> tuple[float, int, int]:
    matchup = payload.get("matchup")
    if not isinstance(matchup, dict):
        raise ValueError("Checkpoint match payload is missing matchup.")
    games = int(matchup["games"])
    wins = int(matchup["candidate_wins"])
    draws = int(matchup["draws"])
    failures = int(matchup.get("timed_out_games", 0)) + int(matchup.get("stalled_games", 0))
    return (wins + 0.5 * draws) / games, games, failures


def baseline_score_share(payload: dict[str, Any]) -> tuple[float, int, int]:
    opponents = payload.get("opponents")
    if not isinstance(opponents, list) or len(opponents) != 1 or not isinstance(opponents[0], dict):
        raise ValueError("Baseline match payload must contain exactly one opponent block.")
    block = opponents[0]
    games = int(block["games"])
    wins = int(block["model_wins"])
    draws = int(block["draws"])
    failures = int(block.get("timed_out_games", 0)) + int(block.get("stalled_games", 0))
    return (wins + 0.5 * draws) / games, games, failures


def summarize_broad_gate(match_dir: Path) -> dict[str, Any]:
    results: dict[str, Any] = {}
    gates: dict[str, bool] = {}
    for opponent in OPPONENTS:
        candidate_payload = json.loads(
            (match_dir / f"broad_candidate_{opponent}.json").read_text(encoding="utf-8")
        )
        parent_payload = json.loads(
            (match_dir / f"broad_parent_{opponent}.json").read_text(encoding="utf-8")
        )
        reader = baseline_score_share if opponent in {"greedy", "shallow"} else checkpoint_score_share
        candidate_share, candidate_games, candidate_failures = reader(candidate_payload)
        parent_share, parent_games, parent_failures = reader(parent_payload)
        if candidate_games != 40 or parent_games != 40:
            raise ValueError("Every broad block must contain exactly 40 games.")
        absolute_gate = candidate_share >= ABSOLUTE_FLOORS[opponent]
        control_gate = candidate_share - parent_share >= CONTROL_DELTAS[opponent]
        integrity_gate = candidate_failures == 0 and parent_failures == 0
        gates[f"{opponent}_absolute_floor"] = absolute_gate
        gates[f"{opponent}_matched_control_delta"] = control_gate
        gates[f"{opponent}_integrity"] = integrity_gate
        results[opponent] = {
            "candidate_score_share": candidate_share,
            "parent_score_share": parent_share,
            "candidate_minus_parent": candidate_share - parent_share,
            "absolute_floor": ABSOLUTE_FLOORS[opponent],
            "required_control_delta": CONTROL_DELTAS[opponent],
            "candidate_failures": candidate_failures,
            "parent_failures": parent_failures,
        }
    return {
        "gate": "canonical_entity_broad_promotion_v1",
        "games_per_model_opponent": 40,
        "results": results,
        "gates": gates,
        "all_gates_passed": all(gates.values()),
    }


def main() -> None:
    args = build_parser().parse_args()
    payload = summarize_broad_gate(Path(args.match_dir))
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
