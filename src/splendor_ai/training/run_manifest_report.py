"""CLI entry point for summarizing replay corpus manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Print a compact report for one or more replay corpus manifest files."
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="Corpus directory or manifest.json path. Repeat to compare multiple corpora.",
    )
    parser.add_argument(
        "--no-matchups",
        action="store_true",
        help="Only print one summary row per manifest.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest_paths = [_resolve_manifest_path(path) for path in args.paths]
    manifests = [(path, _load_manifest(path)) for path in manifest_paths]
    print_manifest_report(manifests, include_matchups=not args.no_matchups)


def print_manifest_report(
    manifests: list[tuple[Path, dict[str, Any]]],
    include_matchups: bool = True,
) -> None:
    if not manifests:
        raise ValueError("No manifests to report.")

    rows = [_summary_row(path, manifest) for path, manifest in manifests]
    headers = ("path", "source", "seeds", "games", "steps", "W-L-D", "stall", "timeout", "fallback")
    widths = [
        max(len(str(row[key])) for row in rows + [dict(zip(headers, headers, strict=True))])
        for key in headers
    ]
    print(_format_row(headers, widths))
    print(_format_row(tuple("-" * width for width in widths), widths))
    for (path, manifest), row in zip(manifests, rows, strict=True):
        print(_format_row(tuple(str(row[key]) for key in headers), widths))
        if include_matchups:
            for matchup in manifest.get("matchups", []):
                print(f"  - {_format_matchup(matchup)}")


def _resolve_manifest_path(raw_path: str | Path) -> Path:
    path = Path(raw_path)
    if path.is_dir():
        path = path / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    return path


def _load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Manifest must be a JSON object: {path}")
    return payload


def _summary_row(path: Path, manifest: dict[str, Any]) -> dict[str, str]:
    results = manifest.get("results", {})
    seed_block = manifest.get("seed_block", {})
    seed_start = seed_block.get("seed_start")
    seed_end = seed_block.get("seed_end_inclusive")
    seeds = "n/a" if seed_start is None or seed_end is None else f"{seed_start}-{seed_end}"
    wins = results.get("seat0_wins", 0)
    losses = results.get("seat1_wins", 0)
    draws = results.get("draws", 0)
    return {
        "path": str(path.parent if path.name == "manifest.json" else path),
        "source": str(manifest.get("source_type", "unknown")),
        "seeds": seeds,
        "games": str(results.get("games", seed_block.get("games", 0))),
        "steps": str(results.get("total_steps", 0)),
        "W-L-D": f"{wins}-{losses}-{draws}",
        "stall": str(results.get("stalled_games", 0)),
        "timeout": str(results.get("timed_out_games", 0)),
        "fallback": str(results.get("loop_fallback_triggers", 0)),
    }


def _format_row(values: tuple[str, ...], widths: list[int]) -> str:
    return "  ".join(value.ljust(width) for value, width in zip(values, widths, strict=True))


def _format_matchup(matchup: dict[str, Any]) -> str:
    seat0 = matchup.get("seat0", {})
    seat1 = matchup.get("seat1", {})
    seat0_label = _short_actor_label(str(seat0.get("label", seat0.get("bot_class", "seat0"))))
    seat1_label = _short_actor_label(str(seat1.get("label", seat1.get("bot_class", "seat1"))))
    wins = matchup.get("seat0_wins", 0)
    losses = matchup.get("seat1_wins", 0)
    draws = matchup.get("draws", 0)
    seeds = f"{matchup.get('seed_start')}-{matchup.get('seed_end_inclusive')}"
    return (
        f"{seat0_label} vs {seat1_label}:"
        f" games={matchup.get('games', 0)}"
        f" W-L-D={wins}-{losses}-{draws}"
        f" seeds={seeds}"
        f" stall={matchup.get('stalled_games', 0)}"
        f" timeout={matchup.get('timed_out_games', 0)}"
        f" fallback={matchup.get('loop_fallback_triggers', 0)}"
    )


def _short_actor_label(label: str) -> str:
    parts = label.split(":", maxsplit=2)
    if len(parts) == 3 and parts[2].endswith(".pt"):
        checkpoint_parent = Path(parts[2]).parent.name
        return f"{parts[0]}:{parts[1]}:{checkpoint_parent}"
    return label


if __name__ == "__main__":
    main()
