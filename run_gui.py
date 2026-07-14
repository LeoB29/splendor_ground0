"""Workspace-friendly entry point for the Splendor GUI."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parent
    src_path = repo_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    parser = argparse.ArgumentParser(description="Play Splendor against a bot.")
    parser.add_argument(
        "--opponent",
        choices=("greedy", "random", "search", "model"),
        default="greedy",
        help="Opponent to preselect for the first game.",
    )
    parser.add_argument(
        "--model-path",
        default="",
        help="Checkpoint path when --opponent model (kind auto-detected).",
    )
    parser.add_argument("--seed", default="0", help="Initial game seed.")
    parser.add_argument(
        "--seat",
        choices=("0", "1"),
        default="0",
        help="Your seat: 0 moves first, 1 moves second.",
    )
    args = parser.parse_args()

    from splendor_ai.gui import launch_gui

    launch_gui(
        opponent=args.opponent,
        model_path=args.model_path,
        seed=args.seed,
        human_seat=args.seat,
    )


if __name__ == "__main__":
    main()
