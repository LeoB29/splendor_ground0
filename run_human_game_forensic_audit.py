"""Workspace entry point for held-out human-game forensic auditing."""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    src_path = Path(__file__).resolve().parent / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    from splendor_ai.training.run_human_game_forensic_audit import main as run_main

    run_main()


if __name__ == "__main__":
    main()
