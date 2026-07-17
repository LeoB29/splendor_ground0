"""Workspace entry point for the one-shot matched-rollout gate."""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from splendor_ai.eval.run_one_shot_rollout_gate import main as run_main

    run_main()


if __name__ == "__main__":
    main()
