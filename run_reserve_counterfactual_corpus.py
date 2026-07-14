"""Workspace entry point for reserve counterfactual corpus compilation."""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    src = Path(__file__).resolve().parent / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from splendor_ai.training.run_reserve_counterfactual_corpus import main as run_main

    run_main()


if __name__ == "__main__":
    main()
