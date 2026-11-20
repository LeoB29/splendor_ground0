from __future__ import annotations

import argparse
import json

from splendor_ai.eval.run_expanded_root_resolver_fresh_champion_comparison import (
    preflight,
    run_gate,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run guarded Experiment 134 expanded-resolver comparison."
    )
    parser.add_argument(
        "--launch",
        action="store_true",
        help="Run the expensive deterministic smoke and conditional 400-game block.",
    )
    args = parser.parse_args()
    result = run_gate() if args.launch else preflight()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
