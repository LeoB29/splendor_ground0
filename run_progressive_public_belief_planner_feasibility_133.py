"""CLI for experiment 133."""

from __future__ import annotations

import argparse
import json

from splendor_ai.eval.run_progressive_public_belief_planner_feasibility import (
    preflight,
    run_experiment,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--launch",
        action="store_true",
        help="Run the frozen 36-state no-training evidence panel.",
    )
    args = parser.parse_args()
    result = run_experiment() if args.launch else preflight()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

