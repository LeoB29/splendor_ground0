from __future__ import annotations

import argparse
import json

from splendor_ai.alphazero.action_fragmentation_audit import preflight, run_audit


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the no-training AlphaZero action-fragmentation audit 135."
    )
    parser.add_argument(
        "--launch",
        action="store_true",
        help="Stream the frozen experiment-114 replay and write the result.",
    )
    args = parser.parse_args()
    result = run_audit() if args.launch else preflight()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

