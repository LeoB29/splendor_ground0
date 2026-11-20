from __future__ import annotations

import argparse
import json

from splendor_ai.alphazero.blind_reserve_refill_audit import preflight, run_audit


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run AlphaZero blind-reserve/refill-coverage audit 136."
    )
    parser.add_argument(
        "--launch",
        action="store_true",
        help="Reconstruct the frozen replay sources and write the audit result.",
    )
    args = parser.parse_args()
    print(json.dumps(run_audit() if args.launch else preflight(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

