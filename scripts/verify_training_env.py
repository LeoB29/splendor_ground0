"""Quick environment check for training on the target machine."""

from __future__ import annotations

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify the local training environment.")
    parser.add_argument("--expect-cuda", action="store_true", help="Fail if CUDA is not available.")
    args = parser.parse_args()

    try:
        import torch
    except Exception as exc:  # pragma: no cover - diagnostic script
        print(f"[verify] failed to import torch: {exc}")
        return 1

    print(f"[verify] torch version: {torch.__version__}")
    print(f"[verify] cuda available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"[verify] cuda device count: {torch.cuda.device_count()}")
        print(f"[verify] cuda device[0]: {torch.cuda.get_device_name(0)}")

    try:
        import splendor_ai  # noqa: F401
    except Exception as exc:  # pragma: no cover - diagnostic script
        print(f"[verify] failed to import splendor_ai: {exc}")
        return 1

    print("[verify] splendor_ai import: ok")

    if args.expect_cuda and not torch.cuda.is_available():
        print("[verify] expected CUDA, but torch.cuda.is_available() is False")
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
