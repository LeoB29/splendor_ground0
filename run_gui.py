"""Workspace-friendly entry point for the Splendor GUI."""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parent
    src_path = repo_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    from splendor_ai.gui import launch_gui

    launch_gui()


if __name__ == "__main__":
    main()
