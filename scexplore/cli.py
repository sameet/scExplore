"""CLI entry point for launching scExplore in a browser."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys


def main() -> int:
    app_path = Path(__file__).resolve().parent.parent / "app.py"
    cmd = [sys.executable, "-m", "streamlit", "run", str(app_path), *sys.argv[1:]]
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
