from __future__ import annotations

"""Convenience launcher for the model parameter Streamlit app."""

import subprocess
import sys
from pathlib import Path
import logging




def main() -> int:
    root = Path(__file__).resolve().parents[1]
    app = root / "src" / "model_visualizer" / "app.py"
    # Use the current interpreter so Streamlit runs inside the active virtualenv.
    command = [sys.executable, "-m", "streamlit", "run", str(app)]
    return subprocess.call(command, cwd=root)


if __name__ == "__main__":
    raise SystemExit(main())
