"""
srtctl-rate-match CLI entry point.

Delegates to tools/rate_matching/cli.py which contains the full implementation.
This shim exists so the installed entry point (srtctl-rate-match) works.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Force unbuffered stdout/stderr so output appears immediately in terminals
# (without this, non-TTY contexts fully buffer and output appears to hang)
os.environ["PYTHONUNBUFFERED"] = "1"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)

# Add tools/rate_matching to path so its imports work
_TOOLS_DIR = Path(__file__).resolve().parents[3] / "tools" / "rate_matching"
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))


def main() -> None:
    from cli import main as _main
    _main()


if __name__ == "__main__":
    main()
