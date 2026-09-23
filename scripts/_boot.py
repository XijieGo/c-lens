"""Add this repository's local source trees to the script import path."""

from __future__ import annotations

import sys
from pathlib import Path


def _find_root() -> Path:
    """Locate the checkout even when a nested launcher is copied, not linked."""
    here = Path(__file__).resolve().parent
    for candidate in (here, *here.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "src").is_dir():
            return candidate
    raise RuntimeError(
        "could not locate the concept-lens checkout; run this script from a "
        "working tree containing pyproject.toml and src/"
    )


ROOT = _find_root()
SRC = ROOT / "src"
JLENS = SRC / "jlens"
for path in (SRC, JLENS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
