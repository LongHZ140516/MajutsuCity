"""Shared repository-relative runtime path resolution."""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def env_path(variable: str, default: str) -> Path:
    """Resolve an optional environment override against the repository root."""

    value = os.environ.get(variable, default)
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()
