"""Resolve the evolvable workspace, materializing the packaged seed on first use.

Resolution order:
  1. an explicit path (``--workspace`` / arg)
  2. ``$MATHAGENT_WORKSPACE``
  3. ``./workspace`` if it already exists (dev / persistent local copy)
  4. otherwise copy the packaged seed → ``./workspace`` and use that

This makes ``pip install mathagent`` self-contained: a fresh install has no repo
checkout, so the bundled ``seed_workspace`` is copied into the user's CWD on first run.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

_SEED = Path(__file__).resolve().parent / "seed_workspace"


def resolve_workspace(explicit: str | os.PathLike | None = None) -> str:
    if explicit:
        return str(explicit)
    env = os.environ.get("MATHAGENT_WORKSPACE")
    if env:
        return env
    local = Path.cwd() / "workspace"
    if local.exists():
        return str(local)
    shutil.copytree(_SEED, local)
    return str(local)
