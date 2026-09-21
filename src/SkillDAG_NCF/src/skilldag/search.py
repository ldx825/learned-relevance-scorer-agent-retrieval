"""Default paths shared by graph and CLI.

The default skill library follows the Graph-of-Skills data layout populated by
``scripts/setup.sh``. Pass ``--skills-dir`` for custom skill libraries.
"""
from __future__ import annotations

from pathlib import Path

_CONTAINER_SKILLDAG_BODIES = Path("/var/lib/skilldag/bodies")
_REPO_ROOT = Path(__file__).resolve().parents[2]
_GOS_DEFAULT_SKILLS_DIR = _REPO_ROOT / "data" / "skillsets" / "skills_200"
DEFAULT_SKILLS_DIR = (
    _CONTAINER_SKILLDAG_BODIES
    if _CONTAINER_SKILLDAG_BODIES.exists()
    else _GOS_DEFAULT_SKILLS_DIR
)
