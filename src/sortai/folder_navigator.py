"""Document archive traversal helpers."""

from __future__ import annotations

from pathlib import Path


def list_children(path: Path) -> list[str]:
    """Return sorted names of immediate sub-directories of *path*."""
    return sorted(p.name for p in path.iterdir() if p.is_dir())


def is_leaf(path: Path) -> bool:
    """Return True if *path* has no sub-directories."""
    return not any(p.is_dir() for p in path.iterdir())
