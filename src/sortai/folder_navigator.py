"""Document archive traversal helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def list_children(path: Path) -> list[str]:
    """Return sorted names of immediate sub-directories of *path*."""
    return sorted(p.name for p in path.iterdir() if p.is_dir())


def is_leaf(path: Path) -> bool:
    """Return True if *path* has no sub-directories."""
    return not any(p.is_dir() for p in path.iterdir())


@dataclass
class FolderInfo:
    name: str
    subfolders: list[str]
    description: str | None


def list_children_with_info(
    path: Path,
    description_filename: str = "folder-description.md",
    subfolder_preview_count: int = 5,
) -> list[FolderInfo]:
    """Return FolderInfo for each immediate sub-directory of *path*."""
    infos = []
    for child_name in list_children(path):
        child_path = path / child_name
        subfolders = list_children(child_path)[:subfolder_preview_count]
        desc_path = child_path / description_filename
        description = desc_path.read_text(encoding="utf-8").strip() if desc_path.is_file() else None
        infos.append(FolderInfo(name=child_name, subfolders=subfolders, description=description))
    return infos
