"""File move operations and JSON-lines decision logging."""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime
from pathlib import Path


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def move_file(src: Path, dest_dir: Path, new_name: str, dry_run: bool) -> Path:
    """Move *src* into *dest_dir* with name *new_name*.

    Creates *dest_dir* if needed; appends _2, _3 … on collision.
    Returns the final destination path (even in dry_run mode).
    Does NOT move anything when dry_run=True.
    """
    stem = Path(new_name).stem
    suffix = Path(new_name).suffix or ".pdf"
    dest = dest_dir / new_name
    counter = 2
    while dest.exists():
        dest = dest_dir / f"{stem}_{counter}{suffix}"
        counter += 1
    if not dry_run:
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), dest)
    return dest


def log_decision(
    src: Path,
    dest: Path,
    summary: str,
    dry_run: bool,
    log_path: Path,
    archive_root: Path | None = None,
    interactions: list | None = None,
) -> None:
    """Append a JSON-lines entry to *log_path* and regenerate the HTML report."""
    entry = {
        "timestamp": datetime.now().isoformat(),
        "original_path": os.path.abspath(src),
        "new_path": os.path.abspath(dest),
        "archive_root": os.path.abspath(archive_root) if archive_root else None,
        "summary": summary,
        "dry_run": dry_run,
        "interactions": interactions or [],
    }
    _append_and_render(log_path, entry)


def log_error(
    src: Path,
    reason: str,
    log_path: Path,
    archive_root: Path | None = None,
    interactions: list | None = None,
) -> None:
    """Append a classification-error entry to *log_path* and regenerate the HTML report."""
    entry = {
        "timestamp": datetime.now().isoformat(),
        "original_path": os.path.abspath(src),
        "new_path": "",
        "archive_root": os.path.abspath(archive_root) if archive_root else None,
        "summary": "",
        "dry_run": False,
        "error": True,
        "error_reason": reason,
        "interactions": interactions or [],
    }
    _append_and_render(log_path, entry)


def log_memory_update(
    original_filename: str,
    previous_folder: str,
    new_folder: str,
    user_hint: str,
    new_rule: str | None,
    log_path: Path,
    interactions: list | None = None,
) -> None:
    """Append a memory-update entry to *log_path* and regenerate the HTML report."""
    entry = {
        "timestamp": datetime.now().isoformat(),
        "type": "memory_update",
        "original_filename": original_filename,
        "previous_folder": previous_folder,
        "new_folder": new_folder,
        "user_hint": user_hint,
        "new_rule": new_rule or "",
        "interactions": interactions or [],
    }
    _append_and_render(log_path, entry)


def load_jsonl_entries(log_path: Path) -> list[dict]:
    """Parse all valid JSON-lines entries from *log_path*, skipping malformed lines."""
    entries: list[dict] = []
    if not log_path.exists():
        return entries
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if line := line.strip():
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return entries


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _append_and_render(log_path: Path, entry: dict) -> None:
    """Append *entry* as a JSON line to *log_path* and regenerate the HTML report."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    # Lazy import avoids a circular dependency: report.py imports load_jsonl_entries
    # from this module.
    from sortai.report import render_html_report

    render_html_report(log_path)
