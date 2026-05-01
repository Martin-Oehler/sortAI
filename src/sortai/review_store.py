"""Thread-safe review queue — persisted as logs/review_queue.json."""

from __future__ import annotations

import json
import threading
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass
class ReviewItem:
    id: str
    timestamp: str
    original_filename: str
    staging_path: str
    proposed_folder: str
    proposed_filename: str
    summary: str
    interactions: list
    status: str  # "pending" | "accepted" | "rejected"
    resolved_path: Optional[str]


def make_review_item(
    original_filename: str,
    staging_path: Path,
    proposed_folder: str,
    proposed_filename: str,
    summary: str,
    interactions: list,
) -> ReviewItem:
    return ReviewItem(
        id=str(uuid.uuid4()),
        timestamp=datetime.now().isoformat(),
        original_filename=original_filename,
        staging_path=str(staging_path),
        proposed_folder=proposed_folder,
        proposed_filename=proposed_filename,
        summary=summary,
        interactions=interactions,
        status="pending",
        resolved_path=None,
    )


class ReviewStore:
    """Mutable JSON array of ReviewItems, written atomically on every change."""

    def __init__(self, queue_path: Path) -> None:
        self._path = queue_path
        self._lock = threading.Lock()
        self._items: list[ReviewItem] = []
        if queue_path.exists():
            self._load()

    def add(self, item: ReviewItem) -> None:
        with self._lock:
            self._items.append(item)
            self._save()

    def get(self, item_id: str) -> ReviewItem:
        with self._lock:
            for item in self._items:
                if item.id == item_id:
                    return item
        raise KeyError(item_id)

    def list_pending(self) -> list[ReviewItem]:
        with self._lock:
            return [i for i in self._items if i.status == "pending"]

    def list_all(self) -> list[ReviewItem]:
        with self._lock:
            return list(self._items)

    def reload(self) -> None:
        """Re-read the queue file from disk (picks up changes from other processes)."""
        with self._lock:
            if self._path.exists():
                self._load()

    def mark_accepted(self, item_id: str, resolved_path: str) -> None:
        with self._lock:
            for item in self._items:
                if item.id == item_id:
                    item.status = "accepted"
                    item.resolved_path = resolved_path
                    break
            self._save()

    def mark_rejected(self, item_id: str, resolved_path: str) -> None:
        with self._lock:
            for item in self._items:
                if item.id == item_id:
                    item.status = "rejected"
                    item.resolved_path = resolved_path
                    break
            self._save()

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps([asdict(i) for i in self._items], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        tmp.replace(self._path)

    def _load(self) -> None:
        data = json.loads(self._path.read_text(encoding="utf-8"))
        self._items = [ReviewItem(**d) for d in data]
