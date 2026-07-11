"""Rotating file logging for headless / autostart runs (stdlib-only)."""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path


class _StreamToLogger:
    """File-like object that routes writes to a logger, line by line.

    Replaces sys.stdout / sys.stderr under pythonw, where they are None —
    rich resolves sys.stdout at print time, so this captures console output.
    """

    def __init__(self, logger: logging.Logger, level: int = logging.INFO) -> None:
        self._logger = logger
        self._level = level
        self._buffer = ""

    def write(self, text: str) -> int:
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if line.strip():
                self._logger.log(self._level, line.rstrip())
        return len(text)

    def flush(self) -> None:
        if self._buffer.strip():
            self._logger.log(self._level, self._buffer.rstrip())
        self._buffer = ""

    def isatty(self) -> bool:
        return False


def setup_file_logging(
    path: Path,
    *,
    max_bytes: int = 10_485_760,
    backup_count: int = 3,
    redirect_std: bool = False,
) -> RotatingFileHandler:
    """Attach a rotating file handler to the root logger.

    With redirect_std=True (tray app), sys.stdout / sys.stderr are replaced
    with logger-backed streams so rich and raw prints land in the file too.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    handler = RotatingFileHandler(
        path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
    )
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))

    root = logging.getLogger()
    root.addHandler(handler)
    root.setLevel(logging.INFO)

    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logging.getLogger(name).setLevel(logging.INFO)

    if redirect_std:
        sys.stdout = _StreamToLogger(logging.getLogger("sortai.stdout"), logging.INFO)
        sys.stderr = _StreamToLogger(logging.getLogger("sortai.stderr"), logging.ERROR)

    return handler
