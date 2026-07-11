"""Tests for sortai.logging_setup — rotating file handler and std redirection."""
from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

import pytest

from sortai.logging_setup import setup_file_logging


@pytest.fixture(autouse=True)
def restore_logging_state():
    """Snapshot root handlers/level and std streams; restore after each test."""
    root = logging.getLogger()
    saved_handlers = root.handlers[:]
    saved_level = root.level
    saved_stdout, saved_stderr = sys.stdout, sys.stderr
    yield
    for handler in root.handlers[:]:
        if handler not in saved_handlers:
            root.removeHandler(handler)
            handler.close()  # release the file so tmp_path can be removed on Windows
    root.setLevel(saved_level)
    sys.stdout, sys.stderr = saved_stdout, saved_stderr


class TestSetupFileLogging:
    def test_handler_configuration(self, tmp_path: Path):
        handler = setup_file_logging(tmp_path / "sub" / "dashboard.log")
        assert isinstance(handler, RotatingFileHandler)
        assert handler.maxBytes == 10_485_760
        assert handler.backupCount == 3
        assert handler.encoding == "utf-8"
        assert handler in logging.getLogger().handlers
        assert (tmp_path / "sub").is_dir()  # parent created

    def test_log_lines_land_in_file(self, tmp_path: Path):
        log_file = tmp_path / "dashboard.log"
        setup_file_logging(log_file)
        logging.getLogger("sortai.test").info("hello from the test")
        content = log_file.read_text(encoding="utf-8")
        assert "hello from the test" in content
        assert "INFO sortai.test" in content

    def test_rotation_with_tiny_max_bytes(self, tmp_path: Path):
        log_file = tmp_path / "dashboard.log"
        setup_file_logging(log_file, max_bytes=200, backup_count=2)
        log = logging.getLogger("sortai.test")
        for i in range(50):
            log.info("line %03d — padding padding padding", i)
        assert log_file.exists()
        assert (tmp_path / "dashboard.log.1").exists()
        assert log_file.stat().st_size < 10_000

    def test_uvicorn_loggers_get_info_level(self, tmp_path: Path):
        setup_file_logging(tmp_path / "dashboard.log")
        for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
            assert logging.getLogger(name).level == logging.INFO

    def test_std_streams_untouched_by_default(self, tmp_path: Path):
        before_out, before_err = sys.stdout, sys.stderr
        setup_file_logging(tmp_path / "dashboard.log")
        assert sys.stdout is before_out
        assert sys.stderr is before_err

    def test_redirect_std_captures_print(self, tmp_path: Path):
        log_file = tmp_path / "dashboard.log"
        setup_file_logging(log_file, redirect_std=True)
        print("printed to stdout")
        print("printed to stderr", file=sys.stderr)
        sys.stdout.flush()
        sys.stderr.flush()
        content = log_file.read_text(encoding="utf-8")
        assert "INFO sortai.stdout: printed to stdout" in content
        assert "ERROR sortai.stderr: printed to stderr" in content

    def test_redirected_stream_is_not_a_tty(self, tmp_path: Path):
        setup_file_logging(tmp_path / "dashboard.log", redirect_std=True)
        assert sys.stdout.isatty() is False
