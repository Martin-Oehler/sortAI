"""Tests for sortai.tray — headless helpers (no pystray required)."""
from __future__ import annotations

import socket
from pathlib import Path

import pytest

from sortai.tray import _port_in_use, parse_args


class TestParseArgs:
    def test_defaults(self):
        args = parse_args([])
        assert args.config == Path("config/config.toml")
        assert args.port is None
        assert args.log_file == Path("logs/dashboard.log")

    def test_overrides(self):
        args = parse_args(
            ["--config", "other.toml", "--port", "9000", "--log-file", "x/y.log"]
        )
        assert args.config == Path("other.toml")
        assert args.port == 9000
        assert args.log_file == Path("x/y.log")


class TestPortInUse:
    def test_true_for_bound_listening_port(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            sock.listen(1)
            port = sock.getsockname()[1]
            assert _port_in_use(port) is True

    def test_false_for_closed_port(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        sock.close()
        assert _port_in_use(port) is False


class TestMakeIcon:
    def test_generates_64px_image(self):
        pytest.importorskip("PIL")
        from sortai.tray import _make_icon

        img = _make_icon()
        assert img.size == (64, 64)
        assert img.mode == "RGBA"
