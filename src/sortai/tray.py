"""System tray app: run the dashboard (with inbox watching) without a console.

Installed as the ``sortai-tray`` gui-script (pythonw subsystem — no console
window). Uses argparse instead of click: click's echo helpers break when
sys.stdout is None, which is the case under pythonw before the log redirect
kicks in.
"""

from __future__ import annotations

import argparse
import logging
import signal
import socket
import sys
import threading
import webbrowser
from pathlib import Path

logger = logging.getLogger("sortai.tray")

_ICON_BLUE = (43, 108, 176)


def parse_args(argv: "list[str] | None" = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="sortai-tray",
        description="Run the sortAI dashboard from the system tray.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/config.toml"),
        help="Path to config.toml (default: config/config.toml)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Dashboard port (default: from config or 8765)",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=Path("logs/dashboard.log"),
        help="Rotating log file (default: logs/dashboard.log)",
    )
    return parser.parse_args(argv)


def _message_box(text: str, title: str = "sortAI") -> None:
    """Show a native message box (the tray app has no console to print to)."""
    try:
        import ctypes

        ctypes.windll.user32.MessageBoxW(0, text, title, 0x40)  # MB_ICONINFORMATION
    except (ImportError, AttributeError, OSError):
        logger.error("%s: %s", title, text)


def _port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, port)) == 0


def _make_icon():
    """Generate the tray icon: a folder glyph on a blue tile (no package data)."""
    from PIL import Image, ImageDraw

    img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle((2, 2, 62, 62), radius=12, fill=_ICON_BLUE)
    # folder: tab + body
    draw.rounded_rectangle((14, 18, 32, 30), radius=3, fill="white")
    draw.rounded_rectangle((14, 24, 50, 48), radius=4, fill="white")
    draw.line((20, 36, 44, 36), fill=_ICON_BLUE, width=3)
    return img


def main(argv: "list[str] | None" = None) -> int:
    args = parse_args(argv)

    # First, before anything can fail: under pythonw sys.stdout/sys.stderr are
    # None, so uncaught output must have somewhere to go.
    from sortai.logging_setup import setup_file_logging

    setup_file_logging(args.log_file, redirect_std=True)
    logger.info("sortai-tray starting (config=%s)", args.config)

    try:
        import pystray
    except ImportError:
        _message_box(
            "The tray app needs the optional 'tray' dependencies.\n\n"
            'Install them with:  pip install -e ".[tray]"',
            "sortAI — missing dependencies",
        )
        return 1

    from sortai.config import Config

    try:
        cfg = Config.load(args.config)
    except Exception as exc:
        logger.exception("failed to load config %s", args.config)
        _message_box(f"Could not load {args.config}:\n\n{exc}", "sortAI — config error")
        return 1

    port = args.port if args.port is not None else cfg.dashboard.port
    url = f"http://localhost:{port}"

    if _port_in_use(port):
        logger.info("port %d already in use — assuming sortAI is running, opening browser", port)
        _message_box(f"sortAI already appears to be running on port {port}.\nOpening the dashboard instead.")
        webbrowser.open(url)
        return 0

    from sortai.dashboard_server import build_runtime, create_server

    store, port, watcher, pipeline_sem = build_runtime(
        cfg, port=port, watch=True, review_mode=cfg.review_mode
    )
    server = create_server(
        cfg, store, port, watcher=watcher, pipeline_sem=pipeline_sem, uvicorn_log_config=None
    )

    # Non-daemon: the process stays alive until the server shuts down cleanly.
    server_thread = threading.Thread(target=server.run, name="sortai-uvicorn")
    server_thread.start()
    logger.info("dashboard serving at %s (watching %s)", url, cfg.inbox)

    def _open_dashboard(icon, item) -> None:
        webbrowser.open(url)

    def _quit(icon, item) -> None:
        logger.info("quit requested from tray menu")
        # handle_exit flushes SSE clients and sets should_exit (thread-safe).
        server.handle_exit(signal.SIGINT, None)
        icon.stop()

    icon = pystray.Icon(
        "sortai",
        _make_icon(),
        "sortAI Dashboard",
        menu=pystray.Menu(
            pystray.MenuItem("Open Dashboard", _open_dashboard, default=True),
            pystray.MenuItem("Quit", _quit),
        ),
    )

    # pystray owns the Win32 message loop on the main thread; uvicorn is the
    # one running in a worker thread.
    icon.run()

    server_thread.join(timeout=10)
    if server_thread.is_alive():
        server.handle_exit(signal.SIGINT, None)  # second call sets force_exit
        server_thread.join(timeout=10)
    logger.info("sortai-tray exited")
    return 0


if __name__ == "__main__":
    sys.exit(main())
