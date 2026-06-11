"""Inbox watcher — monitors a folder and processes new PDFs automatically."""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from rich.console import Console

from sortai.config import Config
from sortai.llm_client import LMStudioClient
from sortai.processor import process_document

if TYPE_CHECKING:
    from sortai.review_store import ReviewStore

console = Console()

# Seconds to wait after a file event before processing (debounce).
_DEBOUNCE_SECONDS = 2.0


class Watcher:
    """Watch *cfg.inbox* for new PDFs and run the full pipeline on each one."""

    def __init__(
        self,
        cfg: Config,
        verbose: bool = False,
        review_mode: bool = False,
        review_store: "ReviewStore | None" = None,
        pipeline_sem: "threading.Semaphore | None" = None,
    ) -> None:
        self.cfg = cfg
        self.verbose = verbose
        self.review_mode = review_mode
        self.review_store = review_store
        self._pipeline_sem = pipeline_sem
        self._pending: dict[Path, float] = {}
        self._lock = threading.Lock()
        self._queue: queue.Queue[Path] = queue.Queue()
        self._stop = threading.Event()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run_once(self) -> None:
        """Process all existing PDFs in the inbox, then return."""
        pdfs = sorted(self.cfg.inbox.glob("*.pdf"))
        if not pdfs:
            console.print("[dim]No PDFs found in inbox.[/dim]")
            return
        console.print(f"[cyan]Processing {len(pdfs)} existing PDF(s)…[/cyan]")
        for pdf in pdfs:
            worker = threading.Thread(target=self._process, args=(pdf,), daemon=True)
            worker.start()
            try:
                worker.join()
            except KeyboardInterrupt:
                console.print("\n[yellow]Interrupted — waiting for current file to finish…[/yellow]")
                worker.join()
                console.print("[dim]Stopped.[/dim]")
                return

    def watch(self) -> None:
        """Block until Ctrl-C, processing PDFs as they appear in the inbox."""
        from watchdog.observers import Observer

        handler = _PDFHandler(self._on_event)
        observer = Observer()
        observer.schedule(handler, str(self.cfg.inbox), recursive=False)
        observer.start()
        self.run_once()

        worker = threading.Thread(target=self._worker, daemon=True, name="sortai-worker")
        worker.start()

        console.print(
            f"[bold green]Watching[/bold green] {self.cfg.inbox} "
            f"{'[dim](dry run)[/dim] ' if self.cfg.dry_run else ''}"
            "— press Ctrl-C to stop."
        )

        try:
            while observer.is_alive() and not self._stop.is_set():
                self._flush_pending()
                time.sleep(0.5)
        except KeyboardInterrupt:
            pass
        finally:
            self._stop.set()
            observer.stop()
            observer.join()
            worker.join()
            console.print("\n[dim]Watcher stopped.[/dim]")

    def stop(self) -> None:
        """Signal the watcher to stop (can be called from another thread)."""
        self._stop.set()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _worker(self) -> None:
        """Daemon thread: pull paths from the queue and process them one by one."""
        while not self._stop.is_set():
            try:
                pdf = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            self._process(pdf)
            self._queue.task_done()

    def _on_event(self, path: Path) -> None:
        """Called by the watchdog thread when a PDF appears."""
        with self._lock:
            self._pending[path] = time.monotonic() + _DEBOUNCE_SECONDS

    def _flush_pending(self) -> None:
        """Enqueue any pending paths whose debounce timer has elapsed."""
        now = time.monotonic()
        ready: list[Path] = []
        with self._lock:
            for path, due in list(self._pending.items()):
                if now >= due:
                    ready.append(path)
                    del self._pending[path]
        for path in ready:
            self._queue.put(path)

    def _process(self, pdf_path: Path) -> None:
        """Run the full pipeline on a single PDF; log errors but don't crash."""
        if not pdf_path.exists():
            console.print(f"[yellow]Skipping (gone):[/yellow] {pdf_path.name}")
            return

        console.print(f"[cyan]Processing[/cyan] {pdf_path.name} …")
        client = LMStudioClient.from_config(self.cfg)
        review_store = self.review_store if self.review_mode else None
        try:
            outcome = process_document(
                self.cfg,
                client,
                pdf_path,
                review_store=review_store,
                verbose=self.verbose,
                pipeline_sem=self._pipeline_sem,
            )
        except Exception as exc:  # noqa: BLE001
            console.print(f"[bold red]Error processing {pdf_path.name}:[/bold red] {exc}")
            return

        label = "[dim](dry run)[/dim] " if outcome.dry_run else ""
        if outcome.status == "error":
            console.print(f"[yellow]Cannot classify {pdf_path.name}:[/yellow] {outcome.error_reason}")
        elif outcome.status == "staged":
            console.print(
                f"[yellow]⚠ Staged for review:[/yellow] {label}{pdf_path.name} "
                f"→ [dim]{outcome.final_path}[/dim]"
            )
        else:
            console.print(f"[bold green]→[/bold green] {label}{outcome.final_path}")


class _PDFHandler:
    """Minimal watchdog event handler that forwards PDF created/moved events."""

    def __init__(self, callback: Callable[[Path], None]) -> None:
        from watchdog.events import FileCreatedEvent, FileMovedEvent

        self._callback = callback
        self._FileCreatedEvent = FileCreatedEvent
        self._FileMovedEvent = FileMovedEvent

    def dispatch(self, event) -> None:
        if isinstance(event, self._FileCreatedEvent):
            path = Path(event.src_path)
        elif isinstance(event, self._FileMovedEvent):
            path = Path(event.dest_path)
        else:
            return

        if path.suffix.lower() == ".pdf":
            self._callback(path)
