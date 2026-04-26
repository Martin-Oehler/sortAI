"""Inbox watcher — monitors a folder and processes new PDFs automatically."""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable
from pathlib import Path

from rich.console import Console

from sortai.config import Config
from sortai.file_ops import log_decision, log_error, move_file
from sortai.llm_client import LMStudioClient
from sortai.pipeline import ClassificationError, Pipeline

console = Console()

# Seconds to wait after a file event before processing (debounce).
_DEBOUNCE_SECONDS = 2.0


class Watcher:
    """Watch *cfg.inbox* for new PDFs and run the full pipeline on each one."""

    def __init__(self, cfg: Config, verbose: bool = False) -> None:
        self.cfg = cfg
        self.verbose = verbose
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
            self._process(pdf)

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
            while observer.is_alive():
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
        client = LMStudioClient(
            base_url=self.cfg.lm_studio.base_url,
            model_name=self.cfg.lm_studio.model,
            prompts_dir=self.cfg.prompts_dir,
            temperature=self.cfg.lm_studio.temperature,
            max_tokens=self.cfg.lm_studio.max_tokens,
            reasoning=self.cfg.lm_studio.reasoning,
        )
        try:
            with client:
                pipeline = Pipeline(self.cfg, client, verbose=self.verbose)
                target_folder, filename, summary, _ = pipeline.run(pdf_path)

            dest = move_file(
                src=pdf_path.resolve(),
                dest_dir=target_folder,
                new_name=filename,
                dry_run=self.cfg.dry_run,
            )
            log_decision(
                src=pdf_path.resolve(),
                dest=dest,
                summary=summary,
                dry_run=self.cfg.dry_run,
                log_path=self.cfg.log_file,
                archive_root=self.cfg.archive,
            )
            label = "[dim](dry run)[/dim] " if self.cfg.dry_run else ""
            console.print(f"[bold green]→[/bold green] {label}{dest}")
        except ClassificationError as exc:
            console.print(f"[yellow]Cannot classify {pdf_path.name}:[/yellow] {exc}")
            log_error(
                src=pdf_path.resolve(),
                reason=str(exc),
                log_path=self.cfg.log_file,
                archive_root=self.cfg.archive,
            )
        except Exception as exc:  # noqa: BLE001
            console.print(f"[bold red]Error processing {pdf_path.name}:[/bold red] {exc}")


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
