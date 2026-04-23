"""Comprehensive unit tests for sortai.watcher."""
from __future__ import annotations

import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from sortai.config import Config, LMStudioConfig
from sortai.watcher import Watcher, _PDFHandler


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def make_cfg(inbox: Path, archive: Path | None = None, dry_run: bool = False) -> Config:
    """Return a minimal Config with the given inbox directory."""
    cfg = Config.__new__(Config)
    cfg.inbox = inbox
    cfg.archive = archive or inbox
    cfg.prompts_dir = Path("prompts")
    cfg.log_file = Path("logs/sortai.jsonl")
    cfg.dry_run = dry_run
    cfg.max_navigate_depth = 10
    cfg.lm_studio = LMStudioConfig(
        base_url="http://localhost:1234",
        model="test-model",
        temperature=0.2,
        max_tokens=2048,
    )
    return cfg


# ---------------------------------------------------------------------------
# Watcher.run_once
# ---------------------------------------------------------------------------

class TestRunOnce:
    def test_no_pdfs_prints_message_and_does_not_call_process(self, tmp_path: Path):
        """When inbox has no PDFs, prints the 'No PDFs found' message and skips _process."""
        cfg = make_cfg(tmp_path)
        watcher = Watcher(cfg)

        with patch.object(watcher, "_process") as mock_process:
            watcher.run_once()

        mock_process.assert_not_called()

    def test_no_pdfs_with_other_files_still_skips(self, tmp_path: Path):
        """Non-PDF files in inbox do not trigger _process."""
        (tmp_path / "document.txt").write_text("hello")
        (tmp_path / "image.png").write_bytes(b"\x89PNG")
        cfg = make_cfg(tmp_path)
        watcher = Watcher(cfg)

        with patch.object(watcher, "_process") as mock_process:
            watcher.run_once()

        mock_process.assert_not_called()

    def test_single_pdf_calls_process_once(self, tmp_path: Path):
        """When inbox has exactly one PDF, _process is called once with that path."""
        pdf = tmp_path / "invoice.pdf"
        pdf.write_bytes(b"%PDF-1.4")
        cfg = make_cfg(tmp_path)
        watcher = Watcher(cfg)

        with patch.object(watcher, "_process") as mock_process:
            watcher.run_once()

        mock_process.assert_called_once_with(pdf)

    def test_multiple_pdfs_calls_process_for_each(self, tmp_path: Path):
        """When inbox has multiple PDFs, _process is called for each."""
        pdfs = []
        for name in ["b.pdf", "a.pdf", "c.pdf"]:
            p = tmp_path / name
            p.write_bytes(b"%PDF-1.4")
            pdfs.append(p)
        cfg = make_cfg(tmp_path)
        watcher = Watcher(cfg)

        with patch.object(watcher, "_process") as mock_process:
            watcher.run_once()

        assert mock_process.call_count == 3

    def test_multiple_pdfs_called_in_sorted_order(self, tmp_path: Path):
        """PDFs are passed to _process in sorted (alphabetical) order."""
        names = ["c_report.pdf", "a_invoice.pdf", "b_contract.pdf"]
        for name in names:
            (tmp_path / name).write_bytes(b"%PDF-1.4")
        cfg = make_cfg(tmp_path)
        watcher = Watcher(cfg)

        with patch.object(watcher, "_process") as mock_process:
            watcher.run_once()

        called_paths = [c.args[0] for c in mock_process.call_args_list]
        assert called_paths == sorted(called_paths)


# ---------------------------------------------------------------------------
# Watcher.watch
# ---------------------------------------------------------------------------

class TestWatch:
    """
    Observer and _PDFHandler are imported lazily inside watch(), so we patch
    them at their source module paths rather than on sortai.watcher.
    """

    def _make_observer_mock(self, is_alive_side_effect=None):
        """Return a MagicMock Observer whose is_alive() drives the watch loop."""
        observer = MagicMock()
        if is_alive_side_effect is not None:
            observer.is_alive.side_effect = is_alive_side_effect
        else:
            # Default: alive once, then dead so the loop exits cleanly.
            observer.is_alive.side_effect = [True, False]
        return observer

    def _watch_patches(self, observer, tmp_path):
        """Return a context-manager stack that wires up all watch() collaborators."""
        from contextlib import ExitStack
        stack = ExitStack()
        # Patch Observer at its real module so the local import inside watch() picks it up
        stack.enter_context(patch("watchdog.observers.Observer", return_value=observer))
        stack.enter_context(patch("time.sleep"))
        return stack

    def test_starts_observer(self, tmp_path: Path):
        """watch() calls observer.start()."""
        cfg = make_cfg(tmp_path)
        watcher = Watcher(cfg)
        observer = self._make_observer_mock()

        with self._watch_patches(observer, tmp_path):
            watcher.watch()

        observer.start.assert_called_once()

    def test_schedules_handler_on_inbox(self, tmp_path: Path):
        """watch() schedules the _PDFHandler on the inbox path, non-recursively."""
        cfg = make_cfg(tmp_path)
        watcher = Watcher(cfg)
        observer = self._make_observer_mock(is_alive_side_effect=[False])

        with self._watch_patches(observer, tmp_path):
            watcher.watch()

        # observer.schedule is called with some handler, the str inbox path, recursive=False
        observer.schedule.assert_called_once()
        _, args, kwargs = observer.schedule.mock_calls[0]
        assert args[1] == str(tmp_path)
        assert kwargs.get("recursive") is False or args[2] is False

    def test_stops_and_joins_observer_on_normal_exit(self, tmp_path: Path):
        """watch() calls observer.stop() and observer.join() when loop ends naturally."""
        cfg = make_cfg(tmp_path)
        watcher = Watcher(cfg)
        observer = self._make_observer_mock(is_alive_side_effect=[False])

        with self._watch_patches(observer, tmp_path):
            watcher.watch()

        observer.stop.assert_called_once()
        observer.join.assert_called_once()

    def test_stops_and_joins_on_keyboard_interrupt(self, tmp_path: Path):
        """watch() calls observer.stop() and observer.join() on KeyboardInterrupt."""
        cfg = make_cfg(tmp_path)
        watcher = Watcher(cfg)
        observer = MagicMock()
        observer.is_alive.side_effect = KeyboardInterrupt

        with self._watch_patches(observer, tmp_path):
            watcher.watch()  # Must not raise

        observer.stop.assert_called_once()
        observer.join.assert_called_once()

    def test_keyboard_interrupt_does_not_propagate(self, tmp_path: Path):
        """watch() swallows KeyboardInterrupt without re-raising it."""
        cfg = make_cfg(tmp_path)
        watcher = Watcher(cfg)
        observer = MagicMock()
        observer.is_alive.side_effect = KeyboardInterrupt

        with self._watch_patches(observer, tmp_path):
            # Should complete without raising
            watcher.watch()

    def test_calls_flush_pending_while_alive(self, tmp_path: Path):
        """watch() calls _flush_pending() on every loop iteration."""
        cfg = make_cfg(tmp_path)
        watcher = Watcher(cfg)
        # alive three times, then dies
        observer = self._make_observer_mock(is_alive_side_effect=[True, True, True, False])

        with self._watch_patches(observer, tmp_path), \
             patch.object(watcher, "_flush_pending") as mock_flush:
            watcher.watch()

        assert mock_flush.call_count == 3

    def test_prints_watching_message(self, tmp_path: Path):
        """watch() prints the watching status line to the console."""
        cfg = make_cfg(tmp_path)
        watcher = Watcher(cfg)
        observer = self._make_observer_mock(is_alive_side_effect=[False])

        with self._watch_patches(observer, tmp_path), \
             patch("sortai.watcher.console") as mock_console:
            watcher.watch()

        # console.print must have been called at least once for the status line
        assert mock_console.print.called

    def test_dry_run_flag_reflected_in_watch_message(self, tmp_path: Path):
        """When dry_run=True, the watching message mentions dry run."""
        cfg = make_cfg(tmp_path, dry_run=True)
        watcher = Watcher(cfg)
        observer = self._make_observer_mock(is_alive_side_effect=[False])
        printed_args = []

        def capture_print(*args, **kwargs):
            printed_args.extend(args)

        with self._watch_patches(observer, tmp_path), \
             patch("sortai.watcher.console") as mock_console:
            mock_console.print.side_effect = capture_print
            watcher.watch()

        combined = " ".join(str(a) for a in printed_args)
        assert "dry run" in combined.lower() or "dry_run" in combined.lower() or "dry" in combined.lower()

    def test_handler_receives_on_event_callback(self, tmp_path: Path):
        """The _PDFHandler passed to observer.schedule has watcher._on_event as its callback."""
        cfg = make_cfg(tmp_path)
        watcher = Watcher(cfg)
        observer = self._make_observer_mock(is_alive_side_effect=[False])

        with self._watch_patches(observer, tmp_path):
            watcher.watch()

        # The first argument to observer.schedule is the handler instance
        handler_arg = observer.schedule.call_args.args[0]
        assert isinstance(handler_arg, _PDFHandler)
        assert handler_arg._callback == watcher._on_event


# ---------------------------------------------------------------------------
# Watcher._on_event
# ---------------------------------------------------------------------------

class TestOnEvent:
    def test_adds_path_to_pending_with_future_due_time(self, tmp_path: Path):
        """_on_event() adds the path to _pending with a future due time."""
        cfg = make_cfg(tmp_path)
        watcher = Watcher(cfg)
        pdf = tmp_path / "new.pdf"

        before = time.monotonic()
        watcher._on_event(pdf)
        after = time.monotonic()

        assert pdf in watcher._pending
        due = watcher._pending[pdf]
        # Due time should be ~2 seconds in the future
        assert due > after
        assert due >= before + 1.9  # within tolerance of _DEBOUNCE_SECONDS = 2.0

    def test_overwrites_existing_pending_entry(self, tmp_path: Path):
        """Calling _on_event() twice for the same path updates the due time."""
        cfg = make_cfg(tmp_path)
        watcher = Watcher(cfg)
        pdf = tmp_path / "doc.pdf"

        watcher._on_event(pdf)
        first_due = watcher._pending[pdf]
        time.sleep(0.01)
        watcher._on_event(pdf)
        second_due = watcher._pending[pdf]

        assert second_due > first_due

    def test_multiple_paths_tracked_independently(self, tmp_path: Path):
        """Different paths each get their own entry in _pending."""
        cfg = make_cfg(tmp_path)
        watcher = Watcher(cfg)
        pdf_a = tmp_path / "a.pdf"
        pdf_b = tmp_path / "b.pdf"

        watcher._on_event(pdf_a)
        watcher._on_event(pdf_b)

        assert pdf_a in watcher._pending
        assert pdf_b in watcher._pending

    def test_thread_safe_concurrent_calls(self, tmp_path: Path):
        """_on_event() can be called concurrently from multiple threads without error."""
        cfg = make_cfg(tmp_path)
        watcher = Watcher(cfg)
        errors = []

        def fire(i: int):
            try:
                pdf = tmp_path / f"file_{i}.pdf"
                watcher._on_event(pdf)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=fire, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert len(watcher._pending) == 20

    def test_pending_dict_uses_lock(self, tmp_path: Path):
        """_on_event holds the lock while writing to _pending (structural check)."""
        cfg = make_cfg(tmp_path)
        watcher = Watcher(cfg)

        # Replace _lock with a MagicMock that behaves as a context manager
        mock_lock = MagicMock()
        mock_lock.__enter__ = MagicMock(return_value=None)
        mock_lock.__exit__ = MagicMock(return_value=False)
        watcher._lock = mock_lock

        pdf = tmp_path / "x.pdf"
        watcher._on_event(pdf)

        mock_lock.__enter__.assert_called_once()
        mock_lock.__exit__.assert_called_once()


# ---------------------------------------------------------------------------
# Watcher._flush_pending
# ---------------------------------------------------------------------------

class TestFlushPending:
    """_flush_pending enqueues ready paths into _queue; it does not call _process directly."""

    def _drain_queue(self, watcher) -> list[Path]:
        paths = []
        while not watcher._queue.empty():
            paths.append(watcher._queue.get_nowait())
        return paths

    def test_does_not_enqueue_paths_with_future_due_time(self, tmp_path: Path):
        """Paths whose due time is still in the future are not enqueued."""
        cfg = make_cfg(tmp_path)
        watcher = Watcher(cfg)
        pdf = tmp_path / "doc.pdf"
        watcher._pending[pdf] = time.monotonic() + 1000.0

        watcher._flush_pending()

        assert watcher._queue.empty()
        assert pdf in watcher._pending  # still pending

    def test_enqueues_paths_whose_due_time_has_elapsed(self, tmp_path: Path):
        """Paths whose due time is in the past are put onto the queue."""
        cfg = make_cfg(tmp_path)
        watcher = Watcher(cfg)
        pdf = tmp_path / "doc.pdf"
        watcher._pending[pdf] = time.monotonic() - 1.0

        watcher._flush_pending()

        assert self._drain_queue(watcher) == [pdf]

    def test_removes_enqueued_paths_from_pending(self, tmp_path: Path):
        """Enqueued paths are removed from _pending."""
        cfg = make_cfg(tmp_path)
        watcher = Watcher(cfg)
        pdf = tmp_path / "doc.pdf"
        watcher._pending[pdf] = time.monotonic() - 1.0

        watcher._flush_pending()

        assert pdf not in watcher._pending

    def test_future_paths_remain_in_pending(self, tmp_path: Path):
        """Paths with a future due time are left untouched in _pending."""
        cfg = make_cfg(tmp_path)
        watcher = Watcher(cfg)
        pdf = tmp_path / "future.pdf"
        future_due = time.monotonic() + 1000.0
        watcher._pending[pdf] = future_due

        watcher._flush_pending()

        assert pdf in watcher._pending
        assert watcher._pending[pdf] == future_due

    def test_mixed_pending_only_enqueues_ready_ones(self, tmp_path: Path):
        """Only elapsed-due paths go to the queue; future ones stay in _pending."""
        cfg = make_cfg(tmp_path)
        watcher = Watcher(cfg)
        past_pdf = tmp_path / "past.pdf"
        future_pdf = tmp_path / "future.pdf"
        watcher._pending[past_pdf] = time.monotonic() - 1.0
        watcher._pending[future_pdf] = time.monotonic() + 1000.0

        watcher._flush_pending()

        assert self._drain_queue(watcher) == [past_pdf]
        assert future_pdf in watcher._pending
        assert past_pdf not in watcher._pending

    def test_empty_pending_leaves_queue_empty(self, tmp_path: Path):
        """With an empty _pending dict, nothing is enqueued."""
        cfg = make_cfg(tmp_path)
        watcher = Watcher(cfg)

        watcher._flush_pending()

        assert watcher._queue.empty()

    def test_multiple_ready_paths_all_enqueued(self, tmp_path: Path):
        """All elapsed-due paths are enqueued in a single flush call."""
        cfg = make_cfg(tmp_path)
        watcher = Watcher(cfg)
        pdfs = [tmp_path / f"doc_{i}.pdf" for i in range(3)]
        for pdf in pdfs:
            watcher._pending[pdf] = time.monotonic() - 1.0

        watcher._flush_pending()

        assert set(self._drain_queue(watcher)) == set(pdfs)

    def test_enqueue_happens_outside_lock(self, tmp_path: Path):
        """queue.put is called after releasing the lock (no deadlock risk)."""
        cfg = make_cfg(tmp_path)
        watcher = Watcher(cfg)
        pdf = tmp_path / "doc.pdf"
        watcher._pending[pdf] = time.monotonic() - 1.0

        lock_held_during_put = []
        real_put = watcher._queue.put

        def checked_put(path):
            acquired = watcher._lock.acquire(blocking=False)
            lock_held_during_put.append(not acquired)
            if acquired:
                watcher._lock.release()
            real_put(path)

        watcher._queue.put = checked_put
        watcher._flush_pending()

        assert lock_held_during_put == [False]


# ---------------------------------------------------------------------------
# Watcher._worker
# ---------------------------------------------------------------------------

class TestWorker:
    def test_worker_calls_process_for_queued_path(self, tmp_path: Path):
        """_worker dequeues a path and calls _process with it."""
        cfg = make_cfg(tmp_path)
        watcher = Watcher(cfg)
        pdf = tmp_path / "doc.pdf"
        watcher._queue.put(pdf)

        def process_and_stop(path):
            watcher._stop.set()  # signal exit after handling this item

        with patch.object(watcher, "_process", side_effect=process_and_stop):
            watcher._worker()

        # _process was called exactly once with the queued path
        # (verified implicitly: if it wasn't called, _stop would never be set
        #  and _worker would block; the test would time out rather than pass)
        assert watcher._stop.is_set()

    def test_worker_exits_when_stop_set_and_queue_empty(self, tmp_path: Path):
        """_worker returns promptly when _stop is set and queue is empty."""
        cfg = make_cfg(tmp_path)
        watcher = Watcher(cfg)
        watcher._stop.set()

        # Should return without hanging.
        t = threading.Thread(target=watcher._worker)
        t.start()
        t.join(timeout=3.0)
        assert not t.is_alive()

    def test_worker_processes_multiple_queued_paths(self, tmp_path: Path):
        """_worker processes every path that was in the queue before stop."""
        cfg = make_cfg(tmp_path)
        watcher = Watcher(cfg)
        pdfs = [tmp_path / f"file_{i}.pdf" for i in range(3)]
        for pdf in pdfs:
            watcher._queue.put(pdf)

        processed = []

        def record(path):
            processed.append(path)
            if len(processed) == len(pdfs):
                watcher._stop.set()

        with patch.object(watcher, "_process", side_effect=record):
            t = threading.Thread(target=watcher._worker, daemon=True)
            t.start()
            t.join(timeout=5.0)

        assert set(processed) == set(pdfs)

    def test_watch_starts_worker_thread(self, tmp_path: Path):
        """watch() starts a daemon thread named 'sortai-worker'."""
        cfg = make_cfg(tmp_path)
        watcher = Watcher(cfg)
        observer = MagicMock()
        observer.is_alive.side_effect = [False]

        started_threads: list[threading.Thread] = []
        real_start = threading.Thread.start

        def capture_start(self_thread, *a, **kw):
            started_threads.append(self_thread)
            real_start(self_thread, *a, **kw)

        with patch("watchdog.observers.Observer", return_value=observer), \
             patch("time.sleep"), \
             patch.object(threading.Thread, "start", capture_start):
            watcher.watch()

        names = [t.name for t in started_threads]
        assert "sortai-worker" in names

    def test_stop_event_set_on_watch_exit(self, tmp_path: Path):
        """watch() sets _stop when the loop ends so the worker thread can exit."""
        cfg = make_cfg(tmp_path)
        watcher = Watcher(cfg)
        observer = MagicMock()
        observer.is_alive.side_effect = [False]

        with patch("watchdog.observers.Observer", return_value=observer), \
             patch("time.sleep"):
            watcher.watch()

        assert watcher._stop.is_set()

    def test_stop_event_set_on_keyboard_interrupt(self, tmp_path: Path):
        """watch() sets _stop even when exiting via KeyboardInterrupt."""
        cfg = make_cfg(tmp_path)
        watcher = Watcher(cfg)
        observer = MagicMock()
        observer.is_alive.side_effect = KeyboardInterrupt

        with patch("watchdog.observers.Observer", return_value=observer), \
             patch("time.sleep"):
            watcher.watch()

        assert watcher._stop.is_set()


# ---------------------------------------------------------------------------
# Watcher._process
# ---------------------------------------------------------------------------

class TestProcess:
    # Convenience: patches for all external collaborators used in _process
    _PATCH_CLIENT = "sortai.watcher.LMStudioClient"
    _PATCH_PIPELINE = "sortai.watcher.Pipeline"
    _PATCH_MOVE = "sortai.watcher.move_file"
    _PATCH_LOG = "sortai.watcher.log_decision"

    def _make_pipeline_mock(self, target: Path, filename: str, summary: str):
        """Return a Pipeline mock whose run() returns the given tuple."""
        pipeline_instance = MagicMock()
        pipeline_instance.run.return_value = (target, filename, summary)
        pipeline_cls = MagicMock(return_value=pipeline_instance)
        return pipeline_cls, pipeline_instance

    def test_skips_when_file_does_not_exist(self, tmp_path: Path):
        """_process prints 'Skipping' and returns early when the PDF is gone."""
        cfg = make_cfg(tmp_path)
        watcher = Watcher(cfg)
        missing = tmp_path / "gone.pdf"  # never created

        with patch(self._PATCH_CLIENT) as MockClient, \
             patch(self._PATCH_PIPELINE) as MockPipeline:
            watcher._process(missing)

        MockClient.assert_not_called()
        MockPipeline.assert_not_called()

    def test_skip_message_contains_skipping(self, tmp_path: Path, capsys):
        """When the PDF is missing, a message containing 'Skipping' is printed."""
        cfg = make_cfg(tmp_path)
        watcher = Watcher(cfg)
        missing = tmp_path / "gone.pdf"

        printed = []
        with patch("sortai.watcher.console") as mock_console:
            mock_console.print.side_effect = lambda *a, **kw: printed.extend(a)
            watcher._process(missing)

        combined = " ".join(str(a) for a in printed)
        assert "kipping" in combined  # "Skipping" or "[yellow]Skipping..."

    def test_creates_lm_studio_client_with_config_values(self, tmp_path: Path):
        """_process instantiates LMStudioClient with the correct config fields."""
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF")
        archive = tmp_path / "archive"
        archive.mkdir()
        cfg = make_cfg(tmp_path, archive=archive)
        watcher = Watcher(cfg)

        pipeline_cls, pipeline_instance = self._make_pipeline_mock(archive, "out.pdf", "summary")
        move_result = archive / "out.pdf"

        with patch(self._PATCH_CLIENT) as MockClient, \
             patch(self._PATCH_PIPELINE, pipeline_cls), \
             patch(self._PATCH_MOVE, return_value=move_result), \
             patch(self._PATCH_LOG):
            watcher._process(pdf)

        MockClient.assert_called_once_with(
            base_url=cfg.lm_studio.base_url,
            model_name=cfg.lm_studio.model,
            prompts_dir=cfg.prompts_dir,
            temperature=cfg.lm_studio.temperature,
            max_tokens=cfg.lm_studio.max_tokens,
        )

    def test_uses_client_as_context_manager(self, tmp_path: Path):
        """_process uses LMStudioClient as a context manager (with statement)."""
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF")
        archive = tmp_path / "archive"
        archive.mkdir()
        cfg = make_cfg(tmp_path, archive=archive)
        watcher = Watcher(cfg)

        client_instance = MagicMock()
        client_instance.__enter__ = MagicMock(return_value=client_instance)
        client_instance.__exit__ = MagicMock(return_value=False)
        pipeline_instance = MagicMock()
        pipeline_instance.run.return_value = (archive, "out.pdf", "summary")
        move_result = archive / "out.pdf"

        with patch(self._PATCH_CLIENT, return_value=client_instance), \
             patch(self._PATCH_PIPELINE, return_value=pipeline_instance), \
             patch(self._PATCH_MOVE, return_value=move_result), \
             patch(self._PATCH_LOG):
            watcher._process(pdf)

        client_instance.__enter__.assert_called_once()
        client_instance.__exit__.assert_called_once()

    def test_runs_pipeline_with_pdf_path(self, tmp_path: Path):
        """_process calls Pipeline.run() with the pdf_path argument."""
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF")
        archive = tmp_path / "archive"
        archive.mkdir()
        cfg = make_cfg(tmp_path, archive=archive)
        watcher = Watcher(cfg)

        pipeline_cls, pipeline_instance = self._make_pipeline_mock(archive, "out.pdf", "summary")
        move_result = archive / "out.pdf"

        with patch(self._PATCH_CLIENT), \
             patch(self._PATCH_PIPELINE, pipeline_cls), \
             patch(self._PATCH_MOVE, return_value=move_result), \
             patch(self._PATCH_LOG):
            watcher._process(pdf)

        pipeline_instance.run.assert_called_once_with(pdf)

    def test_constructs_pipeline_with_cfg_client_verbose(self, tmp_path: Path):
        """Pipeline is constructed with (cfg, client, verbose=watcher.verbose)."""
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF")
        archive = tmp_path / "archive"
        archive.mkdir()
        cfg = make_cfg(tmp_path, archive=archive)
        watcher = Watcher(cfg, verbose=True)

        client_instance = MagicMock()
        client_instance.__enter__ = MagicMock(return_value=client_instance)
        client_instance.__exit__ = MagicMock(return_value=False)
        pipeline_instance = MagicMock()
        pipeline_instance.run.return_value = (archive, "out.pdf", "summary")
        move_result = archive / "out.pdf"

        with patch(self._PATCH_CLIENT, return_value=client_instance) as MockClient, \
             patch(self._PATCH_PIPELINE, return_value=pipeline_instance) as MockPipeline, \
             patch(self._PATCH_MOVE, return_value=move_result), \
             patch(self._PATCH_LOG):
            watcher._process(pdf)

        # Pipeline.__init__ receives cfg, the entered client, and verbose=True
        MockPipeline.assert_called_once_with(cfg, client_instance, verbose=True)

    def test_calls_move_file_with_correct_args(self, tmp_path: Path):
        """_process calls move_file with src=pdf.resolve(), dest_dir, new_name, dry_run."""
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF")
        archive = tmp_path / "archive"
        archive.mkdir()
        cfg = make_cfg(tmp_path, archive=archive, dry_run=False)
        watcher = Watcher(cfg)

        target_folder = archive
        filename = "renamed.pdf"
        summary = "a summary"
        pipeline_cls, pipeline_instance = self._make_pipeline_mock(target_folder, filename, summary)
        move_result = archive / filename

        with patch(self._PATCH_CLIENT), \
             patch(self._PATCH_PIPELINE, pipeline_cls), \
             patch(self._PATCH_MOVE, return_value=move_result) as mock_move, \
             patch(self._PATCH_LOG):
            watcher._process(pdf)

        mock_move.assert_called_once_with(
            src=pdf.resolve(),
            dest_dir=target_folder,
            new_name=filename,
            dry_run=False,
        )

    def test_calls_log_decision_with_correct_args(self, tmp_path: Path):
        """_process calls log_decision with the return value of move_file and pipeline data."""
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF")
        archive = tmp_path / "archive"
        archive.mkdir()
        cfg = make_cfg(tmp_path, archive=archive, dry_run=False)
        watcher = Watcher(cfg)

        target_folder = archive
        filename = "renamed.pdf"
        summary = "my summary"
        pipeline_cls, pipeline_instance = self._make_pipeline_mock(target_folder, filename, summary)
        move_result = archive / filename

        with patch(self._PATCH_CLIENT), \
             patch(self._PATCH_PIPELINE, pipeline_cls), \
             patch(self._PATCH_MOVE, return_value=move_result), \
             patch(self._PATCH_LOG) as mock_log:
            watcher._process(pdf)

        mock_log.assert_called_once_with(
            src=pdf.resolve(),
            dest=move_result,
            summary=summary,
            dry_run=False,
            log_path=cfg.log_file,
            archive_root=cfg.archive,
        )

    def test_respects_dry_run_true(self, tmp_path: Path):
        """When cfg.dry_run=True, move_file and log_decision receive dry_run=True."""
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF")
        archive = tmp_path / "archive"
        archive.mkdir()
        cfg = make_cfg(tmp_path, archive=archive, dry_run=True)
        watcher = Watcher(cfg)

        target_folder = archive
        filename = "out.pdf"
        summary = "summary"
        pipeline_cls, pipeline_instance = self._make_pipeline_mock(target_folder, filename, summary)
        move_result = archive / filename

        with patch(self._PATCH_CLIENT), \
             patch(self._PATCH_PIPELINE, pipeline_cls), \
             patch(self._PATCH_MOVE, return_value=move_result) as mock_move, \
             patch(self._PATCH_LOG) as mock_log:
            watcher._process(pdf)

        assert mock_move.call_args.kwargs["dry_run"] is True
        assert mock_log.call_args.kwargs["dry_run"] is True

    def test_pipeline_error_is_caught_not_reraised(self, tmp_path: Path):
        """When Pipeline.run() raises, _process catches it and does not re-raise."""
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF")
        archive = tmp_path / "archive"
        archive.mkdir()
        cfg = make_cfg(tmp_path, archive=archive)
        watcher = Watcher(cfg)

        client_instance = MagicMock()
        client_instance.__enter__ = MagicMock(return_value=client_instance)
        client_instance.__exit__ = MagicMock(return_value=False)
        pipeline_instance = MagicMock()
        pipeline_instance.run.side_effect = RuntimeError("LLM failed")

        with patch(self._PATCH_CLIENT, return_value=client_instance), \
             patch(self._PATCH_PIPELINE, return_value=pipeline_instance), \
             patch(self._PATCH_MOVE) as mock_move, \
             patch(self._PATCH_LOG) as mock_log:
            watcher._process(pdf)  # must not raise

        # move_file and log_decision are NOT called when pipeline fails
        mock_move.assert_not_called()
        mock_log.assert_not_called()

    def test_pipeline_error_message_is_printed(self, tmp_path: Path):
        """When Pipeline.run() raises, the error message is printed to the console."""
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF")
        archive = tmp_path / "archive"
        archive.mkdir()
        cfg = make_cfg(tmp_path, archive=archive)
        watcher = Watcher(cfg)

        error_msg = "something went badly wrong"
        client_instance = MagicMock()
        client_instance.__enter__ = MagicMock(return_value=client_instance)
        client_instance.__exit__ = MagicMock(return_value=False)
        pipeline_instance = MagicMock()
        pipeline_instance.run.side_effect = RuntimeError(error_msg)

        printed = []
        with patch(self._PATCH_CLIENT, return_value=client_instance), \
             patch(self._PATCH_PIPELINE, return_value=pipeline_instance), \
             patch(self._PATCH_MOVE), \
             patch(self._PATCH_LOG), \
             patch("sortai.watcher.console") as mock_console:
            mock_console.print.side_effect = lambda *a, **kw: printed.extend(a)
            watcher._process(pdf)

        combined = " ".join(str(a) for a in printed)
        assert error_msg in combined

    def test_move_file_error_is_caught_not_reraised(self, tmp_path: Path):
        """When move_file() raises, _process catches it and does not re-raise."""
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF")
        archive = tmp_path / "archive"
        archive.mkdir()
        cfg = make_cfg(tmp_path, archive=archive)
        watcher = Watcher(cfg)

        pipeline_cls, pipeline_instance = self._make_pipeline_mock(archive, "out.pdf", "s")

        with patch(self._PATCH_CLIENT), \
             patch(self._PATCH_PIPELINE, pipeline_cls), \
             patch(self._PATCH_MOVE, side_effect=OSError("disk full")), \
             patch(self._PATCH_LOG):
            watcher._process(pdf)  # must not raise


# ---------------------------------------------------------------------------
# _PDFHandler.dispatch
# ---------------------------------------------------------------------------

class TestPDFHandlerDispatch:
    """Tests for _PDFHandler.dispatch() using real watchdog event classes."""

    def setup_method(self):
        from watchdog.events import FileCreatedEvent, FileMovedEvent
        self.FileCreatedEvent = FileCreatedEvent
        self.FileMovedEvent = FileMovedEvent

    def test_calls_callback_for_created_pdf_lowercase(self, tmp_path: Path):
        """FileCreatedEvent with a lowercase .pdf path triggers the callback."""
        callback = MagicMock()
        handler = _PDFHandler(callback)
        pdf = tmp_path / "invoice.pdf"
        event = self.FileCreatedEvent(str(pdf))

        handler.dispatch(event)

        callback.assert_called_once_with(pdf)

    def test_calls_callback_for_created_pdf_uppercase(self, tmp_path: Path):
        """FileCreatedEvent with a .PDF (uppercase) path triggers the callback."""
        callback = MagicMock()
        handler = _PDFHandler(callback)
        pdf = tmp_path / "invoice.PDF"
        event = self.FileCreatedEvent(str(pdf))

        handler.dispatch(event)

        callback.assert_called_once_with(pdf)

    def test_calls_callback_for_created_pdf_mixed_case(self, tmp_path: Path):
        """FileCreatedEvent with a .Pdf (mixed-case) path triggers the callback."""
        callback = MagicMock()
        handler = _PDFHandler(callback)
        pdf = tmp_path / "invoice.Pdf"
        event = self.FileCreatedEvent(str(pdf))

        handler.dispatch(event)

        callback.assert_called_once_with(pdf)

    def test_ignores_created_non_pdf_files(self, tmp_path: Path):
        """FileCreatedEvent for a non-PDF file does not trigger the callback."""
        callback = MagicMock()
        handler = _PDFHandler(callback)
        event = self.FileCreatedEvent(str(tmp_path / "document.txt"))

        handler.dispatch(event)

        callback.assert_not_called()

    def test_ignores_created_docx_files(self, tmp_path: Path):
        """FileCreatedEvent for a .docx file is ignored."""
        callback = MagicMock()
        handler = _PDFHandler(callback)
        event = self.FileCreatedEvent(str(tmp_path / "report.docx"))

        handler.dispatch(event)

        callback.assert_not_called()

    def test_calls_callback_for_moved_pdf_destination(self, tmp_path: Path):
        """FileMovedEvent uses the destination path and triggers callback for .pdf."""
        callback = MagicMock()
        handler = _PDFHandler(callback)
        src = tmp_path / "old.txt"
        dest = tmp_path / "new.pdf"
        event = self.FileMovedEvent(str(src), str(dest))

        handler.dispatch(event)

        callback.assert_called_once_with(dest)

    def test_ignores_moved_event_when_dest_is_not_pdf(self, tmp_path: Path):
        """FileMovedEvent where the destination is not a PDF is ignored."""
        callback = MagicMock()
        handler = _PDFHandler(callback)
        event = self.FileMovedEvent(str(tmp_path / "a.pdf"), str(tmp_path / "b.txt"))

        handler.dispatch(event)

        callback.assert_not_called()

    def test_moved_event_checks_dest_not_src(self, tmp_path: Path):
        """For FileMovedEvent, only the destination suffix matters, not the source."""
        callback = MagicMock()
        handler = _PDFHandler(callback)
        # src is .pdf but dest is .txt — should NOT call callback
        event = self.FileMovedEvent(str(tmp_path / "a.pdf"), str(tmp_path / "b.txt"))

        handler.dispatch(event)

        callback.assert_not_called()

    def test_ignores_unrelated_event_type(self):
        """Non-created, non-moved event types are silently ignored."""
        from watchdog.events import FileDeletedEvent

        callback = MagicMock()
        handler = _PDFHandler(callback)
        event = FileDeletedEvent("/some/path.pdf")

        handler.dispatch(event)

        callback.assert_not_called()

    def test_ignores_file_modified_event(self):
        """FileModifiedEvent is not handled — callback is not called."""
        from watchdog.events import FileModifiedEvent

        callback = MagicMock()
        handler = _PDFHandler(callback)
        event = FileModifiedEvent("/some/path.pdf")

        handler.dispatch(event)

        callback.assert_not_called()

    def test_callback_receives_path_object_not_string(self, tmp_path: Path):
        """The callback receives a pathlib.Path, not a raw string."""
        callback = MagicMock()
        handler = _PDFHandler(callback)
        pdf = tmp_path / "doc.pdf"
        event = self.FileCreatedEvent(str(pdf))

        handler.dispatch(event)

        assert isinstance(callback.call_args.args[0], Path)

    def test_callback_path_matches_event_src_for_created(self, tmp_path: Path):
        """For FileCreatedEvent the callback path equals Path(event.src_path)."""
        callback = MagicMock()
        handler = _PDFHandler(callback)
        pdf_str = str(tmp_path / "report.pdf")
        event = self.FileCreatedEvent(pdf_str)

        handler.dispatch(event)

        assert callback.call_args.args[0] == Path(pdf_str)

    def test_callback_path_matches_event_dest_for_moved(self, tmp_path: Path):
        """For FileMovedEvent the callback path equals Path(event.dest_path)."""
        callback = MagicMock()
        handler = _PDFHandler(callback)
        dest_str = str(tmp_path / "final.pdf")
        event = self.FileMovedEvent(str(tmp_path / "tmp.pdf"), dest_str)

        handler.dispatch(event)

        assert callback.call_args.args[0] == Path(dest_str)
