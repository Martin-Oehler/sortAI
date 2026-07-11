"""FastAPI dashboard server — audit log viewer and review interface."""

from __future__ import annotations

import asyncio
import sys
import threading as _threading
import traceback
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

if TYPE_CHECKING:
    from sortai.config import Config
    from sortai.review_store import ReviewItem, ReviewStore


def _async_exception_handler(loop: asyncio.AbstractEventLoop, context: dict) -> None:
    exc = context.get("exception")
    if isinstance(exc, (ConnectionResetError, ConnectionAbortedError)):
        return  # Windows proactor cleanup noise — peer already closed the socket
    loop.default_exception_handler(context)


def _log_exception(context: str) -> None:
    """Report a swallowed background-thread exception to stderr."""
    print(f"sortai dashboard: {context}:", file=sys.stderr)
    traceback.print_exc()


def create_app(
    cfg: "Config",
    store: "ReviewStore",
    watcher=None,
    pipeline_sem: "_threading.Semaphore | None" = None,
) -> FastAPI:
    sse_clients: list[asyncio.Queue] = []
    pipeline_sem = pipeline_sem or _threading.Semaphore(1)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        import threading

        app.state.loop = asyncio.get_running_loop()
        app.state.loop.set_exception_handler(_async_exception_handler)
        observer = _start_file_watcher(cfg, _broadcast)

        watcher_thread = None
        if watcher is not None:
            watcher_thread = threading.Thread(
                target=watcher.watch, daemon=True, name="sortai-watcher"
            )
            watcher_thread.start()

        try:
            yield
        finally:
            observer.stop()
            observer.join(timeout=2)
            if watcher is not None:
                watcher.stop()
                watcher_thread.join(timeout=5)

    app = FastAPI(title="sortAI Dashboard", lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")

    app.state.loop = None
    app.state.sse_clients = sse_clients
    app.state.pipeline_sem = pipeline_sem

    # ------------------------------------------------------------------
    # SSE broadcast
    # ------------------------------------------------------------------

    def _broadcast(event_type: str) -> None:
        loop = app.state.loop
        if loop is None:
            return

        async def _push() -> None:
            for q in list(sse_clients):
                await q.put(event_type)

        asyncio.run_coroutine_threadsafe(_push(), loop)

    # ------------------------------------------------------------------
    # Source resolution shared by /reveal, /reveal-target, /api/reprocess
    # and the file-serving routes.
    # ------------------------------------------------------------------

    def _get_queue_item(item_id) -> "ReviewItem":
        try:
            return store.get(item_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="Item not found")

    def _resolve_source(
        item_type, item_id, log_idx, *, staged_statuses: tuple = ("pending",)
    ) -> "tuple[Path, ReviewItem | None]":
        """Resolve a file path from a queue item or a log-entry index.

        Returns (path, item) — *item* is the ReviewItem for queue sources and
        None for log sources. Raises HTTPException when resolution fails.
        Queue items whose status is in *staged_statuses* resolve to their
        staging path; otherwise the resolved (final) path is used.
        """
        if item_type == "queue":
            item = _get_queue_item(item_id)
            if item.status in staged_statuses:
                return Path(item.staging_path), item
            if item.resolved_path:
                return Path(item.resolved_path), item
            raise HTTPException(status_code=404, detail="No file path")
        if item_type == "log":
            from sortai.file_ops import load_jsonl_entries
            entries = load_jsonl_entries(cfg.log_file)
            if log_idx is None or log_idx < 0 or log_idx >= len(entries):
                raise HTTPException(status_code=404, detail="Log entry not found")
            new_path = entries[log_idx].get("new_path", "")
            if not new_path:
                raise HTTPException(status_code=404, detail="No file path in log entry")
            return Path(new_path), None
        raise HTTPException(status_code=400, detail="Invalid item type")

    # ------------------------------------------------------------------
    # Routes
    # ------------------------------------------------------------------

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(Path(__file__).parent / "static" / "index.html")

    @app.get("/api/queue")
    def get_queue() -> list:
        store.reload()
        return [asdict(i) for i in store.list_all()]

    @app.get("/api/log")
    def get_log() -> list:
        from sortai.file_ops import load_jsonl_entries
        return load_jsonl_entries(cfg.log_file)

    @app.get("/files/queue/{item_id}")
    def serve_queue_file(item_id: str) -> FileResponse:
        path, _item = _resolve_source("queue", item_id, None)
        if not path.exists():
            raise HTTPException(status_code=404, detail="File not found on disk")
        return FileResponse(str(path), media_type="application/pdf")

    @app.get("/files/log/{log_idx}")
    def serve_log_file(log_idx: int) -> FileResponse:
        path, _item = _resolve_source("log", None, log_idx)
        if not path.exists():
            raise HTTPException(status_code=404, detail="File not found on disk")
        return FileResponse(str(path), media_type="application/pdf")

    @app.post("/reveal")
    async def reveal_file(request: Request) -> JSONResponse:
        import platform
        import subprocess

        data = await request.json()
        path, _item = _resolve_source(data.get("type"), data.get("id"), data.get("log_idx"))

        import os
        path = Path(os.path.abspath(path))
        if not path.exists():
            raise HTTPException(status_code=404, detail="File not found on disk")

        system = platform.system()
        try:
            if system == "Windows":
                subprocess.Popen(f'explorer /select,"{path}"', shell=True)
            elif system == "Darwin":
                subprocess.Popen(["open", "-R", str(path)])
            else:
                subprocess.Popen(["xdg-open", str(path.parent)])
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Could not open explorer: {exc}")

        return JSONResponse({"status": "ok"})

    @app.post("/reveal-target")
    async def reveal_target_folder(request: Request) -> JSONResponse:
        import platform
        import subprocess

        data = await request.json()
        item_type = data.get("type")
        item_id = data.get("id")
        log_idx = data.get("log_idx")

        if item_type == "queue":
            item = _get_queue_item(item_id)
            if item.status in ("pending", "reprocessing"):
                folder = Path(cfg.archive) / item.proposed_folder
            elif item.resolved_path:
                folder = Path(item.resolved_path).parent
            else:
                raise HTTPException(status_code=404, detail="No target path")
        else:
            path, _item = _resolve_source(item_type, item_id, log_idx)
            folder = path.parent

        import os
        folder = Path(os.path.abspath(folder))
        # Walk up to nearest existing ancestor if the proposed folder doesn't exist yet.
        while not folder.exists() and folder != folder.parent:
            folder = folder.parent
        if not folder.exists():
            raise HTTPException(status_code=404, detail="Target folder not found on disk")

        system = platform.system()
        try:
            if system == "Windows":
                subprocess.Popen(f'explorer "{folder}"', shell=True)
            elif system == "Darwin":
                subprocess.Popen(["open", str(folder)])
            else:
                subprocess.Popen(["xdg-open", str(folder)])
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Could not open explorer: {exc}")

        return JSONResponse({"status": "ok"})

    @app.post("/api/reprocess")
    async def reprocess_file(request: Request) -> JSONResponse:
        import threading

        data = await request.json()
        item_type = data.get("type")
        item_id = data.get("id")
        log_idx = data.get("log_idx")
        hint = (data.get("hint") or "").strip() or None

        source_path, item = _resolve_source(
            item_type, item_id, log_idx, staged_statuses=("pending", "reprocessing")
        )
        original_filename = item.original_filename if item is not None else source_path.name
        original_item_id: str | None = item_id if item_type == "queue" else None

        source_path = source_path.resolve()
        if not source_path.exists():
            raise HTTPException(status_code=404, detail="File not found on disk")

        if original_item_id:
            store.mark_reprocessing(original_item_id)
            _broadcast("queue_updated")

        original_proposed_folder: str | None = None
        if original_item_id:
            try:
                _original_item = store.get(original_item_id)
                original_proposed_folder = _original_item.proposed_folder
            except KeyError:
                pass

        def _run_pipeline() -> None:
            from sortai.llm_client import LMStudioClient
            from sortai.processor import process_document

            client = LMStudioClient.from_config(cfg)
            try:
                outcome = process_document(
                    cfg,
                    client,
                    source_path,
                    review_store=store,
                    user_hint=hint,
                    pipeline_sem=pipeline_sem,
                    dry_run=False,
                    original_filename=original_filename,
                    previous_proposed_folder=original_proposed_folder,
                )
            except Exception:
                _log_exception(f"reprocessing of {source_path.name} failed")
                outcome = None
            if outcome is None or outcome.status == "error":
                if original_item_id:
                    store.mark_pending(original_item_id)
                    _broadcast("queue_updated")
                return

            if original_item_id:
                store.remove(original_item_id)
            _broadcast("queue_updated")

        threading.Thread(target=_run_pipeline, daemon=True).start()
        return JSONResponse({"status": "reprocessing"}, status_code=202)

    @app.post("/api/accept/{item_id}")
    def accept_item(item_id: str) -> JSONResponse:
        import threading as _t
        store.reload()
        item = _get_queue_item(item_id)
        if item.status != "pending":
            raise HTTPException(status_code=400, detail="Item is not pending")

        from sortai.file_ops import log_decision, move_file
        dest_dir = cfg.archive / item.proposed_folder
        src = Path(item.staging_path)
        dest = move_file(src=src, dest_dir=dest_dir, new_name=item.proposed_filename, dry_run=False)
        log_decision(
            src=src,
            dest=dest,
            summary=item.summary,
            dry_run=False,
            log_path=cfg.log_file,
            archive_root=cfg.archive,
            interactions=item.interactions,
        )
        store.mark_accepted(item_id, str(dest))
        _broadcast("queue_updated")

        if item.user_hint and item.previous_proposed_folder and cfg.enable_memory:
            _t.Thread(
                target=_run_learning,
                args=(item, str(dest), cfg, pipeline_sem, _broadcast),
                daemon=True,
                name="sortai-learn",
            ).start()

        return JSONResponse({"status": "accepted", "resolved_path": str(dest)})

    @app.post("/api/reject/{item_id}")
    def reject_item(item_id: str) -> JSONResponse:
        store.reload()
        item = _get_queue_item(item_id)
        if item.status != "pending":
            raise HTTPException(status_code=400, detail="Item is not pending")

        from sortai.file_ops import move_file
        src = Path(item.staging_path)
        dest = move_file(src=src, dest_dir=cfg.rejected_dir, new_name=item.original_filename, dry_run=False)
        store.mark_rejected(item_id, str(dest))
        _broadcast("queue_updated")
        return JSONResponse({"status": "rejected", "resolved_path": str(dest)})

    @app.get("/api/memory")
    def get_memory() -> JSONResponse:
        from sortai.memory import load_rules
        return JSONResponse({"rules": load_rules(cfg.memory_path)})

    @app.delete("/api/memory/{rule_idx}")
    def delete_memory_rule(rule_idx: int) -> JSONResponse:
        from sortai.memory import load_rules, save_rules
        memory_path = cfg.memory_path
        if not memory_path.exists():
            raise HTTPException(status_code=404, detail="No memory file")
        rules = load_rules(memory_path)
        if rule_idx < 0 or rule_idx >= len(rules):
            raise HTTPException(status_code=404, detail="Rule not found")
        rules.pop(rule_idx)
        save_rules(memory_path, rules)
        _broadcast("memory_updated")
        return JSONResponse({"rules": rules})

    @app.get("/api/events")
    async def events(request: Request) -> StreamingResponse:
        q: asyncio.Queue = asyncio.Queue()
        sse_clients.append(q)

        async def generate():
            try:
                while True:
                    try:
                        event = await asyncio.wait_for(q.get(), timeout=30.0)
                    except asyncio.TimeoutError:
                        yield ": keepalive\n\n"
                        continue
                    if event is None:  # shutdown sentinel
                        break
                    yield f"event: {event}\ndata: {{}}\n\n"
            finally:
                if q in sse_clients:
                    sse_clients.remove(q)

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app


# ------------------------------------------------------------------
# File watcher (log/queue change notifications)
# ------------------------------------------------------------------


def _start_file_watcher(cfg: "Config", broadcast: Callable[[str], None]):
    from watchdog.events import FileSystemEventHandler
    from watchdog.observers import Observer

    log_dir = cfg.log_file.parent
    log_dir.mkdir(parents=True, exist_ok=True)

    class _Handler(FileSystemEventHandler):
        def _notify(self, path: str) -> None:
            name = Path(path).name
            if name == cfg.log_file.name:
                broadcast("log_updated")
            elif name == cfg.queue_path.name:
                broadcast("queue_updated")

        def on_modified(self, event) -> None:  # type: ignore[override]
            if not event.is_directory:
                self._notify(event.src_path)

        def on_created(self, event) -> None:  # type: ignore[override]
            if not event.is_directory:
                self._notify(event.src_path)

        def on_moved(self, event) -> None:  # type: ignore[override]
            if not event.is_directory:
                self._notify(event.dest_path)

    observer = Observer()
    observer.schedule(_Handler(), str(log_dir), recursive=False)
    observer.start()
    return observer


def _run_learning(
    item,
    resolved_path: str,
    cfg,
    pipeline_sem: _threading.Semaphore,
    broadcast: Callable[[str], None],
) -> None:
    """Background thread: learn from a user correction, then consolidate memory."""
    from sortai.file_ops import log_memory_update
    from sortai.llm_client import LMStudioClient
    from sortai.memory import consolidate_memory, learn_from_correction
    from sortai.pdf_reader import extract_text

    try:
        doc_text = extract_text(resolved_path)
    except Exception:
        doc_text = ""

    client = LMStudioClient.from_config(cfg)
    pipeline_sem.acquire()
    try:
        client.load_model()

        rule, learn_interactions = learn_from_correction(
            client,
            doc_text=doc_text,
            summary=item.summary,
            previous_folder=item.previous_proposed_folder,
            user_hint=item.user_hint,
            new_folder=item.proposed_folder,
        )

        consolidate_interactions: list = []
        if rule:
            consolidate_interactions = consolidate_memory(client, cfg.memory_path, rule)
            broadcast("memory_updated")

        all_interactions = learn_interactions + consolidate_interactions
        log_memory_update(
            original_filename=item.original_filename,
            previous_folder=item.previous_proposed_folder,
            new_folder=item.proposed_folder,
            user_hint=item.user_hint,
            new_rule=rule,
            log_path=cfg.log_file,
            interactions=all_interactions,
        )
        broadcast("log_updated")
    except Exception:
        _log_exception(f"learning from correction of {item.original_filename} failed")
    finally:
        pipeline_sem.release()


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

# Sentinel: distinguish "not passed" (keep uvicorn's default console logging)
# from an explicit log_config=None (propagate records to the root logger).
_UNSET = object()


def build_runtime(
    cfg: "Config",
    *,
    port: "int | None" = None,
    watch: bool = False,
    review_mode: bool = False,
):
    """Assemble the runtime objects the dashboard server needs.

    Returns (store, effective_port, watcher, pipeline_sem) — watcher and
    pipeline_sem are None unless *watch* is True.
    """
    from sortai.review_store import ReviewStore

    store = ReviewStore(cfg.queue_path)
    effective_port = port if port is not None else cfg.dashboard.port

    watcher = None
    pipeline_sem = None
    if watch:
        from sortai.watcher import Watcher

        # Shared with the dashboard so reprocess/learning and the watcher
        # never run the LLM pipeline concurrently.
        pipeline_sem = _threading.Semaphore(1)
        watcher = Watcher(
            cfg,
            review_mode=review_mode,
            review_store=store if review_mode else None,
            pipeline_sem=pipeline_sem,
        )
    return store, effective_port, watcher, pipeline_sem


def create_server(
    cfg: "Config",
    store: "ReviewStore",
    port: int,
    watcher=None,
    pipeline_sem: "_threading.Semaphore | None" = None,
    uvicorn_log_config=_UNSET,
):
    """Build a configured uvicorn.Server (not yet running).

    Pass uvicorn_log_config=None when file logging is active so uvicorn skips
    its own dictConfig and its records propagate to the root logger.
    """
    import uvicorn

    app = create_app(cfg, store, watcher=watcher, pipeline_sem=pipeline_sem)

    if uvicorn_log_config is not _UNSET:
        # File-logging mode: no dictConfig and no log_level override, so the
        # levels set by setup_file_logging stay in effect and uvicorn's
        # records propagate to the root logger's file handler.
        kwargs = {"log_config": uvicorn_log_config, "log_level": None}
    else:
        kwargs = {"log_level": "warning"}
    config = uvicorn.Config(app, host="127.0.0.1", port=port, **kwargs)
    server = uvicorn.Server(config)

    _original_handle_exit = server.handle_exit

    def _handle_exit(sig, frame):
        loop = app.state.loop
        if loop is not None:
            async def _close_sse():
                for q in list(app.state.sse_clients):
                    await q.put(None)
            try:
                fut = asyncio.run_coroutine_threadsafe(_close_sse(), loop)
                fut.result(timeout=2.0)
            except Exception:
                pass
        _original_handle_exit(sig, frame)

    server.handle_exit = _handle_exit
    return server


def run(
    cfg: "Config",
    store: "ReviewStore",
    port: int,
    open_browser: bool,
    watcher=None,
    pipeline_sem: "_threading.Semaphore | None" = None,
    uvicorn_log_config=_UNSET,
) -> None:
    server = create_server(
        cfg,
        store,
        port,
        watcher=watcher,
        pipeline_sem=pipeline_sem,
        uvicorn_log_config=uvicorn_log_config,
    )

    if open_browser:
        import threading
        import webbrowser

        def _open():
            import time
            time.sleep(1.0)
            webbrowser.open(f"http://localhost:{port}")
        threading.Thread(target=_open, daemon=True).start()

    try:
        server.run()
    except KeyboardInterrupt:
        pass
