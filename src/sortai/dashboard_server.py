"""FastAPI dashboard server — audit log viewer and review interface."""

from __future__ import annotations

import asyncio
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse

if TYPE_CHECKING:
    from sortai.config import Config
    from sortai.review_store import ReviewStore

# Module-level state set by create_app().
_cfg: "Config | None" = None
_store: "ReviewStore | None" = None
_sse_clients: list[asyncio.Queue] = []
_loop: asyncio.AbstractEventLoop | None = None


def create_app(cfg: "Config", store: "ReviewStore") -> FastAPI:
    global _cfg, _store
    _cfg = cfg
    _store = store

    app = FastAPI(title="sortAI Dashboard")

    @app.on_event("startup")
    async def _startup() -> None:
        global _loop
        _loop = asyncio.get_event_loop()
        _start_file_watcher()

    # ------------------------------------------------------------------
    # Routes
    # ------------------------------------------------------------------

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        html_path = Path(__file__).parent / "static" / "index.html"
        return HTMLResponse(html_path.read_text(encoding="utf-8"))

    @app.get("/api/queue")
    def get_queue() -> list:
        return [asdict(i) for i in _store.list_all()]  # type: ignore[union-attr]

    @app.get("/api/log")
    def get_log() -> list:
        from sortai.file_ops import load_jsonl_entries
        return load_jsonl_entries(_cfg.log_file)  # type: ignore[union-attr]

    @app.get("/files/queue/{item_id}")
    def serve_queue_file(item_id: str) -> FileResponse:
        try:
            item = _store.get(item_id)  # type: ignore[union-attr]
        except KeyError:
            raise HTTPException(status_code=404, detail="Item not found")
        if item.status == "pending":
            path = Path(item.staging_path)
        elif item.resolved_path:
            path = Path(item.resolved_path)
        else:
            raise HTTPException(status_code=404, detail="No file path")
        if not path.exists():
            raise HTTPException(status_code=404, detail="File not found on disk")
        return FileResponse(str(path), media_type="application/pdf")

    @app.get("/files/log/{log_idx}")
    def serve_log_file(log_idx: int) -> FileResponse:
        from sortai.file_ops import load_jsonl_entries
        entries = load_jsonl_entries(_cfg.log_file)  # type: ignore[union-attr]
        if log_idx < 0 or log_idx >= len(entries):
            raise HTTPException(status_code=404, detail="Log entry not found")
        entry = entries[log_idx]
        new_path = entry.get("new_path", "")
        if not new_path:
            raise HTTPException(status_code=404, detail="No file path in log entry")
        path = Path(new_path)
        if not path.exists():
            raise HTTPException(status_code=404, detail="File not found on disk")
        return FileResponse(str(path), media_type="application/pdf")

    @app.post("/api/accept/{item_id}")
    def accept_item(item_id: str) -> JSONResponse:
        try:
            item = _store.get(item_id)  # type: ignore[union-attr]
        except KeyError:
            raise HTTPException(status_code=404, detail="Item not found")
        if item.status != "pending":
            raise HTTPException(status_code=400, detail="Item is not pending")

        from sortai.file_ops import log_decision, move_file
        dest_dir = _cfg.archive / item.proposed_folder  # type: ignore[union-attr]
        src = Path(item.staging_path)
        dest = move_file(src=src, dest_dir=dest_dir, new_name=item.proposed_filename, dry_run=False)
        log_decision(
            src=src,
            dest=dest,
            summary=item.summary,
            dry_run=False,
            log_path=_cfg.log_file,  # type: ignore[union-attr]
            archive_root=_cfg.archive,  # type: ignore[union-attr]
        )
        _store.mark_accepted(item_id, str(dest))  # type: ignore[union-attr]
        _broadcast("queue_updated")
        return JSONResponse({"status": "accepted", "resolved_path": str(dest)})

    @app.post("/api/reject/{item_id}")
    def reject_item(item_id: str) -> JSONResponse:
        try:
            item = _store.get(item_id)  # type: ignore[union-attr]
        except KeyError:
            raise HTTPException(status_code=404, detail="Item not found")
        if item.status != "pending":
            raise HTTPException(status_code=400, detail="Item is not pending")

        from sortai.file_ops import move_file
        rejected_dir = (
            _cfg.review.rejected_dir  # type: ignore[union-attr]
            if _cfg.review.rejected_dir  # type: ignore[union-attr]
            else _cfg.inbox.parent / "_rejected"  # type: ignore[union-attr]
        )
        src = Path(item.staging_path)
        dest = move_file(src=src, dest_dir=rejected_dir, new_name=item.original_filename, dry_run=False)
        _store.mark_rejected(item_id, str(dest))  # type: ignore[union-attr]
        _broadcast("queue_updated")
        return JSONResponse({"status": "rejected", "resolved_path": str(dest)})

    @app.get("/api/events")
    async def events(request: Request) -> StreamingResponse:
        q: asyncio.Queue = asyncio.Queue()
        _sse_clients.append(q)

        async def generate():
            try:
                while True:
                    if await request.is_disconnected():
                        break
                    try:
                        event = await asyncio.wait_for(q.get(), timeout=20)
                        yield f"event: {event}\ndata: {{}}\n\n"
                    except asyncio.TimeoutError:
                        yield ": keepalive\n\n"
            finally:
                if q in _sse_clients:
                    _sse_clients.remove(q)

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app


# ------------------------------------------------------------------
# File watcher + SSE broadcast
# ------------------------------------------------------------------


def _start_file_watcher() -> None:
    from watchdog.events import FileSystemEventHandler
    from watchdog.observers import Observer

    log_dir = _cfg.log_file.parent  # type: ignore[union-attr]
    log_dir.mkdir(parents=True, exist_ok=True)

    class _Handler(FileSystemEventHandler):
        def on_modified(self, event) -> None:  # type: ignore[override]
            if event.is_directory:
                return
            name = Path(event.src_path).name
            if name == _cfg.log_file.name:  # type: ignore[union-attr]
                _broadcast("log_updated")
            elif name == "review_queue.json":
                _broadcast("queue_updated")

    observer = Observer()
    observer.schedule(_Handler(), str(log_dir), recursive=False)
    observer.daemon = True  # type: ignore[assignment]
    observer.start()


def _broadcast(event_type: str) -> None:
    if _loop is None:
        return

    async def _push() -> None:
        for q in list(_sse_clients):
            await q.put(event_type)

    asyncio.run_coroutine_threadsafe(_push(), _loop)


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------


def run(cfg: "Config", store: "ReviewStore", port: int, open_browser: bool) -> None:
    import webbrowser

    import uvicorn

    app = create_app(cfg, store)

    if open_browser:
        import threading
        def _open():
            import time
            time.sleep(1.0)
            webbrowser.open(f"http://localhost:{port}")
        threading.Thread(target=_open, daemon=True).start()

    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
