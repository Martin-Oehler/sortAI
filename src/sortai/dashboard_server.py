"""FastAPI dashboard server — audit log viewer and review interface."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

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

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        global _loop
        _loop = asyncio.get_running_loop()
        observer = _start_file_watcher()
        try:
            yield
        finally:
            observer.stop()
            observer.join(timeout=2)

    app = FastAPI(title="sortAI Dashboard", lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")

    # ------------------------------------------------------------------
    # Routes
    # ------------------------------------------------------------------

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(Path(__file__).parent / "static" / "index.html")

    @app.get("/api/queue")
    def get_queue() -> list:
        _store.reload()  # type: ignore[union-attr]
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

    @app.post("/reveal")
    async def reveal_file(request: Request) -> JSONResponse:
        import platform
        import subprocess

        data = await request.json()
        item_type = data.get("type")
        item_id = data.get("id")
        log_idx = data.get("log_idx")

        if item_type == "queue":
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
        elif item_type == "log":
            from sortai.file_ops import load_jsonl_entries
            entries = load_jsonl_entries(_cfg.log_file)  # type: ignore[union-attr]
            if log_idx is None or log_idx < 0 or log_idx >= len(entries):
                raise HTTPException(status_code=404, detail="Log entry not found")
            entry = entries[log_idx]
            new_path = entry.get("new_path", "")
            if not new_path:
                raise HTTPException(status_code=404, detail="No file path in log entry")
            path = Path(new_path)
        else:
            raise HTTPException(status_code=400, detail="Invalid item type")

        path = path.resolve()
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

    @app.post("/api/reprocess")
    async def reprocess_file(request: Request) -> JSONResponse:
        import shutil
        import threading
        import uuid as _uuid

        data = await request.json()
        item_type = data.get("type")
        item_id = data.get("id")
        log_idx = data.get("log_idx")
        hint = (data.get("hint") or "").strip() or None

        if item_type == "queue":
            try:
                item = _store.get(item_id)  # type: ignore[union-attr]
            except KeyError:
                raise HTTPException(status_code=404, detail="Item not found")
            if item.status == "pending":
                source_path = Path(item.staging_path)
            elif item.resolved_path:
                source_path = Path(item.resolved_path)
            else:
                raise HTTPException(status_code=404, detail="No file path")
            original_filename = item.original_filename
        elif item_type == "log":
            from sortai.file_ops import load_jsonl_entries
            entries = load_jsonl_entries(_cfg.log_file)  # type: ignore[union-attr]
            if log_idx is None or log_idx < 0 or log_idx >= len(entries):
                raise HTTPException(status_code=404, detail="Log entry not found")
            entry = entries[log_idx]
            new_path = entry.get("new_path", "")
            if not new_path:
                raise HTTPException(status_code=404, detail="No file path in log entry")
            source_path = Path(new_path)
            original_filename = source_path.name
        else:
            raise HTTPException(status_code=400, detail="Invalid item type")

        source_path = source_path.resolve()
        if not source_path.exists():
            raise HTTPException(status_code=404, detail="File not found on disk")

        staging_dir = (
            _cfg.dashboard.staging_dir  # type: ignore[union-attr]
            if _cfg.dashboard.staging_dir  # type: ignore[union-attr]
            else _cfg.inbox.parent / "_review"  # type: ignore[union-attr]
        )
        staging_dir.mkdir(parents=True, exist_ok=True)
        staged_path = staging_dir / f"{_uuid.uuid4()}_{original_filename}"
        shutil.copy2(source_path, staged_path)

        def _run_pipeline() -> None:
            from sortai.llm_client import LMStudioClient
            from sortai.pipeline import ClassificationError, Pipeline
            from sortai.review_store import make_review_item

            client = LMStudioClient(
                base_url=_cfg.lm_studio.base_url,  # type: ignore[union-attr]
                model_name=_cfg.lm_studio.model,  # type: ignore[union-attr]
                prompts_dir=_cfg.prompts_dir,  # type: ignore[union-attr]
                temperature=_cfg.lm_studio.temperature,  # type: ignore[union-attr]
                max_tokens=_cfg.lm_studio.max_tokens,  # type: ignore[union-attr]
                context_length=_cfg.lm_studio.context_length,  # type: ignore[union-attr]
            )
            try:
                with client:
                    pipeline = Pipeline(_cfg, client)  # type: ignore[arg-type]
                    target_folder, filename, summary, interactions = pipeline.run(staged_path, user_hint=hint)
            except ClassificationError:
                staged_path.unlink(missing_ok=True)
                return
            except Exception:
                staged_path.unlink(missing_ok=True)
                return

            rel_folder = str(target_folder.relative_to(_cfg.archive))  # type: ignore[union-attr]
            new_item = make_review_item(
                original_filename=original_filename,
                staging_path=staged_path,
                proposed_folder=rel_folder,
                proposed_filename=filename,
                summary=summary,
                interactions=interactions,
            )
            _store.add(new_item)  # type: ignore[union-attr]
            _broadcast("queue_updated")

        threading.Thread(target=_run_pipeline, daemon=True).start()
        from fastapi.responses import JSONResponse as _JSONResponse
        return _JSONResponse({"status": "reprocessing"}, status_code=202)

    @app.post("/api/accept/{item_id}")
    def accept_item(item_id: str) -> JSONResponse:
        _store.reload()  # type: ignore[union-attr]
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
            interactions=item.interactions,
        )
        _store.mark_accepted(item_id, str(dest))  # type: ignore[union-attr]
        _broadcast("queue_updated")
        return JSONResponse({"status": "accepted", "resolved_path": str(dest)})

    @app.post("/api/reject/{item_id}")
    def reject_item(item_id: str) -> JSONResponse:
        _store.reload()  # type: ignore[union-attr]
        try:
            item = _store.get(item_id)  # type: ignore[union-attr]
        except KeyError:
            raise HTTPException(status_code=404, detail="Item not found")
        if item.status != "pending":
            raise HTTPException(status_code=400, detail="Item is not pending")

        from sortai.file_ops import move_file
        rejected_dir = (
            _cfg.dashboard.rejected_dir  # type: ignore[union-attr]
            if _cfg.dashboard.rejected_dir  # type: ignore[union-attr]
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
                    get_task = asyncio.ensure_future(q.get())
                    disconnect_task = asyncio.ensure_future(request.is_disconnected())
                    done, pending = await asyncio.wait(
                        {get_task, disconnect_task},
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    for t in pending:
                        t.cancel()
                    if disconnect_task in done and disconnect_task.result():
                        break
                    if get_task in done:
                        event = get_task.result()
                        if event is None:  # shutdown sentinel
                            break
                        yield f"event: {event}\ndata: {{}}\n\n"
                    else:
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


def _start_file_watcher():
    from watchdog.events import FileSystemEventHandler
    from watchdog.observers import Observer

    log_dir = _cfg.log_file.parent  # type: ignore[union-attr]
    log_dir.mkdir(parents=True, exist_ok=True)

    class _Handler(FileSystemEventHandler):
        def _notify(self, path: str) -> None:
            name = Path(path).name
            if name == _cfg.log_file.name:  # type: ignore[union-attr]
                _broadcast("log_updated")
            elif name == "review_queue.json":
                _broadcast("queue_updated")

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

    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)

    _original_handle_exit = server.handle_exit

    def _handle_exit(sig, frame):
        if _loop is not None:
            async def _close_sse():
                for q in list(_sse_clients):
                    await q.put(None)
            try:
                asyncio.run_coroutine_threadsafe(_close_sse(), _loop)
            except RuntimeError:
                pass
        _original_handle_exit(sig, frame)

    server.handle_exit = _handle_exit

    try:
        server.run()
    except KeyboardInterrupt:
        pass
