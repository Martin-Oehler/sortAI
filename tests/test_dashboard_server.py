"""Tests for sortai.dashboard_server — FastAPI routes via TestClient."""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from sortai.config import Config
from sortai.dashboard_server import _run_learning, create_app
from sortai.review_store import ReviewItem, ReviewStore, make_review_item


@pytest.fixture()
def cfg(tmp_path: Path) -> Config:
    inbox = tmp_path / "inbox"
    archive = tmp_path / "archive"
    inbox.mkdir()
    archive.mkdir()
    return Config(inbox=inbox, archive=archive, log_file=tmp_path / "logs" / "sortai.jsonl")


@pytest.fixture()
def store(cfg: Config) -> ReviewStore:
    return ReviewStore(cfg.queue_path)


@pytest.fixture()
def client(cfg: Config, store: ReviewStore) -> TestClient:
    # No context manager: lifespan (watchdog observer) is not started; the
    # SSE broadcast no-ops because app.state.loop stays None.
    return TestClient(create_app(cfg, store))


def _add_staged_item(cfg: Config, store: ReviewStore, name: str = "doc.pdf", **kwargs) -> ReviewItem:
    cfg.staging_dir.mkdir(parents=True, exist_ok=True)
    staged = cfg.staging_dir / name
    staged.write_bytes(b"%PDF-1.4 test")
    item = make_review_item(
        original_filename=name,
        staging_path=staged,
        proposed_folder="invoices/2026",
        proposed_filename="2026-01-01_invoice.pdf",
        summary="an invoice",
        interactions=[],
        **kwargs,
    )
    store.add(item)
    return item


def _wait_for(predicate, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return False


# ---------------------------------------------------------------------------
# Log and queue endpoints
# ---------------------------------------------------------------------------


class TestLogAndQueue:
    def test_queue_empty(self, client: TestClient):
        resp = client.get("/api/queue")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_queue_returns_item_dicts(self, cfg, store, client):
        item = _add_staged_item(cfg, store)
        data = client.get("/api/queue").json()
        assert len(data) == 1
        assert data[0]["id"] == item.id
        assert data[0]["status"] == "pending"
        assert data[0]["proposed_folder"] == "invoices/2026"

    def test_log_empty_when_no_file(self, client: TestClient):
        assert client.get("/api/log").json() == []

    def test_log_returns_jsonl_entries(self, cfg, client):
        cfg.log_file.parent.mkdir(parents=True, exist_ok=True)
        entries = [
            {"timestamp": "2026-01-01T00:00:00", "original_path": "a.pdf", "new_path": "x/a.pdf"},
            {"timestamp": "2026-01-02T00:00:00", "original_path": "b.pdf", "new_path": "y/b.pdf"},
        ]
        cfg.log_file.write_text(
            "\n".join(json.dumps(e) for e in entries) + "\n", encoding="utf-8"
        )
        assert client.get("/api/log").json() == entries


# ---------------------------------------------------------------------------
# File serving
# ---------------------------------------------------------------------------


class TestFileServing:
    def test_serve_queue_file(self, cfg, store, client):
        item = _add_staged_item(cfg, store)
        resp = client.get(f"/files/queue/{item.id}")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "application/pdf"
        assert resp.content == b"%PDF-1.4 test"

    def test_serve_queue_file_unknown_id(self, client):
        resp = client.get("/files/queue/nope")
        assert resp.status_code == 404
        assert resp.json()["detail"] == "Item not found"

    def test_serve_log_file(self, cfg, client):
        pdf = cfg.archive / "sorted.pdf"
        pdf.write_bytes(b"%PDF-1.4 sorted")
        cfg.log_file.parent.mkdir(parents=True, exist_ok=True)
        cfg.log_file.write_text(json.dumps({"new_path": str(pdf)}) + "\n", encoding="utf-8")
        resp = client.get("/files/log/0")
        assert resp.status_code == 200
        assert resp.content == b"%PDF-1.4 sorted"

    def test_serve_log_file_index_out_of_range(self, client):
        resp = client.get("/files/log/99")
        assert resp.status_code == 404
        assert resp.json()["detail"] == "Log entry not found"


# ---------------------------------------------------------------------------
# Accept / reject flow
# ---------------------------------------------------------------------------


class TestAcceptReject:
    def test_accept_moves_file_and_logs(self, cfg, store, client):
        item = _add_staged_item(cfg, store)
        resp = client.post(f"/api/accept/{item.id}")
        assert resp.status_code == 200
        dest = cfg.archive / "invoices" / "2026" / "2026-01-01_invoice.pdf"
        assert resp.json() == {"status": "accepted", "resolved_path": str(dest)}
        assert dest.exists()
        assert not Path(item.staging_path).exists()
        assert store.get(item.id).status == "accepted"
        log = client.get("/api/log").json()
        assert len(log) == 1
        assert log[0]["new_path"] == str(dest)
        assert log[0]["summary"] == "an invoice"

    def test_accept_non_pending_item_is_400(self, cfg, store, client):
        item = _add_staged_item(cfg, store)
        assert client.post(f"/api/accept/{item.id}").status_code == 200
        resp = client.post(f"/api/accept/{item.id}")
        assert resp.status_code == 400
        assert resp.json()["detail"] == "Item is not pending"

    def test_accept_unknown_item_is_404(self, client):
        assert client.post("/api/accept/nope").status_code == 404

    def test_reject_moves_file_to_rejected_dir(self, cfg, store, client):
        item = _add_staged_item(cfg, store)
        resp = client.post(f"/api/reject/{item.id}")
        assert resp.status_code == 200
        dest = cfg.rejected_dir / item.original_filename
        assert resp.json() == {"status": "rejected", "resolved_path": str(dest)}
        assert dest.exists()
        assert store.get(item.id).status == "rejected"

    def test_reject_unknown_item_is_404(self, client):
        assert client.post("/api/reject/nope").status_code == 404


# ---------------------------------------------------------------------------
# Memory endpoints
# ---------------------------------------------------------------------------


class TestMemoryEndpoints:
    def test_get_memory_no_file(self, client):
        assert client.get("/api/memory").json() == {"rules": []}

    def test_get_memory_returns_rules(self, cfg, client):
        from sortai.memory import save_rules
        save_rules(cfg.memory_path, ["rule one", "rule two"])
        assert client.get("/api/memory").json() == {"rules": ["rule one", "rule two"]}

    def test_delete_memory_rule(self, cfg, client):
        from sortai.memory import load_rules, save_rules
        save_rules(cfg.memory_path, ["rule one", "rule two"])
        resp = client.delete("/api/memory/0")
        assert resp.status_code == 200
        assert resp.json() == {"rules": ["rule two"]}
        assert load_rules(cfg.memory_path) == ["rule two"]

    def test_delete_memory_rule_out_of_range(self, cfg, client):
        from sortai.memory import save_rules
        save_rules(cfg.memory_path, ["rule one"])
        assert client.delete("/api/memory/5").status_code == 404

    def test_delete_memory_rule_no_file(self, client):
        resp = client.delete("/api/memory/0")
        assert resp.status_code == 404
        assert resp.json()["detail"] == "No memory file"


# ---------------------------------------------------------------------------
# Source resolution (/reveal and /api/reprocess error paths)
# ---------------------------------------------------------------------------


class TestSourceResolution:
    def test_reveal_invalid_type_is_400(self, client):
        resp = client.post("/reveal", json={"type": "bogus"})
        assert resp.status_code == 400
        assert resp.json()["detail"] == "Invalid item type"

    def test_reveal_unknown_queue_item_is_404(self, client):
        resp = client.post("/reveal", json={"type": "queue", "id": "nope"})
        assert resp.status_code == 404
        assert resp.json()["detail"] == "Item not found"

    def test_reveal_queue_item_without_path_is_404(self, cfg, store, client):
        item = _add_staged_item(cfg, store)
        # Simulate an item that finished without a resolved path.
        item.status = "accepted"
        store.mark_accepted(item.id, "")
        resp = client.post("/reveal", json={"type": "queue", "id": item.id})
        assert resp.status_code == 404
        assert resp.json()["detail"] == "No file path"

    def test_reveal_target_queue_item_without_path_is_404(self, cfg, store, client):
        item = _add_staged_item(cfg, store)
        item.status = "accepted"
        store.mark_accepted(item.id, "")
        resp = client.post("/reveal-target", json={"type": "queue", "id": item.id})
        assert resp.status_code == 404
        assert resp.json()["detail"] == "No target path"

    def test_reveal_missing_file_on_disk_is_404(self, cfg, store, client):
        item = _add_staged_item(cfg, store)
        Path(item.staging_path).unlink()
        resp = client.post("/reveal", json={"type": "queue", "id": item.id})
        assert resp.status_code == 404
        assert resp.json()["detail"] == "File not found on disk"

    def test_reprocess_invalid_type_is_400(self, client):
        resp = client.post("/api/reprocess", json={"type": "bogus"})
        assert resp.status_code == 400
        assert resp.json()["detail"] == "Invalid item type"

    def test_reprocess_unknown_queue_item_is_404(self, client):
        resp = client.post("/api/reprocess", json={"type": "queue", "id": "nope"})
        assert resp.status_code == 404

    def test_reprocess_log_index_out_of_range_is_404(self, client):
        resp = client.post("/api/reprocess", json={"type": "log", "log_idx": 3})
        assert resp.status_code == 404
        assert resp.json()["detail"] == "Log entry not found"

    def test_reprocess_missing_file_on_disk_is_404(self, cfg, store, client):
        item = _add_staged_item(cfg, store)
        Path(item.staging_path).unlink()
        resp = client.post("/api/reprocess", json={"type": "queue", "id": item.id})
        assert resp.status_code == 404
        assert resp.json()["detail"] == "File not found on disk"


# ---------------------------------------------------------------------------
# Reprocess pipeline failure: error is logged and item returns to pending
# ---------------------------------------------------------------------------


class TestReprocessFailure:
    def test_pipeline_exception_logged_and_item_restored(self, cfg, store, client, capsys):
        item = _add_staged_item(cfg, store)
        mock_process = MagicMock(side_effect=RuntimeError("LM Studio unreachable"))

        with patch("sortai.llm_client.LMStudioClient.from_config", return_value=MagicMock()), \
             patch("sortai.processor.process_document", mock_process):
            resp = client.post("/api/reprocess", json={"type": "queue", "id": item.id})
            assert resp.status_code == 202
            assert resp.json() == {"status": "reprocessing"}
            assert _wait_for(lambda: mock_process.called)
            assert _wait_for(lambda: store.get(item.id).status == "pending")

        assert "reprocessing of doc.pdf failed" in capsys.readouterr().err

    def test_error_outcome_restores_item(self, cfg, store, client):
        from sortai.processor import Outcome
        item = _add_staged_item(cfg, store)
        outcome = Outcome(status="error", source=Path(item.staging_path), error_reason="unreadable")
        mock_process = MagicMock(return_value=outcome)

        with patch("sortai.llm_client.LMStudioClient.from_config", return_value=MagicMock()), \
             patch("sortai.processor.process_document", mock_process):
            resp = client.post("/api/reprocess", json={"type": "queue", "id": item.id})
            assert resp.status_code == 202
            assert _wait_for(lambda: store.get(item.id).status == "pending")

        assert mock_process.call_args.kwargs["previous_proposed_folder"] == "invoices/2026"


# ---------------------------------------------------------------------------
# Learning failures are reported, not swallowed
# ---------------------------------------------------------------------------


class TestRunLearning:
    def test_exception_is_logged_and_sem_released(self, cfg, capsys):
        item = make_review_item(
            original_filename="doc.pdf",
            staging_path=cfg.staging_dir / "doc.pdf",
            proposed_folder="invoices",
            proposed_filename="invoice.pdf",
            summary="an invoice",
            interactions=[],
            user_hint="it is an invoice",
            previous_proposed_folder="misc",
        )
        fake_client = MagicMock()
        fake_client.load_model.side_effect = RuntimeError("model load failed")
        sem = threading.Semaphore(1)
        events: list[str] = []

        with patch("sortai.llm_client.LMStudioClient.from_config", return_value=fake_client):
            _run_learning(item, str(cfg.archive / "missing.pdf"), cfg, sem, events.append)

        assert "learning from correction of doc.pdf failed" in capsys.readouterr().err
        assert sem.acquire(blocking=False)  # released despite the failure
        sem.release()
        assert events == []  # nothing broadcast on failure


# ---------------------------------------------------------------------------
# App instances are independent (no module-level state)
# ---------------------------------------------------------------------------


class TestAppIsolation:
    def test_two_apps_do_not_share_state(self, tmp_path: Path):
        def make(name: str):
            inbox = tmp_path / name / "inbox"
            archive = tmp_path / name / "archive"
            inbox.mkdir(parents=True)
            archive.mkdir(parents=True)
            cfg = Config(inbox=inbox, archive=archive, log_file=tmp_path / name / "logs" / "sortai.jsonl")
            store = ReviewStore(cfg.queue_path)
            return cfg, store, create_app(cfg, store)

        cfg1, store1, app1 = make("one")
        cfg2, store2, app2 = make("two")
        client1, client2 = TestClient(app1), TestClient(app2)

        item = _add_staged_item(cfg1, store1)
        assert [i["id"] for i in client1.get("/api/queue").json()] == [item.id]
        assert client2.get("/api/queue").json() == []

        # Per-app state objects are distinct.
        assert app1.state.sse_clients is not app2.state.sse_clients
        assert app1.state.pipeline_sem is not app2.state.pipeline_sem

    def test_lifespan_sets_loop_and_serves_index(self, cfg, store):
        app = create_app(cfg, store)
        with TestClient(app) as client:
            assert app.state.loop is not None
            resp = client.get("/")
            assert resp.status_code == 200
            assert b"<html" in resp.content.lower()
