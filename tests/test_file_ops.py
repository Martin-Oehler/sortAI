"""Unit tests for sortai.file_ops."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from sortai.file_ops import (
    _html_path,
    log_decision,
    move_file,
    render_html_report,
)


# ---------------------------------------------------------------------------
# move_file
# ---------------------------------------------------------------------------


class TestMoveFile:
    def test_moves_file_to_destination(self, tmp_path: Path):
        src = tmp_path / "src" / "doc.pdf"
        src.parent.mkdir()
        src.write_bytes(b"%PDF")
        dest_dir = tmp_path / "archive"

        result = move_file(src, dest_dir, "doc.pdf", dry_run=False)

        assert result == dest_dir / "doc.pdf"
        assert result.exists()
        assert not src.exists()

    def test_creates_dest_dir_if_missing(self, tmp_path: Path):
        src = tmp_path / "doc.pdf"
        src.write_bytes(b"%PDF")
        dest_dir = tmp_path / "new" / "nested"

        move_file(src, dest_dir, "doc.pdf", dry_run=False)

        assert dest_dir.is_dir()

    def test_dry_run_does_not_move(self, tmp_path: Path):
        src = tmp_path / "doc.pdf"
        src.write_bytes(b"%PDF")
        dest_dir = tmp_path / "archive"

        result = move_file(src, dest_dir, "doc.pdf", dry_run=True)

        assert src.exists()
        assert not dest_dir.exists()
        assert result == dest_dir / "doc.pdf"

    def test_dry_run_returns_would_be_dest(self, tmp_path: Path):
        src = tmp_path / "doc.pdf"
        src.write_bytes(b"%PDF")
        dest_dir = tmp_path / "out"

        result = move_file(src, dest_dir, "report.pdf", dry_run=True)

        assert result == dest_dir / "report.pdf"

    def test_collision_renames_with_counter(self, tmp_path: Path):
        dest_dir = tmp_path / "out"
        dest_dir.mkdir()
        (dest_dir / "doc.pdf").write_bytes(b"%PDF")

        src = tmp_path / "doc.pdf"
        src.write_bytes(b"%PDF-new")

        result = move_file(src, dest_dir, "doc.pdf", dry_run=False)

        assert result == dest_dir / "doc_2.pdf"
        assert result.exists()

    def test_collision_increments_until_free(self, tmp_path: Path):
        dest_dir = tmp_path / "out"
        dest_dir.mkdir()
        (dest_dir / "doc.pdf").write_bytes(b"%PDF")
        (dest_dir / "doc_2.pdf").write_bytes(b"%PDF")

        src = tmp_path / "doc.pdf"
        src.write_bytes(b"new")

        result = move_file(src, dest_dir, "doc.pdf", dry_run=False)

        assert result == dest_dir / "doc_3.pdf"

    def test_returns_final_dest_path(self, tmp_path: Path):
        src = tmp_path / "file.pdf"
        src.write_bytes(b"%PDF")
        dest_dir = tmp_path / "out"

        result = move_file(src, dest_dir, "file.pdf", dry_run=False)

        assert isinstance(result, Path)
        assert result.name == "file.pdf"


# ---------------------------------------------------------------------------
# log_decision
# ---------------------------------------------------------------------------


class TestLogDecision:
    def _call(self, tmp_path: Path, **kwargs) -> Path:
        log_path = tmp_path / "logs" / "sortai.jsonl"
        defaults = dict(
            src=tmp_path / "inbox" / "doc.pdf",
            dest=tmp_path / "archive" / "doc.pdf",
            summary="Test summary",
            dry_run=False,
            log_path=log_path,
        )
        defaults.update(kwargs)
        log_decision(**defaults)
        return log_path

    def test_creates_log_file_if_missing(self, tmp_path: Path):
        log_path = self._call(tmp_path)
        assert log_path.exists()

    def test_appends_json_lines_entry(self, tmp_path: Path):
        log_path = self._call(tmp_path)
        lines = [l for l in log_path.read_text().splitlines() if l.strip()]
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert isinstance(entry, dict)

    def test_entry_has_all_required_fields(self, tmp_path: Path):
        log_path = self._call(tmp_path)
        entry = json.loads(log_path.read_text().strip())
        for field in ("timestamp", "original_path", "new_path", "summary", "dry_run"):
            assert field in entry

    def test_dry_run_recorded_in_entry(self, tmp_path: Path):
        log_path = self._call(tmp_path, dry_run=True)
        entry = json.loads(log_path.read_text().strip())
        assert entry["dry_run"] is True

    def test_multiple_calls_append_multiple_lines(self, tmp_path: Path):
        log_path = tmp_path / "logs" / "sortai.jsonl"
        src = tmp_path / "a.pdf"
        dest = tmp_path / "b.pdf"
        for _ in range(3):
            log_decision(src=src, dest=dest, summary="s", dry_run=False, log_path=log_path)
        lines = [l for l in log_path.read_text().splitlines() if l.strip()]
        assert len(lines) == 3

    def test_renders_html_after_logging(self, tmp_path: Path):
        log_path = self._call(tmp_path)
        html_path = log_path.with_name(log_path.stem + "_report.html")
        assert html_path.exists()

    def test_summary_present_in_entry(self, tmp_path: Path):
        log_path = self._call(tmp_path, summary="Invoice from Acme Corp")
        entry = json.loads(log_path.read_text().strip())
        assert entry["summary"] == "Invoice from Acme Corp"


# ---------------------------------------------------------------------------
# render_html_report
# ---------------------------------------------------------------------------


class TestRenderHtmlReport:
    def _make_log(self, tmp_path: Path, entries: list[dict]) -> Path:
        log_path = tmp_path / "sortai.jsonl"
        with log_path.open("w", encoding="utf-8") as fh:
            for e in entries:
                fh.write(json.dumps(e) + "\n")
        return log_path

    def test_creates_html_file(self, tmp_path: Path):
        log_path = self._make_log(tmp_path, [])
        render_html_report(log_path)
        assert _html_path(log_path).exists()

    def test_html_contains_entry_data(self, tmp_path: Path):
        entry = {
            "timestamp": "2026-04-22T12:00:00",
            "original_path": str(tmp_path / "inbox" / "invoice.pdf"),
            "new_path": str(tmp_path / "archive" / "invoice_acme.pdf"),
            "summary": "Acme Corp invoice April 2026",
            "dry_run": False,
        }
        log_path = self._make_log(tmp_path, [entry])
        render_html_report(log_path)
        html = _html_path(log_path).read_text(encoding="utf-8")
        assert "invoice.pdf" in html
        assert "invoice_acme.pdf" in html
        assert "Acme Corp invoice April 2026" in html

    def test_html_is_valid_structure(self, tmp_path: Path):
        log_path = self._make_log(tmp_path, [])
        render_html_report(log_path)
        html = _html_path(log_path).read_text(encoding="utf-8")
        assert html.strip().startswith("<!DOCTYPE html>")
        assert "</html>" in html

    def test_handles_empty_log_gracefully(self, tmp_path: Path):
        log_path = self._make_log(tmp_path, [])
        render_html_report(log_path)
        html = _html_path(log_path).read_text(encoding="utf-8")
        assert "Total: <strong>0</strong>" in html

    def test_handles_corrupt_jsonl_line_gracefully(self, tmp_path: Path):
        log_path = tmp_path / "sortai.jsonl"
        good = {"timestamp": "2026-04-22T10:00:00", "original_path": "/a.pdf",
                "new_path": "/b.pdf", "summary": "ok", "dry_run": False}
        log_path.write_text("not valid json\n" + json.dumps(good) + "\n", encoding="utf-8")
        render_html_report(log_path)
        html = _html_path(log_path).read_text(encoding="utf-8")
        assert "Total: <strong>1</strong>" in html

    def test_dry_run_row_has_dry_run_class(self, tmp_path: Path):
        entry = {
            "timestamp": "2026-04-22T10:00:00",
            "original_path": "/inbox/x.pdf",
            "new_path": "/archive/x.pdf",
            "summary": "dry summary",
            "dry_run": True,
        }
        log_path = self._make_log(tmp_path, [entry])
        render_html_report(log_path)
        html = _html_path(log_path).read_text(encoding="utf-8")
        assert "dry-run" in html
        assert ">Yes<" in html

    def test_destination_link_uses_file_uri(self, tmp_path: Path):
        dest = tmp_path / "archive" / "out.pdf"
        entry = {
            "timestamp": "2026-04-22T10:00:00",
            "original_path": str(tmp_path / "in.pdf"),
            "new_path": str(dest),
            "summary": "s",
            "dry_run": False,
        }
        log_path = self._make_log(tmp_path, [entry])
        render_html_report(log_path)
        html = _html_path(log_path).read_text(encoding="utf-8")
        assert 'href="file:///' in html

    def test_html_escapes_special_chars_in_summary(self, tmp_path: Path):
        entry = {
            "timestamp": "2026-04-22T10:00:00",
            "original_path": "/a.pdf",
            "new_path": "/b.pdf",
            "summary": "Invoice <Acme & Sons> \"2026\"",
            "dry_run": False,
        }
        log_path = self._make_log(tmp_path, [entry])
        render_html_report(log_path)
        html = _html_path(log_path).read_text(encoding="utf-8")
        assert "<Acme" not in html
        assert "&lt;Acme" in html


# ---------------------------------------------------------------------------
# _html_path
# ---------------------------------------------------------------------------


class TestHtmlPath:
    def test_derived_correctly(self, tmp_path: Path):
        log = tmp_path / "logs" / "sortai.jsonl"
        assert _html_path(log) == tmp_path / "logs" / "sortai_report.html"

    def test_different_stem(self, tmp_path: Path):
        log = tmp_path / "mylog.jsonl"
        assert _html_path(log).name == "mylog_report.html"

    def test_file_uri_starts_with_triple_slash(self, tmp_path: Path):
        p = tmp_path / "test.pdf"
        uri = p.as_uri()
        assert uri.startswith("file:///")
        assert "test.pdf" in uri
