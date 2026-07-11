"""Unit tests for sortai.report."""
from __future__ import annotations

import json
from pathlib import Path

from sortai.report import (
    _html_path,
    dest_label,
    render_html_report,
)


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

    def test_newest_entry_appears_first(self, tmp_path: Path):
        entries = [
            {"timestamp": "2026-01-01T10:00:00", "original_path": "/old.pdf",
             "new_path": "/archive/old.pdf", "summary": "old", "dry_run": False},
            {"timestamp": "2026-04-22T10:00:00", "original_path": "/new.pdf",
             "new_path": "/archive/new.pdf", "summary": "new", "dry_run": False},
        ]
        log_path = self._make_log(tmp_path, entries)
        render_html_report(log_path)
        html = _html_path(log_path).read_text(encoding="utf-8")
        assert html.index("new.pdf") < html.index("old.pdf")

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

    def test_shows_relative_path_when_archive_root_present(self, tmp_path: Path):
        archive = tmp_path / "archive"
        dest = archive / "Finance" / "Invoices" / "acme_2026.pdf"
        entry = {
            "timestamp": "2026-04-22T10:00:00",
            "original_path": str(tmp_path / "inbox" / "acme.pdf"),
            "new_path": str(dest),
            "archive_root": str(archive),
            "summary": "s",
            "dry_run": False,
        }
        log_path = self._make_log(tmp_path, [entry])
        render_html_report(log_path)
        html = _html_path(log_path).read_text(encoding="utf-8")
        assert "Finance/Invoices/acme_2026.pdf" in html

    def test_falls_back_to_filename_without_archive_root(self, tmp_path: Path):
        dest = tmp_path / "archive" / "Finance" / "acme_2026.pdf"
        entry = {
            "timestamp": "2026-04-22T10:00:00",
            "original_path": str(tmp_path / "inbox" / "acme.pdf"),
            "new_path": str(dest),
            "summary": "s",
            "dry_run": False,
        }
        log_path = self._make_log(tmp_path, [entry])
        render_html_report(log_path)
        html = _html_path(log_path).read_text(encoding="utf-8")
        assert "acme_2026.pdf" in html

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
# dest_label
# ---------------------------------------------------------------------------


class TestDestLabel:
    def test_returns_empty_for_empty_path(self):
        assert dest_label("", None) == ""

    def test_returns_filename_without_archive_root(self, tmp_path: Path):
        assert dest_label(str(tmp_path / "a" / "b.pdf"), None) == "b.pdf"

    def test_returns_relative_path_with_archive_root(self, tmp_path: Path):
        archive = tmp_path / "archive"
        path = str(archive / "Finance" / "Invoices" / "acme.pdf")
        assert dest_label(path, str(archive)) == "Finance/Invoices/acme.pdf"

    def test_falls_back_to_filename_when_not_under_archive(self, tmp_path: Path):
        assert dest_label(str(tmp_path / "other" / "doc.pdf"), str(tmp_path / "archive")) == "doc.pdf"

    def test_posix_separators_on_all_platforms(self, tmp_path: Path):
        archive = tmp_path / "archive"
        path = str(archive / "A" / "B" / "c.pdf")
        label = dest_label(path, str(archive))
        assert "\\" not in label


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
