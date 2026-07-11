"""Unit tests for sortai.file_ops."""
from __future__ import annotations

import json
from pathlib import Path

from sortai.file_ops import (
    load_jsonl_entries,
    log_decision,
    move_file,
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
        for field in ("timestamp", "original_path", "new_path", "summary", "dry_run", "interactions"):
            assert field in entry

    def test_interactions_stored_in_entry(self, tmp_path: Path):
        interaction = {"stage": "summarize", "step": 0, "prompt": "p", "answer": "a", "reasoning": "r"}
        log_path = self._call(tmp_path, interactions=[interaction])
        entry = json.loads(log_path.read_text().strip())
        assert entry["interactions"] == [interaction]

    def test_interactions_defaults_to_empty_list(self, tmp_path: Path):
        log_path = self._call(tmp_path)
        entry = json.loads(log_path.read_text().strip())
        assert entry["interactions"] == []

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
# load_jsonl_entries
# ---------------------------------------------------------------------------


class TestLoadJsonlEntries:
    def test_returns_empty_list_for_missing_file(self, tmp_path: Path):
        assert load_jsonl_entries(tmp_path / "missing.jsonl") == []

    def test_parses_valid_entries(self, tmp_path: Path):
        log = tmp_path / "log.jsonl"
        log.write_text('{"a": 1}\n{"b": 2}\n', encoding="utf-8")
        assert load_jsonl_entries(log) == [{"a": 1}, {"b": 2}]

    def test_skips_corrupt_lines(self, tmp_path: Path):
        log = tmp_path / "log.jsonl"
        log.write_text('bad json\n{"ok": true}\n', encoding="utf-8")
        assert load_jsonl_entries(log) == [{"ok": True}]

    def test_skips_blank_lines(self, tmp_path: Path):
        log = tmp_path / "log.jsonl"
        log.write_text('\n{"x": 1}\n\n', encoding="utf-8")
        assert load_jsonl_entries(log) == [{"x": 1}]
