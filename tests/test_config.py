"""Tests for sortai.config — focused on the derived-path properties."""
from __future__ import annotations

from pathlib import Path

import pytest

from sortai.config import Config, DashboardConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_cfg(tmp_path: Path, **overrides) -> Config:
    """Return a Config with inbox/archive under *tmp_path*."""
    inbox = tmp_path / "scans" / "inbox"
    archive = tmp_path / "archive"
    inbox.mkdir(parents=True)
    archive.mkdir(parents=True)
    defaults = dict(
        inbox=inbox,
        archive=archive,
        log_file=tmp_path / "logs" / "sortai.jsonl",
    )
    defaults.update(overrides)
    return Config(**defaults)


# ---------------------------------------------------------------------------
# staging_dir / rejected_dir — dashboard override vs default
# ---------------------------------------------------------------------------

class TestStagingDir:
    def test_default_is_review_sibling_of_inbox(self, tmp_path: Path):
        cfg = make_cfg(tmp_path)
        assert cfg.staging_dir == cfg.inbox.parent / "_review"

    def test_dashboard_override_wins(self, tmp_path: Path):
        custom = tmp_path / "my-staging"
        cfg = make_cfg(tmp_path, dashboard=DashboardConfig(staging_dir=custom))
        assert cfg.staging_dir == custom

    def test_override_does_not_affect_rejected_dir(self, tmp_path: Path):
        custom = tmp_path / "my-staging"
        cfg = make_cfg(tmp_path, dashboard=DashboardConfig(staging_dir=custom))
        assert cfg.rejected_dir == cfg.inbox.parent / "_rejected"


class TestRejectedDir:
    def test_default_is_rejected_sibling_of_inbox(self, tmp_path: Path):
        cfg = make_cfg(tmp_path)
        assert cfg.rejected_dir == cfg.inbox.parent / "_rejected"

    def test_dashboard_override_wins(self, tmp_path: Path):
        custom = tmp_path / "my-rejected"
        cfg = make_cfg(tmp_path, dashboard=DashboardConfig(rejected_dir=custom))
        assert cfg.rejected_dir == custom

    def test_override_does_not_affect_staging_dir(self, tmp_path: Path):
        custom = tmp_path / "my-rejected"
        cfg = make_cfg(tmp_path, dashboard=DashboardConfig(rejected_dir=custom))
        assert cfg.staging_dir == cfg.inbox.parent / "_review"


# ---------------------------------------------------------------------------
# queue_path / memory_path / report_path
# ---------------------------------------------------------------------------

class TestQueuePath:
    def test_lives_next_to_log_file(self, tmp_path: Path):
        cfg = make_cfg(tmp_path)
        assert cfg.queue_path == cfg.log_file.parent / "review_queue.json"

    def test_follows_log_file_location(self, tmp_path: Path):
        cfg = make_cfg(tmp_path, log_file=tmp_path / "elsewhere" / "audit.jsonl")
        assert cfg.queue_path == tmp_path / "elsewhere" / "review_queue.json"


class TestMemoryPath:
    def test_lives_in_archive_root(self, tmp_path: Path):
        cfg = make_cfg(tmp_path)
        assert cfg.memory_path == cfg.archive / "classification-memory.md"


class TestReportPath:
    def test_derived_from_log_file_stem(self, tmp_path: Path):
        cfg = make_cfg(tmp_path)
        assert cfg.report_path == cfg.log_file.parent / "sortai_report.html"

    def test_follows_custom_log_file_name(self, tmp_path: Path):
        cfg = make_cfg(tmp_path, log_file=tmp_path / "logs" / "audit.jsonl")
        assert cfg.report_path == tmp_path / "logs" / "audit_report.html"

    def test_matches_file_ops_html_path(self, tmp_path: Path):
        """Config.report_path must agree with the path file_ops actually writes."""
        from sortai.file_ops import _html_path

        cfg = make_cfg(tmp_path)
        assert cfg.report_path == _html_path(cfg.log_file)


# ---------------------------------------------------------------------------
# Config.load — overrides parsed from TOML reach the derived properties
# ---------------------------------------------------------------------------

class TestLoadDerivedPaths:
    def _write_toml(self, tmp_path: Path, extra: str = "") -> Path:
        inbox = tmp_path / "scans" / "inbox"
        archive = tmp_path / "archive"
        inbox.mkdir(parents=True)
        archive.mkdir(parents=True)
        toml = tmp_path / "config.toml"
        toml.write_text(
            f'inbox = {str(inbox)!r}\n'
            f'archive = {str(archive)!r}\n'
            f'log_file = {str(tmp_path / "logs" / "sortai.jsonl")!r}\n'
            f'[lm_studio]\nmodel = "test-model"\n'
            f'{extra}',
            encoding="utf-8",
        )
        return toml

    def test_defaults_without_dashboard_section(self, tmp_path: Path):
        cfg = Config.load(self._write_toml(tmp_path))
        assert cfg.staging_dir == tmp_path / "scans" / "_review"
        assert cfg.rejected_dir == tmp_path / "scans" / "_rejected"
        assert cfg.queue_path == tmp_path / "logs" / "review_queue.json"
        assert cfg.memory_path == tmp_path / "archive" / "classification-memory.md"
        assert cfg.report_path == tmp_path / "logs" / "sortai_report.html"

    def test_dashboard_overrides_from_toml(self, tmp_path: Path):
        staging = tmp_path / "custom-staging"
        rejected = tmp_path / "custom-rejected"
        extra = (
            f'[dashboard]\n'
            f'staging_dir = {str(staging)!r}\n'
            f'rejected_dir = {str(rejected)!r}\n'
        )
        cfg = Config.load(self._write_toml(tmp_path, extra))
        assert cfg.staging_dir == staging
        assert cfg.rejected_dir == rejected
