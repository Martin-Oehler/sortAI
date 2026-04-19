"""Tests for sortai CLI commands: inspect and tree."""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
from click.testing import CliRunner

from sortai.cli import main

# Resolve project root relative to this file so tests work regardless of CWD.
PROJECT_ROOT = Path(__file__).parent.parent
TEST_INBOX = PROJECT_ROOT / "test" / "inbox"
TEST_ARCHIVE = PROJECT_ROOT / "test" / "archive"
REAL_PDF = TEST_INBOX / "Bank Statement Example Final.pdf"


@pytest.fixture()
def config_file(tmp_path: Path) -> Path:
    """Write a temporary config.toml that points at the real test fixtures.

    Inbox and archive are written as absolute POSIX paths so Config.load()
    finds them regardless of the process CWD.
    """
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        textwrap.dedent(f"""\
            inbox  = {TEST_INBOX.as_posix()!r}
            archive = {TEST_ARCHIVE.as_posix()!r}
            dry_run = true

            [lm_studio]
            model = "test-model"
        """),
        encoding="utf-8",
    )
    return cfg


class TestInspectCommand:
    def test_exits_zero(self, config_file: Path) -> None:
        """inspect exits with code 0 for a valid PDF."""
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["--config", str(config_file), "inspect", str(REAL_PDF)],
            catch_exceptions=False,
        )
        assert result.exit_code == 0, result.output

    def test_output_contains_bank_statement(self, config_file: Path) -> None:
        """inspect output contains text extracted from the PDF."""
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["--config", str(config_file), "inspect", str(REAL_PDF)],
            catch_exceptions=False,
        )
        assert "Bank Statement" in result.output

    def test_output_contains_archive_listing(self, config_file: Path) -> None:
        """inspect output lists the top-level archive directories."""
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["--config", str(config_file), "inspect", str(REAL_PDF)],
            catch_exceptions=False,
        )
        # The archive root has at least Banken and Gesundheit sub-folders.
        assert "Banken" in result.output
        assert "Gesundheit" in result.output

    def test_max_chars_flag_limits_output(self, config_file: Path) -> None:
        """inspect with -n 10 truncates the displayed text."""
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["--config", str(config_file), "inspect", str(REAL_PDF), "-n", "10"],
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        assert "omitted" in result.output


class TestTreeCommand:
    def test_exits_zero(self, config_file: Path) -> None:
        """tree exits with code 0."""
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["--config", str(config_file), "tree"],
            catch_exceptions=False,
        )
        assert result.exit_code == 0, result.output

    def test_output_contains_banken(self, config_file: Path) -> None:
        """tree output contains the 'Banken' archive folder."""
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["--config", str(config_file), "tree"],
            catch_exceptions=False,
        )
        assert "Banken" in result.output

    def test_output_contains_gesundheit(self, config_file: Path) -> None:
        """tree output contains the 'Gesundheit' archive folder."""
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["--config", str(config_file), "tree"],
            catch_exceptions=False,
        )
        assert "Gesundheit" in result.output
