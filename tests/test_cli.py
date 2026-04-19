"""Tests for sortai CLI commands: extract and tree."""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
from click.testing import CliRunner

from sortai.cli import main

PROJECT_ROOT = Path(__file__).parent.parent
TEST_INBOX = PROJECT_ROOT / "tests" / "fixtures" / "inbox"
TEST_ARCHIVE = PROJECT_ROOT / "tests" / "fixtures" / "archive"
REAL_PDF = TEST_INBOX / "bank_statement.pdf"


@pytest.fixture()
def config_file(tmp_path: Path) -> Path:
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


class TestExtractCommand:
    def test_exits_zero(self, config_file: Path) -> None:
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["--config", str(config_file), "extract", str(REAL_PDF)],
            catch_exceptions=False,
        )
        assert result.exit_code == 0, result.output

    def test_output_contains_account_statement(self, config_file: Path) -> None:
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["--config", str(config_file), "extract", str(REAL_PDF)],
            catch_exceptions=False,
        )
        assert "Account Statement" in result.output

    def test_output_does_not_contain_archive_listing(self, config_file: Path) -> None:
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["--config", str(config_file), "extract", str(REAL_PDF)],
            catch_exceptions=False,
        )
        assert "Banken" not in result.output
        assert "Gesundheit" not in result.output

    def test_max_chars_flag_limits_output(self, config_file: Path) -> None:
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["--config", str(config_file), "extract", str(REAL_PDF), "-n", "10"],
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        assert "omitted" in result.output


class TestTreeCommand:
    def test_exits_zero(self, config_file: Path) -> None:
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["--config", str(config_file), "tree"],
            catch_exceptions=False,
        )
        assert result.exit_code == 0, result.output

    def test_output_contains_bank(self, config_file: Path) -> None:
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["--config", str(config_file), "tree"],
            catch_exceptions=False,
        )
        assert "bank" in result.output

    def test_output_contains_invoices(self, config_file: Path) -> None:
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["--config", str(config_file), "tree"],
            catch_exceptions=False,
        )
        assert "invoices" in result.output
