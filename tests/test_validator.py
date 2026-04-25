"""Unit tests for sortai.validator."""

from __future__ import annotations

import json
from datetime import datetime
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from sortai.config import Config, LMStudioConfig
from sortai.validator import (
    SampleEntry,
    SampleSet,
    ValidationResult,
    _compare_folders,
    _run_single,
    load_test_set,
    print_results_table,
    print_score,
    run_validation,
    sample_pdfs,
    write_test_set,
)

# Aliases so test code reads naturally.
TestEntry = SampleEntry
TestSet = SampleSet


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_config(archive: Path) -> Config:
    cfg = Config.__new__(Config)
    cfg.inbox = archive
    cfg.archive = archive
    cfg.prompts_dir = Path("prompts")
    cfg.log_file = Path("logs/sortai.jsonl")
    cfg.dry_run = False
    cfg.max_navigate_depth = 10
    cfg.lm_studio = LMStudioConfig(model="test-model")
    return cfg


def make_pdf(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"%PDF-1.4 fake")
    return path


def rich_console_to_string() -> tuple:
    """Return a (Console, get_output) pair backed by a StringIO."""
    from rich.console import Console

    buf = StringIO()
    c = Console(file=buf, highlight=False, markup=False)
    return c, lambda: buf.getvalue()


# ---------------------------------------------------------------------------
# sample_pdfs
# ---------------------------------------------------------------------------

class TestSamplePdfs:
    def test_samples_n_pdfs(self, tmp_path: Path):
        for i in range(10):
            make_pdf(tmp_path / f"file_{i}.pdf")
        result = sample_pdfs(tmp_path, 5)
        assert result["n"] == 5
        assert len(result["entries"]) == 5

    def test_caps_at_available_pdfs(self, tmp_path: Path):
        make_pdf(tmp_path / "a.pdf")
        make_pdf(tmp_path / "b.pdf")
        result = sample_pdfs(tmp_path, 10)
        assert result["n"] == 2
        assert len(result["entries"]) == 2

    def test_raises_if_no_pdfs(self, tmp_path: Path):
        with pytest.raises(ValueError, match="No PDFs found"):
            sample_pdfs(tmp_path, 5)

    def test_ground_truth_is_relative_posix(self, tmp_path: Path):
        make_pdf(tmp_path / "bank" / "statements" / "jan.pdf")
        result = sample_pdfs(tmp_path, 1)
        assert result["entries"][0]["ground_truth_folder"] == "bank/statements"

    def test_ground_truth_at_root_is_dot(self, tmp_path: Path):
        make_pdf(tmp_path / "doc.pdf")
        result = sample_pdfs(tmp_path, 1)
        assert result["entries"][0]["ground_truth_folder"] == "."

    def test_archive_root_is_absolute(self, tmp_path: Path):
        make_pdf(tmp_path / "doc.pdf")
        result = sample_pdfs(tmp_path, 1)
        assert Path(result["archive_root"]).is_absolute()

    def test_created_at_is_isoformat(self, tmp_path: Path):
        make_pdf(tmp_path / "doc.pdf")
        result = sample_pdfs(tmp_path, 1)
        datetime.fromisoformat(result["created_at"])  # does not raise

    def test_entries_have_required_keys(self, tmp_path: Path):
        make_pdf(tmp_path / "doc.pdf")
        result = sample_pdfs(tmp_path, 1)
        entry = result["entries"][0]
        assert "path" in entry
        assert "ground_truth_folder" in entry

    def test_path_is_absolute(self, tmp_path: Path):
        make_pdf(tmp_path / "sub" / "doc.pdf")
        result = sample_pdfs(tmp_path, 1)
        assert Path(result["entries"][0]["path"]).is_absolute()


# ---------------------------------------------------------------------------
# write_test_set / load_test_set
# ---------------------------------------------------------------------------

class TestWriteAndLoadTestSet:
    def _make_test_set(self) -> TestSet:
        return TestSet(
            archive_root="/archive",
            created_at="2026-04-23T12:00:00",
            n=1,
            entries=[TestEntry(path="/archive/bank/doc.pdf", ground_truth_folder="bank")],
        )

    def test_roundtrip(self, tmp_path: Path):
        ts = self._make_test_set()
        out = tmp_path / "test_set.json"
        write_test_set(ts, out)
        loaded = load_test_set(out)
        assert loaded == ts

    def test_output_is_valid_json(self, tmp_path: Path):
        out = tmp_path / "test_set.json"
        write_test_set(self._make_test_set(), out)
        data = json.loads(out.read_text())
        assert "archive_root" in data
        assert "entries" in data
        assert "created_at" in data
        assert "n" in data

    def test_load_raises_on_missing_file(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            load_test_set(tmp_path / "nonexistent.json")


# ---------------------------------------------------------------------------
# _compare_folders
# ---------------------------------------------------------------------------

class TestCompareFolders:
    def test_exact_match(self, tmp_path: Path):
        predicted = tmp_path / "bank" / "statements"
        exact, prefix = _compare_folders(predicted, "bank/statements", tmp_path)
        assert exact is True
        assert prefix is True

    def test_prefix_match_predicted_deeper(self, tmp_path: Path):
        predicted = tmp_path / "bank" / "statements" / "2024"
        exact, prefix = _compare_folders(predicted, "bank/statements", tmp_path)
        assert exact is False
        assert prefix is True

    def test_prefix_match_gt_deeper(self, tmp_path: Path):
        predicted = tmp_path / "bank"
        exact, prefix = _compare_folders(predicted, "bank/statements", tmp_path)
        assert exact is False
        assert prefix is True

    def test_no_match(self, tmp_path: Path):
        predicted = tmp_path / "invoices"
        exact, prefix = _compare_folders(predicted, "bank/statements", tmp_path)
        assert exact is False
        assert prefix is False

    def test_root_exact(self, tmp_path: Path):
        predicted = tmp_path
        exact, prefix = _compare_folders(predicted, ".", tmp_path)
        assert exact is True
        assert prefix is True

    def test_predicted_outside_archive(self, tmp_path: Path):
        other = tmp_path.parent / "other"
        exact, prefix = _compare_folders(other, "bank", tmp_path)
        assert exact is False
        assert prefix is False


# ---------------------------------------------------------------------------
# _run_single
# ---------------------------------------------------------------------------

class TestRunSingle:
    def _make_entry(self, path: str, gt: str) -> TestEntry:
        return TestEntry(path=path, ground_truth_folder=gt)

    def test_success_returns_result(self, tmp_path: Path):
        target = tmp_path / "bank" / "statements"
        pipeline = MagicMock()
        pipeline.run.return_value = (target, "doc.pdf", "a summary", [])
        entry = self._make_entry(str(tmp_path / "doc.pdf"), "bank/statements")

        result = _run_single(entry, tmp_path, pipeline)

        assert result["exact_match"] is True
        assert result["predicted_folder"] == "bank/statements"
        assert result["summary"] == "a summary"
        assert result["error"] == ""
        assert result["interactions"] == []

    def test_error_returns_error_result(self, tmp_path: Path):
        pipeline = MagicMock()
        pipeline.run.side_effect = RuntimeError("LLM unavailable")
        entry = self._make_entry(str(tmp_path / "doc.pdf"), "bank")

        result = _run_single(entry, tmp_path, pipeline)

        assert result["exact_match"] is False
        assert result["prefix_match"] is False
        assert result["predicted_folder"] == ""
        assert "LLM unavailable" in result["error"]
        assert result["interactions"] == []

    def test_prefix_match_detected(self, tmp_path: Path):
        target = tmp_path / "bank"
        pipeline = MagicMock()
        pipeline.run.return_value = (target, "doc.pdf", "summary", [])
        entry = self._make_entry(str(tmp_path / "doc.pdf"), "bank/statements")

        result = _run_single(entry, tmp_path, pipeline)

        assert result["exact_match"] is False
        assert result["prefix_match"] is True


# ---------------------------------------------------------------------------
# run_validation
# ---------------------------------------------------------------------------

class TestRunValidation:
    def _make_test_set(self, tmp_path: Path, n: int = 2) -> TestSet:
        entries = []
        for i in range(n):
            pdf = make_pdf(tmp_path / "archive" / "sub" / f"doc_{i}.pdf")
            entries.append(TestEntry(
                path=str(pdf),
                ground_truth_folder="sub",
            ))
        return TestSet(
            archive_root=str(tmp_path / "archive"),
            created_at=datetime.now().isoformat(),
            n=n,
            entries=entries,
        )

    def _make_client_mock(self, tmp_path: Path) -> MagicMock:
        client = MagicMock()
        client.__enter__ = MagicMock(return_value=client)
        client.__exit__ = MagicMock(return_value=False)
        return client

    def test_processes_all_entries(self, tmp_path: Path):
        archive = tmp_path / "archive"
        archive.mkdir()
        cfg = make_config(archive)
        test_set = self._make_test_set(tmp_path, n=3)

        target = archive / "sub"
        with patch("sortai.validator.LMStudioClient") as MockClient, \
             patch("sortai.validator.Pipeline") as MockPipeline:
            mock_client = self._make_client_mock(tmp_path)
            MockClient.return_value = mock_client
            mock_pipeline = MagicMock()
            mock_pipeline.run.return_value = (target, "doc.pdf", "summary", [])
            MockPipeline.return_value = mock_pipeline

            results = run_validation(test_set, cfg)

        assert len(results) == 3

    def test_always_uses_dry_run(self, tmp_path: Path):
        archive = tmp_path / "archive"
        archive.mkdir()
        cfg = make_config(archive)
        cfg.dry_run = False  # explicitly off
        test_set = self._make_test_set(tmp_path, n=1)

        captured_cfg = []
        with patch("sortai.validator.LMStudioClient") as MockClient, \
             patch("sortai.validator.Pipeline") as MockPipeline:
            mock_client = self._make_client_mock(tmp_path)
            MockClient.return_value = mock_client

            def capture_pipeline(cfg_arg, client_arg, **kwargs):
                captured_cfg.append(cfg_arg)
                m = MagicMock()
                m.run.return_value = (archive / "sub", "doc.pdf", "s", [])
                return m

            MockPipeline.side_effect = capture_pipeline
            run_validation(test_set, cfg)

        assert captured_cfg[0].dry_run is True

    def test_model_loaded_and_unloaded_once(self, tmp_path: Path):
        archive = tmp_path / "archive"
        archive.mkdir()
        cfg = make_config(archive)
        test_set = self._make_test_set(tmp_path, n=3)

        with patch("sortai.validator.LMStudioClient") as MockClient, \
             patch("sortai.validator.Pipeline") as MockPipeline:
            mock_client = self._make_client_mock(tmp_path)
            MockClient.return_value = mock_client
            mock_pipeline = MagicMock()
            mock_pipeline.run.return_value = (archive / "sub", "doc.pdf", "s", [])
            MockPipeline.return_value = mock_pipeline

            run_validation(test_set, cfg)

        mock_client.__enter__.assert_called_once()
        mock_client.__exit__.assert_called_once()


# ---------------------------------------------------------------------------
# print_results_table / print_score (smoke tests)
# ---------------------------------------------------------------------------

def _make_results(
    exact: bool = True,
    prefix: bool = True,
    error: str = "",
) -> list[ValidationResult]:
    return [
        ValidationResult(
            path="/archive/bank/doc.pdf",
            ground_truth_folder="bank",
            predicted_folder="bank" if not error else "",
            exact_match=exact,
            prefix_match=prefix,
            error=error,
            summary="A document summary",
            interactions=[],
        )
    ]


class TestPrintResultsTable:
    def test_renders_without_error(self):
        from rich.console import Console
        buf = StringIO()
        c = Console(file=buf, highlight=False)
        print_results_table(_make_results(), console=c)

    def test_verbose_shows_summary(self):
        from rich.console import Console
        buf = StringIO()
        # Wide console so the summary column is not truncated or wrapped.
        c = Console(file=buf, highlight=False, no_color=True, width=200)
        print_results_table(_make_results(), verbose=True, console=c)
        assert "A document summary" in buf.getvalue()

    def test_non_verbose_omits_summary(self):
        from rich.console import Console
        buf = StringIO()
        c = Console(file=buf, highlight=False, markup=False)
        print_results_table(_make_results(), verbose=False, console=c)
        assert "A document summary" not in buf.getvalue()

    def test_error_result_shows_error_text(self):
        from rich.console import Console
        buf = StringIO()
        c = Console(file=buf, highlight=False, markup=False)
        print_results_table(_make_results(exact=False, prefix=False, error="LLM down"), console=c)
        assert "ERROR" in buf.getvalue() or "LLM down" in buf.getvalue()


class TestPrintScore:
    def _console_and_output(self):
        from rich.console import Console
        buf = StringIO()
        c = Console(file=buf, highlight=False, markup=False)
        return c, buf

    def test_all_correct(self):
        c, buf = self._console_and_output()
        results = _make_results(exact=True, prefix=True) * 10
        print_score(results, console=c)
        out = buf.getvalue()
        assert "10/10" in out

    def test_mixed_results(self):
        c, buf = self._console_and_output()
        exact_results = _make_results(exact=True, prefix=True) * 3
        partial_results = _make_results(exact=False, prefix=True) * 2
        no_match = _make_results(exact=False, prefix=False) * 5
        results = exact_results + partial_results + no_match
        print_score(results, console=c)
        out = buf.getvalue()
        assert "3/10" in out
        assert "5/10" in out

    def test_with_errors(self):
        c, buf = self._console_and_output()
        results = _make_results(exact=False, prefix=False, error="fail") * 2
        print_score(results, console=c)
        out = buf.getvalue()
        assert "2 error" in out

    def test_empty_results(self):
        c, buf = self._console_and_output()
        print_score([], console=c)
        out = buf.getvalue()
        assert "No results" in out


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------

class TestValidateCLI:
    def _write_config(self, config_path: Path, inbox: Path, archive: Path) -> None:
        # TOML requires forward slashes (backslashes are escape sequences).
        inbox_s = inbox.as_posix()
        archive_s = archive.as_posix()
        config_path.write_text(
            f'inbox = "{inbox_s}"\narchive = "{archive_s}"\n'
            '[lm_studio]\nmodel = "test"\n',
            encoding="utf-8",
        )

    def test_validate_sample_writes_file(self, tmp_path: Path):
        from click.testing import CliRunner
        from sortai.cli import main

        # Build minimal archive with PDFs
        archive = tmp_path / "archive"
        (archive / "bank").mkdir(parents=True)
        make_pdf(archive / "bank" / "statement.pdf")
        inbox = tmp_path / "inbox"
        inbox.mkdir()

        config_path = tmp_path / "config.toml"
        self._write_config(config_path, inbox, archive)

        output = tmp_path / "test_set.json"
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["--config", str(config_path), "validate", "sample", str(output), "-n", "1"],
        )
        assert result.exit_code == 0, result.output
        assert output.exists()
        data = json.loads(output.read_text())
        assert data["n"] == 1

    def test_validate_sample_no_pdfs_exits_nonzero(self, tmp_path: Path):
        from click.testing import CliRunner
        from sortai.cli import main

        archive = tmp_path / "archive"
        archive.mkdir()
        inbox = tmp_path / "inbox"
        inbox.mkdir()

        config_path = tmp_path / "config.toml"
        self._write_config(config_path, inbox, archive)

        output = tmp_path / "test_set.json"
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["--config", str(config_path), "validate", "sample", str(output)],
        )
        assert result.exit_code != 0

    def test_validate_run_shows_accuracy(self, tmp_path: Path):
        from click.testing import CliRunner
        from sortai.cli import main

        archive = tmp_path / "archive"
        sub = archive / "bank"
        sub.mkdir(parents=True)
        pdf = make_pdf(sub / "doc.pdf")
        inbox = tmp_path / "inbox"
        inbox.mkdir()

        config_path = tmp_path / "config.toml"
        self._write_config(config_path, inbox, archive)

        test_set = TestSet(
            archive_root=str(archive),
            created_at=datetime.now().isoformat(),
            n=1,
            entries=[TestEntry(path=str(pdf), ground_truth_folder="bank")],
        )
        test_set_file = tmp_path / "test_set.json"
        write_test_set(test_set, test_set_file)

        with patch("sortai.validator.LMStudioClient") as MockClient, \
             patch("sortai.validator.Pipeline") as MockPipeline:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            MockClient.return_value = mock_client
            mock_pipeline = MagicMock()
            mock_pipeline.run.return_value = (sub, "doc.pdf", "summary", [])
            MockPipeline.return_value = mock_pipeline

            runner = CliRunner()
            result = runner.invoke(
                main,
                ["--config", str(config_path), "validate", "run", str(test_set_file)],
            )

        assert result.exit_code == 0, result.output
        assert "Accuracy" in result.output

    def test_validate_run_writes_results_json(self, tmp_path: Path):
        from click.testing import CliRunner
        from sortai.cli import main

        archive = tmp_path / "archive"
        sub = archive / "bank"
        sub.mkdir(parents=True)
        pdf = make_pdf(sub / "doc.pdf")
        inbox = tmp_path / "inbox"
        inbox.mkdir()

        config_path = tmp_path / "config.toml"
        self._write_config(config_path, inbox, archive)

        test_set = TestSet(
            archive_root=str(archive),
            created_at=datetime.now().isoformat(),
            n=1,
            entries=[TestEntry(path=str(pdf), ground_truth_folder="bank")],
        )
        test_set_file = tmp_path / "val.json"
        write_test_set(test_set, test_set_file)

        fake_interaction = {"stage": "summarize", "step": 1,
                            "prompt": "p", "answer": "a", "reasoning": ""}

        with patch("sortai.validator.LMStudioClient") as MockClient, \
             patch("sortai.validator.Pipeline") as MockPipeline:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            MockClient.return_value = mock_client
            mock_pipeline = MagicMock()
            mock_pipeline.run.return_value = (sub, "doc.pdf", "summary", [fake_interaction])
            MockPipeline.return_value = mock_pipeline

            runner = CliRunner()
            cli_result = runner.invoke(
                main,
                ["--config", str(config_path), "validate", "run", str(test_set_file)],
            )

        assert cli_result.exit_code == 0, cli_result.output

        results_path = tmp_path / "val_results.json"
        assert results_path.exists(), "results JSON was not written"
        data = json.loads(results_path.read_text(encoding="utf-8"))
        assert isinstance(data, list)
        assert len(data) == 1
        entry = data[0]
        assert entry["ground_truth_folder"] == "bank"
        assert isinstance(entry["interactions"], list)
        assert entry["interactions"][0]["stage"] == "summarize"
