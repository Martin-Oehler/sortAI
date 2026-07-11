"""Tests for sortai.processor.process_document — the shared orchestration service."""
from __future__ import annotations

import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from sortai.config import Config
from sortai.pipeline import ClassificationError
from sortai.processor import Outcome, process_document

_PATCH_PIPELINE = "sortai.processor.Pipeline"
_PATCH_MOVE = "sortai.processor.move_file"
_PATCH_LOG = "sortai.processor.log_decision"
_PATCH_LOG_ERR = "sortai.processor.log_error"


@pytest.fixture()
def cfg(tmp_path: Path) -> Config:
    inbox = tmp_path / "inbox"
    archive = tmp_path / "archive"
    inbox.mkdir()
    archive.mkdir()
    return Config(inbox=inbox, archive=archive, log_file=tmp_path / "logs" / "sortai.jsonl")


@pytest.fixture()
def pdf(cfg: Config) -> Path:
    p = cfg.inbox / "doc.pdf"
    p.write_bytes(b"%PDF-1.4")
    return p


def _pipeline_mock(target: Path, filename: str = "out.pdf", summary: str = "a summary"):
    instance = MagicMock()
    instance.run.return_value = (target, filename, summary, [{"stage": "summarize"}])
    return MagicMock(return_value=instance), instance


class TestNormalMove:
    def test_moves_and_logs(self, cfg: Config, pdf: Path):
        target = cfg.archive / "bank" / "2026"
        pipeline_cls, _ = _pipeline_mock(target, "statement.pdf", "bank statement")
        dest = target / "statement.pdf"
        client = MagicMock()

        with patch(_PATCH_PIPELINE, pipeline_cls), \
             patch(_PATCH_MOVE, return_value=dest) as mock_move, \
             patch(_PATCH_LOG) as mock_log:
            outcome = process_document(cfg, client, pdf)

        assert outcome.status == "moved"
        assert outcome.final_path == dest
        assert outcome.proposed_folder == "bank/2026"
        assert outcome.proposed_filename == "statement.pdf"
        assert outcome.summary == "bank statement"
        assert outcome.dry_run is False
        mock_move.assert_called_once_with(
            src=pdf.resolve(), dest_dir=target, new_name="statement.pdf", dry_run=False
        )
        mock_log.assert_called_once_with(
            src=pdf.resolve(),
            dest=dest,
            summary="bank statement",
            dry_run=False,
            log_path=cfg.log_file,
            archive_root=cfg.archive,
            interactions=[{"stage": "summarize"}],
        )

    def test_loads_model_and_runs_pipeline_with_hint(self, cfg: Config, pdf: Path):
        pipeline_cls, instance = _pipeline_mock(cfg.archive)
        client = MagicMock()

        with patch(_PATCH_PIPELINE, pipeline_cls), \
             patch(_PATCH_MOVE, return_value=cfg.archive / "out.pdf"), \
             patch(_PATCH_LOG):
            process_document(cfg, client, pdf, user_hint="it is a bill", verbose=True)

        client.load_model.assert_called_once()
        pipeline_cls.assert_called_once_with(cfg, client, verbose=True)
        instance.run.assert_called_once_with(pdf, user_hint="it is a bill")

    def test_pipeline_sem_acquired_and_released(self, cfg: Config, pdf: Path):
        pipeline_cls, _ = _pipeline_mock(cfg.archive)
        sem = threading.Semaphore(1)

        with patch(_PATCH_PIPELINE, pipeline_cls), \
             patch(_PATCH_MOVE, return_value=cfg.archive / "out.pdf"), \
             patch(_PATCH_LOG):
            process_document(cfg, MagicMock(), pdf, pipeline_sem=sem)

        assert sem.acquire(blocking=False)  # released again after processing
        sem.release()


class TestDryRun:
    def test_cfg_dry_run_propagates(self, cfg: Config, pdf: Path):
        cfg.dry_run = True
        pipeline_cls, _ = _pipeline_mock(cfg.archive)

        with patch(_PATCH_PIPELINE, pipeline_cls), \
             patch(_PATCH_MOVE, return_value=cfg.archive / "out.pdf") as mock_move, \
             patch(_PATCH_LOG) as mock_log:
            outcome = process_document(cfg, MagicMock(), pdf)

        assert outcome.dry_run is True
        assert mock_move.call_args.kwargs["dry_run"] is True
        assert mock_log.call_args.kwargs["dry_run"] is True

    def test_explicit_dry_run_overrides_cfg(self, cfg: Config, pdf: Path):
        cfg.dry_run = True
        pipeline_cls, _ = _pipeline_mock(cfg.archive)

        with patch(_PATCH_PIPELINE, pipeline_cls), \
             patch(_PATCH_MOVE, return_value=cfg.archive / "out.pdf") as mock_move, \
             patch(_PATCH_LOG):
            outcome = process_document(cfg, MagicMock(), pdf, dry_run=False)

        assert outcome.dry_run is False
        assert mock_move.call_args.kwargs["dry_run"] is False


class TestReviewStaging:
    def test_stages_and_adds_review_item(self, cfg: Config, pdf: Path):
        target = cfg.archive / "invoices"
        pipeline_cls, _ = _pipeline_mock(target, "invoice.pdf", "an invoice")
        store = MagicMock()
        staged = cfg.staging_dir / pdf.name

        with patch(_PATCH_PIPELINE, pipeline_cls), \
             patch(_PATCH_MOVE, return_value=staged) as mock_move, \
             patch(_PATCH_LOG) as mock_log:
            outcome = process_document(
                cfg, MagicMock(), pdf,
                review_store=store,
                user_hint="utility bill",
                previous_proposed_folder="misc",
            )

        assert outcome.status == "staged"
        assert outcome.final_path == staged
        mock_move.assert_called_once_with(
            src=pdf.resolve(), dest_dir=cfg.staging_dir, new_name=pdf.name, dry_run=False
        )
        mock_log.assert_not_called()  # staged items are logged on accept, not here
        store.add.assert_called_once()
        item = store.add.call_args.args[0]
        assert item is outcome.review_item
        assert item.original_filename == pdf.name
        assert item.staging_path == str(staged)
        assert item.proposed_folder == "invoices"
        assert item.proposed_filename == "invoice.pdf"
        assert item.summary == "an invoice"
        assert item.status == "pending"
        assert item.user_hint == "utility bill"
        assert item.previous_proposed_folder == "misc"

    def test_dry_run_stages_without_adding_item(self, cfg: Config, pdf: Path):
        cfg.dry_run = True
        pipeline_cls, _ = _pipeline_mock(cfg.archive / "invoices")
        store = MagicMock()
        staged = cfg.staging_dir / pdf.name

        with patch(_PATCH_PIPELINE, pipeline_cls), \
             patch(_PATCH_MOVE, return_value=staged) as mock_move:
            outcome = process_document(cfg, MagicMock(), pdf, review_store=store)

        assert outcome.status == "staged"
        assert outcome.review_item is None
        assert mock_move.call_args.kwargs["dry_run"] is True
        store.add.assert_not_called()

    def test_file_already_in_staging_dir_is_not_moved(self, cfg: Config):
        """Dashboard reprocess: a file already staged stays in place."""
        cfg.staging_dir.mkdir(parents=True)
        staged_pdf = cfg.staging_dir / "already_staged.pdf"
        staged_pdf.write_bytes(b"%PDF-1.4")
        pipeline_cls, _ = _pipeline_mock(cfg.archive / "bank")
        store = MagicMock()

        with patch(_PATCH_PIPELINE, pipeline_cls), \
             patch(_PATCH_MOVE) as mock_move:
            outcome = process_document(
                cfg, MagicMock(), staged_pdf,
                review_store=store,
                original_filename="original_name.pdf",
            )

        mock_move.assert_not_called()
        assert outcome.status == "staged"
        assert outcome.final_path == staged_pdf.resolve()
        item = store.add.call_args.args[0]
        assert item.original_filename == "original_name.pdf"
        assert item.staging_path == str(staged_pdf.resolve())


class TestPipelineError:
    def test_classification_error_is_logged_and_reported(self, cfg: Config, pdf: Path):
        client = MagicMock()
        pipeline_instance = MagicMock()
        pipeline_instance.run.side_effect = ClassificationError("handwritten note")

        with patch(_PATCH_PIPELINE, return_value=pipeline_instance), \
             patch(_PATCH_MOVE) as mock_move, \
             patch(_PATCH_LOG) as mock_log, \
             patch(_PATCH_LOG_ERR) as mock_log_err:
            outcome = process_document(cfg, client, pdf)

        assert outcome.status == "error"
        assert outcome.error_reason == "handwritten note"
        assert outcome.final_path is None
        mock_log_err.assert_called_once_with(
            src=pdf.resolve(),
            reason="handwritten note",
            log_path=cfg.log_file,
            archive_root=cfg.archive,
        )
        mock_move.assert_not_called()
        mock_log.assert_not_called()

    def test_other_exceptions_propagate(self, cfg: Config, pdf: Path):
        pipeline_instance = MagicMock()
        pipeline_instance.run.side_effect = RuntimeError("LM Studio unreachable")

        with patch(_PATCH_PIPELINE, return_value=pipeline_instance), \
             pytest.raises(RuntimeError, match="LM Studio unreachable"):
            process_document(cfg, MagicMock(), pdf)

    def test_sem_released_on_error(self, cfg: Config, pdf: Path):
        pipeline_instance = MagicMock()
        pipeline_instance.run.side_effect = ClassificationError("nope")
        sem = threading.Semaphore(1)

        with patch(_PATCH_PIPELINE, return_value=pipeline_instance), \
             patch(_PATCH_LOG_ERR):
            outcome = process_document(cfg, MagicMock(), pdf, pipeline_sem=sem)

        assert outcome.status == "error"
        assert sem.acquire(blocking=False)
        sem.release()
