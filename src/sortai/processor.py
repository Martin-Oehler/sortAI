"""Document-processing service shared by the CLI, watcher, and dashboard.

Owns the "run pipeline → stage for review or move → log" orchestration so
each entry point only handles its own UI (console output, SSE, exit codes).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from sortai.file_ops import log_decision, log_error, move_file
from sortai.pipeline import ClassificationError, Pipeline

if TYPE_CHECKING:
    from sortai.config import Config
    from sortai.llm_client import LMStudioClient
    from sortai.review_store import ReviewItem, ReviewStore


@dataclass
class Outcome:
    """Result of processing a single document.

    status is one of:
    - "moved":  the file was moved into the archive (or would be, in dry-run)
                and the decision was logged. final_path is the destination.
    - "staged": the file was staged for review (or would be, in dry-run).
                final_path is the staging path; review_item is the queue entry
                (None in dry-run, where no entry is added).
    - "error":  the pipeline raised a ClassificationError; the error was
                logged and error_reason holds the message.
    """

    status: str  # "moved" | "staged" | "error"
    source: Path
    dry_run: bool = False
    final_path: Optional[Path] = None
    proposed_folder: Optional[str] = None  # relative to cfg.archive, posix-style
    proposed_filename: Optional[str] = None
    summary: Optional[str] = None
    interactions: list = field(default_factory=list)
    review_item: Optional["ReviewItem"] = None
    error_reason: Optional[str] = None


def process_document(
    cfg: "Config",
    client: "LMStudioClient",
    pdf_path: Path,
    *,
    review_store: "ReviewStore | None" = None,
    user_hint: str | None = None,
    verbose: bool = False,
    pipeline_sem: "threading.Semaphore | None" = None,
    dry_run: bool | None = None,
    original_filename: str | None = None,
    previous_proposed_folder: str | None = None,
) -> Outcome:
    """Run the LLM pipeline on *pdf_path*, then stage for review or move + log.

    - When *review_store* is given, the file is staged into ``cfg.staging_dir``
      and a pending ReviewItem is added (skipped in dry-run). A file already
      inside the staging dir is left in place (dashboard reprocess).
    - Otherwise the file is moved to the pipeline's target folder and the
      decision is appended to the JSONL log.
    - *pipeline_sem*, when given, serialises the LLM portion (model load +
      pipeline run) against other pipeline users.
    - *dry_run* overrides ``cfg.dry_run`` when not None (the dashboard always
      passes False, matching its accept/reject endpoints).
    - *original_filename* overrides the name recorded on the review item and
      used for the staged file (dashboard reprocess of an already-renamed file).
    - A ClassificationError is logged via ``log_error`` and reported as
      status="error"; any other exception propagates to the caller.
    """
    effective_dry_run = cfg.dry_run if dry_run is None else dry_run
    src = pdf_path.resolve()
    name = original_filename or pdf_path.name

    if pipeline_sem:
        pipeline_sem.acquire()
    try:
        client.load_model()
        pipeline = Pipeline(cfg, client, verbose=verbose)
        target_folder, filename, summary, interactions = pipeline.run(pdf_path, user_hint=user_hint)
    except ClassificationError as exc:
        log_error(
            src=src,
            reason=str(exc),
            log_path=cfg.log_file,
            archive_root=cfg.archive,
        )
        return Outcome(
            status="error",
            source=src,
            dry_run=effective_dry_run,
            error_reason=str(exc),
        )
    finally:
        if pipeline_sem:
            pipeline_sem.release()

    proposed_folder = target_folder.relative_to(cfg.archive).as_posix()

    if review_store is not None:
        staging_dir = cfg.staging_dir
        if src.parent == staging_dir.resolve():
            staged = src  # already in staging (reprocess) — leave it in place
        else:
            staged = move_file(
                src=src,
                dest_dir=staging_dir,
                new_name=name,
                dry_run=effective_dry_run,
            )
        item: "ReviewItem | None" = None
        if not effective_dry_run:
            from sortai.review_store import make_review_item

            item = make_review_item(
                original_filename=name,
                staging_path=staged,
                proposed_folder=proposed_folder,
                proposed_filename=filename,
                summary=summary,
                interactions=interactions,
                user_hint=user_hint,
                previous_proposed_folder=previous_proposed_folder,
            )
            review_store.add(item)
        return Outcome(
            status="staged",
            source=src,
            dry_run=effective_dry_run,
            final_path=staged,
            proposed_folder=proposed_folder,
            proposed_filename=filename,
            summary=summary,
            interactions=interactions,
            review_item=item,
        )

    dest = move_file(
        src=src,
        dest_dir=target_folder,
        new_name=filename,
        dry_run=effective_dry_run,
    )
    log_decision(
        src=src,
        dest=dest,
        summary=summary,
        dry_run=effective_dry_run,
        log_path=cfg.log_file,
        archive_root=cfg.archive,
        interactions=interactions,
    )
    return Outcome(
        status="moved",
        source=src,
        dry_run=effective_dry_run,
        final_path=dest,
        proposed_folder=proposed_folder,
        proposed_filename=filename,
        summary=summary,
        interactions=interactions,
    )
