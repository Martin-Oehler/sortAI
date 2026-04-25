"""Ground truth validation for the sortAI pipeline."""

from __future__ import annotations

import dataclasses
import json
import random
from datetime import datetime
from pathlib import Path
from typing import TypedDict

from rich.console import Console
from rich.table import Table

from sortai.llm_client import LMStudioClient
from sortai.pipeline import Pipeline


class SampleEntry(TypedDict):
    path: str               # absolute path to the PDF
    ground_truth_folder: str  # relative to archive_root, posix (e.g. "bank/statements")


class SampleSet(TypedDict):
    archive_root: str
    created_at: str         # ISO 8601
    n: int
    entries: list[SampleEntry]


class ValidationResult(TypedDict):
    path: str
    ground_truth_folder: str
    predicted_folder: str   # "" on error
    exact_match: bool
    prefix_match: bool      # predicted is ancestor/descendant of ground truth
    error: str              # "" if success
    summary: str            # LLM summary, "" on error
    interactions: list      # list[StageInteraction]; [] on error


# Aliases for backwards-compatibility in tests and user-facing code.
TestEntry = SampleEntry
TestSet = SampleSet


def sample_pdfs(archive: Path, n: int) -> SampleSet:
    """Randomly sample up to *n* PDFs from *archive* and return a TestSet.

    Raises ValueError if no PDFs are found.
    """
    all_pdfs = list(archive.rglob("*.pdf"))
    if not all_pdfs:
        raise ValueError(f"No PDFs found in archive: {archive}")

    selected = random.sample(all_pdfs, min(n, len(all_pdfs)))
    entries: list[TestEntry] = []
    for pdf in selected:
        rel = pdf.parent.relative_to(archive)
        entries.append(
            TestEntry(
                path=str(pdf.resolve()),
                ground_truth_folder=rel.as_posix(),
            )
        )

    return SampleSet(
        archive_root=str(archive.resolve()),
        created_at=datetime.now().isoformat(),
        n=len(entries),
        entries=entries,
    )


def write_test_set(test_set: SampleSet, output: Path) -> None:
    """Serialise *test_set* to *output* as formatted JSON."""
    output.write_text(json.dumps(test_set, indent=2), encoding="utf-8")


def load_test_set(path: Path) -> SampleSet:
    """Load and return a TestSet from a JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def _compare_folders(
    predicted: Path,
    ground_truth_rel: str,
    archive_root: Path,
) -> tuple[bool, bool]:
    """Return (exact_match, prefix_match).

    prefix_match is True when predicted and ground truth share a common
    ancestor/descendant relationship (one is a subfolder of the other).
    """
    try:
        predicted_rel = predicted.relative_to(archive_root).as_posix()
    except ValueError:
        return False, False

    gt = ground_truth_rel

    exact = predicted_rel == gt

    if exact:
        return True, True

    # One is a subdirectory of the other
    prefix = predicted_rel.startswith(gt + "/") or gt.startswith(predicted_rel + "/")
    return False, prefix


def _run_single(
    entry: SampleEntry,
    archive_root: Path,
    pipeline: Pipeline,
) -> ValidationResult:
    """Run the pipeline on one entry; catch all exceptions into an error result."""
    try:
        target_folder, _filename, summary, interactions = pipeline.run(Path(entry["path"]))
        exact, prefix = _compare_folders(target_folder, entry["ground_truth_folder"], archive_root)
        try:
            predicted_rel = target_folder.relative_to(archive_root).as_posix()
        except ValueError:
            predicted_rel = str(target_folder)
        return ValidationResult(
            path=entry["path"],
            ground_truth_folder=entry["ground_truth_folder"],
            predicted_folder=predicted_rel,
            exact_match=exact,
            prefix_match=prefix,
            error="",
            summary=summary,
            interactions=interactions,
        )
    except Exception as exc:  # noqa: BLE001
        return ValidationResult(
            path=entry["path"],
            ground_truth_folder=entry["ground_truth_folder"],
            predicted_folder="",
            exact_match=False,
            prefix_match=False,
            error=str(exc),
            summary="",
            interactions=[],
        )


def run_validation(
    test_set: SampleSet,
    cfg,
    verbose: bool = False,
    console: Console | None = None,
) -> list[ValidationResult]:
    """Run the pipeline (always dry-run) on every entry in *test_set*.

    Loads the model once, processes all files, unloads when done.
    """
    if console is None:
        console = Console()

    # Force dry_run regardless of user config to guarantee no file moves.
    cfg_dry = dataclasses.replace(cfg, dry_run=True)

    archive_root = cfg_dry.archive
    total = test_set["n"]

    client = LMStudioClient(
        base_url=cfg_dry.lm_studio.base_url,
        model_name=cfg_dry.lm_studio.model,
        prompts_dir=cfg_dry.prompts_dir,
        temperature=cfg_dry.lm_studio.temperature,
        max_tokens=cfg_dry.lm_studio.max_tokens,
        reasoning=cfg_dry.lm_studio.reasoning,
    )

    results: list[ValidationResult] = []
    with client:
        pipeline = Pipeline(cfg_dry, client, verbose=verbose)
        for i, entry in enumerate(test_set["entries"], 1):
            name = Path(entry["path"]).name
            console.print(f"[cyan]Validating[/cyan] {i}/{total}: {name}")
            result = _run_single(entry, archive_root, pipeline)
            results.append(result)

    return results


def print_results_table(
    results: list[ValidationResult],
    verbose: bool = False,
    console: Console | None = None,
) -> None:
    """Render a Rich table of per-file results."""
    if console is None:
        console = Console()

    table = Table(title="Validation Results", show_lines=False)
    table.add_column("#", style="dim", justify="right")
    table.add_column("File", style="cyan")
    table.add_column("Ground Truth", style="green")
    table.add_column("Predicted")
    table.add_column("Match", justify="center")
    table.add_column("Partial", justify="center")
    if verbose:
        table.add_column("Summary")

    for i, r in enumerate(results, 1):
        filename = Path(r["path"]).name

        if r["error"]:
            predicted_cell = f"[bold red]ERROR: {r['error'][:60]}[/bold red]"
            match_cell = "[red]N[/red]"
            partial_cell = "[red]N[/red]"
        elif r["exact_match"]:
            predicted_cell = f"[green]{r['predicted_folder']}[/green]"
            match_cell = "[green]Y[/green]"
            partial_cell = "[green]Y[/green]"
        elif r["prefix_match"]:
            predicted_cell = f"[yellow]{r['predicted_folder']}[/yellow]"
            match_cell = "[red]N[/red]"
            partial_cell = "[yellow]~[/yellow]"
        else:
            predicted_cell = f"[red]{r['predicted_folder']}[/red]"
            match_cell = "[red]N[/red]"
            partial_cell = "[red]N[/red]"

        row = [str(i), filename, r["ground_truth_folder"], predicted_cell, match_cell, partial_cell]
        if verbose:
            s = r["summary"]
            row.append(s[:80] + ("..." if len(s) > 80 else ""))

        table.add_row(*row)

    console.print(table)


def print_score(
    results: list[ValidationResult],
    console: Console | None = None,
) -> None:
    """Print the headline accuracy figures."""
    if console is None:
        console = Console()

    total = len(results)
    if total == 0:
        console.print("[yellow]No results to score.[/yellow]")
        return

    exact = sum(1 for r in results if r["exact_match"])
    partial = sum(1 for r in results if r["prefix_match"])
    errors = sum(1 for r in results if r["error"])

    console.print(
        f"\n[bold]Accuracy:[/bold] {exact}/{total} ({exact / total:.1%}) exact"
        f"  |  {partial}/{total} ({partial / total:.1%}) partial"
        f"  |  {errors} error(s)"
    )
