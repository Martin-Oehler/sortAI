"""Command-line interface for sortAI."""

from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

console = Console()

DEFAULT_CONFIG = Path("config/config.toml")


def _load_config(config_path: Path, dry_run_override: bool):
    """Load config, applying CLI dry-run override if set."""
    from sortai.config import Config

    cfg = Config.load(config_path)
    if dry_run_override:
        cfg.dry_run = True
    return cfg


@click.group()
@click.option(
    "--config",
    "config_path",
    default=str(DEFAULT_CONFIG),
    show_default=True,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Path to config.toml",
)
@click.option("--dry-run", is_flag=True, default=False, help="Simulate actions without moving files.")
@click.pass_context
def main(ctx: click.Context, config_path: Path, dry_run: bool) -> None:
    """sortAI — sort documents into an archive using an LLM."""
    ctx.ensure_object(dict)
    ctx.obj["config_path"] = config_path
    ctx.obj["dry_run"] = dry_run


@main.command("config")
@click.pass_context
def show_config(ctx: click.Context) -> None:
    """Show the current configuration."""
    cfg = _load_config(ctx.obj["config_path"], ctx.obj["dry_run"])

    table = Table(title="sortAI configuration", show_header=False)
    table.add_column("Key", style="bold cyan")
    table.add_column("Value")

    table.add_row("inbox", str(cfg.inbox))
    table.add_row("archive", str(cfg.archive))
    table.add_row("prompts_dir", str(cfg.prompts_dir))
    table.add_row("log_file", str(cfg.log_file))
    table.add_row("dry_run", str(cfg.dry_run))
    table.add_row("max_navigate_depth", str(cfg.max_navigate_depth))
    table.add_row("lm_studio.base_url", cfg.lm_studio.base_url)
    table.add_row("lm_studio.model", cfg.lm_studio.model)
    table.add_row("lm_studio.temperature", str(cfg.lm_studio.temperature))
    table.add_row("lm_studio.max_tokens", str(cfg.lm_studio.max_tokens))

    console.print(table)


# ---------------------------------------------------------------------------
# Stub commands — to be implemented in later phases
# ---------------------------------------------------------------------------

@main.command("inspect")
@click.argument("pdf_file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.pass_context
def inspect_pdf(ctx: click.Context, pdf_file: Path) -> None:
    """[Phase 2] Show extracted text and archive listing for a PDF."""
    console.print("[yellow]Not yet implemented (Phase 2).[/yellow]")


@main.command("tree")
@click.pass_context
def show_tree(ctx: click.Context) -> None:
    """[Phase 2] Pretty-print the document archive folder tree."""
    console.print("[yellow]Not yet implemented (Phase 2).[/yellow]")


@main.command("ping")
@click.pass_context
def ping_lm_studio(ctx: click.Context) -> None:
    """[Phase 3] Test LM Studio connection (load model, send hello, unload)."""
    console.print("[yellow]Not yet implemented (Phase 3).[/yellow]")


@main.command("process")
@click.argument("pdf_file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.pass_context
def process_pdf(ctx: click.Context, pdf_file: Path) -> None:
    """[Phase 4/5] Run the full pipeline on a single PDF."""
    console.print("[yellow]Not yet implemented (Phase 4).[/yellow]")


@main.command("log")
@click.option("-n", "count", default=20, show_default=True, help="Number of entries to show.")
@click.pass_context
def show_log(ctx: click.Context, count: int) -> None:
    """[Phase 5] Show recent log entries."""
    console.print("[yellow]Not yet implemented (Phase 5).[/yellow]")


@main.command("watch")
@click.option("--once", is_flag=True, default=False, help="Process existing files then exit.")
@click.pass_context
def watch_inbox(ctx: click.Context, once: bool) -> None:
    """[Phase 6] Monitor the inbox folder and process new PDFs automatically."""
    console.print("[yellow]Not yet implemented (Phase 6).[/yellow]")
