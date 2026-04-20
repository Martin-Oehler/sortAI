"""Command-line interface for sortAI."""

from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console
from rich.table import Table
from rich.tree import Tree

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

@main.command("extract")
@click.argument("pdf_file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("-n", "max_chars", default=500, show_default=True, help="Max characters of extracted text to display.")
@click.pass_context
def extract_pdf(ctx: click.Context, pdf_file: Path, max_chars: int) -> None:
    """Extract and display text content from a PDF."""
    from sortai.pdf_reader import extract_text

    text = extract_text(pdf_file)
    console.print(f"\n[bold cyan]Extracted text[/bold cyan] ({len(text)} chars total):\n")
    console.print(text[:max_chars])
    if len(text) > max_chars:
        console.print(f"\n[dim]… {len(text) - max_chars} more chars omitted[/dim]")


def _build_rich_tree(branch: Tree, path: Path) -> None:
    from sortai.folder_navigator import list_children

    for name in list_children(path):
        sub = branch.add(f"[green]{name}[/green]")
        _build_rich_tree(sub, path / name)


@main.command("tree")
@click.pass_context
def show_tree(ctx: click.Context) -> None:
    """Pretty-print the document archive folder tree."""
    cfg = _load_config(ctx.obj["config_path"], ctx.obj["dry_run"])
    root = Tree(f"[bold]{cfg.archive}[/bold]")
    _build_rich_tree(root, cfg.archive)
    console.print(root)


@main.command("ping")
@click.pass_context
def ping_lm_studio(ctx: click.Context) -> None:
    """Test LM Studio connection: load model, send hello, print response, unload."""
    cfg = _load_config(ctx.obj["config_path"], ctx.obj["dry_run"])
    from sortai.llm_client import LMStudioClient

    client = LMStudioClient(
        base_url=cfg.lm_studio.base_url,
        model_name=cfg.lm_studio.model,
        prompts_dir=cfg.prompts_dir,
        temperature=cfg.lm_studio.temperature,
        max_tokens=cfg.lm_studio.max_tokens,
    )

    try:
        console.print(f"[cyan]Loading model[/cyan] [bold]{cfg.lm_studio.model}[/bold] …")
        with client:
            console.print("[cyan]Sending hello…[/cyan]")
            reply = client.complete("Hello! Please respond with a single short sentence.")
            console.print(f"\n[bold green]Response:[/bold green] {reply}\n")
        console.print("[cyan]Model unloaded.[/cyan]")
    except RuntimeError as exc:
        console.print(f"\n[bold red]Error:[/bold red] {exc}")
        raise SystemExit(1)


@main.command("process")
@click.argument("pdf_file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.pass_context
def process_pdf(ctx: click.Context, pdf_file: Path) -> None:
    """Run the full LLM pipeline on a single PDF (prints proposed destination)."""
    cfg = _load_config(ctx.obj["config_path"], ctx.obj["dry_run"])
    from sortai.llm_client import LMStudioClient
    from sortai.pipeline import Pipeline

    client = LMStudioClient(
        base_url=cfg.lm_studio.base_url,
        model_name=cfg.lm_studio.model,
        prompts_dir=cfg.prompts_dir,
        temperature=cfg.lm_studio.temperature,
        max_tokens=cfg.lm_studio.max_tokens,
    )

    try:
        console.print(f"[cyan]Loading model[/cyan] [bold]{cfg.lm_studio.model}[/bold] …")
        with client:
            pipeline = Pipeline(cfg, client)
            console.print(f"[cyan]Processing[/cyan] {pdf_file.name} …")
            target_folder, filename = pipeline.run(pdf_file)
        console.print("[cyan]Model unloaded.[/cyan]")
        dest = target_folder / filename
        console.print(f"\n[bold green]→[/bold green] {dest}\n")
    except RuntimeError as exc:
        console.print(f"\n[bold red]Error:[/bold red] {exc}")
        raise SystemExit(1)


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
