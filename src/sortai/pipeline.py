"""Three-stage LLM pipeline: summarize → navigate → name."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TypedDict

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule

from sortai.config import Config
from sortai.folder_navigator import is_leaf, list_children
from sortai.llm_client import LLMResponse, LMStudioClient
from sortai.pdf_reader import extract_text


class ClassificationError(Exception):
    """Raised when the model determines the document cannot be classified."""


class StageInteraction(TypedDict):
    stage: str
    step: int
    prompt: str
    answer: str
    reasoning: str

# Max characters of document text forwarded to the LLM per call.
_MAX_TEXT_CHARS = 4000


def _truncate(text: str) -> str:
    if len(text) <= _MAX_TEXT_CHARS:
        return text
    return text[:_MAX_TEXT_CHARS] + "\n…[truncated]"


def _sanitize_filename(raw: str) -> str:
    """Keep only lowercase alphanum/hyphen/underscore; strip leading/trailing junk."""
    lines = raw.strip().lower().splitlines()
    name = lines[0].strip() if lines else ""
    name = re.sub(r"[^a-z0-9_\-]", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    if not name:
        name = "document"
    if not name.endswith(".pdf"):
        name += ".pdf"
    return name


class Pipeline:
    def __init__(self, config: Config, client: LMStudioClient, verbose: bool = False) -> None:
        self.config = config
        self.client = client
        self.verbose = verbose
        self._console = Console()

    def _log_exchange(self, stage: str, prompt: str, response: str) -> None:
        self._console.print(Rule(f"[bold cyan]{stage}[/bold cyan]"))
        self._console.print(Panel(Markdown(prompt), title="[dim]prompt[/dim]", border_style="dim"))
        self._console.print(Panel(response.strip(), title="[dim]response[/dim]", border_style="green"))

    def summarize(self, text: str) -> tuple[str, list[StageInteraction]]:
        """Stage 1 — summarize the document text.

        Raises ClassificationError if the model determines the content is unclassifiable.
        """
        template = self.client.load_prompt("summarize")
        prompt = template.replace("{{document_text}}", _truncate(text))
        schema = {
            "type": "object",
            "properties": {
                "can_classify": {"type": "boolean"},
                "summary": {"type": "string"},
                "reason": {"type": "string"},
            },
            "required": ["can_classify", "summary", "reason"],
            "additionalProperties": False,
        }
        resp = self.client.complete_structured(prompt, schema)
        parsed = json.loads(resp.content)
        interactions = [StageInteraction(stage="summarize", step=1,
            prompt=prompt, answer=resp.content, reasoning=resp.reasoning)]
        if self.verbose:
            self._log_exchange("Stage 1 — Summarize", prompt, resp.content)
        if not parsed["can_classify"]:
            raise ClassificationError(parsed.get("reason") or "model refused to classify")
        return parsed["summary"].strip(), interactions

    def navigate_to_folder(self, text: str, summary: str) -> tuple[Path, list[StageInteraction]]:
        """Stage 2 — walk the archive tree to find the best target folder."""
        template = self.client.load_prompt("navigate")
        current = self.config.archive
        short_text = _truncate(text)
        step = 0
        interactions: list[StageInteraction] = []

        for _ in range(self.config.max_navigate_depth):
            children = list_children(current)
            if not children or is_leaf(current):
                break

            folder_listing = "\n".join(f"- {c}" for c in children)
            prompt = (
                template
                .replace("{{current_folder}}", str(current))
                .replace("{{folder_listing}}", folder_listing)
                .replace("{{summary}}", summary)
                .replace("{{document_text}}", short_text)
            )
            schema = {
                "type": "object",
                "properties": {
                    "reasoning": {"type": "string"},
                    "choice": {"type": "string", "enum": children + ["."]},
                },
                "required": ["reasoning", "choice"],
                "additionalProperties": False,
            }
            resp = self.client.complete_structured(prompt, schema)
            parsed = json.loads(resp.content)
            choice = parsed["choice"]
            step += 1
            interactions.append(StageInteraction(stage="navigate", step=step,
                prompt=prompt, answer=choice, reasoning=parsed.get("reasoning", "")))
            if self.verbose:
                self._log_exchange(f"Stage 2 — Navigate (step {step})", prompt, choice)

            if choice == ".":
                break

            current = current / choice

        return current, interactions

    def choose_filename(self, text: str, summary: str, target: Path) -> tuple[str, list[StageInteraction]]:
        """Stage 3 — pick a sanitised filename (with .pdf extension)."""
        existing = sorted(p.name for p in target.iterdir() if p.is_file()) if target.exists() else []
        existing_files = "\n".join(f"- {f}" for f in existing[:20]) or "(none)"

        template = self.client.load_prompt("name_file")
        prompt = (
            template
            .replace("{{target_folder}}", str(target))
            .replace("{{existing_files}}", existing_files)
            .replace("{{summary}}", summary)
            .replace("{{document_text}}", _truncate(text))
        )
        resp = self.client.complete(prompt)
        raw = resp.content.strip()
        result = _sanitize_filename(raw)
        interactions = [StageInteraction(stage="choose_filename", step=1,
            prompt=prompt, answer=raw, reasoning=resp.reasoning)]
        if self.verbose:
            self._log_exchange("Stage 3 — Name file", prompt, raw)
        return result, interactions

    def run(self, pdf_path: Path) -> tuple[Path, str, str, list[StageInteraction]]:
        """Run all three stages and return (target_folder, filename, summary, interactions)."""
        text = extract_text(pdf_path)
        summary, s1 = self.summarize(text)
        target_folder, s2 = self.navigate_to_folder(text, summary)
        filename, s3 = self.choose_filename(text, summary, target_folder)
        return target_folder, filename, summary, s1 + s2 + s3
