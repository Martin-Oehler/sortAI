"""Three-stage LLM pipeline: summarize → navigate → name."""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import TypedDict

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule

from sortai.config import Config
from sortai.folder_navigator import is_leaf, list_children, list_children_with_info
from sortai.llm_client import LLMResponse, LMStudioClient
from sortai.pdf_reader import extract_text, render_pages


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

_VISION_PLACEHOLDER = "(See attached PDF page images.)"


def _apply_hint(prompt: str, hint: str | None) -> str:
    if hint:
        return prompt.replace("{{user_hint}}", hint)
    return re.sub(r"[^\n]*\{\{user_hint\}\}[^\n]*\n?", "", prompt)


def _apply_memory(prompt: str, memory: str | None) -> str:
    if memory and memory.strip():
        return prompt.replace("{{memory}}", memory.strip())
    return re.sub(r"[^\n]*\{\{memory\}\}[^\n]*\n?", "", prompt)


def _truncate(text: str) -> str:
    if len(text) <= _MAX_TEXT_CHARS:
        return text
    return text[:_MAX_TEXT_CHARS] + "\n…[truncated]"


def _apply_document(prompt: str, text: str, images: list[str] | None) -> str:
    """Replace {{document_text}} with either the vision placeholder or the extracted text."""
    if images:
        return prompt.replace("{{document_text}}", _VISION_PLACEHOLDER)
    return prompt.replace("{{document_text}}", _truncate(text))


def _log_exchange(console: Console, stage: str, prompt: str, response: str, reasoning: str = "") -> None:
    """Pretty-print a prompt/response exchange (verbose mode)."""
    console.print(Rule(f"[bold cyan]{stage}[/bold cyan]"))
    console.print(Panel(Markdown(prompt), title="[dim]prompt[/dim]", border_style="dim"))
    if reasoning:
        console.print(Panel(reasoning.strip(), title="[dim]reasoning[/dim]", border_style="yellow"))
    console.print(Panel(response.strip(), title="[dim]response[/dim]", border_style="green"))


_CHAR_MAP = {
    'ä': 'ae', 'ö': 'oe', 'ü': 'ue', 'ß': 'ss',
    'æ': 'ae', 'ø': 'oe', 'å': 'aa',
}


def _sanitize_filename(raw: str) -> str:
    """Transliterate common non-ASCII chars, then keep only lowercase alphanum/hyphen/underscore."""
    name = raw.strip().lower()
    for char, replacement in _CHAR_MAP.items():
        name = name.replace(char, replacement)
    name = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode('ascii')
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

    def _log_exchange(self, stage: str, prompt: str, response: str, reasoning: str = "") -> None:
        _log_exchange(self._console, stage, prompt, response, reasoning)

    def summarize(self, text: str, user_hint: str | None = None, images: list[str] | None = None) -> tuple[str, list[StageInteraction]]:
        """Stage 1 — summarize the document text.

        Raises ClassificationError if the model determines the content is unclassifiable.
        """
        template = self.client.load_prompt("summarize")
        prompt = _apply_hint(_apply_document(template, text, images), user_hint)
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
        resp = self.client.complete_structured(prompt, schema, images=images)
        parsed = json.loads(resp.content)
        interactions = [StageInteraction(stage="summarize", step=1,
            prompt=prompt, answer=parsed.get("summary", ""), reasoning=parsed.get("reason", ""))]
        if self.verbose:
            if parsed["can_classify"]:
                self._log_exchange("Stage 1 — Summarize", prompt, parsed["summary"],
                    reasoning=parsed.get("reason", ""))
            else:
                self._log_exchange("Stage 1 — Summarize", prompt,
                    f"[CANNOT CLASSIFY] {parsed.get('reason', '')}")
        if not parsed["can_classify"]:
            raise ClassificationError(parsed.get("reason") or "model refused to classify")
        return parsed["summary"].strip(), interactions

    def navigate_to_folder(self, text: str, summary: str, user_hint: str | None = None, images: list[str] | None = None) -> tuple[Path, list[StageInteraction]]:
        """Stage 2 — walk the archive tree to find the best target folder."""
        template = self.client.load_prompt("navigate")
        current = self.config.archive
        step = 0
        interactions: list[StageInteraction] = []

        if self.config.enable_memory:
            from sortai.memory import load_memory_text  # local import avoids cycle
            memory = load_memory_text(self.config.memory_path)
        else:
            memory = None

        for _ in range(self.config.max_navigate_depth):
            children = list_children(current)
            if not children or is_leaf(current):
                break

            folder_infos = list_children_with_info(
                current,
                self.config.folder_description_filename,
                self.config.subfolder_preview_count,
            )
            listing_lines = []
            for info in folder_infos:
                line = f"- {info.name}"
                if info.subfolders:
                    line += f" (contains: {', '.join(info.subfolders)})"
                if info.description:
                    line += f" — {info.description}"
                listing_lines.append(line)
            folder_listing = "\n".join(listing_lines)
            prompt = _apply_hint(
                _apply_memory(
                    _apply_document(
                        template
                        .replace("{{current_folder}}", str(current))
                        .replace("{{folder_listing}}", folder_listing)
                        .replace("{{summary}}", summary),
                        text,
                        images,
                    ),
                    memory,
                ),
                user_hint,
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
            resp = self.client.complete_structured(prompt, schema, images=images)
            parsed = json.loads(resp.content)
            choice = parsed["choice"]
            step += 1
            interactions.append(StageInteraction(stage="navigate", step=step,
                prompt=prompt, answer=choice, reasoning=parsed.get("reasoning", "")))
            if self.verbose:
                self._log_exchange(f"Stage 2 — Navigate (step {step})", prompt, choice,
                    reasoning=parsed.get("reasoning", ""))

            if choice == ".":
                break

            current = current / choice

        return current, interactions

    def choose_filename(self, text: str, summary: str, target: Path, user_hint: str | None = None, images: list[str] | None = None) -> tuple[str, list[StageInteraction]]:
        """Stage 3 — pick a sanitised filename (with .pdf extension)."""
        existing = sorted(p.name for p in target.iterdir() if p.is_file()) if target.exists() else []
        existing_files = "\n".join(f"- {f}" for f in existing[:20]) or "(none)"

        template = self.client.load_prompt("name_file")
        prompt = _apply_hint(
            _apply_document(
                template
                .replace("{{target_folder}}", str(target))
                .replace("{{existing_files}}", existing_files)
                .replace("{{summary}}", summary),
                text,
                images,
            ),
            user_hint,
        )
        schema = {
            "type": "object",
            "properties": {
                "reasoning": {"type": "string"},
                "filename": {"type": "string", "pattern": "^[a-z0-9][a-z0-9_-]*$"},
            },
            "required": ["reasoning", "filename"],
            "additionalProperties": False,
        }
        resp = self.client.complete_structured(prompt, schema, images=images)
        parsed = json.loads(resp.content)
        raw = parsed["filename"]
        result = _sanitize_filename(raw)
        interactions = [StageInteraction(stage="choose_filename", step=1,
            prompt=prompt, answer=raw, reasoning=parsed.get("reasoning", ""))]
        if self.verbose:
            self._log_exchange("Stage 3 — Name file", prompt, raw,
                reasoning=parsed.get("reasoning", ""))
        return result, interactions

    def run(self, pdf_path: Path, user_hint: str | None = None) -> tuple[Path, str, str, list[StageInteraction]]:
        """Run all three stages and return (target_folder, filename, summary, interactions)."""
        if self.config.lm_studio.use_vision:
            images = render_pages(pdf_path, self.config.lm_studio.vision_max_pages)
            text = ""
        else:
            images = None
            text = extract_text(pdf_path)
        summary, s1 = self.summarize(text, user_hint, images=images)
        target_folder, s2 = self.navigate_to_folder(text, summary, user_hint, images=images)
        filename, s3 = self.choose_filename(text, summary, target_folder, user_hint, images=images)
        return target_folder, filename, summary, s1 + s2 + s3
