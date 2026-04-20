"""Three-stage LLM pipeline: summarize → navigate → name."""

from __future__ import annotations

import re
from pathlib import Path

from sortai.config import Config
from sortai.folder_navigator import is_leaf, list_children
from sortai.llm_client import LMStudioClient
from sortai.pdf_reader import extract_text

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
    def __init__(self, config: Config, client: LMStudioClient) -> None:
        self.config = config
        self.client = client

    def summarize(self, text: str) -> str:
        """Stage 1 — summarize the document text."""
        template = self.client.load_prompt("summarize")
        prompt = template.replace("{{document_text}}", _truncate(text))
        return self.client.complete(prompt).strip()

    def navigate_to_folder(self, text: str, summary: str) -> Path:
        """Stage 2 — walk the archive tree to find the best target folder."""
        template = self.client.load_prompt("navigate")
        current = self.config.archive
        short_text = _truncate(text)

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
            choice = self.client.complete(prompt).strip().splitlines()[0].strip()

            if choice == "." or choice not in children:
                break

            current = current / choice

        return current

    def choose_filename(self, text: str, summary: str, target: Path) -> str:
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
        raw = self.client.complete(prompt).strip()
        return _sanitize_filename(raw)

    def run(self, pdf_path: Path) -> tuple[Path, str]:
        """Run all three stages and return (target_folder, filename)."""
        text = extract_text(pdf_path)
        summary = self.summarize(text)
        target_folder = self.navigate_to_folder(text, summary)
        filename = self.choose_filename(text, summary, target_folder)
        return target_folder, filename
