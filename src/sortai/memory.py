"""Classification memory — rule-file format and learning functions.

Owns the markdown format of ``classification-memory.md`` (a numbered rule
list under a ``# Classification Memory`` header) plus the two LLM calls that
maintain it: :func:`learn_from_correction` and :func:`consolidate_memory`.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from rich.console import Console

from sortai.llm_client import LMStudioClient
from sortai.pipeline import StageInteraction, _apply_document, _log_exchange

MEMORY_HEADER = "# Classification Memory"

_console = Console()


# ---------------------------------------------------------------------------
# Rule-file format
# ---------------------------------------------------------------------------


def load_memory_text(path: Path) -> str | None:
    """Return the raw memory-file text (for prompt injection), or None if absent."""
    return path.read_text(encoding="utf-8") if path.exists() else None


def load_rules(path: Path) -> list[str]:
    """Parse the numbered rule list from a classification-memory file.

    Returns [] if the file does not exist. Blank lines and markdown headers
    (lines starting with ``#``) are skipped; a leading ``N. `` enumeration
    prefix is stripped; any other non-empty line is kept verbatim.
    """
    if not path.exists():
        return []
    rules: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r"^\d+\.\s+(.+)$", line)
        rules.append(m.group(1) if m else line)
    return rules


def save_rules(path: Path, rules: list[str]) -> None:
    """Write *rules* as a numbered markdown list under the standard header."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [MEMORY_HEADER + "\n"]
    for i, rule in enumerate(rules, 1):
        lines.append(f"{i}. {rule}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _format_numbered(rules: list[str]) -> str:
    return "\n".join(f"{i + 1}. {r}" for i, r in enumerate(rules))


# ---------------------------------------------------------------------------
# LLM learning functions
# ---------------------------------------------------------------------------


def learn_from_correction(
    client: LMStudioClient,
    doc_text: str,
    summary: str,
    previous_folder: str,
    user_hint: str,
    new_folder: str,
    verbose: bool = False,
) -> tuple[str | None, list[StageInteraction]]:
    """Ask the LLM whether a user correction yields a generalizable rule.

    Returns (rule, interactions). rule is None if the LLM decides not to learn.
    """
    template = client.load_prompt("learn")
    prompt = (
        _apply_document(template, doc_text, None)
        .replace("{{previous_folder}}", previous_folder)
        .replace("{{user_hint}}", user_hint)
        .replace("{{new_folder}}", new_folder)
        .replace("{{summary}}", summary)
    )
    schema = {
        "type": "object",
        "properties": {
            "reasoning": {"type": "string"},
            "should_learn": {"type": "boolean"},
            "rule": {"type": "string"},
        },
        "required": ["reasoning", "should_learn", "rule"],
        "additionalProperties": False,
    }
    resp = client.complete_structured(prompt, schema)
    parsed = json.loads(resp.content)
    rule = parsed["rule"].strip() if parsed["should_learn"] and parsed["rule"].strip() else None
    reasoning = parsed.get("reasoning", "")
    interactions = [StageInteraction(
        stage="learn",
        step=1,
        prompt=prompt,
        answer=parsed["rule"] if rule else "(nothing learned)",
        reasoning=reasoning,
    )]
    if verbose:
        _log_exchange(_console, "Memory — Learn", prompt,
            rule or "(nothing to learn)", reasoning=reasoning)
    return rule, interactions


def consolidate_memory(
    client: LMStudioClient,
    memory_path: Path,
    new_rule: str,
    verbose: bool = False,
) -> list[StageInteraction]:
    """Append new_rule to memory_path, then ask the LLM to consolidate.

    Writes the consolidated memory back and returns the interactions.
    """
    existing_rules = load_rules(memory_path)
    existing_rules.append(new_rule)
    current_memory = _format_numbered(existing_rules)

    template = client.load_prompt("consolidate")
    prompt = (
        template
        .replace("{{new_rule}}", new_rule)
        .replace("{{current_memory}}", current_memory)
    )
    schema = {
        "type": "object",
        "properties": {
            "reasoning": {"type": "string"},
            "rules": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["reasoning", "rules"],
        "additionalProperties": False,
    }
    resp = client.complete_structured(prompt, schema)
    parsed = json.loads(resp.content)
    consolidated = [r.strip() for r in parsed["rules"] if r.strip()]
    reasoning = parsed.get("reasoning", "")

    save_rules(memory_path, consolidated)

    interactions = [StageInteraction(
        stage="consolidate",
        step=1,
        prompt=prompt,
        answer=_format_numbered(consolidated),
        reasoning=reasoning,
    )]
    if verbose:
        _log_exchange(_console, "Memory — Consolidate", prompt,
            _format_numbered(consolidated), reasoning=reasoning)
    return interactions
