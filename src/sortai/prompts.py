"""Prompt loading and single-pass template rendering.

Prompt files live in ``prompts/*.md`` and use ``{{placeholder}}`` tokens.
:func:`render` substitutes every placeholder in a single pass so that
replacement text (e.g. untrusted document content) is never rescanned for
further placeholders — a document that literally contains ``{{user_hint}}``
cannot corrupt the rendered prompt. Optional placeholders, passed as ``None``,
have their entire line removed.
"""

from __future__ import annotations

import re
from pathlib import Path

_PLACEHOLDER = re.compile(r"\{\{(\w+)\}\}")


def load_prompt(prompts_dir: Path, name: str) -> str:
    """Return the contents of ``{prompts_dir}/{name}.md``."""
    return (Path(prompts_dir) / f"{name}.md").read_text(encoding="utf-8")


def render(template: str, **values: str | None) -> str:
    """Substitute ``{{placeholder}}`` tokens in *template* in a single pass.

    Each keyword maps a placeholder name to its value. A value of ``None``
    marks an optional placeholder that is absent: the whole line containing it
    is removed. Any string value (including "") is substituted literally.

    All substitutions happen in one pass over the template, so replacement
    values are never rescanned. Placeholders with no matching keyword are left
    untouched.
    """
    absent = [key for key, value in values.items() if value is None]
    if absent:
        line_pattern = re.compile(
            r"[^\n]*\{\{(?:" + "|".join(re.escape(k) for k in absent) + r")\}\}[^\n]*\n?"
        )
        template = line_pattern.sub("", template)

    def _replace(match: re.Match[str]) -> str:
        value = values.get(match.group(1))
        return value if value is not None else match.group(0)

    return _PLACEHOLDER.sub(_replace, template)
