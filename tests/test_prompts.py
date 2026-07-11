"""Unit tests for sortai.prompts — prompt loading and single-pass rendering."""
from __future__ import annotations

from pathlib import Path

import pytest

from sortai.prompts import load_prompt, render


# ---------------------------------------------------------------------------
# load_prompt
# ---------------------------------------------------------------------------

class TestLoadPrompt:
    def test_reads_correct_file(self, tmp_path: Path):
        (tmp_path / "classify.md").write_text("Classify the document.", encoding="utf-8")
        assert load_prompt(tmp_path, "classify") == "Classify the document."

    def test_reads_by_name(self, tmp_path: Path):
        (tmp_path / "summary.md").write_text("Summarize this.", encoding="utf-8")
        assert load_prompt(tmp_path, "summary") == "Summarize this."

    def test_accepts_str_dir(self, tmp_path: Path):
        (tmp_path / "x.md").write_text("hello", encoding="utf-8")
        assert load_prompt(str(tmp_path), "x") == "hello"

    def test_missing_prompt_raises_file_not_found(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            load_prompt(tmp_path, "nonexistent")


# ---------------------------------------------------------------------------
# render
# ---------------------------------------------------------------------------

class TestRender:
    def test_substitutes_single_placeholder(self):
        assert render("Hello {{name}}!", name="world") == "Hello world!"

    def test_substitutes_multiple_placeholders(self):
        result = render("{{a}}-{{b}}-{{a}}", a="x", b="y")
        assert result == "x-y-x"

    def test_empty_string_value_substitutes_empty(self):
        assert render("[{{x}}]", x="") == "[]"

    def test_none_value_strips_whole_line(self):
        template = "keep\nHINT: {{hint}}\nkeep2"
        assert render(template, hint=None) == "keep\nkeep2"

    def test_none_value_strips_only_its_line_not_neighbours(self):
        template = "A: {{a}}\nB: {{b}}\nC: {{a}}"
        # b absent → its line goes; a is substituted on the remaining lines
        assert render(template, a="1", b=None) == "A: 1\nC: 1"

    def test_present_value_keeps_line(self):
        template = "HINT: {{hint}}"
        assert render(template, hint="do this") == "HINT: do this"

    def test_unknown_placeholder_left_untouched(self):
        assert render("{{known}} {{unknown}}", known="k") == "k {{unknown}}"

    def test_replacement_text_not_rescanned(self):
        """A value containing a placeholder token must not be re-substituted."""
        result = render("{{doc}} | {{hint}}", doc="evil {{hint}} text", hint="SAFE")
        assert result == "evil {{hint}} text | SAFE"

    def test_replacement_text_placeholder_survives_when_optional_absent(self):
        """Document content with {{hint}} is untouched even when hint is stripped."""
        template = "DOC: {{doc}}\nHINT: {{hint}}"
        result = render(template, doc="has {{hint}} inside", hint=None)
        # Stripping the last line consumes only its own (absent) newline; the
        # preceding line keeps its trailing newline.
        assert result == "DOC: has {{hint}} inside\n"

    def test_multiple_absent_placeholders_stripped(self):
        template = "A: {{a}}\nB: {{b}}\nC: {{c}}"
        assert render(template, a=None, b="keep", c=None) == "B: keep\n"

    def test_no_placeholders_returns_template_unchanged(self):
        assert render("plain text", unused="x") == "plain text"
