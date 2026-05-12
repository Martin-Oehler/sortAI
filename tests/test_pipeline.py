"""Comprehensive unit tests for sortai.pipeline.Pipeline."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from sortai.config import Config, LMStudioConfig
from sortai.llm_client import LLMResponse
from sortai.pipeline import Pipeline, StageInteraction, _sanitize_filename


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def make_config(archive: Path, max_navigate_depth: int = 10) -> Config:
    """Return a minimal Config whose archive points to *archive*."""
    cfg = Config.__new__(Config)
    cfg.inbox = archive
    cfg.archive = archive
    cfg.prompts_dir = Path("prompts")
    cfg.log_file = Path("logs/sortai.jsonl")
    cfg.dry_run = False
    cfg.max_navigate_depth = max_navigate_depth
    cfg.lm_studio = LMStudioConfig()
    cfg.folder_description_filename = "folder-description.md"
    cfg.subfolder_preview_count = 5
    return cfg


def _r(content: str, reasoning: str = "") -> LLMResponse:
    """Shorthand to create an LLMResponse with plain content."""
    return LLMResponse(content=content, reasoning=reasoning)


def _rj(data: dict) -> LLMResponse:
    """Shorthand to create an LLMResponse with JSON content (structured output)."""
    return LLMResponse(content=json.dumps(data), reasoning="")


def _summarize_resp(summary: str = "summary", reason: str = "", can_classify: bool = True) -> LLMResponse:
    return _rj({"can_classify": can_classify, "summary": summary, "reason": reason})


def _navigate_resp(choice: str = ".", reasoning: str = "") -> LLMResponse:
    return _rj({"reasoning": reasoning, "choice": choice})


def _filename_resp(filename: str = "document", reasoning: str = "") -> LLMResponse:
    return _rj({"reasoning": reasoning, "filename": filename})


# ---------------------------------------------------------------------------
# _sanitize_filename
# ---------------------------------------------------------------------------

class TestSanitizeFilename:
    def test_plain_string_gets_pdf_extension(self):
        assert _sanitize_filename("invoice") == "invoice.pdf"

    def test_already_has_pdf_extension(self):
        # Dots are replaced by underscores, so "invoice.pdf" → "invoice_pdf" → "invoice_pdf.pdf"
        result = _sanitize_filename("invoice.pdf")
        assert result == "invoice_pdf.pdf"

    def test_uppercase_is_lowercased(self):
        assert _sanitize_filename("Invoice") == "invoice.pdf"

    def test_spaces_replaced_with_underscore(self):
        assert _sanitize_filename("my document") == "my_document.pdf"

    def test_illegal_chars_replaced(self):
        result = _sanitize_filename("hello/world:foo")
        assert "/" not in result
        assert ":" not in result
        assert result.endswith(".pdf")

    def test_leading_trailing_whitespace_stripped(self):
        assert _sanitize_filename("  invoice  ") == "invoice.pdf"

    def test_empty_string_returns_document_pdf(self):
        assert _sanitize_filename("") == "document.pdf"

    def test_only_illegal_chars_returns_document_pdf(self):
        # After substitution every char becomes "_"; strip("_") → ""; fallback "document"
        assert _sanitize_filename("!!!") == "document.pdf"

    def test_multiple_underscores_collapsed(self):
        result = _sanitize_filename("a   b")
        # Three spaces → three underscores → collapsed to one
        assert result == "a_b.pdf"

    def test_newline_replaced_by_underscore(self):
        # Structured output gives a single stem; newline is just another illegal char.
        result = _sanitize_filename("first_line\nsecond_line")
        assert result == "first_line_second_line.pdf"

    def test_dot_in_name_replaced(self):
        # Dots are not in the allowed set [a-z0-9_-], so they become underscores.
        # "my.file.pdf" → "my_file_pdf" → append .pdf → "my_file_pdf.pdf"
        result = _sanitize_filename("my.file.pdf")
        assert result == "my_file_pdf.pdf"

    def test_german_umlauts_transliterated(self):
        assert _sanitize_filename("für_märz") == "fuer_maerz.pdf"

    def test_sharp_s_transliterated(self):
        assert _sanitize_filename("straße") == "strasse.pdf"

    def test_accented_chars_decomposed(self):
        assert _sanitize_filename("café") == "cafe.pdf"


# ---------------------------------------------------------------------------
# Pipeline.summarize
# ---------------------------------------------------------------------------

class TestSummarize:
    def test_uses_summarize_template(self, tmp_path: Path):
        client = MagicMock()
        client.load_prompt.side_effect = lambda name: f"TMPL:{name}:{{{{document_text}}}}"
        client.complete_structured.return_value = _summarize_resp("summary text")
        pipeline = Pipeline(make_config(tmp_path), client)

        result, interactions = pipeline.summarize("hello doc")

        client.load_prompt.assert_called_once_with("summarize")
        assert client.complete_structured.call_args[0][0] == "TMPL:summarize:hello doc"
        assert result == "summary text"

    def test_text_substituted_into_template(self, tmp_path: Path):
        client = MagicMock()
        client.load_prompt.return_value = "Please summarize: {{document_text}}"
        client.complete_structured.return_value = _summarize_resp("the summary")
        pipeline = Pipeline(make_config(tmp_path), client)

        pipeline.summarize("my important text")

        assert client.complete_structured.call_args[0][0] == "Please summarize: my important text"

    def test_long_text_is_truncated(self, tmp_path: Path):
        client = MagicMock()
        client.load_prompt.return_value = "{{document_text}}"
        client.complete_structured.return_value = _summarize_resp("ok")
        pipeline = Pipeline(make_config(tmp_path), client)

        long_text = "x" * 5000
        pipeline.summarize(long_text)

        sent = client.complete_structured.call_args[0][0]
        assert len(sent) < 5000
        assert "truncated" in sent

    def test_returns_stripped_completion(self, tmp_path: Path):
        client = MagicMock()
        client.load_prompt.return_value = "{{document_text}}"
        client.complete_structured.return_value = _summarize_resp("\n  result with spaces \n")
        pipeline = Pipeline(make_config(tmp_path), client)

        result, _ = pipeline.summarize("text")
        assert result == "result with spaces"

    def test_returns_interaction_with_correct_fields(self, tmp_path: Path):
        client = MagicMock()
        client.load_prompt.return_value = "Summarize: {{document_text}}"
        client.complete_structured.return_value = _summarize_resp("my summary", reason="thought process")
        pipeline = Pipeline(make_config(tmp_path), client)

        _, interactions = pipeline.summarize("doc text")

        assert len(interactions) == 1
        ix = interactions[0]
        assert ix["stage"] == "summarize"
        assert ix["step"] == 1
        assert ix["answer"] == "my summary"
        assert ix["reasoning"] == "thought process"
        assert "doc text" in ix["prompt"]

    def test_interaction_reasoning_empty_when_absent(self, tmp_path: Path):
        client = MagicMock()
        client.load_prompt.return_value = "{{document_text}}"
        client.complete_structured.return_value = _summarize_resp("summary", reason="")
        pipeline = Pipeline(make_config(tmp_path), client)

        _, interactions = pipeline.summarize("text")
        assert interactions[0]["reasoning"] == ""


# ---------------------------------------------------------------------------
# Pipeline.navigate_to_folder
# ---------------------------------------------------------------------------

class TestNavigateToFolder:
    def _make_archive(self, tmp_path: Path, structure: dict) -> Path:
        """Create nested directories from a dict like {"a": {"b": {}}, "c": {}}."""
        def _create(base: Path, tree: dict):
            for name, subtree in tree.items():
                child = base / name
                child.mkdir()
                _create(child, subtree)
        _create(tmp_path, structure)
        return tmp_path

    def test_stops_at_leaf_node(self, tmp_path: Path):
        """If the archive root is a leaf (no sub-dirs), returns root immediately."""
        client = MagicMock()
        client.load_prompt.return_value = "{{current_folder}}{{folder_listing}}{{summary}}{{document_text}}"
        pipeline = Pipeline(make_config(tmp_path), client)

        result, interactions = pipeline.navigate_to_folder("doc text", "summary")

        assert result == tmp_path
        client.complete_structured.assert_not_called()
        assert interactions == []

    def test_llm_returning_dot_stops_navigation(self, tmp_path: Path):
        """LLM returning '.' means 'stay here'; loop breaks."""
        self._make_archive(tmp_path, {"invoices": {}, "contracts": {}})
        client = MagicMock()
        client.load_prompt.return_value = "{{current_folder}}{{folder_listing}}{{summary}}{{document_text}}"
        client.complete_structured.return_value = _navigate_resp(".")
        pipeline = Pipeline(make_config(tmp_path), client)

        result, interactions = pipeline.navigate_to_folder("doc", "sum")

        assert result == tmp_path
        client.complete_structured.assert_called_once()
        assert len(interactions) == 1
        assert interactions[0]["answer"] == "."

    def test_llm_returning_invalid_name_raises(self, tmp_path: Path):
        """LLM returning a folder not in the archive causes FileNotFoundError on next iteration.

        In production, structured output constrains choices to valid children, so this
        cannot happen. The test documents the raw behaviour.
        """
        self._make_archive(tmp_path, {"invoices": {}, "contracts": {}})
        client = MagicMock()
        client.load_prompt.return_value = "{{current_folder}}{{folder_listing}}{{summary}}{{document_text}}"
        client.complete_structured.return_value = _navigate_resp("nonexistent_folder")
        pipeline = Pipeline(make_config(tmp_path), client)

        with pytest.raises(FileNotFoundError):
            pipeline.navigate_to_folder("doc", "sum")

    def test_follows_valid_path_one_level(self, tmp_path: Path):
        """LLM choosing a valid child descends into it."""
        self._make_archive(tmp_path, {"invoices": {}, "contracts": {}})
        client = MagicMock()
        client.load_prompt.return_value = "{{current_folder}}{{folder_listing}}{{summary}}{{document_text}}"
        client.complete_structured.return_value = _navigate_resp("invoices")
        pipeline = Pipeline(make_config(tmp_path), client)

        result, _ = pipeline.navigate_to_folder("doc", "sum")

        assert result == tmp_path / "invoices"

    def test_follows_valid_path_multiple_levels(self, tmp_path: Path):
        """LLM can descend multiple levels through the tree."""
        self._make_archive(tmp_path, {"finance": {"invoices": {}, "receipts": {}}})
        client = MagicMock()
        client.load_prompt.return_value = "{{current_folder}}{{folder_listing}}{{summary}}{{document_text}}"
        client.complete_structured.side_effect = [_navigate_resp("finance"), _navigate_resp("invoices")]
        pipeline = Pipeline(make_config(tmp_path), client)

        result, interactions = pipeline.navigate_to_folder("doc", "sum")

        assert result == tmp_path / "finance" / "invoices"
        assert client.complete_structured.call_count == 2
        assert len(interactions) == 2
        assert interactions[0]["step"] == 1
        assert interactions[1]["step"] == 2

    def test_respects_max_navigate_depth(self, tmp_path: Path):
        """Navigation never exceeds max_navigate_depth iterations."""
        # Build a deep chain: root/a/b/c/d/e (depth 5)
        self._make_archive(tmp_path, {"a": {"b": {"c": {"d": {"e": {}}}}}})
        client = MagicMock()
        client.load_prompt.return_value = "{{current_folder}}{{folder_listing}}{{summary}}{{document_text}}"
        # LLM always picks the first child it sees
        client.complete_structured.side_effect = [
            _navigate_resp(v) for v in ["a", "b", "c", "d", "e", "should_not_reach"]
        ]
        cfg = make_config(tmp_path, max_navigate_depth=3)
        pipeline = Pipeline(cfg, client)

        result, _ = pipeline.navigate_to_folder("doc", "sum")

        # With depth=3 we can take at most 3 steps: root→a→b→c
        assert client.complete_structured.call_count <= 3
        assert result == tmp_path / "a" / "b" / "c"

    def test_template_substitution_in_navigate_prompt(self, tmp_path: Path):
        """navigate prompt receives correct substitutions."""
        self._make_archive(tmp_path, {"docs": {}})
        client = MagicMock()
        template = "FOLDER:{{current_folder}} LIST:{{folder_listing}} SUM:{{summary}} TEXT:{{document_text}}"
        client.load_prompt.return_value = template
        client.complete_structured.return_value = _navigate_resp(".")
        pipeline = Pipeline(make_config(tmp_path), client)

        pipeline.navigate_to_folder("mytext", "mysum")

        prompt_sent = client.complete_structured.call_args[0][0]
        assert f"FOLDER:{tmp_path}" in prompt_sent
        assert "LIST:- docs" in prompt_sent
        assert "SUM:mysum" in prompt_sent
        assert "TEXT:mytext" in prompt_sent

    def test_interaction_stage_and_reasoning(self, tmp_path: Path):
        self._make_archive(tmp_path, {"bank": {}})
        client = MagicMock()
        client.load_prompt.return_value = "{{current_folder}}{{folder_listing}}{{summary}}{{document_text}}"
        client.complete_structured.return_value = _navigate_resp(".", reasoning="no deeper folder")
        pipeline = Pipeline(make_config(tmp_path), client)

        _, interactions = pipeline.navigate_to_folder("doc", "sum")

        assert interactions[0]["stage"] == "navigate"
        assert interactions[0]["reasoning"] == "no deeper folder"


# ---------------------------------------------------------------------------
# Pipeline.choose_filename
# ---------------------------------------------------------------------------

class TestChooseFilename:
    def test_uses_name_file_template(self, tmp_path: Path):
        client = MagicMock()
        client.load_prompt.return_value = "{{target_folder}}{{existing_files}}{{summary}}{{document_text}}"
        client.complete_structured.return_value = _filename_resp("myfile")
        pipeline = Pipeline(make_config(tmp_path), client)

        pipeline.choose_filename("text", "sum", tmp_path)

        client.load_prompt.assert_called_once_with("name_file")

    def test_template_substitution_correct(self, tmp_path: Path):
        (tmp_path / "existing.pdf").write_bytes(b"%PDF")
        client = MagicMock()
        template = "FOLDER:{{target_folder}} FILES:{{existing_files}} SUM:{{summary}} TEXT:{{document_text}}"
        client.load_prompt.return_value = template
        client.complete_structured.return_value = _filename_resp("result")
        pipeline = Pipeline(make_config(tmp_path), client)

        pipeline.choose_filename("mytext", "mysum", tmp_path)

        prompt_sent = client.complete_structured.call_args[0][0]
        assert f"FOLDER:{tmp_path}" in prompt_sent
        assert "FILES:- existing.pdf" in prompt_sent
        assert "SUM:mysum" in prompt_sent
        assert "TEXT:mytext" in prompt_sent

    def test_empty_target_dir_shows_none(self, tmp_path: Path):
        """An empty target directory shows '(none)' for existing files."""
        client = MagicMock()
        client.load_prompt.return_value = "{{existing_files}}"
        client.complete_structured.return_value = _filename_resp("name")
        pipeline = Pipeline(make_config(tmp_path), client)

        pipeline.choose_filename("text", "sum", tmp_path)

        prompt_sent = client.complete_structured.call_args[0][0]
        assert "(none)" in prompt_sent

    def test_nonexistent_target_dir_shows_none(self, tmp_path: Path):
        """If target dir does not exist yet, existing_files is '(none)'."""
        client = MagicMock()
        client.load_prompt.return_value = "{{existing_files}}"
        client.complete_structured.return_value = _filename_resp("name")
        pipeline = Pipeline(make_config(tmp_path), client)

        nonexistent = tmp_path / "new_folder"
        pipeline.choose_filename("text", "sum", nonexistent)

        prompt_sent = client.complete_structured.call_args[0][0]
        assert "(none)" in prompt_sent

    def test_sanitizes_llm_output(self, tmp_path: Path):
        """choose_filename sanitizes the raw LLM response."""
        client = MagicMock()
        client.load_prompt.return_value = "{{document_text}}"
        client.complete_structured.return_value = _filename_resp("  My Invoice File  ")
        pipeline = Pipeline(make_config(tmp_path), client)

        result, _ = pipeline.choose_filename("text", "sum", tmp_path)

        assert result == "my_invoice_file.pdf"

    def test_returns_pdf_extension(self, tmp_path: Path):
        client = MagicMock()
        client.load_prompt.return_value = "{{document_text}}"
        client.complete_structured.return_value = _filename_resp("contract")
        pipeline = Pipeline(make_config(tmp_path), client)

        result, _ = pipeline.choose_filename("text", "sum", tmp_path)

        assert result.endswith(".pdf")

    def test_existing_files_capped_at_20(self, tmp_path: Path):
        """At most 20 existing files are forwarded to the prompt."""
        for i in range(25):
            (tmp_path / f"file_{i:02d}.pdf").write_bytes(b"%PDF")
        client = MagicMock()
        client.load_prompt.return_value = "{{existing_files}}"
        client.complete_structured.return_value = _filename_resp("name")
        pipeline = Pipeline(make_config(tmp_path), client)

        pipeline.choose_filename("text", "sum", tmp_path)

        prompt_sent = client.complete_structured.call_args[0][0]
        # Count bullet points; should be at most 20
        bullets = [line for line in prompt_sent.splitlines() if line.startswith("- ")]
        assert len(bullets) <= 20

    def test_returns_interaction_with_correct_fields(self, tmp_path: Path):
        client = MagicMock()
        client.load_prompt.return_value = "{{document_text}}"
        client.complete_structured.return_value = _filename_resp("my_file", reasoning="based on content")
        pipeline = Pipeline(make_config(tmp_path), client)

        _, interactions = pipeline.choose_filename("text", "sum", tmp_path)

        assert len(interactions) == 1
        ix = interactions[0]
        assert ix["stage"] == "choose_filename"
        assert ix["step"] == 1
        assert ix["reasoning"] == "based on content"


# ---------------------------------------------------------------------------
# Pipeline.run
# ---------------------------------------------------------------------------

class TestRun:
    def test_orchestrates_all_three_stages(self, tmp_path: Path):
        """run() calls extract_text, summarize, navigate_to_folder, choose_filename."""
        (tmp_path / "misc").mkdir()  # give navigate a folder so it calls the LLM
        client = MagicMock()
        client.load_prompt.return_value = "{{document_text}}{{summary}}{{current_folder}}{{folder_listing}}{{target_folder}}{{existing_files}}"
        client.complete_structured.side_effect = [
            _summarize_resp("the summary"),
            _navigate_resp("."),
            _filename_resp("filename"),
        ]
        cfg = make_config(tmp_path)
        pipeline = Pipeline(cfg, client)

        pdf_path = tmp_path / "test.pdf"

        with patch("sortai.pipeline.extract_text", return_value="extracted text") as mock_extract:
            target, name, summary, interactions = pipeline.run(pdf_path)

        mock_extract.assert_called_once_with(pdf_path)
        assert client.complete_structured.call_count == 3
        assert isinstance(target, Path)
        assert name.endswith(".pdf")

    def test_calls_extract_text_on_pdf_path(self, tmp_path: Path):
        client = MagicMock()
        client.load_prompt.return_value = "{{document_text}}{{summary}}{{current_folder}}{{folder_listing}}{{target_folder}}{{existing_files}}"
        client.complete_structured.side_effect = [
            _summarize_resp("result"),
            _filename_resp("result"),
        ]
        pipeline = Pipeline(make_config(tmp_path), client)

        fake_pdf = tmp_path / "my.pdf"

        with patch("sortai.pipeline.extract_text", return_value="text") as mock_extract:
            pipeline.run(fake_pdf)

        mock_extract.assert_called_once_with(fake_pdf)

    def test_returns_tuple_of_path_str_str_list(self, tmp_path: Path):
        client = MagicMock()
        client.load_prompt.return_value = "{{document_text}}{{summary}}{{current_folder}}{{folder_listing}}{{target_folder}}{{existing_files}}"
        # archive has no subdirs → navigate makes 0 LLM calls; only summarize + filename
        client.complete_structured.side_effect = [
            _summarize_resp("summary"),
            _filename_resp("name"),
        ]
        pipeline = Pipeline(make_config(tmp_path), client)

        with patch("sortai.pipeline.extract_text", return_value="text"):
            result = pipeline.run(tmp_path / "doc.pdf")

        assert isinstance(result, tuple)
        assert len(result) == 4
        assert isinstance(result[0], Path)
        assert isinstance(result[1], str)
        assert isinstance(result[2], str)
        assert isinstance(result[3], list)

    def test_summary_passed_to_navigate_and_name(self, tmp_path: Path):
        """The summary returned from stage 1 is used in stages 2 and 3."""
        (tmp_path / "misc").mkdir()  # give navigate a folder so it calls the LLM
        client = MagicMock()
        template = "{{document_text}}{{summary}}{{current_folder}}{{folder_listing}}{{target_folder}}{{existing_files}}"
        client.load_prompt.return_value = template
        client.complete_structured.side_effect = [
            _summarize_resp("UNIQUE_SUMMARY_VALUE"),
            _navigate_resp("."),
            _filename_resp("filename"),
        ]
        pipeline = Pipeline(make_config(tmp_path), client)

        with patch("sortai.pipeline.extract_text", return_value="text"):
            pipeline.run(tmp_path / "doc.pdf")

        # The navigate call (2nd complete_structured) should receive the summary in its prompt
        navigate_prompt = client.complete_structured.call_args_list[1][0][0]
        assert "UNIQUE_SUMMARY_VALUE" in navigate_prompt

        # The name_file call (3rd complete_structured) should too
        name_prompt = client.complete_structured.call_args_list[2][0][0]
        assert "UNIQUE_SUMMARY_VALUE" in name_prompt

    def test_navigate_result_passed_to_choose_filename(self, tmp_path: Path):
        """The folder returned from navigate_to_folder is the target for choose_filename."""
        (tmp_path / "contracts").mkdir()
        client = MagicMock()
        template = "{{document_text}}{{summary}}{{current_folder}}{{folder_listing}}{{target_folder}}{{existing_files}}"
        client.load_prompt.return_value = template
        client.complete_structured.side_effect = [
            _summarize_resp("summary"),
            _navigate_resp("contracts"),
            _filename_resp("doc"),
        ]
        pipeline = Pipeline(make_config(tmp_path), client)

        with patch("sortai.pipeline.extract_text", return_value="text"):
            target, *_ = pipeline.run(tmp_path / "doc.pdf")

        assert target == tmp_path / "contracts"

    def test_run_combines_all_stage_interactions(self, tmp_path: Path):
        """run() returns concatenation of all stage interactions."""
        (tmp_path / "misc").mkdir()
        client = MagicMock()
        client.load_prompt.return_value = "{{document_text}}{{summary}}{{current_folder}}{{folder_listing}}{{target_folder}}{{existing_files}}"
        client.complete_structured.side_effect = [
            _summarize_resp("summary"),
            _navigate_resp("."),
            _filename_resp("name"),
        ]
        pipeline = Pipeline(make_config(tmp_path), client)

        with patch("sortai.pipeline.extract_text", return_value="text"):
            _, _, _, interactions = pipeline.run(tmp_path / "doc.pdf")

        stages = [ix["stage"] for ix in interactions]
        assert "summarize" in stages
        assert "navigate" in stages
        assert "choose_filename" in stages
