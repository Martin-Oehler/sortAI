"""Comprehensive unit tests for sortai.pipeline.Pipeline."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch, call

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
    return cfg


def _r(content: str, reasoning: str = "") -> LLMResponse:
    """Shorthand to create an LLMResponse."""
    return LLMResponse(content=content, reasoning=reasoning)


def make_client(load_prompt_side_effect=None, complete_side_effect=None,
                complete_return_value="result"):
    """Return a MagicMock that looks like an LMStudioClient."""
    client = MagicMock()
    if load_prompt_side_effect is not None:
        client.load_prompt.side_effect = load_prompt_side_effect
    else:
        client.load_prompt.return_value = "template {{document_text}}"
    if complete_side_effect is not None:
        if isinstance(complete_side_effect, list):
            client.complete.side_effect = [
                _r(v) if isinstance(v, str) else v for v in complete_side_effect
            ]
        else:
            client.complete.side_effect = complete_side_effect
    else:
        client.complete.return_value = _r(complete_return_value)
    return client


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

    def test_only_first_line_used(self):
        result = _sanitize_filename("first_line\nsecond_line")
        assert result == "first_line.pdf"

    def test_dot_in_name_replaced(self):
        # Dots are not in the allowed set [a-z0-9_-], so they become underscores.
        # "my.file.pdf" → "my_file_pdf" → append .pdf → "my_file_pdf.pdf"
        result = _sanitize_filename("my.file.pdf")
        assert result == "my_file_pdf.pdf"


# ---------------------------------------------------------------------------
# Pipeline.summarize
# ---------------------------------------------------------------------------

class TestSummarize:
    def test_uses_summarize_template(self, tmp_path: Path):
        client = make_client(load_prompt_side_effect=lambda name: f"TMPL:{name}:{{{{document_text}}}}")
        client.complete.return_value = _r("  summary text  ")
        pipeline = Pipeline(make_config(tmp_path), client)

        result, interactions = pipeline.summarize("hello doc")

        client.load_prompt.assert_called_once_with("summarize")
        client.complete.assert_called_once_with("TMPL:summarize:hello doc")
        assert result == "summary text"  # stripped

    def test_text_substituted_into_template(self, tmp_path: Path):
        client = MagicMock()
        client.load_prompt.return_value = "Please summarize: {{document_text}}"
        client.complete.return_value = _r("the summary")
        pipeline = Pipeline(make_config(tmp_path), client)

        pipeline.summarize("my important text")

        client.complete.assert_called_once_with("Please summarize: my important text")

    def test_long_text_is_truncated(self, tmp_path: Path):
        client = MagicMock()
        client.load_prompt.return_value = "{{document_text}}"
        client.complete.return_value = _r("ok")
        pipeline = Pipeline(make_config(tmp_path), client)

        long_text = "x" * 5000
        pipeline.summarize(long_text)

        sent = client.complete.call_args[0][0]
        assert len(sent) < 5000
        assert "truncated" in sent

    def test_returns_stripped_completion(self, tmp_path: Path):
        client = MagicMock()
        client.load_prompt.return_value = "{{document_text}}"
        client.complete.return_value = _r("\n  result with spaces \n")
        pipeline = Pipeline(make_config(tmp_path), client)

        result, _ = pipeline.summarize("text")
        assert result == "result with spaces"

    def test_returns_interaction_with_correct_fields(self, tmp_path: Path):
        client = MagicMock()
        client.load_prompt.return_value = "Summarize: {{document_text}}"
        client.complete.return_value = _r("my summary", reasoning="thought process")
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
        client.complete.return_value = _r("summary")
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
        client.complete.assert_not_called()
        assert interactions == []

    def test_llm_returning_dot_stops_navigation(self, tmp_path: Path):
        """LLM returning '.' means 'stay here'; loop breaks."""
        self._make_archive(tmp_path, {"invoices": {}, "contracts": {}})
        client = MagicMock()
        client.load_prompt.return_value = "{{current_folder}}{{folder_listing}}{{summary}}{{document_text}}"
        client.complete.return_value = _r(".")
        pipeline = Pipeline(make_config(tmp_path), client)

        result, interactions = pipeline.navigate_to_folder("doc", "sum")

        assert result == tmp_path
        client.complete.assert_called_once()
        assert len(interactions) == 1
        assert interactions[0]["answer"] == "."

    def test_llm_returning_invalid_name_stops_navigation(self, tmp_path: Path):
        """LLM returning a name not in the current listing stops the loop."""
        self._make_archive(tmp_path, {"invoices": {}, "contracts": {}})
        client = MagicMock()
        client.load_prompt.return_value = "{{current_folder}}{{folder_listing}}{{summary}}{{document_text}}"
        client.complete.return_value = _r("nonexistent_folder")
        pipeline = Pipeline(make_config(tmp_path), client)

        result, _ = pipeline.navigate_to_folder("doc", "sum")

        assert result == tmp_path

    def test_follows_valid_path_one_level(self, tmp_path: Path):
        """LLM choosing a valid child descends into it."""
        self._make_archive(tmp_path, {"invoices": {}, "contracts": {}})
        client = MagicMock()
        client.load_prompt.return_value = "{{current_folder}}{{folder_listing}}{{summary}}{{document_text}}"
        # First call: choose "invoices"; second call (if any): invoices is leaf → no call
        client.complete.return_value = _r("invoices")
        pipeline = Pipeline(make_config(tmp_path), client)

        result, _ = pipeline.navigate_to_folder("doc", "sum")

        assert result == tmp_path / "invoices"

    def test_follows_valid_path_multiple_levels(self, tmp_path: Path):
        """LLM can descend multiple levels through the tree."""
        self._make_archive(tmp_path, {"finance": {"invoices": {}, "receipts": {}}})
        client = MagicMock()
        client.load_prompt.return_value = "{{current_folder}}{{folder_listing}}{{summary}}{{document_text}}"
        # Level 1: pick "finance"; level 2: pick "invoices"; invoices is leaf → stop
        client.complete.side_effect = [_r("finance"), _r("invoices")]
        pipeline = Pipeline(make_config(tmp_path), client)

        result, interactions = pipeline.navigate_to_folder("doc", "sum")

        assert result == tmp_path / "finance" / "invoices"
        assert client.complete.call_count == 2
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
        client.complete.side_effect = [_r(v) for v in ["a", "b", "c", "d", "e", "should_not_reach"]]
        cfg = make_config(tmp_path, max_navigate_depth=3)
        pipeline = Pipeline(cfg, client)

        result, _ = pipeline.navigate_to_folder("doc", "sum")

        # With depth=3 we can take at most 3 steps: root→a→b→c
        assert client.complete.call_count <= 3
        assert result == tmp_path / "a" / "b" / "c"

    def test_template_substitution_in_navigate_prompt(self, tmp_path: Path):
        """navigate prompt receives correct substitutions."""
        self._make_archive(tmp_path, {"docs": {}})
        client = MagicMock()
        template = "FOLDER:{{current_folder}} LIST:{{folder_listing}} SUM:{{summary}} TEXT:{{document_text}}"
        client.load_prompt.return_value = template
        client.complete.return_value = _r(".")
        pipeline = Pipeline(make_config(tmp_path), client)

        pipeline.navigate_to_folder("mytext", "mysum")

        prompt_sent = client.complete.call_args[0][0]
        assert f"FOLDER:{tmp_path}" in prompt_sent
        assert "LIST:- docs" in prompt_sent
        assert "SUM:mysum" in prompt_sent
        assert "TEXT:mytext" in prompt_sent

    def test_llm_response_only_first_line_used(self, tmp_path: Path):
        """Only the first line of the LLM response is used as the folder choice."""
        self._make_archive(tmp_path, {"invoices": {}})
        client = MagicMock()
        client.load_prompt.return_value = "{{current_folder}}{{folder_listing}}{{summary}}{{document_text}}"
        # Extra lines should be ignored; first line is "invoices"
        client.complete.return_value = _r("invoices\nsome explanation\nmore text")
        pipeline = Pipeline(make_config(tmp_path), client)

        result, _ = pipeline.navigate_to_folder("doc", "sum")

        assert result == tmp_path / "invoices"

    def test_interaction_stage_and_reasoning(self, tmp_path: Path):
        self._make_archive(tmp_path, {"bank": {}})
        client = MagicMock()
        client.load_prompt.return_value = "{{current_folder}}{{folder_listing}}{{summary}}{{document_text}}"
        client.complete.return_value = _r(".", reasoning="no deeper folder")
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
        client.complete.return_value = _r("myfile")
        pipeline = Pipeline(make_config(tmp_path), client)

        pipeline.choose_filename("text", "sum", tmp_path)

        client.load_prompt.assert_called_once_with("name_file")

    def test_template_substitution_correct(self, tmp_path: Path):
        (tmp_path / "existing.pdf").write_bytes(b"%PDF")
        client = MagicMock()
        template = "FOLDER:{{target_folder}} FILES:{{existing_files}} SUM:{{summary}} TEXT:{{document_text}}"
        client.load_prompt.return_value = template
        client.complete.return_value = _r("result")
        pipeline = Pipeline(make_config(tmp_path), client)

        pipeline.choose_filename("mytext", "mysum", tmp_path)

        prompt_sent = client.complete.call_args[0][0]
        assert f"FOLDER:{tmp_path}" in prompt_sent
        assert "FILES:- existing.pdf" in prompt_sent
        assert "SUM:mysum" in prompt_sent
        assert "TEXT:mytext" in prompt_sent

    def test_empty_target_dir_shows_none(self, tmp_path: Path):
        """An empty target directory shows '(none)' for existing files."""
        client = MagicMock()
        client.load_prompt.return_value = "{{existing_files}}"
        client.complete.return_value = _r("name")
        pipeline = Pipeline(make_config(tmp_path), client)

        pipeline.choose_filename("text", "sum", tmp_path)

        prompt_sent = client.complete.call_args[0][0]
        assert "(none)" in prompt_sent

    def test_nonexistent_target_dir_shows_none(self, tmp_path: Path):
        """If target dir does not exist yet, existing_files is '(none)'."""
        client = MagicMock()
        client.load_prompt.return_value = "{{existing_files}}"
        client.complete.return_value = _r("name")
        pipeline = Pipeline(make_config(tmp_path), client)

        nonexistent = tmp_path / "new_folder"
        pipeline.choose_filename("text", "sum", nonexistent)

        prompt_sent = client.complete.call_args[0][0]
        assert "(none)" in prompt_sent

    def test_sanitizes_llm_output(self, tmp_path: Path):
        """choose_filename sanitizes the raw LLM response."""
        client = MagicMock()
        client.load_prompt.return_value = "{{document_text}}"
        client.complete.return_value = _r("  My Invoice File  ")
        pipeline = Pipeline(make_config(tmp_path), client)

        result, _ = pipeline.choose_filename("text", "sum", tmp_path)

        assert result == "my_invoice_file.pdf"

    def test_returns_pdf_extension(self, tmp_path: Path):
        client = MagicMock()
        client.load_prompt.return_value = "{{document_text}}"
        client.complete.return_value = _r("contract")
        pipeline = Pipeline(make_config(tmp_path), client)

        result, _ = pipeline.choose_filename("text", "sum", tmp_path)

        assert result.endswith(".pdf")

    def test_existing_files_capped_at_20(self, tmp_path: Path):
        """At most 20 existing files are forwarded to the prompt."""
        for i in range(25):
            (tmp_path / f"file_{i:02d}.pdf").write_bytes(b"%PDF")
        client = MagicMock()
        client.load_prompt.return_value = "{{existing_files}}"
        client.complete.return_value = _r("name")
        pipeline = Pipeline(make_config(tmp_path), client)

        pipeline.choose_filename("text", "sum", tmp_path)

        prompt_sent = client.complete.call_args[0][0]
        # Count bullet points; should be at most 20
        bullets = [line for line in prompt_sent.splitlines() if line.startswith("- ")]
        assert len(bullets) <= 20

    def test_returns_interaction_with_correct_fields(self, tmp_path: Path):
        client = MagicMock()
        client.load_prompt.return_value = "{{document_text}}"
        client.complete.return_value = _r("my_file", reasoning="based on content")
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
        client.complete.side_effect = [_r("the summary"), _r("."), _r("filename")]
        cfg = make_config(tmp_path)
        pipeline = Pipeline(cfg, client)

        pdf_path = tmp_path / "test.pdf"

        with patch("sortai.pipeline.extract_text", return_value="extracted text") as mock_extract:
            target, name, summary, interactions = pipeline.run(pdf_path)

        mock_extract.assert_called_once_with(pdf_path)
        # Three LLM completions: summarize, navigate, name
        assert client.complete.call_count == 3
        assert isinstance(target, Path)
        assert name.endswith(".pdf")

    def test_calls_extract_text_on_pdf_path(self, tmp_path: Path):
        client = MagicMock()
        client.load_prompt.return_value = "{{document_text}}{{summary}}{{current_folder}}{{folder_listing}}{{target_folder}}{{existing_files}}"
        client.complete.return_value = _r("result")
        pipeline = Pipeline(make_config(tmp_path), client)

        fake_pdf = tmp_path / "my.pdf"

        with patch("sortai.pipeline.extract_text", return_value="text") as mock_extract:
            pipeline.run(fake_pdf)

        mock_extract.assert_called_once_with(fake_pdf)

    def test_returns_tuple_of_path_str_str_list(self, tmp_path: Path):
        client = MagicMock()
        client.load_prompt.return_value = "{{document_text}}{{summary}}{{current_folder}}{{folder_listing}}{{target_folder}}{{existing_files}}"
        client.complete.side_effect = [_r("summary"), _r("."), _r("name")]
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
        client.complete.side_effect = [_r("UNIQUE_SUMMARY_VALUE"), _r("."), _r("filename")]
        pipeline = Pipeline(make_config(tmp_path), client)

        with patch("sortai.pipeline.extract_text", return_value="text"):
            pipeline.run(tmp_path / "doc.pdf")

        # The navigate call (2nd complete) should receive the summary in its prompt
        navigate_prompt = client.complete.call_args_list[1][0][0]
        assert "UNIQUE_SUMMARY_VALUE" in navigate_prompt

        # The name_file call (3rd complete) should too
        name_prompt = client.complete.call_args_list[2][0][0]
        assert "UNIQUE_SUMMARY_VALUE" in name_prompt

    def test_navigate_result_passed_to_choose_filename(self, tmp_path: Path):
        """The folder returned from navigate_to_folder is the target for choose_filename."""
        (tmp_path / "contracts").mkdir()
        client = MagicMock()
        template = "{{document_text}}{{summary}}{{current_folder}}{{folder_listing}}{{target_folder}}{{existing_files}}"
        client.load_prompt.return_value = template
        # navigate picks "contracts" (leaf), then name_file returns "doc"
        client.complete.side_effect = [_r("summary"), _r("contracts"), _r("doc")]
        pipeline = Pipeline(make_config(tmp_path), client)

        with patch("sortai.pipeline.extract_text", return_value="text"):
            target, *_ = pipeline.run(tmp_path / "doc.pdf")

        assert target == tmp_path / "contracts"

    def test_run_combines_all_stage_interactions(self, tmp_path: Path):
        """run() returns concatenation of all stage interactions."""
        (tmp_path / "misc").mkdir()
        client = MagicMock()
        client.load_prompt.return_value = "{{document_text}}{{summary}}{{current_folder}}{{folder_listing}}{{target_folder}}{{existing_files}}"
        client.complete.side_effect = [_r("summary"), _r("."), _r("name")]
        pipeline = Pipeline(make_config(tmp_path), client)

        with patch("sortai.pipeline.extract_text", return_value="text"):
            _, _, _, interactions = pipeline.run(tmp_path / "doc.pdf")

        stages = [ix["stage"] for ix in interactions]
        assert "summarize" in stages
        assert "navigate" in stages
        assert "choose_filename" in stages
