"""Unit tests for sortai.memory — rule-file format and learning functions."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from sortai.llm_client import LLMResponse
from sortai.memory import (
    MEMORY_HEADER,
    consolidate_memory,
    learn_from_correction,
    load_memory_text,
    load_rules,
    save_rules,
)


def _rj(data: dict) -> LLMResponse:
    """Shorthand for an LLMResponse with JSON content (structured output)."""
    return LLMResponse(content=json.dumps(data), reasoning="")


# ---------------------------------------------------------------------------
# load_memory_text
# ---------------------------------------------------------------------------

class TestLoadMemoryText:
    def test_missing_file_returns_none(self, tmp_path: Path):
        assert load_memory_text(tmp_path / "missing.md") is None

    def test_returns_raw_text_verbatim(self, tmp_path: Path):
        p = tmp_path / "classification-memory.md"
        raw = "# Classification Memory\n\n1. rule one\n2. rule two\n"
        p.write_text(raw, encoding="utf-8")
        assert load_memory_text(p) == raw


# ---------------------------------------------------------------------------
# load_rules
# ---------------------------------------------------------------------------

class TestLoadRules:
    def test_missing_file_returns_empty_list(self, tmp_path: Path):
        assert load_rules(tmp_path / "missing.md") == []

    def test_empty_file_returns_empty_list(self, tmp_path: Path):
        p = tmp_path / "mem.md"
        p.write_text("", encoding="utf-8")
        assert load_rules(p) == []

    def test_header_and_blank_lines_skipped(self, tmp_path: Path):
        p = tmp_path / "mem.md"
        p.write_text("# Classification Memory\n\n1. first rule\n", encoding="utf-8")
        assert load_rules(p) == ["first rule"]

    def test_header_only_file_returns_empty_list(self, tmp_path: Path):
        p = tmp_path / "mem.md"
        p.write_text("# Classification Memory\n\n", encoding="utf-8")
        assert load_rules(p) == []

    def test_numbered_prefix_stripped(self, tmp_path: Path):
        p = tmp_path / "mem.md"
        p.write_text("1. alpha\n2. beta\n10. gamma\n", encoding="utf-8")
        assert load_rules(p) == ["alpha", "beta", "gamma"]

    def test_unnumbered_lines_kept_verbatim(self, tmp_path: Path):
        p = tmp_path / "mem.md"
        p.write_text("- bullet rule\nplain rule\n", encoding="utf-8")
        assert load_rules(p) == ["- bullet rule", "plain rule"]

    def test_lines_are_stripped(self, tmp_path: Path):
        p = tmp_path / "mem.md"
        p.write_text("  1. padded rule  \n", encoding="utf-8")
        assert load_rules(p) == ["padded rule"]


# ---------------------------------------------------------------------------
# save_rules
# ---------------------------------------------------------------------------

class TestSaveRules:
    def test_exact_serialized_format(self, tmp_path: Path):
        p = tmp_path / "mem.md"
        save_rules(p, ["rule one", "rule two"])
        assert p.read_text(encoding="utf-8") == (
            "# Classification Memory\n\n1. rule one\n2. rule two\n"
        )

    def test_empty_rules_writes_header_only(self, tmp_path: Path):
        p = tmp_path / "mem.md"
        save_rules(p, [])
        assert p.read_text(encoding="utf-8") == "# Classification Memory\n\n"

    def test_creates_parent_directories(self, tmp_path: Path):
        p = tmp_path / "deep" / "nested" / "mem.md"
        save_rules(p, ["a rule"])
        assert p.exists()

    def test_header_constant_matches_format(self, tmp_path: Path):
        p = tmp_path / "mem.md"
        save_rules(p, ["x"])
        assert p.read_text(encoding="utf-8").startswith(MEMORY_HEADER)

    def test_round_trip(self, tmp_path: Path):
        p = tmp_path / "mem.md"
        rules = ["bank statements go to finance/bank", "tax letters go to taxes"]
        save_rules(p, rules)
        assert load_rules(p) == rules

    def test_round_trip_empty(self, tmp_path: Path):
        p = tmp_path / "mem.md"
        save_rules(p, [])
        assert load_rules(p) == []


# ---------------------------------------------------------------------------
# learn_from_correction
# ---------------------------------------------------------------------------

class TestLearnFromCorrection:
    def _client(self, response: LLMResponse) -> MagicMock:
        client = MagicMock()
        client.load_prompt.return_value = (
            "DOC:{{document_text}} PREV:{{previous_folder}} HINT:{{user_hint}} "
            "NEW:{{new_folder}} SUM:{{summary}}"
        )
        client.complete_structured.return_value = response
        return client

    def test_returns_rule_when_should_learn(self, tmp_path: Path):
        client = self._client(_rj({"reasoning": "why", "should_learn": True, "rule": "the rule"}))

        rule, interactions = learn_from_correction(
            client, "doc text", "summary", "old/folder", "a hint", "new/folder")

        assert rule == "the rule"
        assert len(interactions) == 1
        assert interactions[0]["stage"] == "learn"
        assert interactions[0]["step"] == 1
        assert interactions[0]["answer"] == "the rule"
        assert interactions[0]["reasoning"] == "why"

    def test_returns_none_when_should_not_learn(self, tmp_path: Path):
        client = self._client(_rj({"reasoning": "", "should_learn": False, "rule": "ignored"}))

        rule, interactions = learn_from_correction(
            client, "doc", "sum", "old", "hint", "new")

        assert rule is None
        assert interactions[0]["answer"] == "(nothing learned)"

    def test_returns_none_when_rule_is_blank(self, tmp_path: Path):
        client = self._client(_rj({"reasoning": "", "should_learn": True, "rule": "   "}))

        rule, _ = learn_from_correction(client, "doc", "sum", "old", "hint", "new")

        assert rule is None

    def test_prompt_placeholders_substituted(self, tmp_path: Path):
        client = self._client(_rj({"reasoning": "", "should_learn": True, "rule": "r"}))

        learn_from_correction(
            client, "mydoc", "mysum", "prev/dir", "myhint", "next/dir")

        client.load_prompt.assert_called_once_with("learn")
        prompt = client.complete_structured.call_args[0][0]
        assert "DOC:mydoc" in prompt
        assert "PREV:prev/dir" in prompt
        assert "HINT:myhint" in prompt
        assert "NEW:next/dir" in prompt
        assert "SUM:mysum" in prompt


# ---------------------------------------------------------------------------
# consolidate_memory
# ---------------------------------------------------------------------------

class TestConsolidateMemory:
    def _client(self, rules: list[str], reasoning: str = "") -> MagicMock:
        client = MagicMock()
        client.load_prompt.return_value = "NEW:{{new_rule}} MEM:{{current_memory}}"
        client.complete_structured.return_value = _rj({"reasoning": reasoning, "rules": rules})
        return client

    def test_writes_consolidated_rules_to_file(self, tmp_path: Path):
        mem = tmp_path / "mem.md"
        client = self._client(["merged rule"])

        consolidate_memory(client, mem, "new rule")

        assert mem.read_text(encoding="utf-8") == "# Classification Memory\n\n1. merged rule\n"

    def test_creates_file_when_missing(self, tmp_path: Path):
        mem = tmp_path / "sub" / "mem.md"
        client = self._client(["only rule"])

        consolidate_memory(client, mem, "only rule")

        assert load_rules(mem) == ["only rule"]

    def test_existing_rules_plus_new_rule_in_prompt(self, tmp_path: Path):
        mem = tmp_path / "mem.md"
        save_rules(mem, ["existing one", "existing two"])
        client = self._client(["a", "b", "c"])

        consolidate_memory(client, mem, "brand new")

        client.load_prompt.assert_called_once_with("consolidate")
        prompt = client.complete_structured.call_args[0][0]
        assert "NEW:brand new" in prompt
        assert "1. existing one" in prompt
        assert "2. existing two" in prompt
        assert "3. brand new" in prompt

    def test_blank_rules_from_llm_dropped(self, tmp_path: Path):
        mem = tmp_path / "mem.md"
        client = self._client(["keep me", "  ", ""])

        consolidate_memory(client, mem, "rule")

        assert load_rules(mem) == ["keep me"]

    def test_returns_interaction(self, tmp_path: Path):
        mem = tmp_path / "mem.md"
        client = self._client(["one", "two"], reasoning="merged duplicates")

        interactions = consolidate_memory(client, mem, "rule")

        assert len(interactions) == 1
        ix = interactions[0]
        assert ix["stage"] == "consolidate"
        assert ix["step"] == 1
        assert ix["answer"] == "1. one\n2. two"
        assert ix["reasoning"] == "merged duplicates"

    def test_round_trip_through_dashboard_format(self, tmp_path: Path):
        """File written by consolidate_memory parses back with load_rules."""
        mem = tmp_path / "mem.md"
        client = self._client(["rule a", "rule b"])

        consolidate_memory(client, mem, "rule a")

        assert load_rules(mem) == ["rule a", "rule b"]
