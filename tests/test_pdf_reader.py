"""Tests for sortai.pdf_reader.extract_text."""
from __future__ import annotations

from pathlib import Path

import pytest

from sortai.pdf_reader import extract_text

# Resolve project root relative to this file so tests work regardless of CWD.
PROJECT_ROOT = Path(__file__).parent.parent
REAL_PDF = PROJECT_ROOT / "tests" / "fixtures" / "inbox" / "bank_statement.pdf"


def test_extract_text_returns_nonempty_string():
    result = extract_text(REAL_PDF)
    assert isinstance(result, str)
    assert len(result) > 0


def test_extract_text_contains_account_statement():
    result = extract_text(REAL_PDF)
    assert "Account Statement" in result
