"""Tests for sortai.pdf_reader.extract_text."""
from __future__ import annotations

from pathlib import Path

import pytest

from sortai.pdf_reader import extract_text

# Resolve project root relative to this file so tests work regardless of CWD.
PROJECT_ROOT = Path(__file__).parent.parent
REAL_PDF = PROJECT_ROOT / "tests" / "fixtures" / "inbox" / "Bank Statement Example Final.pdf"
IMAGE_PDF = PROJECT_ROOT / "tests" / "fixtures" / "inbox" / "dummy_statement.pdf"


def test_extract_text_returns_nonempty_string():
    """extract_text on the real text-based PDF returns a non-empty string."""
    result = extract_text(REAL_PDF)
    assert isinstance(result, str)
    assert len(result) > 0


def test_extract_text_contains_bank_statement():
    """extract_text on the real PDF contains the phrase 'Bank Statement'."""
    result = extract_text(REAL_PDF)
    assert "Bank Statement" in result


def test_extract_text_image_pdf_returns_empty_string():
    """extract_text on an image-only PDF returns an empty string."""
    result = extract_text(IMAGE_PDF)
    assert result == ""
