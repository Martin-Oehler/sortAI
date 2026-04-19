"""PDF text extraction utilities."""

from __future__ import annotations

from pathlib import Path

import pdfplumber


def extract_text(path: Path) -> str:
    """Extract plain text from all pages of a PDF, trimmed of excess whitespace."""
    parts: list[str] = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                parts.append(text.strip())
    return "\n\n".join(parts)
