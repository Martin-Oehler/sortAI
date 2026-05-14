"""PDF text extraction and rendering utilities."""

from __future__ import annotations

import base64
from pathlib import Path

import fitz  # pymupdf
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


def render_pages(path: Path, max_pages: int = 5, dpi: int = 150) -> list[str]:
    """Render up to max_pages PDF pages as base64-encoded JPEG strings."""
    images: list[str] = []
    doc = fitz.open(path)
    try:
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        for i, page in enumerate(doc):
            if i >= max_pages:
                break
            pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
            jpeg_bytes = pix.tobytes("jpeg")
            images.append(base64.b64encode(jpeg_bytes).decode("ascii"))
    finally:
        doc.close()
    return images
