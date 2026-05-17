"""Lightweight text cleaning for SU-GPT (Section 2).

Goal: normalize whitespace and strip obvious artifacts WITHOUT damaging
academic content such as formulas, code, or non-ASCII characters.
"""

from __future__ import annotations

import re
import unicodedata


_WS_INLINE = re.compile(r"[ \t]+")
_BLANK_LINES = re.compile(r"\n{3,}")
_TRAILING_WS = re.compile(r"[ \t]+\n")
# Stray non-breaking-space, zero-widths, and BOM that PDF extraction can leak.
# Written with explicit unicode escapes so this file stays ASCII-clean.
_STRAY_CHARS = re.compile("[\\u00A0\\u200B\\u200C\\u200D\\uFEFF]")


def clean_text(text: str) -> str:
    """Conservative text cleaner that preserves structure, code, and formulas."""
    if not text:
        return ""

    # Normalize unicode (NFC keeps composed forms like accented Turkish chars).
    text = unicodedata.normalize("NFC", text)

    # Strip zero-width / NBSP / BOM characters that PDF extraction can leak.
    text = _STRAY_CHARS.sub("", text)

    # Convert Windows-style line endings to \n.
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Collapse runs of spaces/tabs INSIDE a line (do not touch newlines).
    text = _WS_INLINE.sub(" ", text)

    # Drop trailing whitespace on each line.
    text = _TRAILING_WS.sub("\n", text)

    # Cap consecutive blank lines at two (one empty line between blocks).
    text = _BLANK_LINES.sub("\n\n", text)

    return text.strip()
