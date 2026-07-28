from __future__ import annotations

"""
Tokenization / normalization used by every sparse retriever in the lab.

Two tokenizers live here on purpose:

`tokenize_original` is a byte-for-byte port of `modules.bm25_retriever._tokenize`, the
tokenizer the shipped BM25 baseline actually uses. It exists so `bm25_original` in the
benchmark is the *same* system that runs in production, not a re-interpretation of it.
Do not "improve" it - if it changes, the baseline stops being a baseline.

`tokenize_v2` is the candidate tokenizer. It differs in exactly three ways, each motivated
by an observed adviSU failure mode:

  1. Turkish characters survive. The original regex is `[a-z0-9]+`, so cci/gg/ii/oo/ss/uu
     (c-cedilla, g-breve, dotless-i, o-umlaut, s-cedilla, u-umlaut) act as *separators*:
     "mufredatinda" tokenizes to fragments and a Turkish query can lose most of its content
     words. We ASCII-fold instead of splitting, which also makes a Turkish query match the
     predominantly English corpus ("mufredat" vs "müfredat" both -> "mufredat").
  2. Turkish casing is handled before folding. Python's str.lower() maps the Turkish
     capital dotted I to "i" + combining dot, which then folds to a stray token.
  3. Course codes are emitted in both split and merged form ("cs 455" -> cs, 455, cs455)
     *and* the merged form is recognized in the reverse direction ("cs455" -> cs455, cs, 455)
     so the query and the document match whichever way each happens to be written.
"""

import re
import unicodedata

# --- shipped baseline tokenizer (do not modify) --------------------------------------

_ORIGINAL_WORD_RE = re.compile(r"[a-z0-9]+")
_ORIGINAL_CODE_RE = re.compile(r"\b([a-z]{2,5})\s+(\d{3,5}[a-z]?)\b")


def tokenize_original(text: str) -> list[str]:
    """Exact port of modules.bm25_retriever._tokenize (the production BM25 tokenizer)."""
    text = (text or "").lower()
    tokens = _ORIGINAL_WORD_RE.findall(text)
    for subj, num in _ORIGINAL_CODE_RE.findall(text):
        tokens.append(f"{subj}{num}")
    return tokens


# --- candidate tokenizer -------------------------------------------------------------

# Turkish (and general Latin-1) folding. Applied AFTER casing so we only handle lowercase.
_FOLD = str.maketrans(
    {
        "ç": "c", "ğ": "g", "ı": "i", "ö": "o", "ş": "s", "ü": "u",
        "â": "a", "î": "i", "û": "u", "é": "e", "è": "e", "á": "a", "í": "i", "ó": "o", "ú": "u",
    }
)

_WORD_RE = re.compile(r"[a-z0-9]+")
_SPLIT_CODE_RE = re.compile(r"\b([a-z]{2,5})\s+(\d{3,5}[a-z]?)\b")
_MERGED_CODE_RE = re.compile(r"\b([a-z]{2,5})(\d{3,5}[a-z]?)\b")


def turkish_lower(text: str) -> str:
    """Lowercase without letting the Turkish dotted/dotless I produce combining marks."""
    # Do the two Turkish-specific capitals by hand, then fall back to normal lowering.
    text = text.replace("İ", "i").replace("I", "ı")
    return unicodedata.normalize("NFC", text).lower()


def fold(text: str) -> str:
    """Turkish-aware casefold + ASCII fold. Used for both queries and documents."""
    return turkish_lower(text).translate(_FOLD)


def tokenize_v2(text: str) -> list[str]:
    """Candidate tokenizer: Turkish-safe, course-code aware in both directions."""
    folded = fold(text or "")
    tokens = _WORD_RE.findall(folded)
    # "cs 455" -> also emit "cs455"
    for subj, num in _SPLIT_CODE_RE.findall(folded):
        tokens.append(f"{subj}{num}")
    # "cs455" -> also emit "cs" and "455" so it matches a document that spaces them
    for subj, num in _MERGED_CODE_RE.findall(folded):
        tokens.extend((subj, num))
    return tokens


# --- entity extraction ---------------------------------------------------------------

# A course code is 2-5 letters + 3-5 digits, optionally separated. Used for metadata
# extraction and for the exact-match signal; deliberately strict to avoid matching years.
COURSE_CODE_RE = re.compile(r"\b([a-z]{2,5})\s*(\d{3,5}[a-z]?)\b")
# adviSU curriculum terms are 6 digits: YYYYMM where MM is 01 (fall) or 02 (spring).
TERM_CODE_RE = re.compile(r"\b(20\d{2}0[12])\b")
# A bare 4-digit academic year, e.g. "2024 curriculum" / "2024 mufredati".
YEAR_RE = re.compile(r"\b(20\d{2})\b")


def extract_course_codes(text: str) -> list[str]:
    """Return normalized course codes ("CS 414" / "cs414" -> "CS 414") in order of appearance."""
    out: list[str] = []
    for subj, num in COURSE_CODE_RE.findall(fold(text or "")):
        code = f"{subj.upper()} {num.upper()}"
        if code not in out:
            out.append(code)
    return out


def extract_term_codes(text: str) -> list[str]:
    """Explicit 6-digit curriculum terms, plus bare years widened to their fall term."""
    folded = fold(text or "")
    out = list(dict.fromkeys(TERM_CODE_RE.findall(folded)))
    if not out:
        # "2024" almost always means the 202401 (fall) curriculum in adviSU usage.
        out = [f"{year}01" for year in dict.fromkeys(YEAR_RE.findall(folded))]
    return out
