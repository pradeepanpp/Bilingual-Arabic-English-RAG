"""
Text cleaning and the check that decides whether a PDF page needs OCR.

Arabic PDFs often extract badly even when they contain a text layer:
characters come out in reverse order (common with pdfplumber), or the
font maps letters to replacement characters (U+FFFD). Those pages are
sent to OCR. A page of normal English text is never sent to OCR.
"""

import re
import unicodedata

ARABIC_LETTER = re.compile(r"[ء-ي]")
ARABIC_WORD = re.compile(r"[ء-ي]{2,}")
TATWEEL = "ـ"

MIN_CHARS = 80               # less than this: probably a scanned page or an image
MAX_BROKEN_RATIO = 0.01      # replacement characters per Arabic letter
MIN_ARABIC_WORDS = 20        # enough words to judge the direction of the text


def clean_text(text: str) -> str:
    """
    NFKC turns Arabic presentation forms (ﺗﻨﺎزع) and ligatures (ﬁ) into normal
    characters; tatweel only stretches words; whitespace is collapsed.
    """
    text = unicodedata.normalize("NFKC", text or "").replace(TATWEEL, "")
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n\s*\n+", "\n\n", text).strip()


def is_reversed_arabic(text: str) -> bool:
    """
    The article "ال" starts many Arabic words. Reversed text has it at the
    end instead ("لا"), so more words end with "لا" than start with "ال".
    """
    words = ARABIC_WORD.findall(text)
    if len(words) < MIN_ARABIC_WORDS:
        return False
    starts = sum(w.startswith("ال") for w in words)
    ends = sum(w.endswith("لا") for w in words)
    return ends > starts


def ocr_reason(text: str) -> str:
    """Why this page should be OCR'd, or "" if the extracted text is usable."""
    text = (text or "").strip()
    if len(text) < MIN_CHARS:
        return "too little text"
    arabic = len(ARABIC_LETTER.findall(text))
    if arabic and text.count("�") / arabic > MAX_BROKEN_RATIO:
        return "broken Arabic characters"
    if is_reversed_arabic(text):
        return "reversed Arabic text"
    return ""
