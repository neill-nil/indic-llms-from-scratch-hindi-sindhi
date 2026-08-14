"""
Heuristic quality filters for corpus preprocessing.

Each filter is a standalone function that takes a document string and returns
either the (possibly modified) document, or None to signal rejection.
"""

import re
import unicodedata
from collections import Counter
from typing import Optional


# ─────────────────────────────────────────────────
# Unicode ranges for script detection
# ─────────────────────────────────────────────────
DEVANAGARI_RANGE = re.compile(r'[\u0900-\u097F]')          # Hindi
ARABIC_RANGE     = re.compile(r'[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]')  # Sindhi (Arabic script)


# ─────────────────────────────────────────────────
# 1. LENGTH FILTER
# ─────────────────────────────────────────────────
def length_filter(text: str, min_chars: int = 50, max_chars: int = 500_000) -> Optional[str]:
    """Reject documents that are too short or absurdly long."""
    n = len(text.strip())
    if n < min_chars or n > max_chars:
        return None
    return text


# ─────────────────────────────────────────────────
# 2. SYMBOL RATIO FILTER
# ─────────────────────────────────────────────────
def symbol_ratio_filter(text: str, max_ratio: float = 0.3) -> Optional[str]:
    """Reject documents where special symbols dominate the text.
    
    Uses Unicode category to only count true punctuation (Ps/Pe/Po/Pi/Pf)
    and symbol (S*) characters — Indic combining marks (Mc/Mn) and
    script-specific punctuation like danda are NOT counted as junk.
    """
    if not text.strip():
        return None
    junk_cats = {'Ps', 'Pe', 'Pi', 'Pf', 'Sm', 'Sc', 'Sk', 'So'}
    symbols = sum(1 for c in text if unicodedata.category(c) in junk_cats)
    non_ws = sum(1 for c in text if not c.isspace())
    if non_ws == 0:
        return None
    ratio = symbols / non_ws
    if ratio > max_ratio:
        return None
    return text


# ─────────────────────────────────────────────────
# 3. LINE STRUCTURE FILTER
# ─────────────────────────────────────────────────
def line_structure_filter(text: str, min_avg_line_len: int = 15,
                          max_short_line_ratio: float = 0.7) -> Optional[str]:
    """Reject documents that look like navigation menus or lists.
    
    If most lines are very short (< 15 chars), the document is
    likely a navigation menu, sidebar, or table of contents.
    """
    lines = [l for l in text.split('\n') if l.strip()]
    if not lines:
        return None
    
    avg_len = sum(len(l) for l in lines) / len(lines)
    short_lines = sum(1 for l in lines if len(l.strip()) < min_avg_line_len)
    short_ratio = short_lines / len(lines)
    
    if avg_len < min_avg_line_len and short_ratio > max_short_line_ratio:
        return None
    return text


# ─────────────────────────────────────────────────
# 4. REPETITION FILTER
# ─────────────────────────────────────────────────
def repetition_filter(text: str, ngram_size: int = 5,
                      max_repeated_ratio: float = 0.3) -> Optional[str]:
    """Reject documents with excessive n-gram repetition.
    
    Splits text into word-level n-grams and checks what fraction
    of n-grams appear more than once. Catches boilerplate, cookie
    banners, repeated paragraphs.
    """
    words = text.split()
    if len(words) < ngram_size * 2:
        return text  # Too short to judge
    
    ngrams = [tuple(words[i:i+ngram_size]) for i in range(len(words) - ngram_size + 1)]
    counts = Counter(ngrams)
    repeated = sum(c - 1 for c in counts.values() if c > 1)
    
    if repeated / len(ngrams) > max_repeated_ratio:
        return None
    return text


# ─────────────────────────────────────────────────
# 5. BLOCKLIST FILTER (URLs, boilerplate)
# ─────────────────────────────────────────────────
_URL_PATTERN = re.compile(r'https?://\S+|www\.\S+')
_HTML_ENTITY_PATTERN = re.compile(r'&[a-zA-Z]+;|&#\d+;')
_HTML_TAG_PATTERN = re.compile(r'<[^>]+>')

def blocklist_filter(text: str) -> str:
    """Strip URLs, HTML tags, and HTML entities from text.
    
    Unlike other filters, this one cleans rather than rejects — it
    removes the offending content but keeps the document.
    """
    text = _URL_PATTERN.sub('', text)
    text = _HTML_TAG_PATTERN.sub('', text)
    text = _HTML_ENTITY_PATTERN.sub('', text)
    return text


# ─────────────────────────────────────────────────
# 6. PERSONAL DATA REDACTION
# ─────────────────────────────────────────────────
_EMAIL_PATTERN = re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}')
_PHONE_PATTERN = re.compile(r'(?:\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,5}[-.\s]?\d{3,5}')

def personal_data_filter(text: str) -> str:
    """Redact email addresses and phone numbers.
    
    Replaces them with placeholder tokens so the model never
    memorizes real PII.
    """
    text = _EMAIL_PATTERN.sub('[EMAIL]', text)
    text = _PHONE_PATTERN.sub('[PHONE]', text)
    return text


# ─────────────────────────────────────────────────
# 7. SCRIPT FILTER (language-specific)
# ─────────────────────────────────────────────────
def script_filter(text: str, language: str, min_script_ratio: float = 0.9) -> Optional[str]:
    """Reject documents that are mostly in the wrong script.
    
    For Hindi: at least min_script_ratio of alphabetic chars should be Devanagari.
    For Sindhi: at least min_script_ratio of alphabetic chars should be Arabic script.
    """
    if not text.strip():
        return None
    
    pattern = DEVANAGARI_RANGE if language == 'hindi' else ARABIC_RANGE
    script_chars = len(pattern.findall(text))
    
    # Compare against total alphabetic characters (ignoring punctuation, digits, spaces)
    alpha_chars = sum(1 for c in text if c.isalpha())
    
    if alpha_chars == 0:
        return None
    
    ratio = script_chars / alpha_chars
    if ratio < min_script_ratio:
        return None
    return text


# ─────────────────────────────────────────────────
# 8. UNICODE NORMALIZATION
# ─────────────────────────────────────────────────
def normalize_unicode(text: str) -> str:
    """Apply NFC Unicode normalization and clean whitespace."""
    text = unicodedata.normalize('NFC', text)
    # Collapse multiple blank lines into one
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Collapse multiple spaces into one
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()
