import re
from typing import AsyncGenerator, Tuple, Optional
from unicodedata import normalize

# Pre-compiled regex patterns for better performance
_EMOJI_PATTERN = re.compile(
    "[\U0001f600-\U0001f64f"  # emoticons
    "\U0001f300-\U0001f5ff"  # symbols & pictographs
    "\U0001f680-\U0001f6ff"  # transport & map symbols
    "\U0001f700-\U0001f77f"
    "\U0001f780-\U0001f7ff"
    "\U0001f800-\U0001f8ff"
    "\U0001f900-\U0001f9ff"
    "\U0001fa00-\U0001fa6f"
    "\U0001fa70-\U0001faff"
    "\u2600-\u26ff"
    "\u2700-\u27bf"
    "\U0001f1e6-\U0001f1ff]+",
    flags=re.UNICODE,
)

# Diacritics removal — only applied for English to preserve accented chars for other languages
_DIACRITICS_PATTERN = re.compile(
    r"[\u0302\u0303\u0304\u0305\u0306\u0307\u0308\u030A\u030B\u030C\u0327\u0328\u0329\u032A\u032B\u032C\u032D\u032E\u032F]"
)

_SPECIAL_SYMBOLS_PATTERN = re.compile(r"[♥☆♡©\\\\]")

# Pre-compiled patterns for spacing fixes (Latin script languages)
_LATIN_SPACING_PATTERNS = [
    (re.compile(r" ,"), ","),
    (re.compile(r" \."), "."),
    (re.compile(r" !"), "!"),
    (re.compile(r" \?"), "?"),
    (re.compile(r" ;"), ";"),
    (re.compile(r" :"), ":"),
    (re.compile(r" '"), "'"),
]

_MULTISPACE_PATTERN = re.compile(r"\s+")

# Build translation table for single-character replacements (English-focused characters)
_CHAR_TRANSLATION = str.maketrans({
    "\u2013": "-", "\u2011": "-", "\u2014": "-", "\u00af": " ", "_": " ",
    "\u201c": '"', "\u201d": '"', "\u2018": "'", "\u2019": "'",
    "\u00b4": "'", "\u0060": "'", "[": " ", "]": " ", "|": " ", "/": " ",
    "#": " ", "\u2192": " ", "\u2190": " ",
})

# Multi-character replacements (English-specific expressions)
_EN_EXPR_REPLACEMENTS = (
    ("@", " at "),
    ("e.g.,", "for example, "),
    ("i.e.,", "that is, "),
)

# Sentence boundary pattern - English-centric
_EN_SENTENCE_PATTERN = re.compile(
    r"(?<!Mr\.)(?<!Mrs\.)(?<!Ms\.)(?<!Dr\.)(?<!Prof\.)(?<!Sr\.)(?<!Jr\.)"
    r"(?<!Ph\.D\.)(?<!etc\.)(?<!e\.g\.)(?<!i\.e\.)(?<!vs\.)(?<!Inc\.)"
    r"(?<!Ltd\.)(?<!Co\.)(?<!Corp\.)(?<!St\.)(?<!Ave\.)(?<!Blvd\.)"
    r"(?<!\b[A-Z]\.)(?<=[.!?])\s+"
)



# Common sentence boundary for most languages using Latin script
_LATIN_SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+")

# Arabic script sentence boundary — matches 。? (Arabic question mark), Arabic full stop
_ARABIC_SENTENCE_PATTERN = re.compile(r"(?<=[.!?\u061f\u06d4])\s*")

# Devanagari sentence boundary — matches danda (।) and double danda (॥)
_DEVANAGARI_SENTENCE_PATTERN = re.compile(r"(?<=[.!?\u0964\u0965])\s*")

# CJK sentence boundary — split after 。！？ but not on 、 or ，
_CJK_SENTENCE_PATTERN = re.compile(r"(?<=[\u3002\uff01\uff1f!?])\s*")

# Pause tag pattern
_PAUSE_TAG_PATTERN = re.compile(r'\[pause:(\d+\.?\d*)\]')

# Paragraph split pattern
_PARAGRAPH_PATTERN = re.compile(r"\n\s*\n+")

# Ending punctuation check — language-aware
_EN_ENDING_PUNCTUATION = re.compile(r"[.!?;:,'\")\]}\u2026\u3002\u300d\u3011\u3009\u300b\u300f\u3001\u00bb]|(?:\u00bb)$")
_CJK_ENDING_PUNCTUATION = re.compile(r"[。！？，、…—～\u3002\uff01\uff1f\u3001\u2026\u2014\uff5e]$")

# Languages that should preserve diacritics
_DIACRITIC_PRESERVING_LANGS = frozenset({
    "fr", "de", "es", "it", "pt", "nl", "pl", "sv", "da", "no",
    "fi", "cs", "ro", "hu", "el", "ru", "uk", "vi",
})

# CJK languages
_CJK_LANGS = frozenset({"zh", "ja", "ko"})

# Languages using Arabic script
_ARABIC_LANGS = frozenset({"ar", "ur", "fa"})

# Languages using Devanagari script
_DEVANAGARI_LANGS = frozenset({"hi", "mr", "ne"})


def _uses_latin_spacing(lang: str) -> bool:
    """Check if language uses the generic Latin spacing rules (safe for most scripts except CJK/Arabic/Devanagari)."""
    return lang not in _CJK_LANGS and lang not in _ARABIC_LANGS and lang not in _DEVANAGARI_LANGS


def clean_text(text: str, lang: str = "en") -> str:
    """
    Language-aware text preprocessing for TTS.

    - English: aggressive cleanup (diacritic removal, abbreviation expansion, forced period)
    - CJK (zh, ja, ko): preserves CJK punctuation, no forced period, no diacritic removal
    - French/German/etc: preserves diacritics, language-specific spacing
    - Arabic/Devanagari: preserves script-specific characters, no forced period
    """
    if not text:
        return text

    # 1. Unicode normalization (NFKC for CJK to preserve compatibility; NFKD for others)
    if lang in _CJK_LANGS:
        text = normalize("NFKC", text)
    else:
        text = normalize("NFKD", text)

    # 2. Remove emojis (universal)
    text = _EMOJI_PATTERN.sub("", text)

    # 3. Replace single characters via translation table (safe for all languages)
    text = text.translate(_CHAR_TRANSLATION)

    # 4. Remove diacritics ONLY for English — preserve for other languages
    if lang == "en":
        text = _DIACRITICS_PATTERN.sub("", text)

    # 5. Remove special symbols
    text = _SPECIAL_SYMBOLS_PATTERN.sub("", text)

    # 6. Language-specific replacements
    if lang == "en":
        for old, new in _EN_EXPR_REPLACEMENTS:
            text = text.replace(old, new)

    # 7. Language-specific spacing fixes
    if lang == "fr":
        # French spacing: space before !?;:
        text = re.sub(r"\s+!", " !", text)
        text = re.sub(r"\s+\?", " ?", text)
        text = re.sub(r"\s+;", " ;", text)
        text = re.sub(r"\s+:", " :", text)
        text = re.sub(r" ,", ",", text)
        text = re.sub(r" \.", ".", text)
    elif _uses_latin_spacing(lang):
        for pattern, replacement in _LATIN_SPACING_PATTERNS:
            text = pattern.sub(replacement, text)

    # 8. Remove duplicate quotes and spaces (universal)
    while '""' in text:
        text = text.replace('""', '"')
    while "''" in text:
        text = text.replace("''", "'")
    text = _MULTISPACE_PATTERN.sub(" ", text).strip()

    # 9. Ensure ending punctuation (language-appropriate)
    if text:
        if lang in _CJK_LANGS:
            if not _CJK_ENDING_PUNCTUATION.search(text):
                text += "。"
        elif lang in _ARABIC_LANGS:
            if not text.endswith((".")) and not text.endswith(("!", "؟", "۔")):
                text += "."
        elif lang == "en":
            if not _EN_ENDING_PUNCTUATION.search(text):
                text += "."

    return text


def _get_sentence_pattern(lang: str):
    """Return the appropriate sentence boundary pattern for the language."""
    if lang in _CJK_LANGS:
        return _CJK_SENTENCE_PATTERN
    elif lang in _ARABIC_LANGS:
        return _ARABIC_SENTENCE_PATTERN
    elif lang in _DEVANAGARI_LANGS:
        return _DEVANAGARI_SENTENCE_PATTERN
    elif lang == "en":
        return _EN_SENTENCE_PATTERN
    else:
        # Generic Latin script: simpler sentence splitting
        return _LATIN_SENTENCE_PATTERN


async def smart_split(
    text: str,
    max_chunk_length: int | None = None,
    lang: str = "en",
) -> AsyncGenerator[Tuple[str, Optional[float]], None]:
    """Split text into chunks by paragraphs and sentences.

    Language-aware: uses appropriate sentence boundaries for CJK, English, and other scripts.

    Yields ``(chunk_text, pause_duration_s)``.
    ``pause_duration_s`` is set when the chunk is a silence marker (``[pause:N]``),
    otherwise ``None``.
    """
    from app.core.config import settings  # local import avoids circular dependency
    chunk_limit = max_chunk_length if max_chunk_length is not None else settings.MAX_CHUNK_LENGTH
    sentence_pattern = _get_sentence_pattern(lang)

    # Split by pause tags first
    parts = _PAUSE_TAG_PATTERN.split(text)

    for i, part in enumerate(parts):
        # Every odd index is a pause duration
        if i % 2 == 1:
            try:
                yield "", float(part)
            except ValueError:
                continue
            continue

        if not part.strip():
            continue

        # Split by paragraph
        paragraphs = _PARAGRAPH_PATTERN.split(part.strip())

        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue

            # Split by sentence boundaries (language-aware)
            sentences = sentence_pattern.split(paragraph)

            # Combine sentences into chunks
            current_chunk = ""

            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue

                if len(current_chunk) + len(sentence) + 1 <= chunk_limit:
                    current_chunk += (" " if current_chunk else "") + sentence
                else:
                    if current_chunk:
                        yield current_chunk.strip(), None
                    current_chunk = sentence

            if current_chunk:
                yield current_chunk.strip(), None
