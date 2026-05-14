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

_DIACRITICS_PATTERN = re.compile(
    r"[\u0302\u0303\u0304\u0305\u0306\u0307\u0308\u030A\u030B\u030C\u0327\u0328\u0329\u032A\u032B\u032C\u032D\u032E\u032F]"
)

_SPECIAL_SYMBOLS_PATTERN = re.compile(r"[♥☆♡©\\]")

# Pre-compiled patterns for spacing fixes
_SPACING_PATTERNS = [
    (re.compile(r" ,"), ","),
    (re.compile(r" \."), "."),
    (re.compile(r" !"), "!"),
    (re.compile(r" \?"), "?"),
    (re.compile(r" ;"), ";"),
    (re.compile(r" :"), ":"),
    (re.compile(r" '"), "'"),
]

_MULTISPACE_PATTERN = re.compile(r"\s+")

# Character replacements (as tuple for faster iteration)
_CHAR_REPLACEMENTS = (
    ("–", "-"), ("‑", "-"), ("—", "-"), ("¯", " "), ("_", " "),
    ("\u201C", '"'), ("\u201D", '"'), ("\u2018", "'"), ("\u2019", "'"),
    ("´", "'"), ("`", "'"), ("[", " "), ("]", " "), ("|", " "), ("/", " "),
    ("#", " "), ("→", " "), ("←", " "),
)

_EXPR_REPLACEMENTS = (
    ("@", " at "),
    ("e.g.,", "for example, "),
    ("i.e.,", "that is, "),
)

# Sentence boundary pattern - pre-compiled
_SENTENCE_PATTERN = re.compile(
    r"(?<!Mr\.)(?<!Mrs\.)(?<!Ms\.)(?<!Dr\.)(?<!Prof\.)(?<!Sr\.)(?<!Jr\.)"
    r"(?<!Ph\.D\.)(?<!etc\.)(?<!e\.g\.)(?<!i\.e\.)(?<!vs\.)(?<!Inc\.)"
    r"(?<!Ltd\.)(?<!Co\.)(?<!Corp\.)(?<!St\.)(?<!Ave\.)(?<!Blvd\.)"
    r"(?<!\b[A-Z]\.)(?<=[.!?])\s+"
)

# Pause tag pattern
_PAUSE_TAG_PATTERN = re.compile(r'\[pause:(\d+\.?\d*)\]')

# Paragraph split pattern
_PARAGRAPH_PATTERN = re.compile(r"\n\s*\n+")

# Ending punctuation check
_ENDING_PUNCTUATION_PATTERN = re.compile(r"[.!?;:,'\"')\]}…。」』】〉》›»]$")


def clean_text(text: str) -> str:
    """
    Minimal text preprocessing for TTS.
    Replaces common symbols, removes emojis, and ensures basic punctuation.
    """
    # 1. Unicode normalization
    text = normalize("NFKD", text)

    # 2. Remove emojis
    text = _EMOJI_PATTERN.sub("", text)

    # 3. Replace various dashes and symbols
    for old, new in _CHAR_REPLACEMENTS:
        text = text.replace(old, new)

    # 4. Remove combining diacritics
    text = _DIACRITICS_PATTERN.sub("", text)

    # 5. Remove special symbols
    text = _SPECIAL_SYMBOLS_PATTERN.sub("", text)

    # 6. Replace known expressions
    for old, new in _EXPR_REPLACEMENTS:
        text = text.replace(old, new)

    # 7. Fix spacing around punctuation
    for pattern, replacement in _SPACING_PATTERNS:
        text = pattern.sub(replacement, text)

    # 8. Remove duplicate quotes and spaces
    while '""' in text:
        text = text.replace('""', '"')
    while "''" in text:
        text = text.replace("''", "'")
    text = _MULTISPACE_PATTERN.sub(" ", text).strip()

    # 9. Ensure ending punctuation
    if text and not _ENDING_PUNCTUATION_PATTERN.search(text):
        text += "."

    return text


async def smart_split(
    text: str,
    max_chunk_length: int | None = None,
) -> AsyncGenerator[Tuple[str, Optional[float]], None]:
    """Split text into chunks by paragraphs and sentences.

    Yields ``(chunk_text, pause_duration_s)``.
    ``pause_duration_s`` is set when the chunk is a silence marker (``[pause:N]``),
    otherwise ``None``.
    """
    from app.core.config import settings  # local import avoids circular dependency
    chunk_limit = max_chunk_length if max_chunk_length is not None else settings.MAX_CHUNK_LENGTH

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
                
            # Split by sentence boundaries
            sentences = _SENTENCE_PATTERN.split(paragraph)
            
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
