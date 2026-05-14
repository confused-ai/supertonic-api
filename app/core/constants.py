"""Application-wide constants shared across modules."""

# Audio format → HTTP media type mapping
MEDIA_TYPES: dict[str, str] = {
    "mp3": "audio/mpeg",
    "wav": "audio/wav",
    "flac": "audio/flac",
    "opus": "audio/ogg",
    "aac": "audio/aac",
    "pcm": "audio/pcm",
}

# Maximum allowed voice JSON upload size
MAX_VOICE_JSON_BYTES: int = 10 * 1024 * 1024  # 10 MB

# Supertonic model identifier
MODEL_NAME: str = "supertonic-3"
