from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Server
    LOG_LEVEL: str = "INFO"
    HOST: str = "0.0.0.0"
    PORT: int = 8800

    # Model inference
    MODEL_THREADS: int = 12
    MODEL_INTER_THREADS: int = 12
    FORCE_PROVIDERS: str = "metal"  # auto | cuda | coreml | cpu | metal
    MAX_WORKERS: int = 8
    MAX_CHUNK_LENGTH: int = 300
    SAMPLE_RATE: int = 44100
    DEFAULT_MODEL_VERSION: str = "v3"

    # Synthesis performance
    # Lower = faster but lower quality. Range 5–12. supertonic-3 default is 8.
    # 5 = fastest, 8 = balanced (official default), 12 = highest quality.
    TOTAL_STEPS: int = 8  # Lower = faster, higher = better quality (range 5–12)
    # Number of text chunks to synthesize concurrently per request.
    # Should match MAX_WORKERS to keep all thread-pool slots busy.
    # Set to 1 to disable batching (sequential synthesis).
    SYNTHESIS_BATCH_SIZE: int = 8

    # Audio trimming & gap padding
    GAP_TRIM_MS: int = 100
    DYNAMIC_GAP_TRIM_PADDING_MS: int = 50
    DYNAMIC_GAP_TRIM_PADDING_CHAR_MULTIPLIER: dict[str, float] = {
        ",": 1.2,
        ".": 1.5,
        "!": 1.5,
        "?": 1.5,
        ";": 1.3,
        ":": 1.3,
    }

    # Production
    # CORS: set to specific origins in production, e.g. '["https://myapp.com"]'
    CORS_ORIGINS: list[str] = ["*"]
    # Disable /docs + /redoc + /openapi.json in production
    ENABLE_DOCS: bool = True
    # Max seconds a synthesis request may run before returning 504
    REQUEST_TIMEOUT_S: float = 120.0
    # Per-IP rate limit on all /v1/* routes (format: "<count>/<unit>")
    RATE_LIMIT: str = "500/minute"

    # Language-aware normalization
    # Supported languages for text normalization: en, fr, de, es, it, pt, nl, pl, sv, da, no, fi, cs, ro, hu, el, ru, uk, ar, hi, zh, ja, ko, th, vi, tr, id, ms, tl, na
    # Target RMS level for scipy-based audio normalization (None to disable)
    AUDIO_RMS_TARGET: float | None = None  # e.g. 0.15 for -16 dB LUFS roughly
    # Target peak level for scipy-based audio normalization (None to disable)
    AUDIO_PEAK_TARGET: float | None = 0.95  # scale so max sample = 0.95 * max_int


settings = Settings()

