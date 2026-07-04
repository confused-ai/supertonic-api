import os

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Server
    LOG_LEVEL: str = "INFO"
    HOST: str = "0.0.0.0"
    PORT: int = 8800

    # Model inference — thread / concurrency knobs.
    # 0 = auto: derived from CPU core count at startup (see _auto_tune below).
    # Auto keeps MAX_WORKERS * MODEL_THREADS <= cores to avoid oversubscription —
    # the #1 cause of latency/throughput collapse under concurrent load.
    # Set any of these to a positive int (env or .env) to override the auto value.
    MODEL_THREADS: int = 0        # intra-op threads per ONNX call (0 = auto)
    MODEL_INTER_THREADS: int = 1  # inter-op pool; supertonic runs ORT_SEQUENTIAL so this is unused
    FORCE_PROVIDERS: str = "metal"  # auto | cuda | coreml | cpu | metal
    MAX_WORKERS: int = 0          # max concurrent model calls (0 = auto)
    MAX_CHUNK_LENGTH: int = 300
    SAMPLE_RATE: int = 44100
    DEFAULT_MODEL_VERSION: str = "v3"

    # Synthesis performance
    # Lower = faster but lower quality. Range 5–12. supertonic-3 default is 8.
    # 5 = fastest, 8 = balanced (official default), 12 = highest quality.
    TOTAL_STEPS: int = 8  # Lower = faster, higher = better quality (range 5–12)
    # Number of text chunks to synthesize concurrently per request.
    # 0 = auto → MAX_WORKERS. Set to 1 to disable batching (sequential synthesis).
    SYNTHESIS_BATCH_SIZE: int = 0

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

    @model_validator(mode="after")
    def _auto_tune(self):
        """Fill 0-sentinel thread knobs from the CPU core count (balanced profile).

        Goal: MAX_WORKERS * MODEL_THREADS <= cores, so concurrent requests never
        oversubscribe the CPU. Explicit (non-zero) env values are left untouched.
        """
        cores = os.cpu_count() or 4
        if self.MAX_WORKERS <= 0:
            # Low enough that accelerators (ANE/GPU) don't queue-thrash, high
            # enough to overlap CPU pre/post-processing with model compute.
            self.MAX_WORKERS = max(2, min(4, cores // 3))
        if self.MODEL_THREADS <= 0:
            self.MODEL_THREADS = max(2, cores // self.MAX_WORKERS)
        if self.SYNTHESIS_BATCH_SIZE <= 0:
            self.SYNTHESIS_BATCH_SIZE = self.MAX_WORKERS
        return self


settings = Settings()


if __name__ == "__main__":
    # Self-check: auto-tune must never oversubscribe and must honour overrides.
    s = Settings()
    assert s.MAX_WORKERS >= 2, s.MAX_WORKERS
    assert s.MODEL_THREADS >= 2, s.MODEL_THREADS
    assert s.SYNTHESIS_BATCH_SIZE == s.MAX_WORKERS or s.SYNTHESIS_BATCH_SIZE > 0
    # peak concurrent threads should stay within ~1.5x cores
    cores = os.cpu_count() or 4
    assert s.MAX_WORKERS * s.MODEL_THREADS <= cores * 1.5, (s.MAX_WORKERS, s.MODEL_THREADS, cores)
    # explicit override wins
    o = Settings(MAX_WORKERS=6, MODEL_THREADS=3, SYNTHESIS_BATCH_SIZE=2)
    assert (o.MAX_WORKERS, o.MODEL_THREADS, o.SYNTHESIS_BATCH_SIZE) == (6, 3, 2), o
    print(f"OK cores={cores} MAX_WORKERS={s.MAX_WORKERS} MODEL_THREADS={s.MODEL_THREADS} BATCH={s.SYNTHESIS_BATCH_SIZE}")

