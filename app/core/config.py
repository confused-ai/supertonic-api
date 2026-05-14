from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    LOG_LEVEL: str = "INFO"
    HOST: str = "0.0.0.0"
    PORT: int = 8800
    MODEL_THREADS: int = 12
    MODEL_INTER_THREADS: int = 12
    FORCE_PROVIDERS: str = "metal"  # auto, cuda, coreml, cpu, metal
    MAX_WORKERS: int = 8
    MAX_CHUNK_LENGTH: int = 300
    SAMPLE_RATE: int = 44100

    # Audio trimming & gap padding
    GAP_TRIM_MS: int = 100
    DYNAMIC_GAP_TRIM_PADDING_MS: int = 50
    DYNAMIC_GAP_TRIM_PADDING_CHAR_MULTIPLIER: dict = {
        ",": 1.2,
        ".": 1.5,
        "!": 1.5,
        "?": 1.5,
        ";": 1.3,
        ":": 1.3,
    }

    # Model version (supertonic-3 is current)
    DEFAULT_MODEL_VERSION: str = "v3"

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()

