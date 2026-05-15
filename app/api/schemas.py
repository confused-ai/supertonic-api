from typing import Any, Literal

from pydantic import BaseModel, Field


# Supertonic synthesis supports 0.7–2.0; values outside that range are clamped.
_SPEED_MIN = 0.7
_SPEED_MAX = 2.0


class OpenAIInput(BaseModel):
    """OpenAI-compatible TTS input schema (drop-in for POST /v1/audio/speech)."""
    model: str = Field(default="tts-1", description="TTS model to use")
    input: str = Field(..., min_length=1, max_length=4096, description="Text to convert to speech")
    voice: str = Field(default="alloy", description="Voice to use for synthesis")
    response_format: Literal["mp3", "opus", "aac", "flac", "wav", "pcm"] = Field(
        default="mp3", description="Output audio format"
    )
    # OpenAI range is 0.25–4.0; values are clamped to supertonic's 0.7–2.0 at synthesis time.
    speed: float = Field(default=1.0, ge=0.25, le=4.0, description="Speech speed (0.25–4.0, clamped to 0.7–2.0)")
    # OpenAI 'instructions' field — accepted and silently ignored (not supported by supertonic).
    instructions: str | None = Field(default=None, exclude=True, description="Style instructions (ignored)")
    # Extensions beyond the OpenAI spec:
    normalize: bool = Field(default=True, description="Whether to normalize text before synthesis")
    lang: str = Field(
        default="en",
        description="BCP-47 language code for synthesis (supertonic-3: 31 langs + 'na' fallback). "
                    "Supports expression tags in input: <laugh>, <breath>, <sigh>.",
    )

    @property
    def clamped_speed(self) -> float:
        """Speed clamped to supertonic's supported range."""
        return max(_SPEED_MIN, min(_SPEED_MAX, self.speed))

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "model": "tts-1",
                    "input": "Hello, world! This is a test.",
                    "voice": "alloy",
                    "response_format": "mp3",
                    "speed": 1.0,
                    "lang": "en",
                }
            ]
        }
    }


class VoiceInfo(BaseModel):
    """Info about a single voice."""
    id: str
    name: str
    type: str  # "preset" | "custom" | "mixed"
    description: str | None = None


class VoiceListResponse(BaseModel):
    """Response for GET /v1/voices."""
    voices: list[VoiceInfo]


class VoiceMixRequest(BaseModel):
    """Request to create a mixed voice."""
    voice_a: str = Field(..., description="First voice name/ID")
    voice_b: str = Field(..., description="Second voice name/ID")
    weight: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="Blend weight: 0.0=all voice_a, 1.0=all voice_b",
    )
    name: str = Field(..., min_length=1, max_length=64, description="Name for the new mixed voice")


class VoiceUploadResponse(BaseModel):
    """Response after uploading a custom voice JSON."""
    id: str
    name: str
    message: str


class VoiceDeleteResponse(BaseModel):
    """Response after deleting a custom voice."""
    id: str
    deleted: bool


# ── System ───────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: Literal["healthy", "initializing"]
    version: str
    model: str | None
    uptime_s: float


# ── Models ───────────────────────────────────────────────────────────────────

class ModelObject(BaseModel):
    id: str
    created: int
    owned_by: str
    providers: list[str] | None = None
    version: str | None = None
    languages: int | None = None
    extra: dict[str, Any] | None = Field(default=None, exclude=True)


class ModelsResponse(BaseModel):
    data: list[ModelObject]


# ── Error ────────────────────────────────────────────────────────────────────

class ErrorDetail(BaseModel):
    detail: str
