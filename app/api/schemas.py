from pydantic import BaseModel, Field
from typing import Optional, Literal, List


class OpenAIInput(BaseModel):
    """OpenAI-compatible TTS input schema."""
    model: str = Field(default="tts-1", description="TTS model to use")
    input: str = Field(..., min_length=1, max_length=4096, description="Text to convert to speech")
    voice: str = Field(default="alloy", description="Voice to use for synthesis")
    response_format: Optional[Literal["mp3", "opus", "aac", "flac", "wav", "pcm"]] = Field(
        default="mp3", description="Output audio format"
    )
    speed: Optional[float] = Field(default=1.0, ge=0.5, le=2.0, description="Speech speed multiplier (0.5–2.0)")
    normalize: bool = Field(default=True, description="Whether to normalize text before synthesis")
    lang: str = Field(
        default="en",
        description="BCP-47 language code for synthesis (supertonic-3: 31 langs + 'na' fallback). "
                    "Supports expression tags in input: <laugh>, <breath>, <sigh>.",
    )


class ModelObject(BaseModel):
    """Model object for /v1/models endpoint."""
    id: str
    object: str = "model"
    created: int
    owned_by: str
    providers: Optional[list] = None


class VoiceInfo(BaseModel):
    """Info about a single voice."""
    id: str
    name: str
    type: str  # "preset" | "custom" | "mixed"
    description: Optional[str] = None


class VoiceListResponse(BaseModel):
    """Response for GET /v1/voices."""
    voices: List[VoiceInfo]


class VoiceMixRequest(BaseModel):
    """Request to create a mixed voice."""
    voice_a: str = Field(..., description="First voice name/ID")
    voice_b: str = Field(..., description="Second voice name/ID")
    weight: float = Field(default=0.5, ge=0.0, le=1.0,
                          description="Blend weight: 0.0=all voice_a, 1.0=all voice_b")
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


class ModelList(BaseModel):
    """List of models for /v1/models endpoint."""
    object: str = "list"
    data: list[ModelObject]
