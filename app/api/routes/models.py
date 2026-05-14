import onnxruntime as ort
from fastapi import APIRouter

from app.core.constants import MODEL_NAME

router = APIRouter()

_STATIC_MODELS = [
    {"id": "tts-1", "created": 1677610602, "owned_by": "openai"},
    {"id": "tts-1-hd", "created": 1677610602, "owned_by": "openai"},
    {"id": "tts-2", "created": 1704067200, "owned_by": "openai"},
    {"id": "tts-2-hd", "created": 1704067200, "owned_by": "openai"},
    {
        "id": MODEL_NAME,
        "created": 1746057600,
        "owned_by": "supertone-inc",
        "version": "3.0",
        "languages": 31,
    },
]


@router.get("/v1/models")
async def list_models():
    """List available TTS models."""
    providers = ort.get_available_providers()
    return {
        "data": [
            {**model, "providers": providers} if model["id"] == MODEL_NAME else model
            for model in _STATIC_MODELS
        ]
    }
