import onnxruntime as ort
from fastapi import APIRouter

from app.api.schemas import ModelObject, ModelsResponse
from app.core.constants import MODEL_NAME

router = APIRouter(tags=["Models"])

_STATIC_MODELS: list[ModelObject] = [
    ModelObject(id="tts-1", created=1677610602, owned_by="openai"),
    ModelObject(id="tts-1-hd", created=1677610602, owned_by="openai"),
    ModelObject(id="tts-2", created=1704067200, owned_by="openai"),
    ModelObject(id="tts-2-hd", created=1704067200, owned_by="openai"),
    ModelObject(
        id=MODEL_NAME,
        created=1746057600,
        owned_by="supertone-inc",
        version="3.0",
        languages=31,
    ),
]


@router.get("/v1/models", response_model=ModelsResponse, summary="List available TTS models")
async def list_models() -> ModelsResponse:
    """Returns all supported model IDs. Compatible with the OpenAI `/v1/models` shape."""
    providers = ort.get_available_providers()
    data = [
        m.model_copy(update={"providers": providers}) if m.id == MODEL_NAME else m
        for m in _STATIC_MODELS
    ]
    return ModelsResponse(data=data)
