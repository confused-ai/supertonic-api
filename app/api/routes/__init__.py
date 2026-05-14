"""API route modules — re-exports a single combined router."""

from fastapi import APIRouter

from app.api.routes.models import router as models_router
from app.api.routes.speech import router as speech_router
from app.api.routes.voices import router as voices_router

router = APIRouter()
router.include_router(speech_router)
router.include_router(models_router)
router.include_router(voices_router)

__all__ = ["router"]
