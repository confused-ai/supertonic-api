"""API route modules.

Exports two routers:
- ``public_router`` — system endpoints (health, static files). No rate limit.
- ``v1_router``     — all /v1/* API endpoints. Rate-limited per IP.
"""

from fastapi import APIRouter

from app.api.routes.models import router as models_router
from app.api.routes.speech import router as speech_router
from app.api.routes.system import router as system_router
from app.api.routes.voices import router as voices_router

# System / infra routes — always public
public_router = APIRouter()
public_router.include_router(system_router)

# API routes — rate-limited (dependency injected in main.py)
v1_router = APIRouter()
v1_router.include_router(speech_router)
v1_router.include_router(models_router)
v1_router.include_router(voices_router)

__all__ = ["public_router", "v1_router"]
