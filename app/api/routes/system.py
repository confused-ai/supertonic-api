import time
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse, PlainTextResponse, Response

from app.api.schemas import HealthResponse
from app.core.constants import APP_VERSION
from app.services.tts import tts_service

router = APIRouter(tags=["System"])

_STATIC_DIR = Path("app/static")
_START_TIME = time.time()


@router.get("/", response_class=FileResponse, include_in_schema=False)
async def root():
    """Serve the landing page."""
    p = _STATIC_DIR / "index.html"
    if p.exists():
        return FileResponse(p)
    return {"message": "Go to /docs for API documentation"}


@router.get("/robots.txt", response_class=PlainTextResponse, include_in_schema=False)
async def robots():
    """Serve robots.txt."""
    p = _STATIC_DIR / "robots.txt"
    if p.exists():
        return p.read_text()
    return Response(status_code=404)


@router.get("/sitemap.xml", include_in_schema=False)
async def sitemap():
    """Serve sitemap.xml."""
    p = _STATIC_DIR / "sitemap.xml"
    if p.exists():
        return Response(content=p.read_text(), media_type="application/xml")
    return Response(status_code=404)


@router.get("/health", response_model=HealthResponse, summary="Service health check")
async def health_check() -> HealthResponse:
    """Returns service status, version, and uptime. Used by load balancers."""
    model_ready = tts_service.model is not None
    return HealthResponse(
        status="healthy" if model_ready else "initializing",
        version=APP_VERSION,
        model="supertonic-3" if model_ready else None,
        uptime_s=round(time.time() - _START_TIME, 1),
    )
