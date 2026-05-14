from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from app.api.routes import public_router, v1_router
from app.core.config import settings
from app.core.constants import APP_VERSION
from app.core.errors import register_error_handlers
from app.core.logging import setup_logging
from app.core.middleware import AccessLogMiddleware, RequestIDMiddleware, SecurityHeadersMiddleware
from app.core.ratelimit import rate_limit
from app.services.tts import tts_service

setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await tts_service.initialize()
    yield
    tts_service.shutdown()


app = FastAPI(
    title="Supertonic TTS API",
    description="OpenAI-compatible text-to-speech API powered by Supertonic 3",
    version=APP_VERSION,
    lifespan=lifespan,
    docs_url="/docs" if settings.ENABLE_DOCS else None,
    redoc_url="/redoc" if settings.ENABLE_DOCS else None,
    openapi_url="/openapi.json" if settings.ENABLE_DOCS else None,
)

# Middleware — applied bottom-up: last add_middleware call = outermost layer.
# Request flow: CORS → GZip → RequestID → SecurityHeaders → AccessLog → routes
app.add_middleware(AccessLogMiddleware)
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(RequestIDMiddleware)
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    # allow_credentials requires explicit origins — incompatible with wildcard "*"
    allow_credentials="*" not in settings.CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

register_error_handlers(app)

# Public routes: /, /health, /robots.txt, /sitemap.xml — no rate limit
app.include_router(public_router)

# API routes: all /v1/* — rate-limited per IP
app.include_router(v1_router, dependencies=[Depends(rate_limit)])


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=False,
    )
