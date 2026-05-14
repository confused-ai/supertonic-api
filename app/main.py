from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import FileResponse, PlainTextResponse, Response
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.logging import setup_logging
from app.services.tts import tts_service
from app.api.routes import router as api_router

setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup and shutdown events."""
    await tts_service.initialize()
    yield


app = FastAPI(
    title="Supertonic TTS API",
    description="OpenAI-compatible text-to-speech API powered by Supertonic 3",
    version="3.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)


# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=FileResponse)
async def root():
    """Serve the landing page."""
    static_path = Path("app/static/index.html")
    if static_path.exists():
        return FileResponse(static_path)
    return {"message": "Go to /docs for API documentation"}


@app.get("/robots.txt", response_class=PlainTextResponse)
async def robots():
    """Serve robots.txt for SEO."""
    p = Path("app/static/robots.txt")
    if p.exists():
        return p.read_text()
    return Response(status_code=404)


@app.get("/sitemap.xml")
async def sitemap():
    """Serve sitemap.xml for SEO."""
    p = Path("app/static/sitemap.xml")
    if p.exists():
        return Response(content=p.read_text(), media_type="application/xml")
    return Response(status_code=404)


@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    status = "healthy" if tts_service.model else "initializing"
    return {"status": status}


# Include API routers
app.include_router(api_router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=False,
    )
