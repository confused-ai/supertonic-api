import asyncio
import time

from collections.abc import AsyncGenerator

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response, StreamingResponse

from app.api.schemas import ErrorDetail, OpenAIInput
from app.core.config import settings
from app.core.constants import MEDIA_TYPES
from app.core.logging import logger
from app.services.audio_encoder import AudioEncoder
from app.services.tts import tts_service
from app.utils.text import clean_text

router = APIRouter(tags=["Speech"])

_AUDIO_RESPONSES = {
    200: {"content": {"audio/mpeg": {}, "audio/opus": {}, "audio/wav": {}, "audio/aac": {}, "audio/flac": {}, "audio/pcm": {}}, "description": "Synthesized audio"},
    500: {"model": ErrorDetail, "description": "Synthesis failed"},
    503: {"model": ErrorDetail, "description": "Model not ready"},
    504: {"model": ErrorDetail, "description": "Synthesis timed out"},
}


@router.post(
    "/v1/audio/speech",
    response_class=Response,
    responses=_AUDIO_RESPONSES,
    summary="Generate speech (buffered)",
)
async def generate_speech(data: OpenAIInput):
    """Generate speech from text. Returns the full audio file once synthesis completes."""
    if not tts_service.model:
        raise HTTPException(status_code=503, detail="Model loading")

    normalized_text = clean_text(data.input, lang=data.lang) if data.normalize else data.input

    sample_rate = getattr(tts_service.model, "sample_rate", settings.SAMPLE_RATE)
    media_type = MEDIA_TYPES.get(data.response_format, "audio/wav")
    filename = f"speech.{data.response_format}"

    start = time.perf_counter()
    writer = AudioEncoder(format=data.response_format, sample_rate=sample_rate)
    try:
        try:
            processed = await asyncio.wait_for(
                tts_service.generate_audio(
                    normalized_text,
                    data.voice,
                    writer,
                    speed=data.clamped_speed,
                    output_format=data.response_format,
                    lang=data.lang,
                ),
                timeout=settings.REQUEST_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail="Synthesis timed out")
        except Exception as exc:
            logger.exception(f"Synthesis error: {exc}")
            raise HTTPException(status_code=500, detail="Internal server error")
    finally:
        writer.close()

    if not processed:
        raise HTTPException(status_code=500, detail="No audio generated")

    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info(f"TTS {elapsed_ms:.0f}ms | {len(data.input)} chars | voice={data.voice} | fmt={data.response_format} | speed={data.clamped_speed}")

    return Response(
        content=processed,
        media_type=media_type,
        headers={"Content-Disposition": f'inline; filename="{filename}"'},
    )


@router.post(
    "/v1/audio/speech/stream",
    response_class=StreamingResponse,
    responses={
        **_AUDIO_RESPONSES,
        200: {"content": {"audio/mpeg": {}, "audio/opus": {}, "audio/wav": {}}, "description": "Streamed audio chunks"},
    },
    summary="Generate speech (streaming)",
)
async def generate_speech_stream(data: OpenAIInput):
    """
    Generate speech from text, streamed progressively.

    Chunks are yielded as synthesized — client can start playback immediately.
    Best used with `mp3` for broad MediaSource compatibility.
    """
    if not tts_service.model:
        raise HTTPException(status_code=503, detail="Model loading")

    normalized_text = clean_text(data.input, lang=data.lang) if data.normalize else data.input
    sample_rate = getattr(tts_service.model, "sample_rate", settings.SAMPLE_RATE)
    media_type = MEDIA_TYPES.get(data.response_format, "audio/wav")
    filename = f"speech.{data.response_format}"

    async def _audio_stream() -> AsyncGenerator[bytes, None]:
        """Internal generator that yields encoded audio chunks."""
        writer = AudioEncoder(format=data.response_format, sample_rate=sample_rate)
        try:
            async for chunk in tts_service.generate_audio_stream(
                normalized_text,
                data.voice,
                writer,
                speed=data.clamped_speed,
                output_format=data.response_format,
                lang=data.lang,
            ):
                if chunk and chunk.output:
                    yield chunk.output
        except asyncio.CancelledError:
            logger.info("Stream cancelled by client")
        except Exception as exc:
            logger.exception(f"Streaming synthesis error: {exc}")
        finally:
            writer.close()

    return StreamingResponse(
        _audio_stream(),
        media_type=media_type,
        headers={
            "Content-Disposition": f'inline; filename="{filename}"',
            "X-Accel-Buffering": "no",  # Disable nginx/proxy buffering
            "Cache-Control": "no-cache",
        },
    )
