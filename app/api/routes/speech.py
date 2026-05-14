import asyncio
import time

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from app.api.schemas import OpenAIInput
from app.core.config import settings
from app.core.constants import MEDIA_TYPES
from app.core.logging import logger
from app.services.audio_encoder import AudioEncoder
from app.services.tts import tts_service
from app.utils.text import clean_text

router = APIRouter()


@router.post("/v1/audio/speech")
async def generate_speech(data: OpenAIInput):
    """Generate speech from text using TTS."""
    if not tts_service.model:
        raise HTTPException(status_code=503, detail="Model loading")

    normalized_text = clean_text(data.input) if data.normalize else data.input

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
                    speed=data.speed,
                    output_format=data.response_format,
                    lang=data.lang,
                ),
                timeout=settings.REQUEST_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail="Synthesis timed out")
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception(f"Synthesis error: {exc}")
        raise HTTPException(status_code=500, detail="Internal server error")
    finally:
        writer.close()

    if not processed:
        raise HTTPException(status_code=500, detail="No audio generated")

    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info(f"TTS {elapsed_ms:.0f}ms | {len(data.input)} chars | voice={data.voice} | fmt={data.response_format}")

    return Response(
        content=processed,
        media_type=media_type,
        headers={"Content-Disposition": f'inline; filename="{filename}"'},
    )
