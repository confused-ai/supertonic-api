import time
import traceback

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
    logger.debug(f"Normalized text: {normalized_text[:100]}...")

    sample_rate = getattr(tts_service.model, "sample_rate", settings.SAMPLE_RATE)
    media_type = MEDIA_TYPES.get(data.response_format, "audio/wav")
    filename = f"speech.{data.response_format}"

    start_time = time.time()
    try:
        writer = AudioEncoder(format=data.response_format, sample_rate=sample_rate)
        try:
            processed = await tts_service.generate_audio(
                normalized_text,
                data.voice,
                writer,
                speed=data.speed,
                output_format=data.response_format,
                lang=data.lang,
            )
        finally:
            writer.close()

        elapsed_ms = (time.time() - start_time) * 1000
        logger.info(f"TTS Total Time: {elapsed_ms:.2f}ms for {len(data.input)} chars")

        if not processed:
            raise ValueError("No audio output generated")

        return Response(
            content=processed,
            media_type=media_type,
            headers={"Content-Disposition": f'inline; filename="{filename}"'},
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Synthesis error: {exc}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Internal Server Error")
