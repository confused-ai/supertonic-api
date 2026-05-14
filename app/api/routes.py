import json
import os
import re
import tempfile
import time
import traceback
import onnxruntime as ort
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import Response

from app.core.config import settings
from app.core.voices import OPENAI_TO_SUPERTONIC
from app.api.schemas import (
    OpenAIInput,
    VoiceInfo, VoiceListResponse, VoiceMixRequest,
    VoiceUploadResponse, VoiceDeleteResponse,
)
from app.services.tts import tts_service
from app.utils.text import clean_text
from app.core.logging import logger
from app.services.audio_encoder import AudioEncoder

router = APIRouter()

# Media type mappings
MEDIA_TYPES = {
    "mp3": "audio/mpeg",
    "wav": "audio/wav",
    "flac": "audio/flac",
    "opus": "audio/ogg",
    "aac": "audio/aac",
    "pcm": "audio/pcm",
}


@router.post("/v1/audio/speech")
async def generate_speech(
    data: OpenAIInput,
):
    """Generate speech from text using TTS."""
    try:
        char_count = len(data.input)

        # Check model availability
        if not tts_service.model:
            raise HTTPException(status_code=503, detail="Model loading")

        # Normalize text if requested
        normalized_text = clean_text(data.input) if data.normalize else data.input
        logger.debug(f"Normalized text: {normalized_text[:100]}...")

        sample_rate = getattr(tts_service.model, "sample_rate", settings.SAMPLE_RATE)
        media_type = MEDIA_TYPES.get(data.response_format, "audio/wav")
        filename = f"speech.{data.response_format}"

        start_time = time.time()

        try:
            writer = AudioEncoder(
                format=data.response_format, sample_rate=sample_rate
            )
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

            total_time = (time.time() - start_time) * 1000
            logger.info(f"TTS Total Time: {total_time:.2f}ms for {char_count} chars")

            if not processed:
                raise ValueError("No audio output generated")

            return Response(
                content=processed,
                media_type=media_type,
                headers={"Content-Disposition": f'inline; filename="{filename}"'},
            )

        except Exception as e:
            logger.error(f"Synthesis error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unhandled error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Internal Server Error")


@router.get("/v1/models")
async def list_models():
    """List available TTS models."""
    providers = ort.get_available_providers()
    return {
        "data": [
            {"id": "tts-1", "created": 1677610602, "owned_by": "openai"},
            {"id": "tts-1-hd", "created": 1677610602, "owned_by": "openai"},
            {"id": "tts-2", "created": 1704067200, "owned_by": "openai"},
            {"id": "tts-2-hd", "created": 1704067200, "owned_by": "openai"},
            {
                "id": "supertonic-3",
                "created": 1746057600,
                "owned_by": "supertone",
                "version": "3.0",
                "languages": 31,
                "providers": providers,
            },
        ]
    }


@router.get("/voices")
async def list_voices_legacy():
    """Legacy voice list endpoint."""
    if not tts_service.model:
        raise HTTPException(status_code=503, detail="Model not ready")
    preset_ids = list(OPENAI_TO_SUPERTONIC.keys())
    native = getattr(tts_service.model, "voice_style_names", [])
    custom_ids = tts_service.list_custom_voice_ids()
    return {
        "voices": preset_ids + custom_ids,
        "native_styles": native,
        "custom_voices": custom_ids,
    }


@router.get("/v1/voices", response_model=VoiceListResponse)
async def list_voices_v1():
    """List all available voices: preset + custom."""
    if not tts_service.model:
        raise HTTPException(status_code=503, detail="Model not ready")

    voice_descriptions = {
        "alloy": "F1 \u2014 Sarah, calm female voice",
        "echo": "M1 \u2014 Alex, lively upbeat male voice",
        "fable": "F2 \u2014 Lily, bright cheerful female voice",
        "onyx": "M2 \u2014 James, deep robust male voice",
        "nova": "F3 \u2014 Jessica, professional announcer-style female",
        "shimmer": "M3 \u2014 Robert, polished authoritative male voice",
    }

    voices: list[VoiceInfo] = []
    for name in OPENAI_TO_SUPERTONIC.keys():
        voices.append(VoiceInfo(
            id=name, name=name, type="preset",
            description=voice_descriptions.get(name),
        ))

    for native in getattr(tts_service.model, "voice_style_names", []):
        if native not in OPENAI_TO_SUPERTONIC.values():
            voices.append(VoiceInfo(id=native, name=native, type="preset"))

    for cid in tts_service.list_custom_voice_ids():
        vtype = "mixed" if cid.startswith("mix:") else "custom"
        voices.append(VoiceInfo(id=cid, name=cid, type=vtype))

    return VoiceListResponse(voices=voices)


_MAX_VOICE_JSON_BYTES = 10 * 1024 * 1024  # 10 MB


@router.post("/v1/voices/upload", response_model=VoiceUploadResponse)
async def upload_voice(
    file: UploadFile = File(..., description="Voice style JSON file"),
    name: str = Form(..., min_length=1, max_length=64, description="Name for this voice"),
):
    """Upload a custom voice style JSON and register it for use in TTS.

    Expected JSON format:
      {"style_ttl": {"dims": [...], "data": [...]},
       "style_dp":  {"dims": [...], "data": [...]}}
    """
    if not tts_service.model:
        raise HTTPException(status_code=503, detail="Model not ready")

    # Sanitise name — only alphanumerics, hyphens, underscores, spaces
    if not re.match(r'^[\w\- ]+$', name):
        raise HTTPException(status_code=422, detail="Voice name contains invalid characters")

    # Read file with size cap
    content = await file.read(_MAX_VOICE_JSON_BYTES + 1)
    if len(content) > _MAX_VOICE_JSON_BYTES:
        raise HTTPException(status_code=413, detail="Voice JSON exceeds 10 MB limit")

    # Validate JSON structure
    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=422, detail=f"Invalid JSON: {exc}")

    if not isinstance(data, dict) or "style_ttl" not in data or "style_dp" not in data:
        raise HTTPException(
            status_code=422,
            detail="JSON must contain 'style_ttl' and 'style_dp' keys",
        )
    for key in ("style_ttl", "style_dp"):
        if not isinstance(data[key], dict) or "dims" not in data[key] or "data" not in data[key]:
            raise HTTPException(
                status_code=422,
                detail=f"'{key}' must have 'dims' and 'data' fields",
            )

    # Write to temp file and load via SDK
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        style = tts_service.model.get_voice_style_from_path(tmp_path)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Failed to load voice style: {exc}")
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    voice_id = name  # use name as ID (sanitised above)
    tts_service.register_custom_voice(voice_id, style)
    logger.info(f"Registered custom voice '{voice_id}'")

    return VoiceUploadResponse(
        id=voice_id,
        name=name,
        message=f"Voice '{name}' uploaded and ready",
    )


@router.post("/v1/voices/mix", response_model=VoiceUploadResponse)
async def mix_voices(data: VoiceMixRequest):
    """Create a new voice by blending two existing voices."""
    if not tts_service.model:
        raise HTTPException(status_code=503, detail="Model not ready")

    if not re.match(r'^[\w\- ]+$', data.name):
        raise HTTPException(status_code=422, detail="Voice name contains invalid characters")

    try:
        style = tts_service.mix_voices(data.voice_a, data.voice_b, data.weight)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Mix failed: {exc}")

    voice_id = f"mix:{data.name}"
    tts_service.register_custom_voice(voice_id, style)
    logger.info(f"Registered mixed voice '{voice_id}' ({data.voice_a} x{1-data.weight:.2f} + {data.voice_b} x{data.weight:.2f})")

    return VoiceUploadResponse(
        id=voice_id,
        name=data.name,
        message=f"Mixed voice '{data.name}' created (weight={data.weight})",
    )


@router.delete("/v1/voices/{voice_id:path}", response_model=VoiceDeleteResponse)
async def delete_voice(voice_id: str):
    """Delete a custom or mixed voice."""
    deleted = tts_service.remove_custom_voice(voice_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Voice '{voice_id}' not found")
    logger.info(f"Deleted custom voice '{voice_id}'")
    return VoiceDeleteResponse(id=voice_id, deleted=True)
