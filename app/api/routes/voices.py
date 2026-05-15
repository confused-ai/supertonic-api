import json
import os
import re
import tempfile

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.api.schemas import (
    VoiceDeleteResponse,
    VoiceInfo,
    VoiceListResponse,
    VoiceMixRequest,
    VoiceUploadResponse,
)
from app.core.constants import MAX_VOICE_JSON_BYTES
from app.core.logging import logger
from app.core.voices import OPENAI_TO_SUPERTONIC
from app.services.tts import tts_service

router = APIRouter(tags=["Voices"])

_VOICE_NAME_RE = re.compile(r'^[\w\- ]+$')

_VOICE_DESCRIPTIONS: dict[str, str] = {
    # ── Female voices (F1–F5) ──────────────────────────────────────────────
    "alloy":   "F1 — calm, clear female voice",
    "nova":    "F2 — bright, professional female voice",
    "shimmer": "F3 — soft, expressive female voice",
    "ash":     "F4 — energetic, versatile female voice",
    "ballad":  "F4 — melodic, smooth female voice (shares style with ash)",
    "coral":   "F5 — airy, warm female voice",
    "marin":   "F5 — gentle, natural female voice (shares style with coral)",
    # ── Male voices (M1–M5) ───────────────────────────────────────────────
    "echo":    "M1 — lively, upbeat male voice",
    "fable":   "M2 — warm, narrative male voice",
    "onyx":    "M3 — deep, authoritative male voice",
    "cedar":   "M4 — measured, resonant male voice",
    "sage":    "M4 — calm, steady male voice (shares style with cedar)",
    "verse":   "M5 — dynamic, dramatic male voice",
}


def _assert_model_ready() -> None:
    if not tts_service.model:
        raise HTTPException(status_code=503, detail="Model not ready")


@router.get("/voices")
async def list_voices_legacy():
    """Legacy voice list endpoint (no version prefix)."""
    _assert_model_ready()
    preset_ids = list(OPENAI_TO_SUPERTONIC.keys())
    native = getattr(tts_service.model, "voice_style_names", [])
    custom_ids = tts_service.list_custom_voice_ids()
    return {
        "voices": preset_ids + custom_ids,
        "native_styles": native,
        "custom_voices": custom_ids,
    }


@router.get("/v1/voices", response_model=VoiceListResponse)
async def list_voices():
    """List all available voices: preset + custom."""
    _assert_model_ready()

    voices: list[VoiceInfo] = [
        VoiceInfo(
            id=name,
            name=name,
            type="preset",
            description=_VOICE_DESCRIPTIONS.get(name),
        )
        for name in OPENAI_TO_SUPERTONIC
    ]

    for native in getattr(tts_service.model, "voice_style_names", []):
        if native not in OPENAI_TO_SUPERTONIC.values():
            voices.append(VoiceInfo(id=native, name=native, type="preset"))

    for cid in tts_service.list_custom_voice_ids():
        vtype = "mixed" if cid.startswith("mix:") else "custom"
        voices.append(VoiceInfo(id=cid, name=cid, type=vtype))

    return VoiceListResponse(voices=voices)


@router.post("/v1/voices/upload", response_model=VoiceUploadResponse)
async def upload_voice(
    file: UploadFile = File(..., description="Voice style JSON file"),
    name: str = Form(..., min_length=1, max_length=64, description="Name for this voice"),
):
    """Upload a custom voice style JSON and register it.

    Expected JSON format::

        {"style_ttl": {"dims": [...], "data": [...]},
         "style_dp":  {"dims": [...], "data": [...]}}
    """
    _assert_model_ready()

    if not _VOICE_NAME_RE.match(name):
        raise HTTPException(status_code=422, detail="Voice name contains invalid characters")

    content = await file.read(MAX_VOICE_JSON_BYTES + 1)
    if len(content) > MAX_VOICE_JSON_BYTES:
        raise HTTPException(status_code=413, detail="Voice JSON exceeds 10 MB limit")

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

    tts_service.register_custom_voice(name, style)
    logger.info(f"Registered custom voice '{name}'")

    return VoiceUploadResponse(id=name, name=name, message=f"Voice '{name}' uploaded and ready")


@router.post("/v1/voices/mix", response_model=VoiceUploadResponse)
async def mix_voices(data: VoiceMixRequest):
    """Create a new voice by blending two existing voices."""
    _assert_model_ready()

    if not _VOICE_NAME_RE.match(data.name):
        raise HTTPException(status_code=422, detail="Voice name contains invalid characters")

    try:
        style = tts_service.mix_voices(data.voice_a, data.voice_b, data.weight)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Mix failed: {exc}")

    voice_id = f"mix:{data.name}"
    tts_service.register_custom_voice(voice_id, style)
    logger.info(
        f"Registered mixed voice '{voice_id}' "
        f"({data.voice_a} x{1 - data.weight:.2f} + {data.voice_b} x{data.weight:.2f})"
    )

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
