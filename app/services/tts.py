import asyncio
import os
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import onnxruntime as ort
from supertonic import TTS
from supertonic.core import Style

from app.core.config import settings
from app.core.constants import MODEL_NAME
from app.core.logging import logger
from app.core.voices import OPENAI_TO_SUPERTONIC
from app.inference.base import AudioChunk
from app.services.audio import AudioNormalizer, AudioService
from app.services.audio_encoder import AudioEncoder
from app.utils.text import smart_split

os.environ['HF_HUB_DISABLE_XET'] = '1'

# Dedicated thread pool for CPU-bound synthesis — isolated from the I/O thread pool.
_tts_executor = ThreadPoolExecutor(
    max_workers=settings.MAX_WORKERS,
    thread_name_prefix="tts-worker",
)


class TTSService:
    """Singleton TTS service for audio generation."""

    def __init__(self):
        self.model = None
        self._style_cache: dict = {}
        self._custom_voices: dict = {}
        self._load_lock = threading.Lock()
        # Created lazily inside the running event loop (asyncio.Semaphore is loop-bound).
        self._semaphore: asyncio.Semaphore | None = None
        self._apply_ort_provider_patch()

    @property
    def _chunk_semaphore(self) -> asyncio.Semaphore:
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(settings.MAX_WORKERS)
        return self._semaphore

    def _apply_ort_provider_patch(self):
        """Patch ONNX Runtime InferenceSession to enforce the configured execution provider."""
        try:
            _original_init = ort.InferenceSession.__init__
            force_provider = settings.FORCE_PROVIDERS

            def _patched_init(session_self, path_or_bytes, *args, **kwargs):
                available = ort.get_available_providers()
                kwargs["providers"] = self._select_providers(force_provider, available)
                logger.debug(f"ORT providers: {kwargs['providers']}")
                _original_init(session_self, path_or_bytes, *args, **kwargs)

            ort.InferenceSession.__init__ = _patched_init
            logger.info(f"ONNX Runtime provider patched. Strategy: {force_provider}")
        except Exception as e:
            logger.warning(f"Could not patch onnxruntime: {e}")

    def _select_providers(self, force_provider: str, available: list) -> list:
        """Select ONNX providers based on configuration."""
        if force_provider == "cuda" and "CUDAExecutionProvider" in available:
            return ["CUDAExecutionProvider"]
        if force_provider in ("coreml", "metal"):
            if "CoreMLExecutionProvider" in available:
                return ["CoreMLExecutionProvider"]
            logger.warning("CoreML/Metal requested but not available, falling back to CPU.")
            return ["CPUExecutionProvider"]
        if force_provider == "cpu":
            return ["CPUExecutionProvider"]
        if force_provider == "auto":
            providers = []
            if "CUDAExecutionProvider" in available:
                providers.append("CUDAExecutionProvider")
            if "CoreMLExecutionProvider" in available:
                providers.append("CoreMLExecutionProvider")
            providers.append("CPUExecutionProvider")
            return providers
        return ["CPUExecutionProvider"]

    def _ensure_model_loaded(self):
        """Thread-safe double-checked model load."""
        if self.model is not None:
            return
        with self._load_lock:
            if self.model is not None:
                return
            logger.info(f"Loading {MODEL_NAME}...")
            intra = settings.MODEL_THREADS if settings.MODEL_THREADS > 0 else None
            inter = settings.MODEL_INTER_THREADS if settings.MODEL_INTER_THREADS > 0 else None
            self.model = TTS(
                model=MODEL_NAME,
                auto_download=True,
                intra_op_num_threads=intra,
                inter_op_num_threads=inter,
            )
            self._style_cache.clear()
            logger.info(f"{MODEL_NAME} loaded. sample_rate={self.model.sample_rate}")

    async def initialize(self):
        """Eagerly load the model at startup without blocking the event loop."""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(_tts_executor, self._ensure_model_loaded)
        # Create the semaphore bound to this running loop.
        self._semaphore = asyncio.Semaphore(settings.MAX_WORKERS)
        logger.info("TTS Service ready")

    def get_style(self, voice_name: str):
        """Resolve voice name to a style object, with in-process caching."""
        self._ensure_model_loaded()
        if voice_name in self._custom_voices:
            return self._custom_voices[voice_name]
        cached = self._style_cache.get(voice_name)
        if cached is not None:
            return cached
        available = getattr(self.model, "voice_style_names", [])
        target = (
            voice_name
            if voice_name in available
            else OPENAI_TO_SUPERTONIC.get(voice_name, available[0] if available else "F1")
        )
        style = self.model.get_voice_style(voice_name=target)
        self._style_cache[voice_name] = style
        return style

    def register_custom_voice(self, voice_id: str, style) -> None:
        self._custom_voices[voice_id] = style

    def remove_custom_voice(self, voice_id: str) -> bool:
        existed = voice_id in self._custom_voices
        self._custom_voices.pop(voice_id, None)
        return existed

    def list_custom_voice_ids(self) -> list:
        return list(self._custom_voices.keys())

    def mix_voices(self, voice_a: str, voice_b: str, weight: float = 0.5):
        """Linearly interpolate two voice styles. weight=0.0 → all a, weight=1.0 → all b."""
        style_a = self.get_style(voice_a)
        style_b = self.get_style(voice_b)
        a = 1.0 - weight
        return Style(
            (a * style_a.ttl + weight * style_b.ttl).astype(np.float32),
            (a * style_a.dp + weight * style_b.dp).astype(np.float32),
        )

    async def _process_chunk(
        self,
        chunk_text: str,
        style,
        speed: float,
        writer: AudioEncoder,
        output_format: str,
        lang: str = "en",
        is_last: bool = False,
        normalizer: AudioNormalizer = None,
    ):
        """Synthesize one text chunk and encode to the target format."""
        async with self._chunk_semaphore:
            try:
                if is_last:
                    return await AudioService.convert_audio(
                        AudioChunk(np.array([], dtype=np.float32), sample_rate=self.model.sample_rate),
                        output_format, writer, speed, "", normalizer=normalizer, is_last_chunk=True,
                    )

                if not chunk_text.strip():
                    return None

                loop = asyncio.get_running_loop()
                logger.debug(f"Synthesizing: textlen={len(chunk_text)}, speed={speed}, lang={lang}")

                wav, _ = await loop.run_in_executor(
                    _tts_executor,
                    lambda: self.model.synthesize(chunk_text, voice_style=style, speed=speed, lang=lang),
                )

                logger.debug(f"Synthesized: shape={wav.shape}, dtype={wav.dtype}")

                # supertonic-3 returns (1, num_samples) — take a view into row 0, no copy.
                if wav.ndim == 2:
                    wav = wav[0]

                audio_chunk = AudioChunk(audio=wav, sample_rate=self.model.sample_rate, text=chunk_text)
                return await AudioService.convert_audio(
                    audio_chunk, output_format, writer, speed, chunk_text,
                    is_last_chunk=False, normalizer=normalizer,
                )
            except Exception as e:
                logger.error(f"Failed to process chunk: {e}")
                logger.error(traceback.format_exc())
                return None

    async def generate_audio_stream(
        self,
        text: str,
        voice: str,
        writer: AudioEncoder,
        speed: float = 1.0,
        output_format: str = "wav",
        lang: str = "en",
    ):
        """Yield encoded audio chunks as they are produced."""
        style = self.get_style(voice)  # ensures model loaded
        normalizer = AudioNormalizer()
        normalizer.sample_rate = self.model.sample_rate
        had_chunks = False

        async for chunk_text, pause_duration_s in smart_split(text):
            if pause_duration_s and pause_duration_s > 0:
                silence = np.zeros(int(pause_duration_s * self.model.sample_rate), dtype=np.int16)
                pause_chunk = AudioChunk(audio=silence, sample_rate=self.model.sample_rate)
                formatted_pause = await AudioService.convert_audio(
                    pause_chunk, output_format, writer,
                    speed=speed, is_last_chunk=False, trim_audio=False, normalizer=normalizer,
                )
                if formatted_pause.output:
                    yield formatted_pause
                had_chunks = True
            elif chunk_text.strip():
                processed = await self._process_chunk(
                    chunk_text, style, speed, writer, output_format,
                    lang=lang, is_last=False, normalizer=normalizer,
                )
                if processed and processed.output:
                    yield processed
                had_chunks = True

        if had_chunks:
            final = await self._process_chunk(
                "", style, speed, writer, output_format,
                lang=lang, is_last=True, normalizer=normalizer,
            )
            if final and final.output:
                yield final

    async def generate_audio(
        self,
        text: str,
        voice: str,
        writer: AudioEncoder,
        speed: float = 1.0,
        output_format: str = "wav",
        lang: str = "en",
    ) -> bytes:
        """Collect all streamed chunks and return complete audio bytes."""
        output = bytearray()
        async for chunk in self.generate_audio_stream(text, voice, writer, speed, output_format, lang):
            if chunk.output:
                output.extend(chunk.output)
        return bytes(output)


# Singleton instance
tts_service = TTSService()
