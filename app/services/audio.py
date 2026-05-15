import asyncio
import math

import numpy as np

from app.core.config import settings
from app.core.logging import logger
from app.inference.base import AudioChunk
from app.services.audio_encoder import AudioEncoder

try:
    import scipy  # noqa: F401
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class AudioNormalizer:
    """Handles audio normalization state for a single stream"""

    def __init__(self):
        self.chunk_trim_ms = settings.GAP_TRIM_MS
        self.sample_rate = settings.SAMPLE_RATE

    @property
    def samples_to_trim(self) -> int:
        return int(self.chunk_trim_ms * self.sample_rate / 1000)

    @property
    def samples_to_pad_start(self) -> int:
        return int(50 * self.sample_rate / 1000)

    def _apply_scipy_normalization(self, audio_data: np.ndarray) -> np.ndarray:
        """Apply scipy-powered RMS and/or peak normalization to the audio data.

        Order: RMS normalization first (adjusts overall loudness), then peak
        limiting (ensures no clipping). This guarantees peaks stay within bounds.
        """
        if not HAS_SCIPY:
            return audio_data

        # Ensure float for processing
        if audio_data.dtype == np.int16:
            audio_float = audio_data.astype(np.float32) / 32767.0
        else:
            audio_float = audio_data.astype(np.float32)

        peak_target = settings.AUDIO_PEAK_TARGET
        rms_target = settings.AUDIO_RMS_TARGET

        # 1. RMS normalization first (adjusts overall loudness)
        if rms_target is not None and rms_target > 0:
            current_rms = np.sqrt(np.mean(audio_float ** 2))
            if current_rms > 1e-10:
                gain = rms_target / current_rms
                # Limit gain to avoid excessive amplification (max 6 dB)
                gain = min(gain, 2.0)
                audio_float = audio_float * gain

        # 2. Peak normalization last (ensures no clipping)
        if peak_target is not None and peak_target > 0:
            current_peak = np.max(np.abs(audio_float))
            if current_peak > peak_target:
                gain = peak_target / current_peak
                audio_float = audio_float * gain

        # Clip to valid range and convert back to int16
        return np.clip(audio_float * 32767, -32768, 32767).astype(np.int16)

    def find_first_last_non_silent(
        self,
        audio_data: np.ndarray,
        chunk_text: str,
        speed: float,
        silence_threshold_db: int = -45,
        is_last_chunk: bool = False,
    ) -> tuple[int, int]:
        """Find the first and last non-silent sample indices using vectorized NumPy operations."""
        # Calculate padding multiplier based on last character
        pad_multiplier = 1
        split_character = chunk_text.strip()
        if split_character:
            split_character = split_character[-1]
            pad_multiplier = settings.DYNAMIC_GAP_TRIM_PADDING_CHAR_MULTIPLIER.get(
                split_character, 1
            )

        if not is_last_chunk:
            samples_to_pad_end = max(
                int(
                    (
                        settings.DYNAMIC_GAP_TRIM_PADDING_MS
                        * self.sample_rate
                        * pad_multiplier
                    )
                    / 1000
                )
                - self.samples_to_pad_start,
                0,
            )
        else:
            samples_to_pad_end = self.samples_to_pad_start

        # Vectorized silence detection
        amplitude_threshold = 32767 * (10 ** (silence_threshold_db / 20))

        # Find all non-silent indices at once
        non_silent_mask = np.abs(audio_data) > amplitude_threshold
        non_silent_indices = np.where(non_silent_mask)[0]

        if len(non_silent_indices) == 0:
            return 0, len(audio_data)

        non_silent_index_start = non_silent_indices[0]
        non_silent_index_end = non_silent_indices[-1]

        return max(non_silent_index_start - self.samples_to_pad_start, 0), min(
            non_silent_index_end + math.ceil(samples_to_pad_end / speed),
            len(audio_data),
        )

    def normalize(self, audio_data: np.ndarray) -> np.ndarray:
        """Normalize audio to int16 format with optional scipy-powered RMS/peak normalization."""
        if len(audio_data) == 0:
            return audio_data

        if audio_data.dtype != np.int16:
            audio_data = np.clip(audio_data * 32767, -32768, 32767).astype(np.int16)

        if HAS_SCIPY and (settings.AUDIO_RMS_TARGET is not None or settings.AUDIO_PEAK_TARGET is not None):
            audio_data = self._apply_scipy_normalization(audio_data)

        return audio_data


class AudioService:
    """Service for audio format conversions with streaming support"""

    SUPPORTED_FORMATS = frozenset({"wav", "mp3", "opus", "flac", "aac", "pcm"})

    @staticmethod
    async def convert_audio(
        audio_chunk: AudioChunk,
        output_format: str,
        writer: AudioEncoder,
        speed: float = 1,
        chunk_text: str = "",
        is_last_chunk: bool = False,
        trim_audio: bool = True,
        normalizer: AudioNormalizer = None,
    ) -> AudioChunk:
        """Convert audio chunk to the target format."""
        if output_format not in AudioService.SUPPORTED_FORMATS:
            raise ValueError(f"Format {output_format} not supported")

        loop = asyncio.get_running_loop()

        def _process():
            inner_normalizer = normalizer
            if inner_normalizer is None:
                inner_normalizer = AudioNormalizer()
                inner_normalizer.sample_rate = audio_chunk.sample_rate

            # Normalize once here; trim_audio skips its own normalize since we pass the normalizer.
            norm_audio = inner_normalizer.normalize(audio_chunk.audio)
            result_chunk = AudioChunk(
                audio=norm_audio,
                sample_rate=audio_chunk.sample_rate,
                text=audio_chunk.text,
            )

            if trim_audio:
                result_chunk = AudioService.trim_audio(
                    result_chunk, chunk_text, speed, is_last_chunk, inner_normalizer
                )

            chunk_data = b""
            if len(result_chunk.audio) > 0:
                chunk_data = writer.write_chunk(result_chunk.audio)

            if is_last_chunk:
                final_data = writer.write_chunk(finalize=True)
                result_chunk.output = chunk_data + (final_data if final_data else b"")
            elif chunk_data:
                result_chunk.output = chunk_data

            return result_chunk

        try:
            return await loop.run_in_executor(None, _process)
        except Exception as e:
            logger.error(f"Error converting audio stream to {output_format}: {e}")
            raise ValueError(f"Failed to convert audio stream to {output_format}: {e}")

    @staticmethod
    def trim_audio(
        audio_chunk: AudioChunk,
        chunk_text: str = "",
        speed: float = 1,
        is_last_chunk: bool = False,
        normalizer: AudioNormalizer = None,
    ) -> AudioChunk:
        """Trim silence from audio chunk edges. Assumes audio is already normalized to int16."""
        if normalizer is None:
            normalizer = AudioNormalizer()
            normalizer.sample_rate = audio_chunk.sample_rate
            # Caller did not pre-normalize — do it now.
            audio_chunk.audio = normalizer.normalize(audio_chunk.audio)

        if len(audio_chunk.audio) > (2 * normalizer.samples_to_trim):
            audio_chunk.audio = audio_chunk.audio[
                normalizer.samples_to_trim : -normalizer.samples_to_trim
            ]

        start_index, end_index = normalizer.find_first_last_non_silent(
            audio_chunk.audio, chunk_text, speed, is_last_chunk=is_last_chunk
        )

        start_index = int(start_index)
        end_index = int(end_index)

        audio_chunk.audio = audio_chunk.audio[start_index:end_index]

        return audio_chunk
