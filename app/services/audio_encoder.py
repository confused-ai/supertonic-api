from io import BytesIO
from typing import Optional
import av
import numpy as np
from app.core.logging import logger


class PipeIO:
    """
    A wrapper around a file-like object that exposes only write/tell/flush
    to trick PyAV into treating it as a non-seekable stream (like a pipe).
    """
    __slots__ = ('buffer',)
    
    def __init__(self, buffer):
        self.buffer = buffer

    def write(self, data):
        return self.buffer.write(data)

    def tell(self):
        return self.buffer.tell()
        
    def flush(self):
        return self.buffer.flush()


class AudioEncoder:
    """Handles streaming audio format conversions using PyAV for efficient encoding."""
    
    # Codec mappings as class constant
    CODEC_MAP = {
        "wav": "pcm_s16le",
        "mp3": "mp3",
        "opus": "libopus",
        "flac": "flac",
        "aac": "aac",
    }

    def __init__(self, format: str, sample_rate: int, channels: int = 1):
        self.format = format.lower()
        self.sample_rate = sample_rate
        self.channels = channels
        self.pts = 0

        if self.format == "pcm":
            # PCM is raw audio, no container needed
            self.container = None
            self.stream = None
            self.output_buffer = None
        elif self.format in self.CODEC_MAP:
            self.output_buffer = BytesIO()
            
            container_options = {}
            if self.format == 'mp3':
                container_options = {'write_xing': '0'}

            # libopus only supports specific sample rates; 48000 is the standard.
            stream_rate = 48000 if self.format == "opus" else self.sample_rate

            self.container = av.open(
                PipeIO(self.output_buffer),
                mode="w",
                format=self.format if self.format != "aac" else "adts",
                options=container_options
            )
            self.stream = self.container.add_stream(
                self.CODEC_MAP[self.format],
                rate=stream_rate,
                layout="mono" if self.channels == 1 else "stereo",
            )
            
            if self.format in ['mp3', 'aac', 'opus']:
                self.stream.bit_rate = 128000
        else:
            raise ValueError(f"Unsupported format: {self.format}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def close(self):
        """Close and cleanup resources."""
        if hasattr(self, "output_buffer") and self.output_buffer:
            self.output_buffer.close()

    def write_chunk(
        self, audio_data: Optional[np.ndarray] = None, finalize: bool = False
    ) -> bytes:
        """Write a chunk of audio data and return bytes in the target format.

        Args:
            audio_data: Audio data to write, or None if finalizing
            finalize: Whether this is the final write to close the stream
        """
        if finalize:
            if self.format == "pcm":
                return b""
            
            # Flush stream encoder
            try:
                for packet in self.stream.encode(None):
                    self.container.mux(packet)
            except Exception as e:
                logger.warning(f"Error flushing encoder: {e}")

            # Close container to flush footer
            try:
                self.container.close()
            except Exception as e:
                logger.debug(f"Container close (expected if pipe): {e}")

            # Get the final bytes
            data = self.output_buffer.getvalue()
            self.output_buffer.close()
            return data

        if audio_data is None or len(audio_data) == 0:
            return b""

        if self.format == "pcm":
            # Write raw bytes
            return audio_data.tobytes()

        # Ensure audio_data is int16 as expected by the encoder
        if audio_data.dtype != np.int16:
            audio_data = (audio_data * 32767).astype(np.int16)

        frame = av.AudioFrame.from_ndarray(
            audio_data.reshape(1, -1),
            format="s16",
            layout="mono" if self.channels == 1 else "stereo",
        )
        frame.sample_rate = self.sample_rate
        frame.pts = self.pts
        self.pts += frame.samples

        for packet in self.stream.encode(frame):
            self.container.mux(packet)

        data = self.output_buffer.getvalue()
        # Reset buffer for next chunk
        self.output_buffer.seek(0)
        self.output_buffer.truncate(0)
        return data
