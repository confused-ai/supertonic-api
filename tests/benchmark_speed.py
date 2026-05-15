"""
Benchmark: TOTAL_STEPS vs SYNTHESIS_BATCH_SIZE

Measures end-to-end TTS generation time for different config combinations.
Test text is ~1500 chars to produce multiple chunks.
"""

import asyncio
import time
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Silence noisy loggers before importing app modules
os.environ["LOG_LEVEL"] = "ERROR"

import numpy as np
from app.core.config import settings
from app.services.tts import tts_service
from app.services.audio_encoder import AudioEncoder

# ── Test text (~1500 chars, 10+ sentences → many chunks) ──────────────────
LONG_TEXT = (
    "Welcome to the future of text to speech synthesis. "
    "Our technology converts written words into natural sounding audio with remarkable accuracy. "
    "The system supports multiple languages and voices for diverse applications. "
    "You can adjust the speaking rate between seventy percent and two hundred percent of normal speed. "
    "For developers, we provide a simple API that integrates seamlessly with existing applications. "
    "The neural network architecture ensures smooth and natural prosody across all supported languages. "
    "Each voice has been carefully crafted to deliver consistent quality across different types of content. "
    "Whether you are building assistive technology or creating audio content, our platform delivers results. "
    "The streaming API enables real time audio generation with minimal latency. "
    "Background processing ensures that long form content is handled efficiently. "
    "We have optimized the inference pipeline for both CPU and GPU environments. "
    "Batch processing allows multiple segments to be synthesized in parallel for improved throughput. "
    "The system automatically handles punctuation based pauses and natural breathing points. "
    "All audio output can be configured in multiple formats including MP3 and WAV. "
    "This completes the test sequence for benchmarking purposes."
)

# ── Config combinations ────────────────────────────────────────────────────
CONFIGS = [
    # (total_steps, batch_size, label)
    (5, 1, "baseline (steps=5, batch=1)"),
    (5, 4, "steps=5, batch=4"),
    (5, 8, "steps=5, batch=8"),
    (3, 1, "steps=3, batch=1"),
    (3, 4, "steps=3, batch=4"),
    (3, 8, "steps=3, batch=8"),
]

RUNS_PER_CONFIG = 3
VOICE = "alloy"
FORMAT = "wav"
SAMPLE_RATE = 44100


async def run_single(config_total_steps: int, config_batch_size: int, label: str) -> dict:
    """Run one benchmark iteration and return timing stats."""
    # Patch settings
    original_steps = settings.TOTAL_STEPS
    original_batch = settings.SYNTHESIS_BATCH_SIZE
    settings.TOTAL_STEPS = config_total_steps
    settings.SYNTHESIS_BATCH_SIZE = config_batch_size

    times = []
    output_sizes = []

    for run_idx in range(RUNS_PER_CONFIG + 1):  # +1 for warmup
        writer = AudioEncoder(format=FORMAT, sample_rate=SAMPLE_RATE)
        try:
            start = time.perf_counter()
            result = await tts_service.generate_audio(
                text=LONG_TEXT,
                voice=VOICE,
                writer=writer,
                speed=1.0,
                output_format=FORMAT,
                lang="en",
            )
            elapsed = time.perf_counter() - start

            if run_idx == 0:
                print(f"  Warmup: {elapsed*1000:.0f}ms, {len(result)} bytes")
            else:
                times.append(elapsed)
                output_sizes.append(len(result))
        finally:
            writer.close()

    # Restore original settings
    settings.TOTAL_STEPS = original_steps
    settings.SYNTHESIS_BATCH_SIZE = original_batch

    times_ms = [t * 1000 for t in times]
    return {
        "label": label,
        "steps": config_total_steps,
        "batch": config_batch_size,
        "times_ms": times_ms,
        "avg_ms": sum(times_ms) / len(times_ms),
        "min_ms": min(times_ms),
        "max_ms": max(times_ms),
        "avg_bytes": sum(output_sizes) / len(output_sizes),
    }


async def main():
    print("=" * 72)
    print("  TTS Speed Benchmark: TOTAL_STEPS × SYNTHESIS_BATCH_SIZE")
    print("=" * 72)
    print(f"\n  Test text: {len(LONG_TEXT)} chars")
    print(f"  Voice: {VOICE}, Format: {FORMAT}")
    print(f"  Runs per config: {RUNS_PER_CONFIG}")
    print(f"  Provider: {settings.FORCE_PROVIDERS}")
    print(f"  Thread pool workers: {settings.MAX_WORKERS}")
    print()

    # Ensure model is loaded
    print("  Loading model...", end=" ", flush=True)
    await tts_service.initialize()
    print(f"OK (sample_rate={tts_service.model.sample_rate})")
    print()

    # Verify we can get a style
    print("  Resolving voice...", end=" ", flush=True)
    _ = tts_service.get_style(VOICE)
    print("OK")
    print()

    results = []
    for steps, batch, label in CONFIGS:
        print(f"[{label}]")
        result = await run_single(steps, batch, label)
        results.append(result)
        print(
            f"  → avg={result['avg_ms']:.0f}ms  "
            f"min={result['min_ms']:.0f}ms  "
            f"max={result['max_ms']:.0f}ms  "
            f"output={result['avg_bytes']/1024:.0f}KB"
        )
        print()

    # ── Summary table ──
    print("=" * 72)
    print("  SUMMARY")
    print("=" * 72)
    print(f"  {'Config':<32} {'Avg (ms)':>10} {'Min (ms)':>10} {'Max (ms)':>10}  {'Output':>8}")
    print(f"  {'-'*31}  {'-'*9}  {'-'*9}  {'-'*9}  {'-'*7}")
    baseline = results[0]["avg_ms"]
    for r in results:
        speedup = baseline / r["avg_ms"]
        print(
            f"  {r['label']:<32} {r['avg_ms']:>8.0f}ms  "
            f"{r['min_ms']:>8.0f}ms  "
            f"{r['max_ms']:>8.0f}ms  "
            f"×{speedup:.1f}x  "
            f"{r['avg_bytes']/1024:.0f}KB"
        )

    print()
    print(f"  Baseline (steps=5, batch=1): {baseline:.0f}ms")
    for r in results[1:]:
        speedup = baseline / r["avg_ms"]
        vs_seq = results[0]["avg_ms"] / r["avg_ms"]
        pct = (1 - 1 / vs_seq) * 100
        print(
            f"  {r['label']:<32} → {r['avg_ms']:.0f}ms  "
            f"({pct:+.0f}% vs sequential)"
        )
    print("=" * 72)


if __name__ == "__main__":
    asyncio.run(main())
