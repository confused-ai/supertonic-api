#!/usr/bin/env python3
"""
Unified Supertonic TTS test runner.

Usage:
    python tests/run_all.py                        # unit + integration + eval
    python tests/run_all.py --url http://host:8800 # custom server
    python tests/run_all.py --unit-only            # unit tests only (no server)
    python tests/run_all.py --stress               # include stress suite
    python tests/run_all.py --concurrency 20 --requests 200  # stress params
"""
import argparse
import asyncio
import os
import statistics
import struct
import sys
import time
import unittest
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import httpx
import numpy as np

# ---------------------------------------------------------------------------
# Ensure project root is on path so unit tests can import app modules
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------
GREEN = "\033[92m"
RED   = "\033[91m"
YELLOW = "\033[93m"
CYAN  = "\033[96m"
BOLD  = "\033[1m"
RESET = "\033[0m"

def _ok(msg):  print(f"  {GREEN}✓{RESET} {msg}")
def _fail(msg): print(f"  {RED}✗{RESET} {msg}")
def _info(msg): print(f"  {CYAN}→{RESET} {msg}")
def _head(msg): print(f"\n{BOLD}{CYAN}{msg}{RESET}")
def _warn(msg): print(f"  {YELLOW}⚠{RESET}  {msg}")

# ---------------------------------------------------------------------------
# Result tracking
# ---------------------------------------------------------------------------
@dataclass
class Suite:
    name: str
    passed: int = 0
    failed: int = 0
    errors: list = field(default_factory=list)

    def record(self, ok: bool, label: str, detail: str = ""):
        if ok:
            self.passed += 1
            _ok(label)
        else:
            self.failed += 1
            self.errors.append(f"{label}: {detail}")
            _fail(f"{label}  —  {detail}")

    @property
    def total(self): return self.passed + self.failed

# ==========================================================================
# 1. UNIT TESTS  (pure Python, no server needed)
# ==========================================================================

def run_unit_tests() -> Suite:
    _head("UNIT TESTS")
    suite = Suite("unit")

    # ---- text utilities ----
    from app.utils.text import clean_text, smart_split

    def _clean(label, inp, expected_contains=None, expected_not_contains=None, ends_with_punct=None):
        try:
            result = clean_text(inp)
            ok = True
            detail = ""
            if expected_contains and expected_contains not in result:
                ok, detail = False, f"missing '{expected_contains}' in '{result}'"
            if expected_not_contains and expected_not_contains in result:
                ok, detail = False, f"found '{expected_not_contains}' in '{result}'"
            if ends_with_punct and not result[-1] in ".!?;:,'\"":
                ok, detail = False, f"missing ending punct, got '{result[-3:]}'"
            suite.record(ok, label, detail)
        except Exception as e:
            suite.record(False, label, str(e))

    _clean("clean_text: removes emoji",    "Hello 😀 world",       expected_not_contains="😀")
    _clean("clean_text: en-dash → hyphen", "Well–done",             expected_contains="-")
    _clean("clean_text: adds period",      "Hello world",           ends_with_punct=True)
    _clean("clean_text: no double period", "Hello world.",          expected_not_contains="..")
    _clean("clean_text: e.g. expansion",   "e.g., this works",      expected_contains="for example")
    _clean("clean_text: multi-space",      "hello   world",         expected_not_contains="  ")
    _clean("clean_text: @ expansion",      "reach us @support",     expected_contains="at")

    # edge: empty string should not crash
    try:
        result = clean_text("")
        suite.record(True, "clean_text: empty string no crash")
    except Exception as e:
        suite.record(False, "clean_text: empty string no crash", str(e))

    # smart_split
    async def _collect_split(text, max_chunk=300):
        chunks = []
        async for chunk_text, _, pause in smart_split(text, max_chunk_length=max_chunk):
            chunks.append((chunk_text, pause))
        return chunks

    def _split(label, text, max_chunk=300, min_chunks=1, has_pause=False):
        try:
            loop = asyncio.new_event_loop()
            try:
                chunks = loop.run_until_complete(_collect_split(text, max_chunk))
            finally:
                loop.close()
            ok = len(chunks) >= min_chunks
            detail = "" if ok else f"got {len(chunks)} chunks, expected >={min_chunks}"
            if ok and has_pause:
                ok = any(p is not None for _, p in chunks)
                detail = "no pause chunk found" if not ok else ""
            suite.record(ok, label, detail)
        except Exception as e:
            suite.record(False, label, str(e))

    _split("smart_split: short text → 1 chunk",       "Hello world.", min_chunks=1)
    _split("smart_split: long text → multiple chunks",
           "A " * 200, max_chunk=50, min_chunks=2)
    _split("smart_split: pause tag yields pause",
           "Hello. [pause:1.0] World.", has_pause=True)
    _split("smart_split: empty string → 0 chunks",    "", min_chunks=0)

    # ---- schema validation ----
    from pydantic import ValidationError
    from app.api.schemas import OpenAIInput

    def _schema(label, ok_expected, **kwargs):
        try:
            OpenAIInput(**kwargs)
            suite.record(ok_expected, label,
                         "" if ok_expected else "expected ValidationError but got none")
        except ValidationError:
            suite.record(not ok_expected, label,
                         "" if not ok_expected else "unexpected ValidationError")
        except Exception as e:
            suite.record(False, label, str(e))

    _schema("schema: valid minimal input",         True,  input="Hello")
    _schema("schema: empty input rejected",        False, input="")
    _schema("schema: input > 4096 rejected",       False, input="x" * 4097)
    _schema("schema: speed below 0.5 rejected",    False, input="Hi", speed=0.1)
    _schema("schema: speed above 2.0 rejected",    False, input="Hi", speed=3.0)
    _schema("schema: invalid format rejected",     False, input="Hi", response_format="ogg")
    _schema("schema: valid mp3 format",            True,  input="Hi", response_format="mp3")
    _schema("schema: valid speed boundary 0.5",    True,  input="Hi", speed=0.5)
    _schema("schema: valid speed boundary 2.0",    True,  input="Hi", speed=2.0)

    # ---- audio normalizer ----
    from app.services.audio import AudioNormalizer

    def _norm(label, fn):
        try:
            fn()
            suite.record(True, label)
        except Exception as e:
            suite.record(False, label, str(e))

    def _check_normalize_float32():
        n = AudioNormalizer()
        n.sample_rate = 44100
        arr = np.array([0.0, 0.5, -0.5, 1.0, -1.0], dtype=np.float32)
        out = n.normalize(arr)
        assert out.dtype == np.int16, f"expected int16 got {out.dtype}"
        assert out[2] < 0, "negative float should map to negative int16"

    def _check_normalize_int16_passthrough():
        n = AudioNormalizer()
        n.sample_rate = 44100
        arr = np.array([0, 100, -100], dtype=np.int16)
        out = n.normalize(arr)
        assert out.dtype == np.int16
        np.testing.assert_array_equal(out, arr)

    def _check_silence_detection():
        n = AudioNormalizer()
        n.sample_rate = 44100
        silent = np.zeros(1000, dtype=np.int16)
        start, end = n.find_first_last_non_silent(silent, "text", 1.0)
        assert start == 0 and end == len(silent), f"silent: {start},{end}"

    def _check_non_silent_bounds():
        n = AudioNormalizer()
        n.sample_rate = 44100
        arr = np.zeros(1000, dtype=np.int16)
        arr[300:700] = 10000  # non-silent region
        start, end = n.find_first_last_non_silent(arr, "text.", 1.0)
        assert start <= 300, f"start {start} > 300"
        assert end >= 700, f"end {end} < 700"

    _norm("audio: normalize float32 → int16",       _check_normalize_float32)
    _norm("audio: normalize int16 passthrough",     _check_normalize_int16_passthrough)
    _norm("audio: silence detection returns full",  _check_silence_detection)
    _norm("audio: non-silent bounds detected",      _check_non_silent_bounds)

    return suite


# ==========================================================================
# 2. INTEGRATION TESTS  (live server required)
# ==========================================================================

VOICES = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
FORMATS = ["mp3", "wav", "flac", "opus", "aac", "pcm"]

# Minimal magic-byte checks for audio validity
_MAGIC = {
    "wav":  lambda b: b[:4] == b"RIFF",
    "mp3":  lambda b: b[:3] == b"ID3" or (len(b) >= 2 and b[0] == 0xFF and (b[1] & 0xE0) == 0xE0),
    "flac": lambda b: b[:4] == b"fLaC",
    "opus": lambda b: b[:4] == b"OggS",
    "aac":  lambda b: len(b) > 10,
    "pcm":  lambda b: len(b) > 0,
}

async def run_integration_tests(base_url: str) -> Suite:
    _head("INTEGRATION TESTS")
    suite = Suite("integration")
    timeout = httpx.Timeout(120.0)
    limits = httpx.Limits(max_connections=20)

    async with httpx.AsyncClient(base_url=base_url, timeout=timeout, limits=limits) as c:

        # ---- health / meta ----
        async def get(path, label, expected_status=200, check=None):
            try:
                r = await c.get(path)
                ok = r.status_code == expected_status
                if ok and check:
                    ok, detail = check(r.json()), ""
                else:
                    detail = f"status={r.status_code}" if not ok else ""
                suite.record(ok, label, detail)
            except Exception as e:
                suite.record(False, label, str(e))

        await get("/health",     "GET /health → 200",         check=lambda j: j.get("status") == "healthy")
        await get("/v1/models",  "GET /v1/models → has data", check=lambda j: isinstance(j.get("data"), list))
        await get("/v1/voices",  "GET /v1/voices → has voices",check=lambda j: isinstance(j.get("voices"), list))
        await get("/voices",     "GET /voices legacy → 200",  check=lambda j: isinstance(j.get("voices"), list))

        # ---- speech synthesis ----
        async def speech(label, payload, expected_status=200, save_as=None):
            try:
                r = await c.post("/v1/audio/speech", json=payload)
                ok = r.status_code == expected_status
                detail = f"status={r.status_code}" if not ok else ""
                if ok and expected_status == 200:
                    fmt = payload.get("response_format", "mp3")
                    magic_ok = _MAGIC[fmt](r.content)
                    if not magic_ok:
                        ok, detail = False, f"invalid {fmt} header bytes"
                    elif save_as:
                        path = OUTPUT_DIR / save_as
                        path.write_bytes(r.content)
                suite.record(ok, label, detail)
                return r
            except Exception as e:
                suite.record(False, label, str(e))
                return None

        base = {"input": "Hello, this is a test.", "response_format": "mp3"}

        # All formats
        for fmt in FORMATS:
            await speech(
                f"POST /v1/audio/speech format={fmt}",
                {**base, "response_format": fmt},
                save_as=f"test_{fmt}.{fmt}",
            )

        # All preset voices
        for voice in VOICES:
            await speech(
                f"POST /v1/audio/speech voice={voice}",
                {**base, "voice": voice},
            )

        # Speed boundaries
        await speech("POST /v1/audio/speech speed=0.5", {**base, "speed": 0.5})
        await speech("POST /v1/audio/speech speed=2.0", {**base, "speed": 2.0})

        # normalize=false
        await speech("POST /v1/audio/speech normalize=false",
                     {**base, "input": "Hello world", "normalize": False})

        # pause tag
        await speech("POST /v1/audio/speech pause tag",
                     {**base, "input": "Hello. [pause:0.5] World."})

        # multi-sentence (auto chunking)
        long_input = (
            "The quick brown fox jumps over the lazy dog. "
            "Pack my box with five dozen liquor jugs. "
            "How vexingly quick daft zebras jump."
        )
        await speech("POST /v1/audio/speech multi-sentence", {**base, "input": long_input},
                     save_as="test_multisent.mp3")

        # ---- validation errors ----
        await speech("POST /v1/audio/speech empty input → 422",
                     {"input": "", "response_format": "mp3"}, expected_status=422)
        await speech("POST /v1/audio/speech input > 4096 → 422",
                     {"input": "x" * 4097, "response_format": "mp3"}, expected_status=422)
        await speech("POST /v1/audio/speech bad format → 422",
                     {"input": "Hi", "response_format": "ogg"}, expected_status=422)
        await speech("POST /v1/audio/speech speed too low → 422",
                     {"input": "Hi", "speed": 0.1}, expected_status=422)
        await speech("POST /v1/audio/speech speed too high → 422",
                     {"input": "Hi", "speed": 3.0}, expected_status=422)

        # ---- voice mix ----
        mix_payload = {"voice_a": "alloy", "voice_b": "echo", "weight": 0.5, "name": "test-mix"}
        try:
            r = await c.post("/v1/voices/mix", json=mix_payload)
            ok = r.status_code == 200
            detail = f"status={r.status_code}" if not ok else ""
            suite.record(ok, "POST /v1/voices/mix", detail)
            if ok:
                voice_id = r.json().get("id", "mix:test-mix")
                # use the mixed voice
                await speech("POST /v1/audio/speech with mixed voice",
                             {**base, "voice": voice_id})
                # delete it
                r2 = await c.delete(f"/v1/voices/{voice_id}")
                suite.record(r2.status_code == 200, "DELETE /v1/voices/{id}", f"status={r2.status_code}")
        except Exception as e:
            suite.record(False, "POST /v1/voices/mix", str(e))

        # delete non-existent voice → 404
        try:
            r = await c.delete("/v1/voices/definitely-does-not-exist")
            suite.record(r.status_code == 404, "DELETE /v1/voices/nonexistent → 404",
                         f"status={r.status_code}")
        except Exception as e:
            suite.record(False, "DELETE /v1/voices/nonexistent → 404", str(e))

    return suite


# ==========================================================================
# 3. EVAL TESTS  (latency + audio quality metrics)
# ==========================================================================

@dataclass
class LatencyResult:
    label: str
    ttfb_ms: float
    total_ms: float
    bytes_out: int
    success: bool
    error: str = ""


async def _measure(client: httpx.AsyncClient, label: str, payload: dict) -> LatencyResult:
    t0 = time.perf_counter()
    ttfb = None
    total_bytes = 0
    try:
        async with client.stream("POST", "/v1/audio/speech", json=payload) as r:
            if r.status_code != 200:
                return LatencyResult(label, 0, 0, 0, False, f"HTTP {r.status_code}")
            async for chunk in r.aiter_bytes():
                if ttfb is None:
                    ttfb = (time.perf_counter() - t0) * 1000
                total_bytes += len(chunk)
    except Exception as e:
        return LatencyResult(label, 0, 0, 0, False, str(e))

    total_ms = (time.perf_counter() - t0) * 1000
    return LatencyResult(label, ttfb or total_ms, total_ms, total_bytes, True)


async def run_eval_tests(base_url: str) -> Suite:
    _head("EVAL TESTS  (latency & quality)")
    suite = Suite("eval")

    eval_cases = [
        ("short (10 words)",   "Hello, this is a short synthesis test."),
        ("medium (30 words)",
         "The Supertonic TTS system converts text into natural-sounding speech "
         "using a deep neural network model trained on thousands of hours of audio data."),
        ("numbers & currency", "Please call 555-0199. The total is $19.99 plus €5.00 shipping."),
        ("punctuation heavy",  "Wait... really? Yes! It works; perfectly, every time."),
        ("long (multi-chunk)",
         "This is the first sentence of a longer passage. "
         "It continues with a second sentence here. "
         "And a third sentence follows for good measure. "
         "Finally, we conclude with this fourth sentence to ensure chunking is exercised."),
    ]

    timeout = httpx.Timeout(120.0)
    async with httpx.AsyncClient(base_url=base_url, timeout=timeout) as c:
        results: list[LatencyResult] = []

        for label, text in eval_cases:
            r = await _measure(c, label, {"input": text, "response_format": "mp3", "voice": "alloy"})
            results.append(r)
            if r.success:
                _info(f"{label:<25}  TTFB={r.ttfb_ms:6.0f}ms  total={r.total_ms:6.0f}ms  size={r.bytes_out/1024:5.1f}KB")
                suite.record(r.ttfb_ms < 30_000, f"eval latency: {label}", f"TTFB={r.ttfb_ms:.0f}ms")
                suite.record(r.bytes_out > 100,   f"eval audio bytes: {label}", f"{r.bytes_out}B")
            else:
                suite.record(False, f"eval: {label}", r.error)

        # ---- concurrent burst (5 parallel) ----
        _info("Running 5 concurrent requests…")
        tasks = [
            _measure(c, f"concurrent-{i}", {"input": "Concurrency test.", "response_format": "mp3"})
            for i in range(5)
        ]
        concurrent = await asyncio.gather(*tasks)
        c_ok = [r for r in concurrent if r.success]
        suite.record(len(c_ok) == 5, "eval: 5 concurrent requests all succeed",
                     f"{len(c_ok)}/5 succeeded")

        if c_ok:
            avg_ttfb = statistics.mean(r.ttfb_ms for r in c_ok)
            _info(f"Concurrent avg TTFB: {avg_ttfb:.0f}ms")

        # ---- summary stats ----
        ok_results = [r for r in results if r.success]
        if ok_results:
            ttfbs  = [r.ttfb_ms  for r in ok_results]
            totals = [r.total_ms for r in ok_results]
            print(f"\n  {'Metric':<25} {'p50':>8} {'p95':>8} {'avg':>8}")
            print(f"  {'-'*52}")

            def _pct(data, p):
                s = sorted(data)
                idx = max(0, int(len(s) * p / 100) - 1)
                return s[idx]

            print(f"  {'TTFB (ms)':<25} {_pct(ttfbs,50):>8.0f} {_pct(ttfbs,95):>8.0f} {statistics.mean(ttfbs):>8.0f}")
            print(f"  {'Total latency (ms)':<25} {_pct(totals,50):>8.0f} {_pct(totals,95):>8.0f} {statistics.mean(totals):>8.0f}")

    return suite


# ==========================================================================
# 4. STRESS TEST
# ==========================================================================

async def run_stress_test(base_url: str, concurrency: int, total: int) -> Suite:
    _head(f"STRESS TEST  ({total} requests, {concurrency} concurrent)")
    suite = Suite("stress")

    payload = {
        "input": "Stress test: the quick brown fox jumps over the lazy dog.",
        "response_format": "mp3",
        "voice": "alloy",
    }
    sem = asyncio.Semaphore(concurrency)
    results: list[LatencyResult] = []

    async def _req(i: int):
        async with sem:
            async with httpx.AsyncClient(base_url=base_url, timeout=httpx.Timeout(120.0)) as c:
                return await _measure(c, str(i), payload)

    t0 = time.perf_counter()
    all_results = await asyncio.gather(*[_req(i) for i in range(total)])
    duration = time.perf_counter() - t0

    successes = [r for r in all_results if r.success]
    failures  = [r for r in all_results if not r.success]
    success_rate = len(successes) / total * 100
    throughput   = len(successes) / duration

    suite.record(success_rate >= 95, f"stress: success rate ≥95%", f"{success_rate:.1f}%")

    if successes:
        ttfbs = [r.ttfb_ms for r in successes]
        p95   = sorted(ttfbs)[int(len(ttfbs) * 0.95)]
        suite.record(p95 < 60_000, f"stress: p95 TTFB < 60s", f"p95={p95:.0f}ms")

        _info(f"Duration:      {duration:.1f}s")
        _info(f"Throughput:    {throughput:.2f} req/s")
        _info(f"Success rate:  {success_rate:.1f}%  ({len(successes)}/{total})")
        _info(f"Avg TTFB:      {statistics.mean(ttfbs):.0f}ms")
        _info(f"P95 TTFB:      {p95:.0f}ms")
        if failures:
            _warn(f"Failures: {len(failures)} — sample error: {failures[0].error}")

    return suite


# ==========================================================================
# Main
# ==========================================================================

def _print_summary(suites: list[Suite]):
    _head("SUMMARY")
    total_pass = total_fail = 0
    for s in suites:
        colour = GREEN if s.failed == 0 else RED
        print(f"  {colour}{s.name:<15}{RESET}  {s.passed:>3} passed  {s.failed:>3} failed")
        total_pass += s.passed
        total_fail += s.failed
        for e in s.errors:
            print(f"             {RED}{e}{RESET}")
    print()
    if total_fail == 0:
        print(f"{BOLD}{GREEN}All {total_pass} tests passed.{RESET}")
    else:
        print(f"{BOLD}{RED}{total_fail} test(s) failed out of {total_pass + total_fail}.{RESET}")
    return total_fail == 0


async def _check_server(base_url: str) -> bool:
    try:
        async with httpx.AsyncClient(base_url=base_url, timeout=10.0) as c:
            r = await c.get("/health")
            return r.status_code == 200
    except Exception:
        return False


async def main():
    parser = argparse.ArgumentParser(description="Supertonic TTS unified test runner")
    parser.add_argument("--url",          default="http://localhost:8800", help="API base URL")
    parser.add_argument("--unit-only",    action="store_true", help="Unit tests only (no server)")
    parser.add_argument("--stress",       action="store_true", help="Include stress test suite")
    parser.add_argument("--concurrency",  type=int, default=10, help="Stress test concurrency")
    parser.add_argument("--requests",     type=int, default=50,  help="Stress test total requests")
    args = parser.parse_args()

    suites: list[Suite] = []

    # Unit tests — always run
    suites.append(run_unit_tests())

    if args.unit_only:
        _print_summary(suites)
        sys.exit(0 if suites[0].failed == 0 else 1)

    # Check server
    _info(f"Checking server at {args.url}…")
    if not await _check_server(args.url):
        print(f"\n{RED}Server not reachable at {args.url}. Skipping integration/eval/stress.{RESET}")
        print("Run with --unit-only to skip server tests.\n")
        _print_summary(suites)
        sys.exit(1)
    _info("Server is healthy.\n")

    suites.append(await run_integration_tests(args.url))
    suites.append(await run_eval_tests(args.url))

    if args.stress:
        suites.append(await run_stress_test(args.url, args.concurrency, args.requests))

    ok = _print_summary(suites)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    asyncio.run(main())
