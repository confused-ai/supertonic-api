# 🎙️ Supertonic TTS API

<div align="center">

[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker)](Dockerfile)

**OpenAI-compatible Text-to-Speech API**

[Features](#-features) • [Quick Start](#-quick-start) • [API Reference](#-api-reference) • [Deployment](#-deployment) • [Configuration](#-configuration)

</div>

---

## ✨ Features

- 🚀 **OpenAI-Compatible API** — Drop-in replacement for OpenAI's TTS API, no auth required
- ⚡ **High Performance** — Dedicated thread pool, async synthesis, semaphore-based concurrency control
- 🎵 **Multiple Formats** — MP3, WAV, FLAC, Opus, AAC, PCM via PyAV
- 🗣️ **Multiple Voices** — OpenAI voice names + native Supertonic styles + custom/mixed voice upload
- 🐳 **Docker Ready** — Production containerization with nginx load balancer and persistent model cache
- 📊 **GPU Acceleration** — CUDA, CoreML, and Metal backends via ONNX Runtime
- 🔊 **Smart Text Processing** — Unicode normalization, emoji removal, auto-chunking, pause tags
- 🌍 **31 Languages** — Full multilingual support via supertonic-3

## 📋 Requirements

- Python 3.10+
- ONNX Runtime (CPU/CUDA/CoreML)
- Supertonic TTS library

## 🚀 Quick Start

### Using Docker (Recommended)

```bash
git clone https://github.com/confused-ai/supertonic-api.git
cd supertonic-api

# Start API + nginx load balancer
docker compose up -d

# API available at http://localhost:8800
# Model downloads once and is cached in a Docker volume
```

### Manual Installation

```bash
git clone https://github.com/confused-ai/supertonic-api.git
cd supertonic-api

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8800
```

### Quick Test

```bash
curl -X POST "http://localhost:8800/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello from Supertonic!", "voice": "alloy", "response_format": "mp3"}' \
  --output speech.mp3
```

## 📖 API Reference

### Generate Speech

**POST** `/v1/audio/speech`

```bash
curl -X POST "http://localhost:8800/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{"model": "tts-1", "input": "Your text here...", "voice": "alloy", "response_format": "mp3", "speed": 1.0}' \
  --output output.mp3
```

#### Parameters

| Parameter         | Type    | Default | Description                                            |
| ----------------- | ------- | ------- | ------------------------------------------------------ |
| `model`           | string  | `tts-1` | Accepted for OpenAI compatibility; model is supertonic-3 |
| `input`           | string  | —       | Text to synthesize (1–4096 chars)                      |
| `voice`           | string  | `alloy` | Preset voice ID (see table below) or custom/mixed voice ID                 |
| `response_format` | string  | `mp3`   | mp3, opus, aac, flac, wav, pcm                         |
| `speed`           | float   | `1.0`   | Speed multiplier (0.5–2.0)                             |
| `normalize`       | boolean | `true`  | Unicode normalization, emoji removal, punctuation fix  |
| `lang`            | string  | `en`    | BCP-47 language code (31 languages supported + `na`)   |

### List Models

**GET** `/v1/models`

```bash
curl "http://localhost:8800/v1/models"
```

### Voices

10 native Supertonic styles exposed via 13 OpenAI-compatible voice IDs:

| Voice ID  | Style | Character                                          |
|-----------|-------|----------------------------------------------------|
| `alloy`   | F1    | Calm, clear female                                 |
| `nova`    | F2    | Bright, professional female                        |
| `shimmer` | F3    | Soft, expressive female                            |
| `ash`     | F4    | Energetic, versatile female                        |
| `ballad`  | F4    | Melodic, smooth female *(shares style with ash)*   |
| `coral`   | F5    | Airy, warm female                                  |
| `marin`   | F5    | Gentle, natural female *(shares style with coral)* |
| `echo`    | M1    | Lively, upbeat male                                |
| `fable`   | M2    | Warm, narrative male                               |
| `onyx`    | M3    | Deep, authoritative male                           |
| `cedar`   | M4    | Measured, resonant male                            |
| `sage`    | M4    | Calm, steady male *(shares style with cedar)*      |
| `verse`   | M5    | Dynamic, dramatic male                             |

**GET** `/v1/voices` — full voice list with types (preset / custom / mixed)  
**GET** `/voices` — legacy alias

```bash
curl "http://localhost:8800/v1/voices"
```

### Upload Custom Voice

**POST** `/v1/voices/upload`

```bash
curl -X POST "http://localhost:8800/v1/voices/upload" \
  -F "file=@my_voice.json" \
  -F "name=my-voice"
```

### Mix Two Voices

**POST** `/v1/voices/mix`

```bash
curl -X POST "http://localhost:8800/v1/voices/mix" \
  -H "Content-Type: application/json" \
  -d '{"voice_a": "alloy", "voice_b": "echo", "weight": 0.5, "name": "alloy-echo"}'
```

### Delete Custom Voice

**DELETE** `/v1/voices/{voice_id}`

```bash
curl -X DELETE "http://localhost:8800/v1/voices/mix:alloy-echo"
```

### Health Check

**GET** `/health`

```bash
curl "http://localhost:8800/health"
```

## 🎭 Available Voices

| Voice     | Style | Description                          |
| --------- | ----- | ------------------------------------ |
| `alloy`   | F1    | Sarah — calm female                  |
| `echo`    | M1    | Alex — lively upbeat male            |
| `fable`   | F2    | Lily — bright cheerful female        |
| `onyx`    | M2    | James — deep robust male             |
| `nova`    | F3    | Jessica — professional announcer     |
| `shimmer` | M3    | Robert — polished authoritative male |

You can also use any native supertonic style name directly (e.g. `F4`, `M5`) or a custom/mixed voice ID.

## ⚙️ Configuration

Environment variables can be set in `.env` file:

```env
# Server
HOST=0.0.0.0
PORT=8800
LOG_LEVEL=INFO

# Model Performance
MODEL_THREADS=12        # ONNX intra-op threads
MODEL_INTER_THREADS=4   # ONNX inter-op threads
MAX_WORKERS=8           # Concurrent synthesis workers + semaphore limit

# GPU Acceleration
FORCE_PROVIDERS=auto    # auto | cuda | coreml | metal | cpu

# Audio
SAMPLE_RATE=44100
MAX_CHUNK_LENGTH=300    # Max chars per synthesis chunk

# HuggingFace model cache (mounted as Docker volume)
HF_HOME=/root/.cache/huggingface
```

### GPU Acceleration

Set `FORCE_PROVIDERS` based on your hardware:

| Value    | Description                         |
| -------- | ----------------------------------- |
| `auto`   | Auto-detect best available provider |
| `cuda`   | NVIDIA GPU acceleration             |
| `coreml` | Apple CoreML (M-series chips)       |
| `metal`  | Apple Metal (maps to CoreML)        |
| `cpu`    | CPU only                            |

## 🐳 Deployment

### Docker Compose (Production)

```bash
docker compose up -d --build
```

Services:
- **api** — FastAPI + uvicorn on port 8801 (internal)
- **lb** — nginx reverse proxy on port **8800** (public)
- **hf_cache** — named Docker volume; model downloads once, reused on every restart

To scale API workers:
```bash
docker compose up -d --scale api=2
```

## 📊 Performance

- **Dedicated thread pool** — synthesis runs in isolated `ThreadPoolExecutor`, never blocks the I/O loop
- **Thread-safe model init** — double-checked locking; model loads once across all workers
- **Semaphore-bounded concurrency** — `MAX_WORKERS` cap prevents memory exhaustion under load
- **PyAV streaming encoder** — chunks encoded on-the-fly, no full audio buffering
- **Pre-compiled regex** — text normalization patterns compiled at startup
- **Smart chunking** — long text split at sentence/paragraph boundaries, preserves `[pause:N]` tags

## 🔧 Development

```bash
pip install -r requirements.txt

# Dev server with auto-reload
uvicorn app.main:app --reload --port 8800

# Run all tests (unit + integration + eval)
python tests/run_all.py

# Unit tests only (no server needed)
python tests/run_all.py --unit-only

# With stress test
python tests/run_all.py --stress --concurrency 20 --requests 200

# Custom server
python tests/run_all.py --url http://localhost:8801
```

## 📁 Project Structure

```
supertonic-api/
├── app/
│   ├── api/
│   │   ├── routes/            # Endpoint modules (speech, voices, models)
│   │   └── schemas.py         # Pydantic I/O models
│   ├── core/
│   │   ├── config.py          # pydantic-settings (.env)
│   │   ├── constants.py       # Model name
│   │   └── voices.py          # OpenAI → Supertonic voice map
│   ├── services/
│   │   ├── tts.py             # Singleton TTS service + async generation
│   │   ├── audio.py           # AudioNormalizer, AudioService
│   │   └── audio_encoder.py   # PyAV streaming encoder (mp3/wav/flac/opus/aac/pcm)
│   ├── utils/
│   │   └── text.py            # clean_text(), smart_split()
│   ├── inference/
│   │   └── base.py            # AudioChunk dataclass
│   └── main.py                # FastAPI app + lifespan
├── tests/
│   ├── run_all.py             # Unified test runner
│   └── output/                # Saved test audio files
├── Dockerfile
├── docker-compose.yml         # api + nginx lb + hf_cache volume
nginx.conf
└── requirements.txt
```

## 🤝 Contributing

Contributions welcome. See [CONTRIBUTING.md](CONTRIBUTING.md).

1. Fork → branch → commit → PR
2. Run `python tests/run_all.py --unit-only` before submitting

## 🙏 Acknowledgments

- [Supertonic](https://github.com/supertoneinc/supertonic) - TTS engine
- [FastAPI](https://fastapi.tiangolo.com/) - Web framework
- [PyAV](https://pyav.org/) - Audio encoding

---

<div align="center">

**[⬆ Back to Top](#️-supertonic-tts-api)**

Made with ❤️ by the community

</div>
