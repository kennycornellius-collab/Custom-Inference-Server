# Custom AI Inference Server

A lightweight local REST API for serving quantized Large Language Models. Built to bridge the gap between raw `.gguf` model files and client applications, mimicking enterprise-grade inference infrastructure without relying on third-party APIs.

---

## Overview

A self-hosted inference backend that loads quantized GGUF models into memory once at startup and exposes them over a clean HTTP API. Responses are streamed token-by-token using Python generators and Server-Sent Events, so clients receive output in real time rather than waiting for the full generation to complete.

Request validation is handled by Pydantic before anything reaches the inference engine — malformed payloads, out-of-range temperature values, and oversized token requests are rejected at the schema layer, preventing the engine from receiving invalid inputs.

---

## Architecture

| Component | Technology |
|---|---|
| API Framework | `FastAPI` — async request handling |
| Web Server | `Uvicorn` — ASGI server |
| Inference Engine | `llama-cpp-python` (`cu122`) — CUDA-accelerated bindings for `llama.cpp` |
| Data Validation | `Pydantic` — strict JSON schema enforcement |
| Configuration | `pydantic-settings` — environment variable management via `.env` |

---

## Project Structure

```
custom-inference-server/
├── api/
│   └── routes.py           # Endpoints: GET /health, POST /generate
├── core/
│   ├── config.py           # Environment configuration (host, port, model path, GPU layers)
│   └── model_manager.py    # llama.cpp wrapper, singleton loader, async token stream generator
├── schemas/
│   └── request.py          # Request schema and parameter validation
├── models/                 # Directory for .gguf model weights (gitignored)
├── .env                    # Environment variables (not included)
├── main.py                 # FastAPI application entry point
└── requirements.txt
```

---

## API Reference

### `GET /health`

Returns server status and whether the model loaded successfully.

```json
{
  "status": "online",
  "model_loaded": true
}
```

### `POST /generate`

Streams a text generation response token-by-token.

**Request body:**

| Field | Type | Default | Constraints | Description |
|---|---|---|---|---|
| `prompt` | string | required | — | Input prompt |
| `max_tokens` | int | 512 | 1–8192 | Maximum tokens to generate |
| `temperature` | float | 0.7 | 0.0–2.0 | Sampling temperature |

**Response:** `text/event-stream` — token stream via SSE.

---

## Setup & Usage

### Requirements

Install the CUDA-accelerated wheel directly — do not install the standard `llama-cpp-python` from PyPI as it ships without GPU support:

```bash
pip install fastapi uvicorn pydantic pydantic-settings
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu122
```

### 1. Configure environment

Create a `.env` file in the project root:

```env
MODEL_PATH=./models/your-model.gguf
MODEL_NAME=your-model-name
N_GPU_LAYERS=-1
N_CTX=4096
```

`N_GPU_LAYERS=-1` offloads all layers to GPU VRAM. Set to `0` to run on CPU only.

### 2. Start the server

```bash
python main.py
```

The server starts at `http://127.0.0.1:8000`. Interactive API docs are available at `http://127.0.0.1:8000/docs`.

---

## Changelog

### v2.0 — CUDA Acceleration & Engine Refactor

**GPU Offloading:** Upgraded `llama-cpp-python` to the `cu122` pre-compiled wheel, offloading 100% of model layers to RTX 3060 VRAM.

**Performance:** Achieved a ~700% speed increase over the CPU baseline — scaling from ~9 t/s to ~66 t/s on `Mistral-7B-Instruct-v0.2`.

**Streaming Logic Refactor:** Rewrote the `model_manager.py` generator to support modern asynchronous token streaming and handle empty-packet errors during real-time generation.

**Environment Architecture:** Hardened the Pydantic settings configuration to map `.env` variables (thread counts, batch sizes) directly into the C++ engine boot sequence.

### v1.0 — CPU Baseline

Initial release. Established the foundational asynchronous pipeline from the llama.cpp engine to the FastAPI layer with token streaming, request validation, and singleton model loading.

---

## Roadmap — v3.0

With the core backend stabilized and GPU-accelerated, development will pivot to increasing the features and performance of the inference server