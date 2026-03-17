# Custom AI Inference Server

A lightweight, self-hosted REST API for serving quantized Large Language Models with an OpenAI-compatible interface. Built to bridge the gap between raw `.gguf` model files and client applications — without relying on third-party APIs.

---

## Overview

Loads quantized GGUF models into VRAM once at startup and exposes them over a clean HTTP API that mirrors the OpenAI chat completions spec. Responses are streamed token-by-token using Python generators and Server-Sent Events, so clients receive output in real time rather than waiting for full generation to complete.

Request validation is handled by Pydantic before anything reaches the inference engine — malformed payloads, out-of-range parameter values, and token requests that exceed the server's context window are rejected at the schema layer.

---

## Architecture

| Component | Technology |
|---|---|
| API Framework | `FastAPI` — async request handling with lifespan management |
| Web Server | `Uvicorn` — ASGI server |
| Inference Engine | `llama-cpp-python` (`cu122`) — CUDA-accelerated bindings for `llama.cpp` |
| Data Validation | `Pydantic` — strict JSON schema enforcement with cross-field validators |
| Configuration | `pydantic-settings` — environment variable management via `.env` |

---

## Project Structure

```
custom-inference-server/
├── api/
│   └── routes.py           # Endpoints: /health, /v1/models, /v1/chat/completions
├── core/
│   ├── config.py           # Environment configuration (host, port, model path, GPU layers)
│   └── model_manager.py    # llama.cpp wrapper, async startup, streaming generator
├── schemas/
│   └── request.py          # Chat completion request schema and parameter validation
├── models/                 # Directory for .gguf model weights (gitignored)
├── .env                    # Environment variables (not included)
├── main.py  
├── test.py                 # Testing script that automatically spin up a virtual client and fire requests at the endpoints
└── requirements.txt
```

---

## API Reference
> **Authentication:** All `/v1/` endpoints are secured via middleware. You must pass an API key in the header of your requests: `Authorization: Bearer <YOUR_API_KEY>`.

### `GET /health`

Returns server status and the loaded model name.

```json
{
  "status": "online",
  "model": "your-model-name"
}
```

Returns `503` if the model failed to load.

---

### `GET /v1/models`

Lists the currently loaded model in OpenAI-compatible format.

```json
{
  "object": "list",
  "data": [
    {
      "id": "your-model-name",
      "object": "model",
      "created": 1710000000,
      "owned_by": "custom-inference-server"
    }
  ]
}
```

---

### `POST /v1/chat/completions`

Generates a chat completion. Supports both streaming and non-streaming responses.

**Request body:**

| Field | Type | Default | Constraints | Description |
|---|---|---|---|---|
| `messages` | array | required | — | Conversation history as `[{"role": "...", "content": "..."}]`. Roles: `system`, `user`, `assistant` |
| `model` | string | `"local-model"` | — | Ignored by the engine; present for client compatibility |
| `max_tokens` | int | `null` | ≥ 1, ≤ `N_CTX` | Maximum tokens to generate. Validated against server context limit |
| `temperature` | float | `0.7` | 0.0–2.0 | Sampling temperature |
| `stream` | bool | `false` | — | Stream tokens via SSE if `true` |
| `top_p` | float | `1.0` | 0.0–1.0 | Nucleus sampling threshold |
| `top_k` | int | `40` | ≥ 1 | Top-K sampling |
| `repeat_penalty` | float | `1.1` | ≥ 0.0 | Penalises repeated tokens |
| `frequency_penalty` | float | `0.0` | -2.0–2.0 | Penalises token frequency |
| `presence_penalty` | float | `0.0` | -2.0–2.0 | Penalises token presence |
| `stop` | string[] | `null` | — | Stop sequences |
| `seed` | int | `null` | — | Seed for reproducible outputs |
| `logprobs` | bool | `false` | — | Return log probabilities |

**Streaming response** (`stream: true`): `text/event-stream` — token chunks via SSE, terminated with `data: [DONE]`.

**Non-streaming response** (`stream: false`): standard JSON chat completion object.

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
N_BATCH=512
USE_FLASH_ATTENTION=true
API_KEY=your-secure-api-key-here
```

`N_GPU_LAYERS=-1` offloads all layers to GPU VRAM. Set to `0` to run on CPU only.

### 2. Start the server

```bash
python main.py
```

The server starts at `http://127.0.0.1:8000` by default (configurable via `HOST` and `PORT` in `.env`). Interactive API docs are available at `http://127.0.0.1:8000/docs`.

---

## Changelog

### v3.1 — Security & Integration Testing

**Authentication Middleware:** Implemented dependency-injected API key validation using FastAPI's `Security` and `APIKeyHeader`, protecting all inference endpoints from unauthorized execution.

**Integration Test Suite:** Built a comprehensive testing pipeline using `pytest` and `httpx`. The suite utilizes a shared ASGI lifespan fixture to prevent GPU memory limits during testing, and explicitly asserts 401 Unauthorized bounces, Pydantic 422 context-window rejections, and the precise formatting of `data: [DONE]` signals within Server-Sent Event (SSE) streams.

---

### v3.0 — Async Hardening & OpenAI-Compatible Interface

**Non-blocking startup:** Model loading now runs inside a FastAPI lifespan event via `run_in_executor`, so the server becomes responsive immediately and loads the model in the background rather than blocking the process on import.

**Async lock safety:** `asyncio.Lock()` is now initialised during the async lifespan setup rather than lazily on first request, eliminating a race condition where concurrent cold requests could create independent lock objects.

**Initialisation guard:** Both `generate_stream` and `generate_async` now check for a `None` lock and return a clean error rather than crashing with a `TypeError` if a request arrives before startup completes.

**Inference timeout:** Streaming generation now enforces a 120-second per-chunk timeout via `asyncio.wait_for`, preventing runaway inference from blocking the lock indefinitely.

**OpenAI-compatible chat completions:** Migrated from a simple `/generate` prompt endpoint to a full `/v1/chat/completions` interface with a structured messages array, matching the OpenAI API spec for drop-in client compatibility.

**Extended parameter support:** Added `logprobs`, `frequency_penalty`, `presence_penalty`, `seed`, and `stop` to the request schema — all wired through to the inference engine.

**Context window validation:** `max_tokens` is now validated against the server's configured `N_CTX` at the schema layer via a Pydantic `model_validator`, rejecting oversized requests before they reach the engine.

**Model listing endpoint:** Added `GET /v1/models` returning the loaded model in OpenAI list format, with a stable `created` timestamp set at load time.

---

### v2.0 — CUDA Acceleration & Engine Refactor

**GPU offloading:** Upgraded `llama-cpp-python` to the `cu122` pre-compiled wheel, offloading 100% of model layers to RTX 3060 VRAM.

**Performance:** Achieved a ~700% speed increase over the CPU baseline — scaling from ~9 t/s to ~66 t/s on `Mistral-7B-Instruct-v0.2`.

**Streaming logic refactor:** Rewrote the `model_manager.py` generator to support modern asynchronous token streaming, offloading synchronous llama.cpp inference to a thread pool executor and bridging chunks back to the async context via a thread-safe queue.

**Environment architecture:** Hardened the Pydantic settings configuration to map `.env` variables directly into the C++ engine boot sequence.

---

### v1.0 — CPU Baseline

Initial release. Established the foundational asynchronous pipeline from the llama.cpp engine to the FastAPI layer with token streaming, request validation, and singleton model loading.