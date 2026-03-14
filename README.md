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
| Inference Engine | `llama-cpp-python` — Python bindings for `llama.cpp` |
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
│   └── model_manager.py    # llama.cpp wrapper, singleton loader, token stream generator
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

```bash
pip install fastapi uvicorn llama-cpp-python pydantic pydantic-settings
```

### 1. Configure environment

Create a `.env` file in the project root:

```env
MODEL_PATH=./models/your-model.gguf
MODEL_NAME=your-model-name
N_GPU_LAYERS=-1
N_CTX=4096
```

`N_GPU_LAYERS=-1` offloads all layers to GPU. Set to `0` to run on CPU only.

### 2. Start the server

```bash
python main.py
```

The server starts at `http://127.0.0.1:8000`. Interactive API docs are available at `http://127.0.0.1:8000/docs`.

---

## Roadmap — v2.0 GPU Acceleration

The v1.0 baseline runs on CPU (`BLAS = 0`). The next phase refactors the engine for hardware acceleration and broader model support.

**CUDA Acceleration (RTX 3060):** Replace the CPU-bound `llama-cpp-python` wheel with the pre-compiled `cu121` build and offload all layer calculations to GPU VRAM via `n_gpu_layers=-1`.

**Next-Gen Model Support:** Upgrade to the `Ministral-3` architecture (`Ministral-3-8B-Instruct-2512-Q4_K_M.gguf`) and refactor the `model_manager.py` generator logic to align with breaking changes in the latest `llama.cpp` API.

**Hardware-Specific Optimizations:** Restrict computation to 6 physical cores (`N_THREADS=6`) to prevent Ryzen 5 cache bottlenecking, and increase prompt ingestion speed with `N_BATCH=512`.

**Flash Attention:** Enable `flash_attn=True` for memory-efficient context scaling at longer sequence lengths.
