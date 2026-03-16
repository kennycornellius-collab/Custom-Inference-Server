import time
from fastapi import APIRouter
from fastapi.responses import StreamingResponse, JSONResponse
from schemas.request import ChatCompletionRequest
from core.model_manager import model_engine
from core.config import settings

router = APIRouter()

@router.get("/health")
async def health_check():
    """Honest health check reflecting actual engine status."""
    if model_engine.llm is None:
        return JSONResponse(
            status_code=503, 
            content={"status": "offline", "error": "Model failed to load into memory."}
        )
    return {"status": "online", "model": settings.model_name}

@router.get("/v1/models")
async def list_models():
    """OpenAI compatible stub so UI clients can detect the model."""
    return {
        "object": "list",
        "data": [
            {
                "id": settings.model_name,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "custom-inference-server"
            }
        ]
    }

@router.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """Placeholder for Strike 2: Streaming vs Non-Streaming logic."""
    return StreamingResponse(
        model_engine.generate_stream(request), 
        media_type="text/event-stream"
    )