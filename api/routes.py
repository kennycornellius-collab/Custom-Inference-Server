from fastapi import APIRouter, HTTPException, Security, Depends
from fastapi.security import APIKeyHeader
from fastapi.responses import StreamingResponse, JSONResponse
from schemas.request import ChatCompletionRequest
from core.model_manager import model_engine
from core.config import settings
import time

router = APIRouter()

api_key_header = APIKeyHeader(name="Authorization", auto_error=True)

def verify_api_key(api_key: str = Security(api_key_header)):
    """Middleware to verify the Bearer token."""
    if api_key != f"Bearer {settings.api_key}":
        raise HTTPException(status_code=401, detail="Unauthorized: Invalid API Key")
    return api_key

@router.get("/health")
async def health_check():
    if model_engine.llm is None:
        return JSONResponse(
            status_code=503, 
            content={"status": "offline", "error": "Model failed to load."}
        )
    return {"status": "online", "model": settings.model_name}

@router.get("/v1/models")
async def list_models(key: str = Depends(verify_api_key)):
    return {
        "object": "list",
        "data": [{"id": settings.model_name, "object": "model", "created": model_engine.created_at, "owned_by": "custom-inference-server"}]
    }

@router.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest, key: str = Depends(verify_api_key)):
    if model_engine.llm is None:
        raise HTTPException(status_code=503, detail="Model engine is offline.")

    if request.stream:
        return StreamingResponse(
            model_engine.generate_stream(request), 
            media_type="text/event-stream"
        )
    else:
        try:
            response_data = await model_engine.generate_async(request)
            return JSONResponse(content=response_data)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))