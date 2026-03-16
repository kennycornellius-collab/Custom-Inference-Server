import time
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from schemas.request import ChatCompletionRequest
from core.model_manager import model_engine
from core.config import settings

router = APIRouter()

@router.get("/health")
async def health_check():
    
    if model_engine.llm is None:
        return JSONResponse(
            status_code=503, 
            content={"status": "offline", "error": "Model failed to load into memory."}
        )
    return {"status": "online", "model": settings.model_name}

@router.get("/v1/models")
async def list_models():
    
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
    
    if model_engine.llm is None:
        raise HTTPException(status_code=503, detail="Model engine is offline or crashed.")

    
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