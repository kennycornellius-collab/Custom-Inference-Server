from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from schemas.request import GenerateRequest
from core.model_manager import model_engine

router = APIRouter()

@router.get("/health")
async def health_check():
    return {
        "status": "online", 
        "model_loaded": model_engine.llm is not None
    }

@router.post("/generate")
async def generate_text(request: GenerateRequest):

    if model_engine.llm is None:
        raise HTTPException(status_code=500, detail="Model engine is offline.")

    generator = model_engine.generate_stream(
        prompt=request.prompt,
        max_tokens=request.max_tokens,
        temperature=request.temperature
    )
    return StreamingResponse(generator, media_type="text/event-stream")