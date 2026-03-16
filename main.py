import uvicorn
import asyncio
from fastapi import FastAPI
from contextlib import asynccontextmanager
from api.routes import router
from core.model_manager import model_engine
from core.config import settings

@asynccontextmanager
async def lifespan(app: FastAPI):
    await model_engine.async_setup()
    yield


app = FastAPI(
    title="Custom Local Inference Server",
    description="An OpenAI-compatible API powered by llama-cpp-python.",
    lifespan=lifespan
)


app.include_router(router)

if __name__ == "__main__":
    
    uvicorn.run(
        "main:app", 
        host=settings.host, 
        port=settings.port, 
        reload=False 
    )