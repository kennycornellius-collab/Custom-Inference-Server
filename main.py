import uvicorn
from fastapi import FastAPI
from api.routes import router

app = FastAPI(
    title="Custom AI Inference Server",
    description="A local backend for serving quantized LLMs via llama.cpp.",
    version="1.0.0"
)

app.include_router(router)

if __name__ == "__main__":
    print("Starting Inference Server")
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)