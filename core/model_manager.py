import logging
import json
import time
import asyncio
from llama_cpp import Llama
from core.config import settings
from schemas.request import ChatCompletionRequest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelManager:
    def __init__(self):

        self.llm = None
        self.lock = None
        self.created_at = int(time.time()) 

    def _load_cpp_engine(self):
        logger.info(f"Booting up CUDA Engine. Loading model: {settings.model_name}")
        try:
            self.llm = Llama(
                model_path=settings.model_path,
                n_gpu_layers=settings.n_gpu_layers,
                n_ctx=settings.n_ctx,
                n_threads=settings.n_threads,
                n_batch=settings.n_batch,
                flash_attn=settings.use_flash_attention,
                verbose=False 
            )
            logger.info(f"SUCCESS: {settings.model_name} loaded into VRAM!")
        except Exception as e:
            logger.error(f"CRITICAL ERROR: Failed to load model. Details: {e}")

    async def async_setup(self):
        self.lock = asyncio.Lock()
        loop = asyncio.get_running_loop()

        await loop.run_in_executor(None, self._load_cpp_engine)

    def _get_params(self, request_data: ChatCompletionRequest):
        messages = [{"role": msg.role, "content": msg.content} for msg in request_data.messages]
        return {
            "messages": messages,
            "max_tokens": request_data.max_tokens,
            "temperature": request_data.temperature,
            "top_p": request_data.top_p,
            "top_k": request_data.top_k,
            "repeat_penalty": request_data.repeat_penalty,
            "frequency_penalty": request_data.frequency_penalty,
            "presence_penalty": request_data.presence_penalty,
            "stop": request_data.stop,
            "seed": request_data.seed,
            "logprobs": request_data.logprobs,
        }

    async def generate_stream(self, request_data: ChatCompletionRequest):
        if self.lock is None:
            yield f"data: {json.dumps({'error': 'Server is still initializing.'})}\n\n"
            yield "data: [DONE]\n\n"
            return
        
        if self.llm is None:
            yield f"data: {json.dumps({'error': 'Model not loaded'})}\n\n"
            yield "data: [DONE]\n\n"
            return

        async with self.lock:
            params = self._get_params(request_data)
            loop = asyncio.get_running_loop()
            q = asyncio.Queue()

            def _sync_inference():
                try:
                    streamer = self.llm.create_chat_completion(**params, stream=True)
                    for chunk in streamer:
                        asyncio.run_coroutine_threadsafe(q.put(chunk), loop)
                    asyncio.run_coroutine_threadsafe(q.put(None), loop) 
                except Exception as e:
                    asyncio.run_coroutine_threadsafe(q.put(e), loop)

            inference_future = loop.run_in_executor(None, _sync_inference)

            while True:
                try:
                    chunk = await asyncio.wait_for(q.get(), timeout=120.0)
                except asyncio.TimeoutError:
                    logger.error("Inference timed out.")
                    yield f"data: {json.dumps({'error': 'Inference timed out'})}\n\n"
                    yield "data: [DONE]\n\n"
                    break
                
                if chunk is None:
                    yield "data: [DONE]\n\n"
                    break
                    
                if isinstance(chunk, Exception):
                    logger.error(f"Inference crash: {chunk}")
                    yield f"data: {json.dumps({'error': str(chunk)})}\n\n"
                    yield "data: [DONE]\n\n"
                    break
                
                yield f"data: {json.dumps(chunk)}\n\n"

    def _generate_sync(self, request_data: ChatCompletionRequest):
        if self.llm is None:
            raise RuntimeError("Model not loaded into memory.")
        params = self._get_params(request_data)
        return self.llm.create_chat_completion(**params, stream=False)

    async def generate_async(self, request_data: ChatCompletionRequest):
        if self.lock is None:
            raise RuntimeError("Server is still initializing.")
        async with self.lock:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, self._generate_sync, request_data)
model_engine = ModelManager()