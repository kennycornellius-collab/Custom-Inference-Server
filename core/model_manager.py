import logging
import json
import asyncio
from llama_cpp import Llama
from core.config import settings
from schemas.request import ChatCompletionRequest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelManager:
    def __init__(self):
        logger.info(f"Loading model: {settings.model_name}")
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
            self.llm = None

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
        }

    def generate_stream(self, request_data: ChatCompletionRequest):
        
        if self.llm is None:
            yield f"data: {json.dumps({'error': 'Model not loaded'})}\n\n"
            yield "data: [DONE]\n\n"
            return

        try:
            params = self._get_params(request_data)
            
            streamer = self.llm.create_chat_completion(**params, stream=True)
            
            for chunk in streamer:
                
                yield f"data: {json.dumps(chunk)}\n\n"
                
            yield "data: [DONE]\n\n"
                
        except Exception as e:
            logger.error(f"Inference crash: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            yield "data: [DONE]\n\n"

    def _generate_sync(self, request_data: ChatCompletionRequest):
        
        if self.llm is None:
            raise RuntimeError("Model not loaded into memory.")
        params = self._get_params(request_data)
        return self.llm.create_chat_completion(**params, stream=False)

    async def generate_async(self, request_data: ChatCompletionRequest):
        
        loop = asyncio.get_event_loop()
        
        return await loop.run_in_executor(None, self._generate_sync, request_data)

model_engine = ModelManager()