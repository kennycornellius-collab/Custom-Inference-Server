import logging
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
            logger.info(f"{settings.model_name} loaded into VRAM!")
        except Exception as e:
            logger.error(f"ERROR: Failed to load model. Details: {e}")
            self.llm = None

    def generate_stream(self, request_data: ChatCompletionRequest):
        if self.llm is None:
            yield "Error: The model failed to load into memory."
            return

        try:
            messages = [{"role": msg.role, "content": msg.content} for msg in request_data.messages]

            streamer = self.llm.create_chat_completion(
                messages=messages,
                max_tokens=request_data.max_tokens,
                temperature=request_data.temperature,
                top_p=request_data.top_p,
                frequency_penalty=request_data.frequency_penalty,
                presence_penalty=request_data.presence_penalty,
                stop=request_data.stop,
                stream=True
            )
            
            for chunk in streamer:
                if "choices" in chunk and len(chunk["choices"]) > 0:
                    delta = chunk["choices"][0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        yield content
                        
        except Exception as e:
            logger.error(f"Inference crash during streaming: {e}")
            yield f"\n[Engine Error: {str(e)}]"

model_engine = ModelManager()