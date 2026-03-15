import logging
from llama_cpp import Llama
from core.config import settings

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
                verbose=True 
            )
        except Exception as e:
            logger.error(f"ERROR: {e}")
            self.llm = None

    def generate_stream(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7):
        if self.llm is None:
            yield "Error: The model failed to load into memory."
            return

        try:
            streamer = self.llm(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True
            )
            for output in streamer:
                if "choices" in output and len(output["choices"]) > 0:
                    token = output["choices"][0].get("text", "")
                    if token:
                        yield token
                        
        except Exception as e:
            logger.error(f"Inference crash during streaming: {e}")
            yield f"\n[Engine Error: {str(e)}]"

model_engine = ModelManager()