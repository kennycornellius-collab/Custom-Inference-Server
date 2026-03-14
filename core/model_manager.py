import logging
from llama_cpp import Llama
from core.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelManager:
    def __init__(self):
        
        logger.info(f"Loading model from: {settings.model_path}")
        try:
            self.llm = Llama(
                model_path=settings.model_path,
                n_gpu_layers=settings.n_gpu_layers,
                n_ctx=settings.n_ctx,
                verbose=False  
            )
            logger.info(f"Successfully loaded {settings.model_name} into memory!")
        except Exception as e:
            logger.error(f"ERROR: Failed to load model. Details: {e}")
            self.llm = None

    def generate_stream(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7):
        
        if self.llm is None:
            yield "Error: The model failed to load into memory."
            return

        streamer = self.llm(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True 
        )

        for output in streamer:
            token = output["choices"][0]["text"]
            yield token

model_engine = ModelManager()