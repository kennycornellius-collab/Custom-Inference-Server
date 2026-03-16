import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    # Server config
    host: str = "127.0.0.1"
    port: int = 8000
    
    # Model config
    model_path: str
    model_name: str = "default-model"
    
    # Engine config
    n_gpu_layers: int = -1
    n_ctx: int = Field(default=4096, ge=512) # Forces at least 512 context
    # Dynamically grab CPU threads, fallback to 4 if it fails
    n_threads: int = Field(default_factory=lambda: os.cpu_count() or 4) 
    n_batch: int = Field(default=512, ge=1)
    use_flash_attention: bool = True

    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8",
        protected_namespaces=() 
    )

settings = Settings()