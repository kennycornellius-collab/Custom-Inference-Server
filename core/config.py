from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):

    host: str = "127.0.0.1"
    port: int = 8000

    model_path: str
    model_name: str = "default-model"
 
    n_gpu_layers: int = -1
    n_ctx: int = 4096

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

settings = Settings()