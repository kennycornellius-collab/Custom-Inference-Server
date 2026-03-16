from pydantic import BaseModel, Field, model_validator
from typing import List, Optional, Literal
from core.config import settings

class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "local-model"
    messages: List[Message]
    
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=None, ge=1) 
    stream: Optional[bool] = False
    
    top_p: Optional[float] = Field(default=1.0, ge=0.0, le=1.0)
    top_k: Optional[int] = Field(default=40, ge=1)
    repeat_penalty: Optional[float] = Field(default=1.1, ge=0.0)
    frequency_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)
    presence_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)
    stop: Optional[List[str]] = None
    seed: Optional[int] = None 
    logprobs: Optional[bool] = False 

    @model_validator(mode='after')
    def check_max_tokens(self) -> 'ChatCompletionRequest':
        limit = settings.n_ctx
        if self.max_tokens is not None and self.max_tokens > limit:
            raise ValueError(f"max_tokens ({self.max_tokens}) exceeds server context limit ({limit})")
        return self