from pydantic import BaseModel, Field
from typing import List, Optional, Literal

class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "local-model"
    messages: List[Message]
    
    
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=512, ge=1) 
    stream: Optional[bool] = False
    
   
    top_p: Optional[float] = Field(default=1.0, ge=0.0, le=1.0)
    top_k: Optional[int] = Field(default=40, ge=1)
    repeat_penalty: Optional[float] = Field(default=1.1, ge=0.0)
    frequency_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)
    presence_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)
    stop: Optional[List[str]] = None
    seed: Optional[int] = None 