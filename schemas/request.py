from pydantic import BaseModel, Field

class GenerateRequest(BaseModel):
    prompt: str = Field(
        ..., 
        description="The main text prompt you want to send to the AI."
    )
    
    max_tokens: int = Field(
        default=512, 
        ge=1, 
        le=8192, 
        description="The maximum number of words to generate."
    )
    
    temperature: float = Field(
        default=0.7, 
        ge=0.0, 
        le=2.0, 
        description="Controls the creativity. 0.0 is robotic, 1.0 is creative."
    )