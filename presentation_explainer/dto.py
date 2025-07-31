from pydantic import BaseModel
from typing import Optional, List

class TTSRequest(BaseModel):
    text: str
    language: str = "en"
    voice_cleanup: bool = False
    temperature: float = 0.75
    repetition_penalty: float = 5.0

class TTSResponse(BaseModel):
    success: bool
    message: str
    audio_url: Optional[str] = None
    metrics: Optional[dict] = None
    task_id: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    xtts_model_loaded: bool
    supported_languages: List[str]
    gpu_available: bool