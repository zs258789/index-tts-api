from pydantic import BaseModel, Field
from typing import Optional, Literal

class SpeechRequest(BaseModel):
    """
    Request model for speech synthesis
    """
    model: str = "IndexTTS"
    input: str
    voice: str
    response_format: Literal["mp3", "wav", "ogg"] = "mp3"
    sample_rate: Optional[int] = 24000
    stream: Optional[bool] = False
    speed: Optional[float] = 1.0
    gain: Optional[float] = 0.0