from fastapi import APIRouter, HTTPException, BackgroundTasks, Request
from fastapi.responses import StreamingResponse, FileResponse
import os
import logging
from ..models import SpeechRequest
from ..services.tts_service import TTSService

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize the TTS service
# Note: We could move this to a dependency to allow configuration
tts_service = TTSService()

@router.post("/speech")
async def generate_speech(request: SpeechRequest, background_tasks: BackgroundTasks, req: Request):
    """
    Generate speech from text
    """
    try:
        # Log request details
        logger.info(f"Speech request: model={request.model}, voice={request.voice}, format={request.response_format}")
        
        # Parse the voice from the request
        # Format is expected to be "IndexTTS:voice_name" or just "voice_name"
        model, voice = request.voice.split(":", 1) if ":" in request.voice else ("IndexTTS", request.voice)
        
        # Generate speech
        output_path = tts_service.generate_speech(
            text=request.input,
            voice=voice,
            response_format=request.response_format,
            sample_rate=request.sample_rate,
            speed=request.speed,
            gain=request.gain
        )
        
        # Handle streaming vs non-streaming
        if request.stream:
            logger.info(f"Streaming response: {output_path}")
            
            def iterfile():
                with open(output_path, mode="rb") as file_like:
                    yield from file_like
                # Clean up the file after streaming
                os.remove(output_path)
                
            return StreamingResponse(
                iterfile(),
                media_type=f"audio/{request.response_format}",
                headers={"Content-Disposition": f"attachment; filename=speech.{request.response_format}"}
            )
        else:
            logger.info(f"File response: {output_path}")
            
            # Add cleanup task for non-streaming response
            background_tasks.add_task(os.remove, output_path)
            
            return FileResponse(
                output_path,
                media_type=f"audio/{request.response_format}",
                filename=f"speech.{request.response_format}"
            )
    except Exception as e:
        logger.error(f"Error generating speech: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))