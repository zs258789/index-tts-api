import os
import tempfile
import logging
from indextts.infer import IndexTTS
from ..utils.audio_utils import convert_audio, apply_audio_effects

logger = logging.getLogger(__name__)

class TTSService:
    """
    Service for text-to-speech generation using IndexTTS
    """
    
    def __init__(self, model_dir="checkpoints", cfg_path="checkpoints/config.yaml", is_fp16=True, device=None):
        """
        Initialize the TTS service
        
        Args:
            model_dir (str): Directory containing model files
            cfg_path (str): Path to config file
            is_fp16 (bool): Whether to use FP16 precision
            device (str): Device to use (cpu, cuda, mps)
        """
        logger.info(f"Initializing TTS service with model_dir={model_dir}, cfg_path={cfg_path}")
        self.tts = IndexTTS(
            model_dir=model_dir,
            cfg_path=cfg_path,
            is_fp16=is_fp16,
            device=device
        )
        self.voices_dir = "characters"
        os.makedirs(self.voices_dir, exist_ok=True)
        logger.info(f"Voice directory: {self.voices_dir}")
        
    def generate_speech(self, text, voice, response_format="mp3", sample_rate=24000, speed=1.0, gain=0.0):
        """
        Generate speech from text
        
        Args:
            text (str): Text to synthesize
            voice (str): Voice identifier (file name without extension)
            response_format (str): Output format (mp3, wav, ogg)
            sample_rate (int): Output sample rate
            speed (float): Speed factor
            gain (float): Gain adjustment in dB
            
        Returns:
            str: Path to the generated audio file
        """
        # Get voice file path from voice identifier
        voice_file = os.path.join(self.voices_dir, f"{voice}.wav")
        
        if not os.path.exists(voice_file):
            logger.error(f"Voice file not found: {voice_file}")
            raise ValueError(f"Voice file {voice_file} not found")
        
        # Create a temporary output file for the wav
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wav_output = tmp.name
        
        logger.info(f"Generating speech with voice={voice}, text='{text[:50]}...'")
        # Generate speech
        self.tts.infer(voice_file, text, wav_output)
        
        # Apply audio effects if needed
        if speed != 1.0 or gain != 0.0:
            logger.info(f"Applying audio effects: speed={speed}, gain={gain}")
            effect_output = apply_audio_effects(wav_output, speed, gain)
            if effect_output != wav_output:
                os.remove(wav_output)  # Clean up original if a new file was created
            wav_output = effect_output
        
        # Convert to the desired format if not wav
        if response_format != "wav":
            logger.info(f"Converting to {response_format} with sample rate {sample_rate}")
            output_path = convert_audio(wav_output, response_format, sample_rate)
            os.remove(wav_output)  # Clean up
        else:
            output_path = wav_output
        
        logger.info(f"Speech generation completed: {output_path}")
        return output_path