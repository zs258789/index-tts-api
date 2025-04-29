import subprocess
import os
import tempfile
import logging

logger = logging.getLogger(__name__)

def convert_audio(input_path, output_format, sample_rate=24000):
    """
    Convert audio to the specified format and sample rate.
    
    Args:
        input_path (str): Path to the input audio file
        output_format (str): Target format (mp3, wav, ogg)
        sample_rate (int): Target sample rate
        
    Returns:
        str: Path to the converted audio file
    """
    output_path = os.path.splitext(input_path)[0] + f".{output_format}"
    
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-ar", str(sample_rate)
    ]
    
    # Add format-specific options
    if output_format == "mp3":
        cmd.extend(["-c:a", "libmp3lame", "-q:a", "2"])
    elif output_format == "ogg":
        cmd.extend(["-c:a", "libvorbis", "-q:a", "4"])
    
    cmd.append(output_path)
    
    logger.info(f"Converting audio to {output_format} with sample rate {sample_rate}")
    subprocess.run(cmd, check=True, stderr=subprocess.PIPE)
    return output_path

def apply_audio_effects(input_path, speed=1.0, gain=0.0):
    """
    Apply speed and gain adjustments to audio.
    
    Args:
        input_path (str): Path to the input audio file
        speed (float): Speed factor (1.0 = normal speed)
        gain (float): Gain in dB (0.0 = no change)
        
    Returns:
        str: Path to the processed audio file
    """
    if speed == 1.0 and gain == 0.0:
        return input_path
    
    output_path = tempfile.mktemp(suffix=os.path.splitext(input_path)[1])
    
    cmd = ["ffmpeg", "-y", "-i", input_path]
    
    filter_complex = []
    
    # Apply speed adjustment (using atempo)
    if speed != 1.0:
        # atempo has a range of 0.5 to 2.0
        # For values outside this range, chain multiple atempo filters
        speed_str = ""
        remaining_speed = speed
        
        while remaining_speed > 2.0:
            speed_str += "atempo=2.0,"
            remaining_speed /= 2.0
        
        while remaining_speed < 0.5:
            speed_str += "atempo=0.5,"
            remaining_speed /= 0.5
        
        speed_str += f"atempo={remaining_speed}"
        filter_complex.append(speed_str)
    
    # Apply gain adjustment
    if gain != 0.0:
        filter_complex.append(f"volume={gain}dB")
    
    if filter_complex:
        cmd.extend(["-filter:a", ",".join(filter_complex)])
    
    cmd.append(output_path)
    
    logger.info(f"Applying audio effects: speed={speed}, gain={gain}")
    subprocess.run(cmd, check=True, stderr=subprocess.PIPE)
    return output_path