"""
Audio processing utilities using FFmpeg.
Downloads and converts audio files for STT processing.
"""
import requests
import subprocess
import tempfile
import os
import logging
from pathlib import Path
from twilio.rest import Client
# Assuming these are available from your main app's config
from aiman.config import TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Twilio client to handle authentication
twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

def download_and_convert_audio(media_url: str) -> str:
    """
    Download audio from an authenticated Twilio URL and convert to WAV.
    
    Args:
        media_url: The media URL provided by Twilio's webhook.
    
    Returns:
        Path to the converted WAV file, or None on failure.
    """
    try:
        # Step 1: Download the audio file with authentication
        logger.info(f"Downloading audio from: {media_url}")
        
        # Twilio media URLs require authentication. We can use the configured
        # credentials (Account SID and Auth Token) for the request.
        response = requests.get(
            media_url,
            auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN),
            timeout=30
        )
        
        # Raise an exception for HTTP errors (e.g., 401 Unauthorized)
        response.raise_for_status()

        # Step 2: Create temporary files
        temp_dir = tempfile.mkdtemp()
        input_file = os.path.join(temp_dir, "input.ogg")
        output_file = os.path.join(temp_dir, "output.wav")
        
        # Step 3: Save downloaded content
        with open(input_file, "wb") as f:
            f.write(response.content)
        
        # Step 4: Convert using FFmpeg
        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-i", input_file,
            "-ar", "16000",
            "-ac", "1",
            "-f", "wav",
            output_file
        ]
        
        logger.info("Converting audio with FFmpeg...")
        result = subprocess.run(
            ffmpeg_cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True
        )
        
        if result.returncode != 0:
            logger.error(f"FFmpeg error: {result.stderr}")
            # Ensure temp files are cleaned up before raising
            cleanup_temp_file(input_file)
            cleanup_temp_file(output_file)
            raise RuntimeError(f"Audio conversion failed: {result.stderr}")
        
        # Step 5: Clean up input file
        os.remove(input_file)
        
        logger.info(f"Audio converted successfully: {output_file}")
        return output_file
    
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to download audio from Twilio: {str(e)}")
        # Raise the error so it can be handled by the main webhook function
        raise
    except Exception as e:
        logger.error(f"Audio processing failed: {str(e)}")
        # Raise the error so it can be handled by the main webhook function
        raise

def cleanup_temp_file(file_path: str):
    """Clean up temporary audio files."""
    try:
        # The tempfile.mkdtemp() creates a directory, not just a file.
        # This function should be modified to handle the directory cleanup.
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Cleaned up temp file: {file_path}")
    except Exception as e:
        logger.warning(f"Failed to cleanup {file_path}: {str(e)}")