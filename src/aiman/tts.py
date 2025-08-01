"""
Text-to-Speech using Sarvam AI Bulbul v2.
Converts text to natural-sounding Indian language speech.
"""

import requests
import base64
import tempfile
import os
import logging
from aiman.config import SARVAM_API_KEY

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Sarvam AI TTS Configuration  
TTS_API_URL = "https://api.sarvam.ai/text-to-speech"
TTS_HEADERS = {
    "api-subscription-key": SARVAM_API_KEY,
    "Content-Type": "application/json"
}

# Supported language codes by Sarvam TTS
SUPPORTED_LANGUAGES = {
    "bn-IN", "en-IN", "gu-IN", "hi-IN", "kn-IN", 
    "ml-IN", "mr-IN", "od-IN", "pa-IN", "ta-IN", "te-IN"
}

# Language and speaker mapping (expanded)
VOICE_MAPPING = {
    "hi-IN": {"speaker": "anushka", "name": "Hindi"},
    "bn-IN": {"speaker": "manisha", "name": "Bengali"},
    "ta-IN": {"speaker": "vidya", "name": "Tamil"},
    "te-IN": {"speaker": "abhiash", "name": "Telugu"},
    "mr-IN": {"speaker": "arya", "name": "Marathi"},
    "gu-IN": {"speaker": "karun", "name": "Gujarati"},
    "kn-IN": {"speaker": "hitesh", "name": "Kannada"},
    "ml-IN": {"speaker": "madhur", "name": "Malayalam"},
    "pa-IN": {"speaker": "gurpreet", "name": "Punjabi"},
    "od-IN": {"speaker": "suman", "name": "Odia"},
    "en-IN": {"speaker": "ron", "name": "English"},
    # Add other languages as needed, or fallback below
}

def get_optimal_voice(language_code: str) -> str:
    """Get optimal voice for language."""
    if language_code in VOICE_MAPPING:
        return VOICE_MAPPING[language_code]["speaker"]
    else:
        logger.warning(f"Speaker not found for {language_code}, defaulting to 'meera'")
        return "meera"  # default fallback speaker

def text_to_speech(text: str, language: str = "hi-IN", speaker: str = None) -> str:
    """
    Convert text to speech using Sarvam Bulbul v2.
    
    Args:
        text: Text to convert to speech (max 500 chars)
        language: Language code (must be one of Sarvam's supported codes)
        speaker: Voice speaker name (optional, will auto-select if None)
    
    Returns:
        Path to generated audio file, or None on failure
    """
    try:
        logger.info(f"Converting text to speech: {text[:100]}...")

        # Validate or default language code to supported
        if language not in SUPPORTED_LANGUAGES:
            logger.warning(f"Language '{language}' not supported by Sarvam TTS. Defaulting to 'hi-IN'.")
            language = "hi-IN"

        MAX_TTS_LENGTH = 500  # Increased from 50 to 500 for meaningful TTS output
        clean_text = text.strip()
        if len(clean_text) > MAX_TTS_LENGTH:
            logger.info(f"Input text length {len(clean_text)} exceeds {MAX_TTS_LENGTH}, trimming.")
            clean_text = clean_text[:MAX_TTS_LENGTH]

        # Select speaker if not provided
        if not speaker:
            speaker = get_optimal_voice(language)

        # Prepare payload for Sarvam TTS API
        payload = {
            "inputs": [clean_text],
            "target_language_code": language,
            "speaker": speaker,
            "pitch": 0,           # Neutral pitch
            "pace": 1.0,          # Normal speed
            "loudness": 1.0,      # Normal volume
            "speech_sample_rate": 22050,
            "enable_preprocessing": True,
            "model": "bulbul:v2"  # Use latest Bulbul model
        }

        response = requests.post(
            TTS_API_URL,
            headers=TTS_HEADERS,
            json=payload,
            timeout=60
        )

        response.raise_for_status()
        result = response.json()

        if "audios" in result and result["audios"]:
            audio_base64 = result["audios"][0]
            audio_bytes = base64.b64decode(audio_base64)

            temp_dir = tempfile.mkdtemp(prefix="sarvam_tts_")
            audio_file_path = os.path.join(temp_dir, "tts_output.mp3")

            with open(audio_file_path, "wb") as f:
                f.write(audio_bytes)

            logger.info(f"TTS audio generated: {audio_file_path}")
            return audio_file_path
        else:
            logger.error(f"No audio data received from TTS API. Response: {result}")
            return None

    except requests.HTTPError as e:
        try:
            error_info = response.json()
        except Exception:
            error_info = response.text
        logger.error(f"TTS API HTTP error: {e}, Response: {error_info}")
        return None
    except Exception as e:
        logger.error(f"TTS processing failed: {str(e)}")
        return None
