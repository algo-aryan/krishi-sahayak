import os
import time
import tempfile
import subprocess
from flask import Flask, request, send_file, abort, make_response, Response
from twilio.rest import Client
from twilio.twiml.messaging_response import MessagingResponse  
import logging
import traceback  # For detailed error logging
from datetime import datetime, timedelta
import threading
from collections import defaultdict

from aiman.config import (
    TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_NUMBER,
    FLASK_ENV, FLASK_DEBUG
)
from aiman import audio_pipeline, stt, translator, llm, tts, db

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# In-memory store mapping audio filenames to local file paths

file_storage_lock = threading.Lock()
file_storage = {}

def store_audio_file_safely(filename: str, filepath: str):
    """Thread-safe way to store audio file mapping"""
    with file_storage_lock:
        file_storage[filename] = filepath
        logger.info(f"Stored audio file mapping: {filename} -> {filepath}")

def get_audio_file_safely(filename: str) -> str:
    """Thread-safe way to get audio file path"""
    with file_storage_lock:
        return file_storage.get(filename)


def cleanup_file_later(filename: str, delay_sec: int = 120):
    """Delete temporary audio files after delay."""
    time.sleep(delay_sec)
    path = file_storage.pop(filename, None)
    if path and os.path.exists(path):
        try:
            os.remove(path)
            logger.info(f"Cleaned up temporary audio file: {path}")
        except Exception as e:
            logger.error(f"Failed to clean up audio file {path}: {e}")

@app.route("/audio/<filename>", methods=["GET"])
def serve_audio(filename):
    path = file_storage.get(filename)
    if not path or not os.path.exists(path):
        logger.warning(f"Audio file not found: {filename}")
        abort(404)
    
    logger.info(f"Serving audio file: {filename}")
    
    # Get file size
    file_size = os.path.getsize(path)
    
    # Set proper MIME type for WhatsApp compatibility
    if filename.endswith(".ogg"):
        mimetype = "audio/ogg; codecs=opus"  # ✅ WhatsApp compatible
    else:
        mimetype = "audio/mpeg"
    
    # Handle range requests for audio streaming
    range_header = request.headers.get('Range', None)
    
    if range_header:
        # Parse range header
        byte_start, byte_end = parse_range_header(range_header, file_size)
        
        # Read the requested range
        with open(path, 'rb') as f:
            f.seek(byte_start)
            data = f.read(byte_end - byte_start + 1)
        
        # Create 206 Partial Content response
        response = Response(
            data,
            status=206,
            mimetype=mimetype,
            direct_passthrough=True
        )
        
        # Set required headers for partial content
        response.headers.add('Content-Range', f'bytes {byte_start}-{byte_end}/{file_size}')
        response.headers.add('Accept-Ranges', 'bytes')
        response.headers.add('Content-Length', str(len(data)))
        response.headers.add('Content-Disposition', f'inline; filename="{filename}"')
        
        return response
    else:
        # Regular full file response
        response = make_response(send_file(path, mimetype=mimetype))
        response.headers.add('Accept-Ranges', 'bytes')
        response.headers.add('Content-Length', str(file_size))
        response.headers.add('Content-Disposition', f'inline; filename="{filename}"')
        
        return response

def parse_range_header(range_header, file_size):
    """Parse HTTP Range header and return start, end byte positions"""
    try:
        # Range header format: "bytes=start-end"
        range_match = range_header.replace('bytes=', '')
        
        if '-' in range_match:
            start, end = range_match.split('-', 1)
            
            # Handle different range formats
            if start == '':
                # Suffix-byte-range: bytes=-500 (last 500 bytes)
                start = max(0, file_size - int(end))
                end = file_size - 1
            elif end == '':
                # Range from start to end: bytes=500-
                start = int(start)
                end = file_size - 1
            else:
                # Full range: bytes=500-999
                start = int(start)
                end = int(end)
        else:
            # Single byte position
            start = int(range_match)
            end = file_size - 1
            
        # Ensure end doesn't exceed file size
        end = min(end, file_size - 1)
        
        return start, end
        
    except (ValueError, AttributeError):
        # Invalid range header, return full file
        return 0, file_size - 1



@app.route("/", methods=["GET"])
def health_check():
    return {
        "status": "healthy",
        "service": "AI Manthan Agricultural Chatbot",
        "version": "1.0.0"
    }

@app.route("/whatsapp", methods=["POST"])
def whatsapp_webhook():
    try:
        sender = request.values.get("From", "")
        message_text = request.values.get("Body", "").strip()
        media_url = request.values.get("MediaUrl0", "")
        media_content_type = request.values.get("MediaContentType0", "")

        logger.info(f"Received message from {sender}")

        user_stats = db.get_user_stats(sender)

        if media_url and media_content_type.startswith("audio"):
            response_text = process_voice_message(sender, media_url, user_stats)
        elif message_text:
            response_text = process_text_message(sender, message_text, user_stats)
        else:
            response_text = llm.generate_quick_response("greeting")

        # Only send text if we have a response (voice audio sends separately)
        if response_text:
            send_whatsapp_response(sender, response_text)

        resp = MessagingResponse()
        return str(resp)

    except Exception as e:
        logger.error(f"Webhook error: {str(e)}")
        logger.error(traceback.format_exc())
        error_message = llm.generate_quick_response("error")
        send_whatsapp_response(sender, error_message)
        resp = MessagingResponse()
        return str(resp)

def process_voice_message(sender: str, media_url: str, user_stats: dict) -> str:
    # Welcome new users
    if not user_stats.get("is_returning_user", False):
        welcome_message = llm.generate_quick_response("greeting")
        send_whatsapp_response(sender, welcome_message)
        time.sleep(1) # Small delay to ensure message order
    
    wav_file = None
    try:
        logger.info("Processing voice message...")
        wav_file = audio_pipeline.download_and_convert_audio(media_url)
        if not wav_file:
            raise RuntimeError("Audio conversion failed")

        # Use enhanced transcription with language verification
        transcription = stt.detect_and_transcribe(wav_file)
        
        regional_text = transcription.get("text", "")
        stt_lang_code = transcription.get("language", "hi-IN")
        confidence = transcription.get("confidence", 0.0)
        
        if not regional_text:
            return llm.generate_quick_response("error")

        logger.info(f"Enhanced STT result: ({stt_lang_code}, conf: {confidence:.2f}): {regional_text}")

        # Continue with translation and response generation
        detected_lang_prefix = stt_lang_code.split('-')[0]
        english_text, _ = translator.to_english(regional_text, source_lang_code=detected_lang_prefix)

        history = db.get_history(sender, limit=3)
        english_response = llm.generate_agricultural_advice(english_text, history)

        regional_response = translator.from_english(english_response, target_language_code=stt_lang_code)

        db.save_user_message(sender, regional_text)
        db.save_bot_response(sender, regional_response)

        # Generate audio SYNCHRONOUSLY before sending
        audio_response_file_mp3 = tts.text_to_speech(regional_response, language=stt_lang_code)
        if not audio_response_file_mp3:
            logger.error("TTS generation failed")
            return regional_response

        audio_response_file_ogg = convert_mp3_to_ogg(audio_response_file_mp3)
        if not audio_response_file_ogg or not os.path.exists(audio_response_file_ogg):
            logger.error("MP3 to OGG conversion failed")
            return regional_response

        # Store file mapping BEFORE sending Twilio message
        filename = os.path.basename(audio_response_file_ogg)
        store_audio_file_safely(filename, audio_response_file_ogg)
        
        # Verify file exists before sending
        if os.path.exists(audio_response_file_ogg):
            send_whatsapp_audio(sender, audio_response_file_ogg)
            return None  # audio sent
        else:
            logger.error(f"Audio file does not exist: {audio_response_file_ogg}")
            return regional_response

    except Exception as e:
        logger.error(f"Voice processing error: {e}")
        return llm.generate_quick_response("error")

    finally:
        if wav_file and os.path.exists(wav_file):
            audio_pipeline.cleanup_temp_file(wav_file)


def convert_mp3_to_ogg(mp3_file_path: str) -> str:
    """Convert MP3 file to OGG with Opus codec for WhatsApp compatibility."""
    try:
        temp_dir = tempfile.mkdtemp(prefix="tts_ogg_")
        ogg_file_path = os.path.join(temp_dir, "tts_output.ogg")

        ffmpeg_cmd = [
            "ffmpeg",
            "-y",                 # Overwrite output files without asking
            "-i", mp3_file_path,  # Input MP3 path
            "-c:a", "libopus",    # Use Opus codec (WhatsApp requirement)
            "-b:a", "32k",        # Bitrate (32k is optimal for voice)
            "-vbr", "on",         # Variable bitrate for better quality
            "-compression_level", "10",  # Maximum compression
            "-frame_duration", "20",     # 20ms frame duration
            "-application", "voip",      # Optimize for voice
            ogg_file_path
        ]

        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"ffmpeg conversion failed: {result.stderr}")
            return mp3_file_path  # fallback to original if failure

        # Verify the output file exists and has content
        if os.path.exists(ogg_file_path) and os.path.getsize(ogg_file_path) > 0:
            logger.info(f"Converted MP3 to OGG/Opus successfully: {ogg_file_path}")
            return ogg_file_path
        else:
            logger.error("OGG conversion produced empty file")
            return mp3_file_path

    except Exception as e:
        logger.error(f"Error in converting MP3 to OGG: {e}")
        return mp3_file_path


def process_text_message(sender: str, message_text: str, user_stats: dict) -> str:
    try:
        logger.info(f"Processing text message: {message_text}")

        # Greeting for new users
        is_greeting = any(
            greeting in message_text.lower()
            for greeting in ["hi", "hello", "नमस्ते", "हैलो", "hey"]
        )
        if not user_stats.get("is_returning_user", False) and is_greeting:
            response = llm.generate_quick_response("greeting")
            db.save_user_message(sender, message_text)
            db.save_bot_response(sender, response)
            return response

        # Translation flow for general text
        english_text, detected_language = translator.to_english(message_text)
        history = db.get_history(sender, limit=3)
        english_response = llm.generate_agricultural_advice(english_text, history)
        regional_response = translator.from_english(
            english_response,
            target_language_code=detected_language
        )

        db.save_user_message(sender, message_text)
        db.save_bot_response(sender, regional_response)

        return regional_response

    except Exception as e:
        logger.error(f"Text processing error: {str(e)}")
        logger.error(traceback.format_exc())
        return llm.generate_quick_response("error")

def send_whatsapp_response(sender: str, message: str):
    try:
        if not message:
            return

        cleaned_number = sender.replace("whatsapp:", "").replace(" ", "")
        if not cleaned_number.startswith("+"):
            formatted_number = f"whatsapp:+91{cleaned_number}"
        else:
            formatted_number = f"whatsapp:{cleaned_number}"

        MAX_MESSAGE_LENGTH = 1600

        # Split message intelligently at spaces to avoid breaking words
        message_chunks = []
        start = 0
        text_length = len(message)

        while start < text_length:
            end = min(start + MAX_MESSAGE_LENGTH, text_length)
            if end < text_length:
                last_space = message.rfind(' ', start, end)
                if last_space != -1 and last_space > start:
                    end = last_space
            chunk = message[start:end].strip()
            if chunk:
                message_chunks.append(chunk)
            start = end

        for chunk in message_chunks:
            twilio_client.messages.create(
                from_=TWILIO_NUMBER,
                to=formatted_number,
                body=chunk
            )
        logger.info(f"Sent {len(message_chunks)} text message chunk(s) to {sender}")

    except Exception as e:
        logger.error(f"Failed to send WhatsApp response: {str(e)}")
        logger.error(traceback.format_exc())

def send_whatsapp_audio(sender: str, audio_file_path: str):
    try:
        # Use BASE_URL or NGROK_URL environment variable for public URL base
        base_url = os.environ.get("BASE_URL") or os.environ.get("NGROK_URL")
        if not base_url:
            logger.error("BASE_URL or NGROK_URL environment variable is not set. Cannot send audio.")
            send_whatsapp_response(sender, "🎵 ऑडियो जवाब उपलब्ध नहीं है। कृपया कुछ समय बाद पुनः प्रयास करें।")
            return

        filename = os.path.basename(audio_file_path)
        
        # File already stored safely in process_voice_message
        
        audio_url = f"{base_url.rstrip('/')}/audio/{filename}"

        cleaned_number = sender.replace("whatsapp:", "").replace(" ", "")
        if not cleaned_number.startswith("+"):
            formatted_number = f"whatsapp:+91{cleaned_number}"
        else:
            formatted_number = f"whatsapp:{cleaned_number}"

        # Add small delay to ensure file is fully written
        time.sleep(0.1)
        
        twilio_client.messages.create(
            from_=TWILIO_NUMBER,
            to=formatted_number,
            media_url=[audio_url]
        )
        logger.info(f"Sent audio message with media URL {audio_url} to {sender}")

        # Cleanup audio file asynchronously after 2 minutes
        cleanup_thread = threading.Thread(target=cleanup_file_later, args=(filename, 120))
        cleanup_thread.start()

    except Exception as e:
        logger.error(f"Failed to send WhatsApp audio: {str(e)}")
        logger.error(traceback.format_exc())
        # Fallback to text if audio send fails
        send_whatsapp_response(sender, "🎵 यहाँ आपका ऑडियो जवाब होगा")


@app.route("/stats", methods=["GET"])
def get_stats():
    try:
        return {
            "status": "operational",
            "total_conversations": db.conversations.count_documents({}),
            "active_users_today": db.conversations.distinct("user_id", {
                "timestamp": {"$gte": datetime.utcnow() - timedelta(days=1)}
            })
        }
    except Exception as e:
        logger.error(f"Stats endpoint error: {str(e)}")
        logger.error(traceback.format_exc())
        return {"error": str(e)}, 500

if __name__ == "__main__":
    logger.info("🚀 Starting AI Manthan Agricultural Chatbot...")
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
        debug=FLASK_DEBUG
    )