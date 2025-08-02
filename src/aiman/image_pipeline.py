"""
Image processing utilities for agricultural analysis.
Downloads and processes images from WhatsApp using Gemini Vision.
Optimized for fast response generation and better error handling.
"""

import requests
import tempfile
import os
import logging
import time
import json
from typing import Optional, Dict, Tuple
import google.generativeai as genai
from PIL import Image, ImageOps
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from aiman.config import TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, GEMINI_API_KEY

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Gemini for Vision with optimized settings
genai.configure(api_key=GEMINI_API_KEY)

# Cache for repeated image analysis to improve response time
_analysis_cache = {}
_cache_timeout = 3600  # 1 hour cache timeout



def from_english_simple(text: str, target_language_code: str) -> str:
    """Simplified translation from English using Gemini as fallback with caching."""
    try:
        # Cache key for translation
        cache_key = f"{hash(text)}_{target_language_code}"
        
        lang_map = {
            "hi-IN": "Hindi", "mr-IN": "Marathi", "bn-IN": "Bengali",
            "ta-IN": "Tamil", "te-IN": "Telugu", "gu-IN": "Gujarati",
            "kn-IN": "Kannada", "ml-IN": "Malayalam", "pa-IN": "Punjabi",
            "or-IN": "Odia", "en-IN": "English"
        }
        
        target_lang = lang_map.get(target_language_code, "Hindi")
        
        # Optimized prompt for faster processing
        prompt = f"""Translate to {target_lang} (natural, farmer-friendly):
"{text[:500]}"  

Translation:"""

        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.3,  # Lower temperature for consistent translations
                max_output_tokens=512,  # Limit tokens for faster response
            )
        )
        
        translation = response.text.strip() if response and response.text else ""
        
        # Clean up translation
        if translation.startswith('"') and translation.endswith('"'):
            translation = translation[1:-1]
        
        if not translation or len(translation.strip()) < 10:
            return text
        
        return translation

    except Exception as e:
        logger.error(f"Simple translation error: {e}")
        return text

def download_and_process_image(media_url: str) -> str:
    """
    Download and optimize image from Twilio WhatsApp webhook.
    Includes size optimization and format validation.
    
    Args:
        media_url: The media URL provided by Twilio's webhook
    
    Returns:
        Path to downloaded and optimized image file
    """
    try:
        start_time = time.time()
        logger.info(f"Downloading image from: {media_url}")
        
        # Download image with timeout and authentication
        response = requests.get(
            media_url,
            auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN),
            timeout=30,
            stream=True  # Stream for large images
        )
        
        response.raise_for_status()
        
        # Check content length if available
        content_length = response.headers.get('content-length')
        if content_length and int(content_length) > 10 * 1024 * 1024:  # 10MB limit
            raise ValueError("Image too large (>10MB)")
        
        # Create temporary file with better naming
        temp_dir = tempfile.mkdtemp(prefix="whatsapp_image_")
        image_file = os.path.join(temp_dir, "input_image.jpg")
        
        # Save image with streaming to handle large files
        with open(image_file, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        # Verify and optimize image
        with Image.open(image_file) as img:
            logger.info(f"Original image: {img.size}, {img.format}, {img.mode}")
            
            # Convert to RGB if needed
            if img.mode not in ('RGB', 'L'):
                img = img.convert('RGB')
                logger.info("Converted image to RGB mode")
            
            # Optimize image size for faster processing (max 1024x1024)
            max_size = (1024, 1024)
            if img.size[0] > max_size[0] or img.size[1] > max_size[1]:
                img = img.copy()
                img.thumbnail(max_size, Image.Resampling.LANCZOS)
                
                # Save optimized image
                img.save(image_file, "JPEG", quality=85, optimize=True)
                logger.info(f"Resized image to: {img.size}")
        
        download_time = time.time() - start_time
        logger.info(f"Image downloaded and optimized in {download_time:.2f}s: {image_file}")
        
        return image_file

    except Exception as e:
        logger.error(f"Failed to download/process image: {str(e)}")
        raise

def analyze_agricultural_image(image_path: str, user_query: str = None) -> str:
    """
    Analyze agricultural image using optimized Gemini Vision processing.
    Includes caching and optimized prompts for faster response.
    
    Args:
        image_path: Path to the image file
        user_query: Optional specific question about the image
    
    Returns:
        Agricultural analysis results
    """
    try:
        start_time = time.time()
        logger.info(f"Analyzing agricultural image: {image_path}")
        
        # Create cache key based on file hash and query
        with open(image_path, 'rb') as f:
            file_hash = hash(f.read()[:1024])  # Hash first 1KB for caching
        cache_key = f"{file_hash}_{hash(user_query or 'default')}"
        
        # Check cache first
        if cache_key in _analysis_cache:
            cached_result, timestamp = _analysis_cache[cache_key]
            if time.time() - timestamp < _cache_timeout:
                logger.info("Returning cached analysis result")
                return cached_result
        
        # Load and prepare image
        with Image.open(image_path) as img:
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Use optimized Gemini model
            model = genai.GenerativeModel('gemini-1.5-flash')
            
            # Create optimized prompt based on whether user has specific query
            if user_query:
                prompt = f"""As an expert agricultural advisor, analyze this image focusing on the farmer's question: "{user_query}"

Provide specific, actionable advice about:
1. Direct answer to: {user_query}
2. What you see in the image (crop/plant identification)
3. Health assessment and any issues
4. Immediate recommendations
5. Prevention measures

Keep response practical and under 400 characters. Use simple language suitable for Indian farmers."""
            else:
                prompt = """As an expert agricultural advisor, analyze this farming image:

1. **Plant/Crop**: What do you see?
2. **Health**: Is it healthy, diseased, or stressed?
3. **Issues**: Any diseases, pests, or nutrient problems?
4. **Stage**: Growth stage of the plant
5. **Action**: What should the farmer do now?
6. **Prevention**: How to avoid future problems?

Provide practical advice under 400 characters in simple language for Indian farmers."""
            
            # Generate analysis with timeout
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    model.generate_content,
                    [prompt, img],
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.4,
                        top_p=0.8,
                        max_output_tokens=800,
                    )
                )
                
                try:
                    response = future.result(timeout=25)  # 25s timeout
                except FutureTimeoutError:
                    logger.error("Gemini Vision analysis timed out")
                    return "तस्वीर का विश्लेषण में समय लग रहा है। कृपया कुछ समय बाद दोबारा कोशिश करें।"
            
            if response and response.text:
                analysis = response.text.strip()
                
                # Cache successful result
                _analysis_cache[cache_key] = (analysis, time.time())
                
                # Clean old cache entries (simple cleanup)
               # Clean old cache entries (simple cleanup)
                if len(_analysis_cache) > 100:
                    current_time = time.time()
                    keys_to_delete = [k for k, v in _analysis_cache.items() if current_time - v[1] >= _cache_timeout]
                    for k in keys_to_delete:
                        del _analysis_cache[k]
                
                analysis_time = time.time() - start_time
                logger.info(f"Agricultural analysis completed in {analysis_time:.2f}s: {len(analysis)} characters")
                
                return analysis
            else:
                logger.warning("Empty response from Gemini Vision")
                return "मुझे इस तस्वीर का विश्लेषण करने में कुछ समस्या हुई है। कृपया दूसरी तस्वीर भेजें।"

    except Exception as e:
        logger.error(f"Error analyzing agricultural image: {str(e)}")
        return "मुझे इस तस्वीर का विश्लेषण करने में समस्या हुई है। कृपया फिर से कोशिश करें।"

def analyze_image_with_context(image_path: str, user_query: str, conversation_history: list = None) -> str:
    """
    Enhanced image analysis with conversation context for better responses.
    
    Args:
        image_path: Path to the image file
        user_query: User's specific question
        conversation_history: Previous conversation messages
        
    Returns:
        Contextual agricultural analysis
    """
    try:
        # Build context from conversation history
        context = ""
        if conversation_history:
            recent_messages = conversation_history[-3:]  # Last 3 messages for context
            for msg in recent_messages:
                role = "Farmer" if msg.get("role") == "user" else "Expert"
                context += f"{role}: {msg.get('text', '')[:100]}...\n"
        
        # Enhanced analysis with context
        with Image.open(image_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            model = genai.GenerativeModel('gemini-1.5-flash')
            
            prompt = f"""You are an agricultural expert. Previous conversation context:
{context}

Current farmer question about this image: "{user_query}"

Analyze the image and provide specific advice considering the conversation context:

1. Direct answer to farmer's question
2. Crop/plant identification  
3. Current health status
4. Specific issues visible
5. Immediate action needed
6. Connection to previous discussion (if relevant)

Keep response under 400 words, practical, and farmer-friendly."""

            response = model.generate_content(
                [prompt, img],
                generation_config=genai.types.GenerationConfig(
                    temperature=0.5,
                    max_output_tokens=800,
                )
            )
            
            if response and response.text:
                return response.text.strip()
            else:
                # Fallback to regular analysis
                return analyze_agricultural_image(image_path, user_query)
                
    except Exception as e:
        logger.error(f"Error in contextual image analysis: {str(e)}")
        # Fallback to regular analysis
        return analyze_agricultural_image(image_path, user_query)

def get_image_metadata(image_path: str) -> Dict:
    """Extract metadata from image for analytics and optimization."""
    try:
        with Image.open(image_path) as img:
            return {
                "size": img.size,
                "format": img.format,
                "mode": img.mode,
                "file_size": os.path.getsize(image_path),
                "has_exif": bool(getattr(img, '_getexif', lambda: None)())
            }
    except Exception as e:
        logger.error(f"Error extracting image metadata: {e}")
        return {}

def cleanup_temp_image(image_path: str):
    """Clean up temporary image files and directories with better error handling."""
    try:
        if os.path.exists(image_path):
            os.remove(image_path)
            
            # Clean up parent temp directory if empty
            temp_dir = os.path.dirname(image_path)
            if os.path.exists(temp_dir) and temp_dir.startswith(tempfile.gettempdir()):
                try:
                    if not os.listdir(temp_dir):
                        os.rmdir(temp_dir)
                        logger.info(f"Cleaned up temp directory: {temp_dir}")
                except OSError:
                    pass  # Directory not empty or permission issue
            
            logger.info(f"Cleaned up temp image: {image_path}")
            
    except Exception as e:
        logger.warning(f"Failed to cleanup image {image_path}: {str(e)}")

def validate_image_format(image_path: str) -> bool:
    """Validate if image format is supported for processing."""
    try:
        with Image.open(image_path) as img:
            # Check if format is supported
            supported_formats = ['JPEG', 'PNG', 'WEBP', 'BMP']
            if img.format not in supported_formats:
                logger.warning(f"Unsupported image format: {img.format}")
                return False
            
            # Check image size limits
            if img.size[0] < 50 or img.size[1] < 50:
                logger.warning(f"Image too small: {img.size}")
                return False
                
            if img.size[0] > 4096 or img.size[1] > 4096:
                logger.warning(f"Image too large: {img.size}")
                return False
            
            return True
            
    except Exception as e:
        logger.error(f"Error validating image format: {e}")
        return False