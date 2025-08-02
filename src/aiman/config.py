"""
Configuration management using environment variables.
Loads secrets from .env file safely.
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

def get_env(key: str, default: str = None, required: bool = True) -> str:
    """
    Get environment variable with error handling.
    
    Args:
        key: Environment variable name
        default: Default value if not found
        required: Whether to exit if not found
    
    Returns:
        Environment variable value
    """
    value = os.getenv(key, default)
    
    if required and value is None:
        print(f"❌ ERROR: Missing required environment variable: {key}")
        print(f"Please add {key} to your .env file")
        sys.exit(1)
    
    return value

# Load all configuration variables
TWILIO_ACCOUNT_SID = get_env("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = get_env("TWILIO_AUTH_TOKEN") 
TWILIO_NUMBER = get_env("TWILIO_NUMBER")

SARVAM_API_KEY = get_env("SARVAM_API_KEY")
GEMINI_API_KEY = get_env("GEMINI_API_KEY")

MONGODB_URI = get_env("MONGODB_URI")

FLASK_ENV = get_env("FLASK_ENV", default="production", required=False)
FLASK_DEBUG = get_env("FLASK_DEBUG", default="false", required=False).lower() == "true"

print("✅ Configuration loaded successfully")