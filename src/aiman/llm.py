"""
Large Language Model integration using Google Gemini 2.5 Flash.
Handles agricultural queries and generates expert responses.
"""
import google.generativeai as genai
import logging
from aiman.config import GEMINI_API_KEY

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY)

# Initialize Gemini model
model = genai.GenerativeModel("gemini-1.5-flash")

# Agricultural expert system prompt
AGRICULTURAL_EXPERT_PROMPT = """You are an expert agricultural advisor with deep knowledge of Indian farming practices, crops, weather patterns, soil management, pest control, and modern agricultural techniques.
Give detailed answer in 250 characters. 

Your expertise includes:
- Crop selection and rotation strategies
- Soil health and fertilizer management  
- Pest and disease identification and treatment
- Weather-based farming decisions
- Organic and sustainable farming practices
- Government schemes and subsidies for farmers
- Market prices and crop economics
- Water management and irrigation techniques

Guidelines for responses:
1. Provide practical, actionable advice
2. Consider Indian climate and soil conditions
3. Suggest cost-effective solutions
4. Mention relevant government schemes when applicable
5. Keep language simple and farmer-friendly
6. Include seasonal timing when relevant
7. Provide multiple options when possible

Always be helpful, accurate, and supportive of farmers' needs."""

def build_conversation_prompt(history: list, user_message: str) -> str:
    """
    Build a conversation prompt with history context.
    
    Args:
        history: List of previous conversation messages
        user_message: Current user message
    
    Returns:
        Complete prompt for the LLM
    """
    prompt = AGRICULTURAL_EXPERT_PROMPT + "\n\nConversation History:\n"
    
    # Add conversation history
    for msg in history[-10:]:  # Keep last 10 messages for context
        role = "Farmer" if msg.get("role") == "user" else "Agricultural Expert"
        prompt += f"{role}: {msg.get('text', '')}\n"
    
    # Add current message
    prompt += f"Farmer: {user_message}\n"
    prompt += "Agricultural Expert:"
    
    return prompt

def generate_agricultural_advice(user_message: str, conversation_history: list = None) -> str:
    """
    Generate agricultural advice using Gemini.
    
    Args:
        user_message: User's question in English
        conversation_history: Previous conversation context
    
    Returns:
        Expert agricultural advice
    """
    try:
        if conversation_history is None:
            conversation_history = []
        
        # Build the prompt with context
        prompt = build_conversation_prompt(conversation_history, user_message)
        
        logger.info(f"Generating advice for: {user_message[:50]}...")
        
        # Generate response using Gemini
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.7,
                top_p=0.8,
                top_k=40,
                max_output_tokens=1024,
            )
        )
        
        if response and response.text:
            advice = response.text.strip()
            logger.info(f"Generated advice: {advice[:50]}...")
            return advice
        else:
            logger.warning("Empty response from Gemini")
            return "मुझे खुशी होगी कि मैं आपकी मदद कर सकूं। कृपया अपना सवाल फिर से पूछें।"
    
    except Exception as e:
        logger.error(f"LLM generation failed: {str(e)}")
        return "मुझे खेद है, मैं अभी आपकी मदद नहीं कर सकता। कृपया बाद में कोशिश करें।"

def generate_quick_response(message_type: str = "greeting") -> str:
    """Generate quick responses for common scenarios."""
    responses = {
        "greeting": "नमस्ते! मैं आपका कृषि सलाहकार हूं। मैं खेती, फसल, मिट्टी, और कृषि की किसी भी समस्या में आपकी मदद कर सकता हूं। आपका क्या सवाल है?",
        "error": "मुझे खेद है, कुछ तकनीकी समस्या हुई है। कृपया अपना संदेश फिर से भेजें।",
        "processing": "आपका संदेश मिल गया है। मैं आपके लिए सबसे अच्छी सलाह तैयार कर रहा हूं...",
        "goodbye": "धन्यवाद! खुशी से आपकी मदद की। यदि कोई और सवाल हो तो बेझिझक पूछें। अच्छी खेती!"
    }
    
    return responses.get(message_type, responses["greeting"])
