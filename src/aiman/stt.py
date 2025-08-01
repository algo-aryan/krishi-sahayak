"""
Enhanced Speech-to-Text using Sarvam AI Sarika v2.5.
Implements multi-language candidate system with improved accuracy for similar languages.
"""
import requests
import logging
from typing import Dict, List, Tuple, Optional
from collections import Counter
from aiman.config import SARVAM_API_KEY

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Sarvam AI API Configuration
STT_API_URL = "https://api.sarvam.ai/speech-to-text"
STT_HEADERS = {
    "api-subscription-key": SARVAM_API_KEY
}

# Language confidence scoring weights
SCRIPT_SIMILARITY_WEIGHTS = {
    "hi": {"mr": 0.8, "ne": 0.7},  # Hindi similar to Marathi, Nepali
    "mr": {"hi": 0.8, "ne": 0.6},  # Marathi similar to Hindi, Nepali
    "ta": {"te": 0.3, "kn": 0.3},  # Tamil different script families
    "te": {"ta": 0.3, "kn": 0.4},
    "bn": {"as": 0.7, "hi": 0.2},  # Bengali script differences
}

# Region-based language probability (for context hints)
REGION_LANGUAGE_HINTS = {
    "महाराष्ट्र": ["mr-IN", "hi-IN"],  # Maharashtra -> Marathi primary
    "कर्नाटक": ["kn-IN", "hi-IN"],     # Karnataka -> Kannada primary  
    "गुजरात": ["gu-IN", "hi-IN"],     # Gujarat -> Gujarati primary
    "तमिलनाडु": ["ta-IN", "hi-IN"],   # Tamil Nadu -> Tamil primary
    "पंजाब": ["pa-IN", "hi-IN"],      # Punjab -> Punjabi primary
    "बंगाल": ["bn-IN", "hi-IN"],      # Bengal -> Bengali primary
}

def transcribe_audio(wav_file_path: str, language_code: str = "hi-IN") -> dict:
    """
    Transcribe audio file to text using Sarvam Sarika API.
    
    Args:
        wav_file_path: Path to WAV audio file
        language_code: Language code (hi-IN, bn-IN, etc.)
    
    Returns:
        Dictionary with transcription results
    """
    try:
        logger.debug(f"Transcribing audio with {language_code}: {wav_file_path}")
        
        # Prepare the request
        files = {
            "file": ("audio.wav", open(wav_file_path, "rb"), "audio/wav")
        }
        
        data = {
            "language_code": language_code,
            "model": "saarika:v2.5",  # Use latest Sarika model
            "with_timestamps": "false"
        }
        
        # Make API request
        response = requests.post(
            STT_API_URL,
            headers=STT_HEADERS,
            files=files,
            data=data,
            timeout=30
        )
        
        # Close the file
        files["file"][1].close()
        
        response.raise_for_status()
        result = response.json()
        
        # Extract transcription
        transcription = {
            "text": result.get("transcript", ""),
            "language": language_code,
            "confidence": result.get("confidence", 0.0)
        }
        
        logger.debug(f"Transcription successful: {transcription['text'][:50]}...")
        return transcription
    
    except requests.RequestException as e:
        logger.error(f"STT API request failed for {language_code}: {str(e)}")
        return {"text": "", "language": language_code, "confidence": 0.0}
    
    except Exception as e:
        logger.error(f"STT processing failed for {language_code}: {str(e)}")
        return {"text": "", "language": language_code, "confidence": 0.0}

def detect_and_transcribe(wav_file_path: str) -> dict:
    """
    Enhanced transcription with multiple language candidates and confidence scoring.
    """
    try:
        # Get multiple transcription candidates
        candidates = get_multiple_language_candidates(wav_file_path)
        
        # Analyze and score candidates
        best_candidate = select_best_language_candidate(candidates)
        
        logger.info(f"Enhanced STT selected: {best_candidate['language']} with confidence: {best_candidate.get('confidence', 'N/A')}")
        
        return {
            "text": best_candidate["text"],
            "language": best_candidate["language"],
            "confidence": best_candidate.get("confidence", 1.0),
            "alternatives": candidates[:3]  # Top 3 alternatives for debugging
        }
        
    except Exception as e:
        logger.error(f"Enhanced STT failed: {e}")
        # Fallback to simple method
        return transcribe_audio(wav_file_path, "hi-IN")

def get_multiple_language_candidates(wav_file_path: str) -> List[Dict]:
    """
    Get transcription candidates from multiple language models.
    """
    candidates = []
    
    # Primary candidates - most common Indian languages
    primary_languages = ["hi-IN", "mr-IN", "kn-IN", "ta-IN", "te-IN", "gu-IN", "bn-IN", "pa-IN"]
    
    # Try each language and collect results
    for lang_code in primary_languages:
        try:
            result = transcribe_audio(wav_file_path, lang_code)
            if result and result.get("text"):
                candidates.append({
                    "text": result["text"],
                    "language": lang_code,
                    "confidence": result.get("confidence", 0.5),
                    "source": "hinted"
                })
        except Exception as e:
            logger.warning(f"Failed to transcribe with {lang_code}: {e}")
            continue
    
    return candidates

def select_best_language_candidate(candidates: List[Dict]) -> Dict:
    """
    Select the best language candidate using multiple scoring factors.
    """
    if not candidates:
        return {"text": "", "language": "hi-IN", "confidence": 0.0}
    
    scored_candidates = []
    
    for candidate in candidates:
        score = calculate_candidate_score(candidate, candidates)
        scored_candidates.append((score, candidate))
    
    # Sort by score descending
    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    
    best_score, best_candidate = scored_candidates[0]
    best_candidate["confidence"] = best_score
    
    logger.info(f"Candidate scores: {[(c['language'], round(s, 3)) for s, c in scored_candidates[:3]]}")
    
    return best_candidate

def calculate_candidate_score(candidate: Dict, all_candidates: List[Dict]) -> float:
    """
    Calculate confidence score for a candidate based on multiple factors.
    """
    text = candidate["text"]
    language = candidate["language"].split("-")[0]  # Extract base language
    base_confidence = candidate.get("confidence", 0.5)
    
    score = base_confidence
    
    # Factor 1: Text length and completeness (longer, complete sentences score higher)
    length_factor = min(len(text.split()) / 10.0, 1.0)  # Normalize to max 1.0
    score += length_factor * 0.2
    
    # Factor 2: Regional context hints
    context_boost = get_regional_context_score(text, language)
    score += context_boost
    
    # Factor 3: Script consistency (check if text matches expected script)
    script_consistency = check_script_consistency(text, language)
    score += script_consistency * 0.3
    
    # Factor 4: Language model perplexity (simulated by word frequency)
    word_naturalness = calculate_word_naturalness(text, language)
    score += word_naturalness * 0.2
    
    # Factor 5: Penalize if too similar to other candidates (might be confusion)
    similarity_penalty = calculate_similarity_penalty(candidate, all_candidates)
    score -= similarity_penalty
    
    return max(0.0, min(1.0, score))  # Clamp between 0 and 1

def get_regional_context_score(text: str, language: str) -> float:
    """
    Boost score if text contains regional hints that match the language.
    """
    score_boost = 0.0
    
    for region, expected_langs in REGION_LANGUAGE_HINTS.items():
        if region in text:
            lang_code = f"{language}-IN"
            if lang_code in expected_langs:
                # Primary language for region gets higher boost
                boost = 0.3 if expected_langs[0] == lang_code else 0.1
                score_boost += boost
                logger.debug(f"Regional boost: {region} -> {language} (+{boost})")
    
    return score_boost

def check_script_consistency(text: str, language: str) -> float:
    """
    Check if the script used in text matches the expected script for language.
    """
    if not text.strip():
        return 0.0
    
    # Count characters by script
    devanagari_chars = sum(1 for c in text if '\u0900' <= c <= '\u097F')
    bengali_chars = sum(1 for c in text if '\u0980' <= c <= '\u09FF')
    tamil_chars = sum(1 for c in text if '\u0B80' <= c <= '\u0BFF')
    telugu_chars = sum(1 for c in text if '\u0C00' <= c <= '\u0C7F')
    kannada_chars = sum(1 for c in text if '\u0C80' <= c <= '\u0CFF')
    gujarati_chars = sum(1 for c in text if '\u0A80' <= c <= '\u0AFF')
    
    total_indic_chars = (devanagari_chars + bengali_chars + tamil_chars + 
                        telugu_chars + kannada_chars + gujarati_chars)
    
    if total_indic_chars == 0:
        return 0.0
    
    # Expected script for each language
    expected_scripts = {
        "hi": devanagari_chars,
        "mr": devanagari_chars, 
        "ne": devanagari_chars,
        "bn": bengali_chars,
        "ta": tamil_chars,
        "te": telugu_chars,
        "kn": kannada_chars,
        "gu": gujarati_chars,
    }
    
    expected_count = expected_scripts.get(language, 0)
    consistency_ratio = expected_count / total_indic_chars if total_indic_chars > 0 else 0
    
    return consistency_ratio

def calculate_word_naturalness(text: str, language: str) -> float:
    """
    Estimate naturalness based on common words and patterns for the language.
    """
    if not text.strip():
        return 0.0
    
    # Common words/patterns for each language (simplified)
    language_patterns = {
        "hi": ["में", "है", "का", "को", "से", "और", "या", "हैं", "था", "गया", "कैसे", "क्या", "कहाँ"],
        "mr": ["मध्ये", "आहे", "चा", "ला", "पासून", "आणि", "किंवा", "त", "होता", "गेला", "ात", "यचा", "ायचा", "कसा", "काय", "कुठे"],
        "ta": ["இல்", "உள்ள", "அந்த", "இந்த", "என்", "ஒரு", "மற்றும்", "அல்லது", "எப்படி", "என்ன"],
        "te": ["లో", "ఉంది", "అని", "ఒక", "మరియు", "లేదా", "అయ్యి", "ఎలా", "ఏమి"],
        "kn": ["ಇನ್", "ಉಳ್ಳ", "ಒಂದು", "ಮತ್ತು", "ಅಥವಾ", "ಹೇಗೆ", "ಏನು"],
        "gu": ["માં", "છે", "નો", "ને", "થી", "અને", "અથવાં", "કેવી", "શું"],
        "bn": ["এ", "আছে", "র", "কে", "থেকে", "এবং", "অথবা", "কিভাবে", "কি"],
        "pa": ["વિચ", "હૈ", "દા", "નુ", "થોં", "તે", "જા", "કિવેં", "કિ"]
    }
    
    if language not in language_patterns:
        return 0.5  # Neutral score for unknown languages
    
    common_words = language_patterns[language]
    text_words = text.split()
    
    matches = sum(1 for word in text_words if any(pattern in word for pattern in common_words))
    
    if len(text_words) == 0:
        return 0.0
        
    naturalness_score = min(matches / len(text_words), 1.0)
    return naturalness_score

def calculate_similarity_penalty(candidate: Dict, all_candidates: List[Dict]) -> float:
    """
    Penalize candidates that are too similar to others (may indicate confusion).
    """
    text = candidate["text"]
    language = candidate["language"]
    
    penalty = 0.0
    
    for other in all_candidates:
        if other["language"] == language:
            continue
            
        # Simple similarity check (can be improved with edit distance)
        other_text = other["text"]
        if len(text) > 0 and len(other_text) > 0:
            # Character-level similarity
            similarity = len(set(text) & set(other_text)) / len(set(text) | set(other_text))
            if similarity > 0.7:  # High similarity threshold
                penalty += 0.1
    
    return min(penalty, 0.3)  # Cap penalty at 0.3
