"""
Translation using AI4Bharat IndicTrans2.
Handles Indian languages to English and vice versa.
"""

import os
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from langdetect import detect, detect_langs, DetectorFactory
import logging
import unicodedata

# Try to import IndicProcessor, fallback to manual approach if not available
try:
    from IndicTransToolkit.processor import IndicProcessor
    INDIC_PROCESSOR_AVAILABLE = True
except ImportError:
    try:
        from IndicTransTokenizer import IndicProcessor
        INDIC_PROCESSOR_AVAILABLE = True
    except ImportError:
        INDIC_PROCESSOR_AVAILABLE = False

DetectorFactory.seed = 0

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

LANG_MAPPING = {
    "hi": {"ai4bharat": "hin_Deva", "sarvam": "hi-IN"},
    "bn": {"ai4bharat": "ben_Beng", "sarvam": "bn-IN"},
    "ta": {"ai4bharat": "tam_Taml", "sarvam": "ta-IN"},
    "te": {"ai4bharat": "tel_Telu", "sarvam": "te-IN"},
    "mr": {"ai4bharat": "mar_Deva", "sarvam": "mr-IN"},
    "gu": {"ai4bharat": "guj_Gujr", "sarvam": "gu-IN"},
    "kn": {"ai4bharat": "kan_Knda", "sarvam": "kn-IN"},
    "ml": {"ai4bharat": "mal_Mlym", "sarvam": "ml-IN"},
    "pa": {"ai4bharat": "pan_Guru", "sarvam": "pa-IN"},
    "or": {"ai4bharat": "ori_Orya", "sarvam": "or-IN"},
    "ur": {"ai4bharat": "urd_Arab", "sarvam": None},
    "en": {"ai4bharat": "eng_Latn", "sarvam": "en-IN"},
}

SCRIPT_MAPPING = {
    "hi": "Devanagari",
    "bn": "Bengali", 
    "ta": "Tamil",
    "te": "Telugu",
    "mr": "Devanagari",
    "gu": "Gujarati",
    "kn": "Kannada",
    "ml": "Malayalam",
    "pa": "Gurmukhi",
    "or": "Oriya",
    "ur": "Arabic",
    "en": "Latn",
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {DEVICE}")

XX2EN_MODEL_NAME = "ai4bharat/indictrans2-indic-en-1B"
xx2en_tokenizer = AutoTokenizer.from_pretrained(XX2EN_MODEL_NAME, trust_remote_code=True)
xx2en_model = AutoModelForSeq2SeqLM.from_pretrained(XX2EN_MODEL_NAME, trust_remote_code=True).to(DEVICE)

EN2XX_MODEL_NAME = "ai4bharat/indictrans2-en-indic-1B"
en2xx_tokenizer = AutoTokenizer.from_pretrained(EN2XX_MODEL_NAME, trust_remote_code=True)
en2xx_model = AutoModelForSeq2SeqLM.from_pretrained(EN2XX_MODEL_NAME, trust_remote_code=True).to(DEVICE)

# Initialize IndicProcessor if available
if INDIC_PROCESSOR_AVAILABLE:
    ip = IndicProcessor(inference=True)
    logger.info("IndicProcessor initialized successfully")
else:
    ip = None
    logger.warning("IndicProcessor not available, using manual preprocessing")

def detect_language(text: str) -> str:
    try:
        detections = detect_langs(text)
        for detection in detections:
            lang_code = detection.lang
            if lang_code in LANG_MAPPING and detection.prob > 0.8:
                logger.info(f"Detected language with high confidence: {lang_code} (prob: {detection.prob})")
                return lang_code
        # fallback to most likely detected lang if no high confidence
        if detections and detections[0].lang in LANG_MAPPING:
            fallback_lang = detections[0].lang
            logger.warning(f"No high-confidence detection. Falling back to most likely: {fallback_lang}")
            return fallback_lang
        logger.warning("Language detection failed or unsupported language detected, defaulting to 'hi'")
        return "hi"
    except Exception as e:
        logger.error(f"Language detection error: {e}, defaulting to 'hi'")
        return "hi"

def to_english(text: str, source_lang_code: str = None) -> tuple[str, str]:
    """
    Translate from Indic to English using proper IndicTrans2 preprocessing.
    Returns translated text and Sarvam language code.
    """
    try:
        # Determine language code
        if source_lang_code:
            src_lang = source_lang_code.split('-')[0]
            if src_lang not in LANG_MAPPING:
                src_lang = None
        else:
            src_lang = None

        detected_lang = detect_language(text)
        final_lang = src_lang or detected_lang or "hi"

        if final_lang not in LANG_MAPPING:
            final_lang = "hi"

        ai4_code = LANG_MAPPING[final_lang]["ai4bharat"]
        sarvam_code = LANG_MAPPING[final_lang].get("sarvam") or "hi-IN"

        # Use IndicProcessor if available, otherwise manual preprocessing
        if INDIC_PROCESSOR_AVAILABLE and ip:
            # Recommended approach using IndicProcessor
            batch = ip.preprocess_batch(
                [text],
                src_lang=ai4_code,
                tgt_lang="eng_Latn"
            )
            inputs = xx2en_tokenizer(
                batch,
                truncation=True,
                padding="longest",
                return_tensors="pt",
                return_attention_mask=True,
            ).to(DEVICE)
        else:
            # Manual approach: prefix with language tags
            tagged_text = f"{ai4_code} eng_Latn {text}"
            inputs = xx2en_tokenizer(
                tagged_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(DEVICE)

        with torch.no_grad():
            outputs = xx2en_model.generate(
                **inputs,
                use_cache=True,
                min_length=0,
                max_length=512,
                num_beams=5,
                num_return_sequences=1,
            )
        
        # Decode translation
        if INDIC_PROCESSOR_AVAILABLE and ip:
            translation = xx2en_tokenizer.batch_decode(
                outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )[0]
            # Post-process with IndicProcessor
            translations = ip.postprocess_batch([translation], lang="eng_Latn")
            translation = translations[0] if translations else translation
        else:
            translation = xx2en_tokenizer.decode(
                outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True
            )

        logger.info(f"Translated '{text[:50]}...' to English: '{translation[:50]}...'")
        return translation, sarvam_code

    except Exception as e:
        logger.error(f"Error during to_english translation: {e}. Falling back to original text.")
        return text, "hi-IN"

def from_english(text: str, target_language_code: str) -> str:
    """
    Translate from English back to target Indic language.
    Uses proper IndicTrans2 preprocessing.
    """
    try:
        ai4_tgt = None
        # find ai4bharat code for Sarvam language code
        for v in LANG_MAPPING.values():
            if v.get("sarvam") == target_language_code:
                ai4_tgt = v["ai4bharat"]
                break
        if not ai4_tgt:
            ai4_tgt = "hin_Deva"  # default to Hindi

        # Use IndicProcessor if available, otherwise manual preprocessing
        if INDIC_PROCESSOR_AVAILABLE and ip:
            # Recommended approach using IndicProcessor
            batch = ip.preprocess_batch(
                [text],
                src_lang="eng_Latn",
                tgt_lang=ai4_tgt
            )
            inputs = en2xx_tokenizer(
                batch,
                truncation=True,
                padding="longest",
                return_tensors="pt",
                return_attention_mask=True,
            ).to(DEVICE)
        else:
            # Manual approach: prefix with language tags
            tagged_text = f"eng_Latn {ai4_tgt} {text}"
            inputs = en2xx_tokenizer(
                tagged_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(DEVICE)

        with torch.no_grad():
            outputs = en2xx_model.generate(
                **inputs,
                use_cache=True,
                min_length=0,
                max_length=256,
                num_beams=5,
                num_return_sequences=1,
            )
        
        # Decode translation
        if INDIC_PROCESSOR_AVAILABLE and ip:
            translation = en2xx_tokenizer.batch_decode(
                outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )[0]
            # Post-process with IndicProcessor
            translations = ip.postprocess_batch([translation], lang=ai4_tgt)
            translation = translations[0] if translations else translation
        else:
            translation = en2xx_tokenizer.decode(
                outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True
            )

        if not translation.strip() or set(translation.strip()) <= {'.'}:
            logger.warning("Translation output invalid or just dots, falling back to English text.")
            translation = text

        logger.info(f"Translated from English to {target_language_code}: {translation[:100]}...")
        return translation

    except Exception as e:
        logger.error(f"Error during from_english translation: {e}. Returning English text fallback.")
        return text
