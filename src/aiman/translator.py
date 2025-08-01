# translator.py

import os
import torch
import logging
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from langdetect import detect_langs, DetectorFactory
import logging
import google.generativeai as genai

# Configure Gemini API key
genai.configure(api_key="GEMINI_API_KEY")  # 🔁 Replace with your real API key

model = genai.GenerativeModel('gemini-1.5-flash')

try:
    from IndicTransToolkit.processor import IndicProcessor
    INDIC_PROCESSOR_AVAILABLE = True
except ImportError:
    try:
        from IndicTransTokenizer import IndicProcessor
        INDIC_PROCESSOR_AVAILABLE = True
    except ImportError:
        INDIC_PROCESSOR_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DetectorFactory.seed = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {DEVICE}")

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

XX2EN_MODEL = "ai4bharat/indictrans2-indic-en-1B"
EN2XX_MODEL = "ai4bharat/indictrans2-en-indic-1B"

xx2en_tokenizer = AutoTokenizer.from_pretrained(XX2EN_MODEL, trust_remote_code=True)
xx2en_model = AutoModelForSeq2SeqLM.from_pretrained(XX2EN_MODEL, trust_remote_code=True).to(DEVICE)

en2xx_tokenizer = AutoTokenizer.from_pretrained(EN2XX_MODEL, trust_remote_code=True)
en2xx_model = AutoModelForSeq2SeqLM.from_pretrained(EN2XX_MODEL, trust_remote_code=True).to(DEVICE)

ip = IndicProcessor(inference=True) if INDIC_PROCESSOR_AVAILABLE else None
if ip:
    logger.info("IndicProcessor initialized")
else:
    logger.warning("IndicProcessor not found. Using fallback preprocessing.")


def detect_language(text: str) -> str:
    try:
        detections = detect_langs(text)
        for det in detections:
            if det.lang in LANG_MAPPING and det.prob > 0.85:
                return det.lang
        for det in detections:
            if det.lang in LANG_MAPPING:
                return det.lang
    except Exception as e:
        logger.warning(f"Language detection failed: {e}")
    return "hi"  # fallback


def safe_generate(model, tokenizer, inputs, max_len=512):
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            use_cache=True,
            min_length=0,
            max_length=max_len,
            num_beams=5,
            num_return_sequences=1,
        )
    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    if not decoded.strip() or set(decoded.strip()) <= {'.'}:
        return ""
    return decoded.strip()


def to_english(text: str, source_lang_code: str = None) -> tuple[str, str]:
    try:
        lang = source_lang_code.split("-")[0] if source_lang_code else detect_language(text)
        ai4_code = LANG_MAPPING.get(lang, LANG_MAPPING["hi"])["ai4bharat"]
        sarvam_code = LANG_MAPPING.get(lang, LANG_MAPPING["hi"])["sarvam"] or "hi-IN"

        prompt = f"Translate the following text to English:\n\n'{text}'"

        import google.generativeai as genai
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)

        translation = response.text.strip() if response and response.text else ""

        if not translation:
            logger.warning("Empty or invalid translation. Returning fallback.")
            return text, sarvam_code

        logger.info(f"Translated to English: {translation}")
        return translation, sarvam_code

    except Exception as e:
        logger.error(f"to_english error: {e}")
        return text, "hi-IN"


def from_english(text: str, target_language_code: str) -> str:
    try:
        ai4_tgt = None
        for v in LANG_MAPPING.values():
            if v.get("sarvam") == target_language_code:
                ai4_tgt = v["ai4bharat"]
                break
        if not ai4_tgt:
            ai4_tgt = "hin_Deva"

        if ip:
            batch = ip.preprocess_batch([text], src_lang="eng_Latn", tgt_lang=ai4_tgt)
            inputs = en2xx_tokenizer(batch, padding=True, return_tensors="pt").to(DEVICE)
        else:
            tagged = f"eng_Latn {ai4_tgt} {text}"
            inputs = en2xx_tokenizer(tagged, return_tensors="pt", padding=True).to(DEVICE)

        translation = safe_generate(en2xx_model, en2xx_tokenizer, inputs, max_len=256)
        if not translation:
            logger.warning("Empty translation. Returning fallback.")
            return text

        if ip:
            translation = ip.postprocess_batch([translation], lang=ai4_tgt)[0]

        logger.info(f"Translated from English to {target_language_code}: {translation}")
        return translation

    except Exception as e:
        logger.error(f"from_english error: {e}")
        return text