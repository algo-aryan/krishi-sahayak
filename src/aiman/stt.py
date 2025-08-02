from __future__ import annotations
import concurrent.futures
import os
import re
import math
from collections import Counter
from typing import Dict, List

import requests
import logging
# from aiman.config import SARVAM_API_KEY # Assuming this is in your project structure

# --- Mock SARVAM_API_KEY for standalone execution ---
SARVAM_API_KEY = os.environ.get("SARVAM_API_KEY", "YOUR_API_KEY_HERE")
if SARVAM_API_KEY == "YOUR_API_KEY_HERE":
    print("Warning: SARVAM_API_KEY not set. Using a placeholder.")
# ----------------------------------------------------


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------


STT_API_URL = "https://api.sarvam.ai/speech-to-text"
STT_HEADERS = {"api-subscription-key": SARVAM_API_KEY}


# Languages you want to support
PRIMARY_LANGS: List[str] = ["hi-IN", "mr-IN", "bn-IN", "ta-IN", "kn-IN"]


# ----------------------------------------------------------------------
# Public entry-point (No changes to signature or return values)
# ----------------------------------------------------------------------


def detect_and_transcribe(wav_file_path: str) -> Dict[str, object]:
    """
    Returns a dict with keys: text, language, confidence, alternatives.
    If the best candidate scores below 0.30, text == "" to signal low confidence.
    """
    candidates = _get_candidates_parallel(wav_file_path, PRIMARY_LANGS)
    if not candidates:
        return {"text": "", "language": "hi-IN", "confidence": 0.0, "alternatives": []}

    # The chosen candidate is now based on the robust composite score
    best = max(candidates, key=lambda c: c["score"])

    if best["score"] < 0.30 or not best["text"].strip():
        return {"text": "", "language": best["language"], "confidence": best["score"], "alternatives": []}

    logger.info(
        "Chosen transcript (%s, conf %.2f): %s",
        best["language"],
        best["score"],
        best["text"][:80] + ("…" if len(best["text"]) > 80 else ""),
    )

    best_alts = sorted(candidates, key=lambda c: c["score"], reverse=True)[:3]
    return {
        # IMPORTANT: Return the text from the best candidate, which might be different
        # from the reference text if the API returned different strings.
        "text": best["text"],
        "language": best["language"],
        "confidence": best["score"],
        "alternatives": best_alts,
    }


# ----------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------


def _get_candidates_parallel(wav_path: str, langs: List[str]) -> List[Dict]:
    """
    Fires parallel requests and uses a single reference text for robust scoring.
    """
    results: List[Dict] = []

    def _one(lang_code: str) -> Dict | None:
        try:
            return _transcribe(wav_path, lang_code)
        except Exception as e:
            logger.warning("STT failure for %s: %s", lang_code, e)
            return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(langs)) as pool:
        futures = [pool.submit(_one, lc) for lc in langs]
        for fut in concurrent.futures.as_completed(futures):
            result = fut.result()
            if result and result.get("text"):
                results.append(result)

    if not results:
        return []

    # **NEW LOGIC**: Find the best transcription text to use as a reference
    # We assume the one with the highest API confidence is the most phonetically accurate
    reference_candidate = max(results, key=lambda c: c.get("sarvam_conf", 0.0))
    reference_text = reference_candidate.get("text", "")

    if not reference_text:
        return []

    # **NEW LOGIC**: Re-score every candidate using the same reference text
    for cand in results:
        cand["score"] = _composite_score(cand, reference_text)

    return results


def _transcribe(wav_path: str, lang_code: str) -> Dict:
    """Call Sarvam STT for a single language hint."""
    with open(wav_path, "rb") as fh:
        files = {"file": ("audio.wav", fh, "audio/wav")}
        data = {"language_code": lang_code, "model": "saarika:v2.5", "with_timestamps": "false"}
        r = requests.post(STT_API_URL, headers=STT_HEADERS, files=files, data=data, timeout=40)
        r.raise_for_status()
        js = r.json()

    return {
        "text": js.get("transcript", "").strip(),
        "language": lang_code,
        "sarvam_conf": float(js.get("confidence", 0.0)),
    }


# ----------------------------------------------------------------------
# Scoring – Now compares against a single reference text
# ----------------------------------------------------------------------

def _composite_score(candidate: Dict, reference_text: str) -> float:
    """
    Calculates a composite score for a language candidate.
    Crucially, it uses a single `reference_text` for all text-based analysis
    to ensure a fair, apples-to-apples comparison.
    """
    lang = candidate.get("language", "").split("-")[0]
    if not lang:
        return 0.0

    # 1. API confidence (still from the original candidate)
    api_conf = candidate.get("sarvam_conf", 0.0)

    # 2. Length factor (based on the reference text)
    length_factor = min(len(reference_text) / 30.0, 1.0)

    # 3. Script consistency (based on the reference text)
    script_score = _script_match_ratio(reference_text, lang)

    # 4. Character Frequency Score (based on the reference text)
    char_freq_score = _char_frequency_score(reference_text, lang)

    # Weighted sum remains the same
    final_score = (
        0.25 * api_conf
        + 0.05 * length_factor
        + 0.45 * script_score
        + 0.25 * char_freq_score
    )

    return min(final_score, 1.0)


def _script_match_ratio(text: str, lang: str) -> float:
    """Return fraction of Indic characters matching the expected script for the language."""
    ranges = {
        "hi": ("\u0900", "\u097F"), "mr": ("\u0900", "\u097F"),
        "bn": ("\u0980", "\u09FF"), "ta": ("\u0B80", "\u0BFF"),
        "kn": ("\u0C80", "\u0CFF"),
    }
    if lang not in ranges: return 0.5
    lo, hi = ranges[lang]
    clean_text = "".join(filter(str.isalpha, text))
    if not clean_text: return 0.0
    indic_chars = [c for c in clean_text if "\u0900" <= c <= "\u0D7F"]
    total = len(indic_chars)
    if total == 0: return 0.5
    match = sum(1 for c in indic_chars if lo <= c <= hi)
    return match / total


# ----------------------------------------------------------------------
# Character Frequency Analysis (Now more impactful)
# ----------------------------------------------------------------------

_CHAR_LOG_FREQUENCIES = {
    'hi': {'क': -2.9, 'ख': -5.1, 'ग': -4.0, 'घ': -6.0, 'च': -4.8, 'छ': -6.2, 'ज': -3.9, 'झ': -6.8, 'ट': -5.5, 'ठ': -6.7, 'ड': -5.4, 'ढ': -7.1, 'ण': -5.3, 'त': -3.0, 'थ': -4.7, 'द': -3.7, 'ध': -5.2, 'न': -2.8, 'प': -3.5, 'फ': -6.1, 'ब': -3.8, 'भ': -4.4, 'म': -3.0, 'य': -3.7, 'र': -2.7, 'ल': -3.3, 'व': -3.9, 'श': -4.5, 'ष': -5.8, 'स': -3.2, 'ह': -3.1, 'ा': -2.1, 'ि': -2.8, 'ी': -2.9, 'ु': -3.9, 'ू': -5.0, 'े': -2.6, 'ै': -4.3, 'ो': -3.7, 'ौ': -5.3, 'ं': -3.3, '्': -3.2, 'अ': -4.0, 'आ': -4.1, 'इ': -4.9, 'ए': -4.5},
    'mr': {'क': -3.0, 'ख': -5.4, 'ग': -4.2, 'घ': -6.2, 'च': -4.5, 'छ': -6.5, 'ज': -4.0, 'झ': -7.0, 'ट': -5.2, 'ठ': -6.5, 'ड': -5.6, 'ढ': -6.8, 'ण': -4.8, 'त': -3.1, 'थ': -4.9, 'द': -4.0, 'ध': -5.5, 'न': -3.0, 'प': -3.7, 'फ': -6.4, 'ब': -4.1, 'भ': -4.6, 'म': -3.1, 'य': -3.8, 'र': -2.8, 'ल': -3.4, 'व': -3.6, 'श': -4.7, 'ष': -5.9, 'स': -3.5, 'ह': -3.6, 'ा': -2.2, 'ि': -2.9, 'ी': -3.0, 'ु': -4.0, 'ू': -5.2, 'े': -2.8, 'ै': -4.8, 'ो': -4.0, 'ौ': -5.8, 'ं': -3.1, '्': -3.4, 'ळ': -4.9, 'आ': -4.2, 'अ':-3.9, 'ए': -4.7},
    'bn': {'ক': -3.3, 'খ': -5.5, 'গ': -4.5, 'ঘ': -6.5, 'চ': -4.7, 'ছ': -6.0, 'জ': -4.2, 'ঝ': -7.2, 'ট': -5.2, 'ঠ': -6.8, 'ড': -5.3, 'ঢ': -7.5, 'ণ': -5.8, 'ত': -3.5, 'থ': -5.0, 'দ': -4.3, 'ধ': -5.7, 'ন': -3.2, 'প': -4.0, 'ফ': -6.2, 'ব': -3.8, 'ভ': -5.1, 'ম': -3.6, 'য': -4.8, 'র': -2.9, 'ল': -3.7, 'শ': -4.6, 'ষ': -6.7, 'স': -3.9, 'হ': -4.1, 'া': -2.4, 'ি': -2.9, 'ী': -3.8, 'ু': -4.2, 'ূ': -6.1, 'ে': -2.8, 'ৈ': -5.4, 'ো': -3.9, 'ৌ': -6.3, 'ং': -4.9, '্': -3.1, 'য়': -4.4},
    'ta': {'க': -3.3, 'ங': -7.8, 'ச': -5.0, 'ஞ': -7.2, 'ட': -4.7, 'ண': -4.3, 'த': -3.6, 'ந': -3.7, 'ப': -4.1, 'ம': -3.4, 'ய': -4.0, 'ர': -3.8, 'ல': -3.5, 'வ': -3.9, 'ழ': -5.6, 'ள': -4.5, 'ற': -4.8, 'ன': -3.9, '்': -2.5, 'ா': -3.0, 'ி': -3.1, 'ீ': -4.2, 'ு': -3.2, 'ூ': -5.3, 'ெ': -3.5, 'ே': -3.6, 'ை': -4.0, 'ொ': -6.5, 'ோ': -4.9, 'ௌ': -6.8},
    'kn': {'ಕ': -3.1, 'ಖ': -6.3, 'ಗ': -4.6, 'ಘ': -7.5, 'ಚ': -4.8, 'ಛ': -7.2, 'ಜ': -5.0, 'ಝ': -8.0, 'ಟ': -6.0, 'ಠ': -7.8, 'ಡ': -5.2, 'ಢ': -7.6, 'ಣ': -5.5, 'ತ': -4.0, 'ಥ': -6.5, 'ದ': -3.5, 'ಧ': -5.8, 'ನ': -3.3, 'ಪ': -4.5, 'ಫ': -7.3, 'ಬ': -4.7, 'ಭ': -5.9, 'ಮ': -3.8, 'ಯ': -4.2, 'ರ': -3.2, 'ಲ': -3.9, 'ವ': -3.7, 'ಶ': -5.3, 'ಷ': -6.2, 'ಸ': -3.6, 'ಹ': -4.4, 'ಳ': -5.1, '್': -3.0, 'ಾ': -2.8, 'ಿ': -3.4, 'ೀ': -4.5, 'ು': -3.9, 'ೂ': -5.7, 'ೆ': -3.8, 'ೇ': -4.1, 'ೈ': -5.4, 'ೊ': -6.0, 'ೋ': -4.9, 'ೌ': -6.7}
}

_ALL_KNOWN_CHARS = {char for lang_map in _CHAR_LOG_FREQUENCIES.values() for char in lang_map}


def _char_frequency_score(text: str, lang: str) -> float:
    """
    Scores text based on character frequency, with heavy penalties for "exclusive"
    characters from other languages.
    """
    if lang not in _CHAR_LOG_FREQUENCIES:
        return 0.5

    freq_map = _CHAR_LOG_FREQUENCIES[lang]
    
    known_chars_in_text = [c for c in text if c in _ALL_KNOWN_CHARS]
    if not known_chars_in_text:
        return 0.2

    log_prob_sum = 0
    
    for char in known_chars_in_text:
        if char in freq_map:
            log_prob_sum += freq_map[char]
        else:
            # HEAVY PENALTY: Character is known in other languages but NOT this one.
            log_prob_sum -= 15.0

    avg_log_prob = log_prob_sum / len(known_chars_in_text)

    score = (avg_log_prob - (-10.0)) / (-3.0 - (-10.0))
    
    return max(0.0, min(1.0, score))