"""
Robust multilingual Speech-to-Text using Sarvam AI (Sarika v2.5)
Handles Hindi, Marathi, Bengali, Tamil out-of-the-box and is easily extensible.
"""

from __future__ import annotations
import concurrent.futures
import os
import re
import unicodedata
from collections import Counter
from typing import Dict, List

import requests
import logging
from aiman.config import SARVAM_API_KEY

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

STT_API_URL = "https://api.sarvam.ai/speech-to-text"
STT_HEADERS = {"api-subscription-key": SARVAM_API_KEY}

# Languages you want to support – just append more BCP-47 codes here.
PRIMARY_LANGS: List[str] = ["hi-IN", "mr-IN", "bn-IN", "ta-IN"]

# Region keywords that can give a small boost if they appear in the text
REGION_HINTS: Dict[str, List[str]] = {
    "महाराष्ट्र": ["mr-IN", "hi-IN"],
    "বাংলা": ["bn-IN"],
    "বঙ্গ": ["bn-IN"],
    "தமிழ்": ["ta-IN"],
    "हिन्दी": ["hi-IN"],
}

# ----------------------------------------------------------------------
# Public entry-point
# ----------------------------------------------------------------------

def detect_and_transcribe(wav_file_path: str) -> Dict[str, object]:
    """
    Returns a dict with keys: text, language, confidence, alternatives.
    If the best candidate scores below 0.30, text == "" to signal low confidence.
    """
    candidates = _get_candidates_parallel(wav_file_path, PRIMARY_LANGS)
    if not candidates:
        return {"text": "", "language": "hi-IN", "confidence": 0.0}

    best = max(candidates, key=lambda c: c["score"])

    # If truly unsure, return empty so webhook asks user to repeat
    if best["score"] < 0.30 or not best["text"].strip():
        return {"text": "", "language": best["language"], "confidence": best["score"]}

    logger.info(
        "Chosen transcript (%s, conf %.2f): %s",
        best["language"],
        best["score"],
        best["text"][:80] + ("…" if len(best["text"]) > 80 else ""),
    )

    # Trim alternatives to top-3 for logging/debug
    best_alts = sorted(candidates, key=lambda c: c["score"], reverse=True)[:3]
    return {
        "text": best["text"],
        "language": best["language"],
        "confidence": best["score"],
        "alternatives": best_alts,
    }

# ----------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------

def _get_candidates_parallel(wav_path: str, langs: List[str]) -> List[Dict]:
    """Fire off parallel requests so we pay only the cost of the slowest call."""
    results: List[Dict] = []

    def _one(lang_code: str) -> Dict:
        try:
            return _transcribe(wav_path, lang_code)
        except Exception as e:
            logger.warning("STT failure for %s: %s", lang_code, e)
            return {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(langs)) as pool:
        futures = [pool.submit(_one, lc) for lc in langs]
        for fut in concurrent.futures.as_completed(futures):
            if fut.result():
                results.append(fut.result())

    # Post-process to attach a composite score
    for cand in results:
        cand["score"] = _composite_score(cand)

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
# Scoring – combine multiple weak signals into a robust metric
# ----------------------------------------------------------------------

def _composite_score(cand: Dict) -> float:
    """
    Combine Sarvam's own confidence with text length, script match,
    naturalness and region hints into a single 0-1 score.
    Weights can be tuned; current weights work well for 4–8 languages.
    """
    text = cand["text"]
    lang = cand["language"].split("-")[0]

    if not text:
        return 0.0

    # 1. API confidence (already 0-1)
    api_conf = cand.get("sarvam_conf", 0.0)

    # 2. Length factor: longer transcriptions are generally more reliable
    length_factor = min(len(text) / 30.0, 1.0)  # 0-1 over 0-30 chars

    # 3. Script consistency
    script_score = _script_match_ratio(text, lang)

    # 4. Word naturalness (simple n-gram keyword hits)
    naturalness = _naturalness_score(text, lang)

    # 5. Region keywords boost
    region_boost = any(k in text for k in REGION_HINTS if cand["language"] in REGION_HINTS[k])
    region_score = 0.1 if region_boost else 0.0

    # Weighted sum
    return min(
        1.0,
        0.45 * api_conf
        + 0.15 * length_factor
        + 0.15 * script_score
        + 0.20 * naturalness
        + region_score,
    )


def _script_match_ratio(text: str, lang: str) -> float:
    """Return fraction of characters matching expected script for the language."""
    ranges = {
        "hi": ("\u0900", "\u097F"),  # Devanagari
        "mr": ("\u0900", "\u097F"),
        "bn": ("\u0980", "\u09FF"),
        "ta": ("\u0B80", "\u0BFF"),
    }
    if lang not in ranges:
        return 0.5
    lo, hi = ranges[lang]
    total = sum(1 for c in text if "\u0900" <= c <= "\u0FFF")  # any Indic block
    match = sum(1 for c in text if lo <= c <= hi)
    return match / total if total else 0.0


_KEYWORDS = {
    "hi": ["है", "में", "क्या", "कैसे"],
    "mr": ["आहे", "कसा", "काय", "मध्ये"],
    "bn": ["কেন", "কী", "কিভাবে", "আছে"],
    "ta": ["எப்படி", "என்ன", "உடன்"],
}


def _naturalness_score(text: str, lang: str) -> float:
    """Crude but fast keyword-hit ratio for language-specific common words."""
    words = re.findall(r"\w+", text)
    if not words or lang not in _KEYWORDS:
        return 0.5
    hits = sum(1 for w in words for k in _KEYWORDS[lang] if k in w)
    return min(hits / len(words), 1.0)
