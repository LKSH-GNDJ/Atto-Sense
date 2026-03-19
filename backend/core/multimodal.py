"""
AttoSense v4 - Multimodal Input Handler
=======================================
Handles all input-specific preprocessing (audio transcription, image resizing,
OCR fallback, language detection, filler-word removal) then delegates NLU
classification to nlu_pipeline.classify().

Audio pipeline:  raw bytes → quality gate → Whisper → clean transcript → pipeline
Vision pipeline: raw bytes → resize → LLM vision → JSON → optional OCR → pipeline
Text pipeline:   raw string → condense if long → language detect → pipeline
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import re
import time
from datetime import datetime, timezone
from typing import Optional

import httpx
from groq import Groq, RateLimitError, APIStatusError, APITimeoutError
from tenacity import (
    retry, stop_after_attempt, wait_exponential,
    retry_if_exception_type, before_sleep_log,
)

from backend.models.schemas import (
    NLUResult, Entity, IntentCategory, VisionAnalysis, ErrorType,
    TranscriptionResult,
)
from backend.core.logging_config import get_logger
from backend.core.calibration import calibrate_confidence, is_low_confidence
from backend.core.nlu_pipeline import classify as pipeline_classify, example_store

log = get_logger("attosense.multimodal")

# ── Constants ──────────────────────────────────────────────────────────────────
TEXT_MODEL     = "llama-3.3-70b-versatile"
VISION_MODEL   = "meta-llama/llama-4-scout-17b-16e-instruct"
AUDIO_MODEL    = "whisper-large-v3"
GROQ_TIMEOUT_S = 45.0
MAX_RETRIES    = 3

MIN_AUDIO_BYTES      = 4_096
MIN_AUDIO_DURATION_S = 1.5
MAX_NO_SPEECH_PROB   = 0.80
MIN_AVG_LOGPROB      = -1.00
MAX_IMAGE_DIMENSION  = 1200
LONG_INPUT_THRESHOLD = 800    # chars above which text is condensed

ERROR_TYPES = [e.value for e in ErrorType]

# ── [O] Circuit Breaker ────────────────────────────────────────────────────────
class _CircuitBreaker:
    FAILURE_THRESHOLD = 5
    RECOVERY_SECONDS  = 30
    def __init__(self):
        self._failures = 0
        self._opened_at: Optional[float] = None
    @property
    def is_open(self):
        if self._opened_at is None: return False
        if time.monotonic() - self._opened_at >= self.RECOVERY_SECONDS:
            log.info("circuit_breaker_half_open"); return False
        return True
    def record_success(self): self._failures = 0; self._opened_at = None
    def record_failure(self):
        self._failures += 1
        if self._failures >= self.FAILURE_THRESHOLD and self._opened_at is None:
            self._opened_at = time.monotonic()
            log.warning("circuit_breaker_opened", extra={"failures": self._failures})
    def raise_if_open(self):
        if self.is_open:
            r = self.RECOVERY_SECONDS - (time.monotonic() - self._opened_at)
            raise RuntimeError(f"Groq temporarily unavailable. Retry in {r:.0f}s.")

_breaker = _CircuitBreaker()

# ── Groq Client ────────────────────────────────────────────────────────────────
_groq_client: Optional[Groq] = None
def get_groq_client() -> Groq:
    global _groq_client
    if _groq_client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key: raise EnvironmentError("GROQ_API_KEY is not set.")
        _groq_client = Groq(
            api_key=api_key,
            timeout=httpx.Timeout(GROQ_TIMEOUT_S, connect=10.0),
            max_retries=0,
        )
        log.info("groq_client_initialized", extra={"timeout_s": GROQ_TIMEOUT_S})
    return _groq_client

# ── Retry policy ───────────────────────────────────────────────────────────────
_RETRYABLE = (RateLimitError, APIStatusError, APITimeoutError, httpx.TimeoutException)
_retry_policy = retry(
    retry=retry_if_exception_type(_RETRYABLE),
    stop=stop_after_attempt(MAX_RETRIES),
    wait=wait_exponential(multiplier=1, min=2, max=20),
    before_sleep=before_sleep_log(log, log.level if hasattr(log, "level") else 20),
    reraise=True,
)

# ── Sync workers (injected into pipeline as callables) ─────────────────────────
@_retry_policy
def _sync_chat(messages: list, max_tokens: int = 800) -> str:
    _breaker.raise_if_open()
    try:
        r = get_groq_client().chat.completions.create(
            model=TEXT_MODEL, messages=messages,
            temperature=0.05, max_tokens=max_tokens,
        ).choices[0].message.content
        _breaker.record_success(); return r
    except Exception: _breaker.record_failure(); raise

@_retry_policy
def _sync_chat_temp(messages: list, max_tokens: int = 400, temperature: float = 0.15) -> str:
    """Variable-temperature variant used by Stage 3 ensemble."""
    _breaker.raise_if_open()
    try:
        r = get_groq_client().chat.completions.create(
            model=TEXT_MODEL, messages=messages,
            temperature=temperature, max_tokens=max_tokens,
        ).choices[0].message.content
        _breaker.record_success(); return r
    except Exception: _breaker.record_failure(); raise

@_retry_policy
def _sync_whisper(audio_bytes: bytes, mime_type: str, ext: str,
                  language: Optional[str], temperature: float = 0.0):
    _breaker.raise_if_open()
    try:
        r = get_groq_client().audio.transcriptions.create(
            model=AUDIO_MODEL,
            file=(f"audio.{ext}", audio_bytes, mime_type),
            language=language,
            response_format="verbose_json",
            temperature=temperature,
        )
        _breaker.record_success(); return r
    except Exception: _breaker.record_failure(); raise

# ── Text helpers ───────────────────────────────────────────────────────────────
_FILLER_RE = re.compile(
    r"\b(um+|uh+|er+|ah+|hmm+|like,?|you know,?|so,?|basically,?|literally,?|right,?)\b",
    re.IGNORECASE,
)
def _clean_transcript(text: str) -> str:
    return re.sub(r"\s{2,}", " ", _FILLER_RE.sub(" ", text)).strip()

_SIGNAL_KEYWORDS = re.compile(
    r"\b(invoice|charge|charged|bill|billing|refund|payment|paid|amount|"
    r"error|crash|bug|broken|fail|issue|problem|not working|"
    r"account|password|login|email|reset|access|"
    r"plan|upgrade|pricing|quote|enterprise|"
    r"complaint|unacceptable|terrible|furious|frustrated|"
    r"manager|escalate|legal|bank|dispute|"
    r"order|ticket|case|reference|inv[-]?\\d*|ord[-]?\\d*)\b",
    re.IGNORECASE,
)

def _condense_long_input(text: str) -> str:
    """Extract key intent-bearing sentences from long emails/tickets."""
    if len(text) <= LONG_INPUT_THRESHOLD:
        return text
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]
    if len(sentences) <= 6:
        return text
    first_3  = sentences[:3]
    last_3   = sentences[-3:]
    signal   = [s for s in sentences[3:-3] if _SIGNAL_KEYWORDS.search(s)]
    seen, unique = set(), []
    for s in first_3 + signal + last_3:
        if s not in seen: seen.add(s); unique.append(s)
    condensed = " ".join(unique)
    log.debug("input_condensed", extra={
        "original_chars": len(text), "condensed_chars": len(condensed),
        "sentences_kept": len(unique), "total": len(sentences),
    })
    return f"[Long email/ticket — key sentences extracted]\n{condensed}"

def _detect_language(text: str, whisper_lang: Optional[str] = None) -> Optional[str]:
    if whisper_lang and whisper_lang.lower() not in ("en", "english", ""):
        return whisper_lang
    try:
        from langdetect import detect
        lang = detect(text[:500])
        return lang if lang != "en" else None
    except Exception: return None

def _language_prefix(lang: Optional[str]) -> str:
    if not lang: return ""
    names = {"es":"Spanish","fr":"French","de":"German","ar":"Arabic","pt":"Portuguese",
             "zh-cn":"Chinese","ja":"Japanese","hi":"Hindi","it":"Italian",
             "ru":"Russian","ko":"Korean","tr":"Turkish"}
    name = names.get(lang, lang.upper())
    return f"NOTE: Customer is writing in {name}. Classify intent regardless. Respond JSON only.\n\n"

# ── Audio quality gate ─────────────────────────────────────────────────────────
class AudioQualityError(ValueError): pass

def _check_audio_quality(audio_bytes: bytes, resp=None) -> None:
    if len(audio_bytes) < MIN_AUDIO_BYTES:
        raise AudioQualityError(
            f"Audio too small ({len(audio_bytes)} bytes). Record at least 2 seconds."
        )
    if resp is None: return
    dur = getattr(resp, "duration", None)
    if dur is not None and float(dur) < MIN_AUDIO_DURATION_S:
        raise AudioQualityError(f"Audio too short ({float(dur):.1f}s).")
    segs = getattr(resp, "segments", None) or []
    if segs:
        avg_nsp = sum(s.get("no_speech_prob", 0) for s in segs) / len(segs)
        avg_lp  = sum(s.get("avg_logprob",    0) for s in segs) / len(segs)
        if avg_nsp > MAX_NO_SPEECH_PROB:
            raise AudioQualityError(f"Audio appears silent (no_speech_prob={avg_nsp:.2f}).")
        if avg_lp < MIN_AVG_LOGPROB:
            raise AudioQualityError(f"Transcription quality too low (avg_logprob={avg_lp:.2f}).")

# ── Image preprocessing ────────────────────────────────────────────────────────
def _preprocess_image(image_b64: str, mime_type: str) -> tuple[str, str]:
    try:
        from PIL import Image
        img = Image.open(io.BytesIO(base64.b64decode(image_b64)))
        w, h = img.size
        if max(w, h) <= MAX_IMAGE_DIMENSION: return image_b64, mime_type
        scale = MAX_IMAGE_DIMENSION / max(w, h)
        img   = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        buf   = io.BytesIO()
        fmt   = "JPEG" if "jpeg" in mime_type or "jpg" in mime_type else "PNG"
        img.save(buf, format=fmt, quality=90)
        log.debug("image_resized", extra={"from": f"{w}x{h}", "to": f"{int(w*scale)}x{int(h*scale)}"})
        return base64.b64encode(buf.getvalue()).decode(), f"image/{'jpeg' if fmt=='JPEG' else 'png'}"
    except Exception as e:
        log.warning("image_preprocess_failed", extra={"error": str(e)})
        return image_b64, mime_type

# ── OCR fallback ───────────────────────────────────────────────────────────────
def _ocr_extract(image_b64: str) -> Optional[str]:
    try:
        import pytesseract; from PIL import Image
        text = pytesseract.image_to_string(
            Image.open(io.BytesIO(base64.b64decode(image_b64)))
        ).strip()
        return text if len(text) > 20 else None
    except Exception: return None

# ── Vision JSON parser ─────────────────────────────────────────────────────────
def _parse_json(raw: str) -> dict:
    cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()
    try: return json.loads(cleaned)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if m: return json.loads(m.group())
        raise ValueError(f"Cannot parse JSON: {raw[:200]}")

_VISION_SYSTEM = (
    "You are AttoSense Vision. Analyse the screenshot/photo.\n"
    "Extract ALL visible text. Identify errors, dialogs, broken UI.\n\n"
    'Return ONLY JSON:\n{\n'
    '  "intent_primary": one of ' + str([e.value for e in IntentCategory]) + ',\n'
    '  "confidence": float 0.0-1.0,\n'
    '  "sentiment": one of ["positive","neutral","negative","frustrated"],\n'
    '  "frustration_score": float 0.0-1.0,\n'
    '  "error_type": one of ' + str(ERROR_TYPES) + ',\n'
    '  "error_detail": string or null,\n'
    '  "visual_summary": string (one sentence),\n'
    '  "screen_type": one of ["error_page","login","dashboard","payment","settings","chat","other"]\n'
    '}\n\n'
    "Frustration: 0.0-0.2 calm  0.3-0.5 minor  0.6-0.8 error  0.9-1.0 critical\n"
    "HTTP 4xx/5xx, crash → technical_support\n"
    "Invoice, payment declined → billing\n"
    "Login, profile settings → account_management\n"
    "Pricing page, plan selector → sales_inquiry\n"
    "Rage text, caps lock → complaint or escalation"
)


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC ASYNC PIPELINES
# ══════════════════════════════════════════════════════════════════════════════

async def classify_text_async(
    message: str,
    context: Optional[list[dict]] = None,
) -> tuple[NLUResult, str, float]:
    t0       = time.perf_counter()
    text_in  = _condense_long_input(message)
    lang     = _detect_language(message)
    lang_pfx = _language_prefix(lang)

    # Prepend language note to the condensed text if needed
    classify_text = lang_pfx + text_in if lang_pfx else text_in

    result = await pipeline_classify(
        text=classify_text,
        modality="text",
        language=lang,
        sync_chat=_sync_chat,
        sync_chat_temp=_sync_chat_temp,
    )
    latency_ms = (time.perf_counter() - t0) * 1000
    return result, TEXT_MODEL, latency_ms


async def classify_audio_async(
    audio_b64: str,
    mime_type: str = "audio/wav",
    language: Optional[str] = None,
) -> tuple[NLUResult, str, float]:
    ext_map     = {"audio/wav":"wav","audio/mp3":"mp3","audio/ogg":"ogg","audio/webm":"webm"}
    ext         = ext_map.get(mime_type, "wav")
    audio_bytes = base64.b64decode(audio_b64)

    _check_audio_quality(audio_bytes)

    t0 = time.perf_counter()
    # Temperature tuning: shorter clips need more temperature diversity
    whisper_temp = 0.2 if len(audio_bytes) / 32_000 < 10 else 0.0

    try:
        resp = await asyncio.wait_for(
            asyncio.to_thread(_sync_whisper, audio_bytes, mime_type, ext, language, whisper_temp),
            timeout=GROQ_TIMEOUT_S,
        )
    except asyncio.TimeoutError:
        raise TimeoutError(f"Whisper call exceeded {GROQ_TIMEOUT_S}s.")

    _check_audio_quality(audio_bytes, resp)

    raw_transcript = getattr(resp, "text", str(resp)).strip()
    whisper_lang   = getattr(resp, "language", None)
    clean          = _clean_transcript(raw_transcript)
    lang           = _detect_language(clean, whisper_lang)
    lang_pfx       = _language_prefix(lang)
    classify_text  = lang_pfx + clean if lang_pfx else clean

    result = await pipeline_classify(
        text=classify_text,
        modality="audio",
        language=lang,
        sync_chat=_sync_chat,
        sync_chat_temp=_sync_chat_temp,
    )
    # Preserve raw transcript in the result
    result = result.model_copy(update={"raw_transcript": raw_transcript})
    latency_ms = (time.perf_counter() - t0) * 1000
    return result, f"{AUDIO_MODEL} → {TEXT_MODEL}", latency_ms


async def classify_vision_async(
    image_b64: str,
    mime_type: str = "image/jpeg",
    caption: Optional[str] = None,
) -> tuple[NLUResult, str, float]:
    image_b64, mime_type = _preprocess_image(image_b64, mime_type)

    t0 = time.perf_counter()
    user_prompt = (
        "Step 1 — Extract ALL visible text.\n"
        "Step 2 — Identify errors, dialogs, warnings, broken UI.\n"
        "Step 3 — Customer caption (weigh highly): "
        + (f'"{caption}"' if caption else "(none)") + "\n"
        "Step 4 — Return ONLY the JSON object."
    )
    msgs = [
        {"role": "system", "content": _VISION_SYSTEM},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_b64}"}},
            {"type": "text", "text": user_prompt},
        ]},
    ]
    try:
        raw = await asyncio.wait_for(
            asyncio.to_thread(
                lambda: get_groq_client().chat.completions.create(
                    model=VISION_MODEL, messages=msgs,
                    temperature=0.05, max_tokens=600,
                ).choices[0].message.content
            ),
            timeout=GROQ_TIMEOUT_S,
        )
        vision_parsed = _parse_json(raw)
    except Exception as exc:
        log.warning("vision_parse_failed", extra={"error": str(exc)})
        vision_parsed = {}

    # Build VisionAnalysis
    try:
        etype = ErrorType(vision_parsed.get("error_type", "none"))
    except ValueError:
        etype = ErrorType.NONE
    vision_analysis = VisionAnalysis(
        frustration_score=max(0.0, min(1.0, float(vision_parsed.get("frustration_score", 0.0)))),
        error_type=etype,
        error_detail=vision_parsed.get("error_detail"),
        visual_summary=vision_parsed.get("visual_summary"),
        screen_type=vision_parsed.get("screen_type"),
    )

    # Text for NLU pipeline: caption + visual_summary + extracted error detail
    nlu_parts = []
    if caption:          nlu_parts.append(caption)
    vs = vision_parsed.get("visual_summary")
    if vs:               nlu_parts.append(vs)
    ed = vision_parsed.get("error_detail")
    if ed:               nlu_parts.append(f"Error: {ed}")
    nlu_text = ". ".join(nlu_parts) if nlu_parts else "Screenshot with no caption provided."

    # OCR fallback if vision confidence is low and image looks text-heavy
    vision_conf  = float(vision_parsed.get("confidence", 0.65))
    frust_score  = vision_analysis.frustration_score
    if vision_conf < 0.70 and frust_score < 0.4:
        ocr_text = _ocr_extract(image_b64)
        if ocr_text:
            nlu_text = f"{nlu_text}. OCR text: {ocr_text[:600]}"
            log.info("vision_ocr_appended", extra={"ocr_chars": len(ocr_text)})

    lang   = _detect_language(nlu_text)
    result = await pipeline_classify(
        text=nlu_text,
        modality="vision",
        language=lang,
        sync_chat=_sync_chat,
        sync_chat_temp=_sync_chat_temp,
    )
    # Attach vision analysis to the NLU result
    result = result.model_copy(update={"vision": vision_analysis})
    latency_ms = (time.perf_counter() - t0) * 1000

    log.info("vision_classified", extra={
        "intent": result.intent.value,
        "confidence": result.confidence,
        "frustration": vision_analysis.frustration_score,
        "screen_type": vision_analysis.screen_type,
        "latency_ms": round(latency_ms, 1),
    })
    return result, VISION_MODEL, latency_ms


async def transcribe_audio_async(
    audio_b64: str,
    mime_type: str = "audio/wav",
    language: Optional[str] = None,
    session_id: Optional[str] = None,
) -> TranscriptionResult:
    ext_map     = {"audio/wav":"wav","audio/mp3":"mp3","audio/ogg":"ogg","audio/webm":"webm"}
    ext         = ext_map.get(mime_type, "wav")
    audio_bytes = base64.b64decode(audio_b64)
    _check_audio_quality(audio_bytes)

    t0 = time.perf_counter()
    try:
        resp = await asyncio.wait_for(
            asyncio.to_thread(_sync_whisper, audio_bytes, mime_type, ext, language, 0.0),
            timeout=GROQ_TIMEOUT_S,
        )
    except asyncio.TimeoutError:
        raise TimeoutError(f"Whisper call exceeded {GROQ_TIMEOUT_S}s.")

    _check_audio_quality(audio_bytes, resp)
    latency_ms = (time.perf_counter() - t0) * 1000
    transcript = getattr(resp, "text", str(resp)).strip()
    lang_det   = getattr(resp, "language", None)
    dur        = getattr(resp, "duration", None)

    log.info("audio_transcribed", extra={
        "language": lang_det, "duration_s": dur, "latency_ms": round(latency_ms, 1)
    })
    return TranscriptionResult(
        success=True, transcript=transcript, language_detected=lang_det,
        duration_seconds=float(dur) if dur else None,
        model_used=AUDIO_MODEL, latency_ms=latency_ms, session_id=session_id,
    )


async def probe_groq() -> bool:
    try:
        await asyncio.wait_for(
            asyncio.to_thread(_sync_chat, [{"role":"user","content":"ping"}], 4),
            timeout=10.0,
        )
        return True
    except Exception as exc:
        log.warning("groq_probe_failed", extra={"error": str(exc)}); return False


def circuit_breaker_status() -> dict:
    return {
        "open":      _breaker.is_open,
        "failures":  _breaker._failures,
        "opened_at": datetime.fromtimestamp(_breaker._opened_at, tz=timezone.utc).isoformat()
                     if _breaker._opened_at else None,
    }
