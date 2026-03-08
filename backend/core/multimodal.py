"""
BotTrain v3.1 - Multimodal Groq SDK Wrapper
Production-hardened:
  - Singleton Groq client (instantiated once, reused across all requests)
  - tenacity retry with exponential backoff (3 attempts, handles 429/503)
  - 45-second hard timeout on every Groq call via asyncio.wait_for
  - Sync shims removed — only async functions exported (no asyncio.run risk)
  - Structured logging on every pipeline call
"""

import os
import base64
import json
import time
import re
import asyncio
from typing import Optional

import httpx
from groq import Groq, RateLimitError, APIStatusError, APITimeoutError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

from backend.models.schemas import (
    NLUResult, Entity, IntentCategory, VisionAnalysis, ErrorType,
    TranscriptionResult,
)
from backend.core.logging_config import get_logger

log = get_logger("bottrain.multimodal")

# ── Constants ──────────────────────────────────────────────────────────────────

TEXT_MODEL       = "llama-3.3-70b-versatile"
VISION_MODEL     = "meta-llama/llama-4-scout-17b-16e-instruct"
AUDIO_MODEL      = "whisper-large-v3"
INBOX_THRESHOLD  = 0.70
GROQ_TIMEOUT_S   = 45.0          # hard ceiling per Groq call
MAX_RETRIES      = 3

INTENT_LIST = [e.value for e in IntentCategory]
ERROR_TYPES = [e.value for e in ErrorType]

# ── Singleton Groq Client ──────────────────────────────────────────────────────

_groq_client: Optional[Groq] = None


def get_groq_client() -> Groq:
    """
    Return the module-level singleton Groq client.
    Instantiated lazily on first call and reused forever.
    The Groq SDK is thread-safe; one client handles all concurrent requests.
    """
    global _groq_client
    if _groq_client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError("GROQ_API_KEY is not set.")
        _groq_client = Groq(
            api_key=api_key,
            timeout=httpx.Timeout(GROQ_TIMEOUT_S, connect=10.0),
            max_retries=0,          # we handle retries ourselves via tenacity
        )
        log.info("groq_client_initialized", extra={"timeout_s": GROQ_TIMEOUT_S})
    return _groq_client


# ── System Prompts ─────────────────────────────────────────────────────────────

NLU_SYSTEM_PROMPT = (
    "You are BotTrain, a zero-shot NLU classifier for SMB customer support.\n\n"
    "Analyze the input and return ONLY valid JSON — no markdown, no explanation.\n\n"
    "Required JSON schema:\n"
    "{\n"
    '  "intent": one of ' + str(INTENT_LIST) + ",\n"
    '  "confidence": float 0.0-1.0,\n'
    '  "entities": [{"label": string, "value": string, "confidence": float}],\n'
    '  "sentiment": one of ["positive","neutral","negative","frustrated"],\n'
    '  "requires_escalation": boolean,\n'
    '  "reasoning": string (one sentence max)\n'
    "}\n\n"
    "Rules:\n"
    "- confidence reflects genuine uncertainty; NEVER default to 1.0\n"
    "- extract entities: ORDER_ID, PRODUCT, DATE, ACCOUNT_NUMBER, AMOUNT, EMAIL, PHONE\n"
    "- requires_escalation = true if sentiment is frustrated OR intent is escalation OR confidence < 0.5\n"
    "- Return ONLY the JSON object."
)

VISION_SYSTEM_PROMPT = (
    "You are BotTrain Vision, a multimodal NLU classifier for SMB customer support.\n\n"
    "Analyze the screenshot/photo and return ONLY valid JSON — no prose, no fences.\n\n"
    "Required JSON schema:\n"
    "{\n"
    '  "intent": one of ' + str(INTENT_LIST) + ",\n"
    '  "confidence": float 0.0-1.0,\n'
    '  "entities": [{"label": string, "value": string, "confidence": float}],\n'
    '  "sentiment": one of ["positive","neutral","negative","frustrated"],\n'
    '  "requires_escalation": boolean,\n'
    '  "reasoning": string (one sentence),\n'
    '  "frustration_score": float 0.0-1.0,\n'
    '  "error_type": one of ' + str(ERROR_TYPES) + ",\n"
    '  "error_detail": string or null,\n'
    '  "visual_summary": string (one sentence)\n'
    "}\n\n"
    "Frustration scoring: 0.0-0.2 calm; 0.3-0.5 minor issue; 0.6-0.8 clear error; 0.9-1.0 critical failure.\n"
    "Return ONLY the JSON object."
)


# ── Retry Policy ───────────────────────────────────────────────────────────────

_RETRYABLE = (RateLimitError, APIStatusError, APITimeoutError, httpx.TimeoutException)

_retry_policy = retry(
    retry=retry_if_exception_type(_RETRYABLE),
    stop=stop_after_attempt(MAX_RETRIES),
    wait=wait_exponential(multiplier=1, min=2, max=20),
    before_sleep=before_sleep_log(log, log.level if hasattr(log, "level") else 20),
    reraise=True,
)


# ── Sync Workers (run in threadpool via asyncio.to_thread) ─────────────────────

@_retry_policy
def _sync_chat(model: str, messages: list, max_tokens: int = 512) -> str:
    """Thread-safe Groq chat completion with retry."""
    return get_groq_client().chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.1,
        max_tokens=max_tokens,
    ).choices[0].message.content


@_retry_policy
def _sync_whisper(audio_bytes: bytes, mime_type: str, ext: str, language: Optional[str]):
    """Thread-safe Whisper transcription with retry."""
    return get_groq_client().audio.transcriptions.create(
        model=AUDIO_MODEL,
        file=(f"audio.{ext}", audio_bytes, mime_type),
        language=language,
        response_format="verbose_json",
    )


# ── JSON Parser ────────────────────────────────────────────────────────────────

def _parse_nlu_json(raw: str) -> dict:
    cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise ValueError(f"Could not parse LLM JSON: {raw[:300]}")


def _build_nlu_result(
    parsed: dict,
    transcript: Optional[str] = None,
    vision_data: Optional[dict] = None,
) -> NLUResult:
    entities = [
        Entity(
            label=e.get("label", "UNKNOWN"),
            value=str(e.get("value", "")),
            confidence=float(e.get("confidence", 0.8)),
        )
        for e in parsed.get("entities", [])
    ]

    raw_intent = parsed.get("intent", "general_inquiry")
    try:
        intent = IntentCategory(raw_intent)
    except ValueError:
        intent = IntentCategory.GENERAL

    confidence = max(0.0, min(1.0, float(parsed.get("confidence", 0.5))))
    low_conf   = confidence < INBOX_THRESHOLD

    vision_analysis = None
    if vision_data is not None:
        try:
            etype = ErrorType(vision_data.get("error_type", "none"))
        except ValueError:
            etype = ErrorType.NONE
        vision_analysis = VisionAnalysis(
            frustration_score=max(0.0, min(1.0, float(vision_data.get("frustration_score", 0.0)))),
            error_type=etype,
            error_detail=vision_data.get("error_detail"),
            visual_summary=vision_data.get("visual_summary"),
        )

    return NLUResult(
        intent=intent,
        confidence=confidence,
        entities=entities,
        sentiment=parsed.get("sentiment", "neutral"),
        requires_escalation=parsed.get("requires_escalation", confidence < 0.5),
        low_confidence=low_conf,
        raw_transcript=transcript,
        reasoning=parsed.get("reasoning"),
        vision=vision_analysis,
    )


# ── Async Text Pipeline ────────────────────────────────────────────────────────

async def classify_text_async(
    message: str,
    context: Optional[list[dict]] = None,
) -> tuple[NLUResult, str, float]:
    messages = [{"role": "system", "content": NLU_SYSTEM_PROMPT}]
    if context:
        messages.extend(context[-6:])
    messages.append({"role": "user", "content": message})

    t0 = time.perf_counter()
    try:
        raw = await asyncio.wait_for(
            asyncio.to_thread(_sync_chat, TEXT_MODEL, messages),
            timeout=GROQ_TIMEOUT_S,
        )
    except asyncio.TimeoutError:
        raise TimeoutError(f"Groq text call exceeded {GROQ_TIMEOUT_S}s timeout.")
    latency_ms = (time.perf_counter() - t0) * 1000

    result = _build_nlu_result(_parse_nlu_json(raw))
    log.info("text_classified", extra={
        "intent": result.intent.value, "confidence": result.confidence,
        "latency_ms": round(latency_ms, 1),
    })
    return result, TEXT_MODEL, latency_ms


# ── Async Transcription ────────────────────────────────────────────────────────

async def transcribe_audio_async(
    audio_b64: str,
    mime_type: str = "audio/wav",
    language: Optional[str] = None,
    session_id: Optional[str] = None,
) -> TranscriptionResult:
    ext_map = {"audio/wav": "wav", "audio/mp3": "mp3", "audio/ogg": "ogg", "audio/webm": "webm"}
    ext         = ext_map.get(mime_type, "wav")
    audio_bytes = base64.b64decode(audio_b64)

    t0 = time.perf_counter()
    try:
        resp = await asyncio.wait_for(
            asyncio.to_thread(_sync_whisper, audio_bytes, mime_type, ext, language),
            timeout=GROQ_TIMEOUT_S,
        )
    except asyncio.TimeoutError:
        raise TimeoutError(f"Whisper call exceeded {GROQ_TIMEOUT_S}s timeout.")
    latency_ms = (time.perf_counter() - t0) * 1000

    transcript       = getattr(resp, "text", str(resp)).strip()
    lang_detected    = getattr(resp, "language", None)
    duration_seconds = getattr(resp, "duration", None)

    log.info("audio_transcribed", extra={
        "language": lang_detected, "duration_s": duration_seconds,
        "latency_ms": round(latency_ms, 1),
    })
    return TranscriptionResult(
        success=True,
        transcript=transcript,
        language_detected=lang_detected,
        duration_seconds=float(duration_seconds) if duration_seconds else None,
        model_used=AUDIO_MODEL,
        latency_ms=latency_ms,
        session_id=session_id,
    )


# ── Async Audio Pipeline ───────────────────────────────────────────────────────

async def classify_audio_async(
    audio_b64: str,
    mime_type: str = "audio/wav",
    language: Optional[str] = None,
) -> tuple[NLUResult, str, float]:
    ext_map = {"audio/wav": "wav", "audio/mp3": "mp3", "audio/ogg": "ogg", "audio/webm": "webm"}
    ext         = ext_map.get(mime_type, "wav")
    audio_bytes = base64.b64decode(audio_b64)

    t0 = time.perf_counter()
    try:
        whisper_resp = await asyncio.wait_for(
            asyncio.to_thread(_sync_whisper, audio_bytes, mime_type, ext, language),
            timeout=GROQ_TIMEOUT_S,
        )
    except asyncio.TimeoutError:
        raise TimeoutError(f"Whisper call exceeded {GROQ_TIMEOUT_S}s timeout.")

    transcript = getattr(whisper_resp, "text", str(whisper_resp)).strip()

    messages = [
        {"role": "system", "content": NLU_SYSTEM_PROMPT},
        {"role": "user",   "content": transcript},
    ]
    try:
        raw = await asyncio.wait_for(
            asyncio.to_thread(_sync_chat, TEXT_MODEL, messages),
            timeout=GROQ_TIMEOUT_S,
        )
    except asyncio.TimeoutError:
        raise TimeoutError(f"Groq chat call exceeded {GROQ_TIMEOUT_S}s timeout.")

    latency_ms = (time.perf_counter() - t0) * 1000
    result = _build_nlu_result(_parse_nlu_json(raw), transcript=transcript)
    log.info("audio_classified", extra={
        "intent": result.intent.value, "confidence": result.confidence,
        "latency_ms": round(latency_ms, 1),
    })
    return result, f"{AUDIO_MODEL} → {TEXT_MODEL}", latency_ms


# ── Async Vision Pipeline ──────────────────────────────────────────────────────

async def classify_vision_async(
    image_b64: str,
    mime_type: str = "image/jpeg",
    caption: Optional[str] = None,
) -> tuple[NLUResult, str, float]:
    user_text = caption or "Analyze this customer support screenshot or image."
    vision_prompt = (
        f"User context: {user_text}\n\n"
        "Carefully examine for: error messages, warning dialogs, UI anomalies, "
        "frustration signals (CAPS text, angry wording, red/warning colors, error screens). "
        "Return ONLY the required JSON object."
    )
    messages = [
        {"role": "system", "content": VISION_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_b64}"}},
                {"type": "text", "text": vision_prompt},
            ],
        },
    ]

    t0 = time.perf_counter()
    try:
        raw = await asyncio.wait_for(
            asyncio.to_thread(_sync_chat, VISION_MODEL, messages, 768),
            timeout=GROQ_TIMEOUT_S,
        )
    except asyncio.TimeoutError:
        raise TimeoutError(f"Groq vision call exceeded {GROQ_TIMEOUT_S}s timeout.")
    latency_ms = (time.perf_counter() - t0) * 1000

    parsed      = _parse_nlu_json(raw)
    vision_keys = {"frustration_score", "error_type", "error_detail", "visual_summary"}
    vision_data = {k: parsed[k] for k in vision_keys if k in parsed}
    result      = _build_nlu_result(parsed, transcript=caption, vision_data=vision_data)

    log.info("vision_classified", extra={
        "intent": result.intent.value,
        "confidence": result.confidence,
        "frustration": result.vision.frustration_score if result.vision else None,
        "latency_ms": round(latency_ms, 1),
    })
    return result, VISION_MODEL, latency_ms


# ── Groq Connectivity Probe (used by /health) ──────────────────────────────────

async def probe_groq() -> bool:
    """Send a minimal completion to verify Groq is reachable. Returns True on success."""
    try:
        await asyncio.wait_for(
            asyncio.to_thread(
                _sync_chat,
                TEXT_MODEL,
                [{"role": "user", "content": "ping"}],
                4,
            ),
            timeout=10.0,
        )
        return True
    except Exception as exc:
        log.warning("groq_probe_failed", extra={"error": str(exc)})
        return False
