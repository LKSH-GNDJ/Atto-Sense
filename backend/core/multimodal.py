"""
AttoSense MK1 — Multimodal Input Handler
Handles audio/vision preprocessing then delegates NLU to nlu_pipeline.classify().
Audio:  quality gate → Whisper → clean transcript → pipeline
Vision: resize → LLM vision → JSON → optional OCR → pipeline
Text:   condense if long → language detect → pipeline
"""
from __future__ import annotations
import asyncio, base64, io, json, os, re, time
from datetime import datetime, timezone
from typing import Optional

import httpx
from groq import Groq, RateLimitError, APIStatusError, APITimeoutError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, before_sleep_log

from backend.models.schemas import NLUResult, Entity, IntentDomain, VisionAnalysis, ErrorType, TranscriptionResult
from backend.core.logging_config import get_logger
from backend.core.calibration import calibrate_confidence, is_low_confidence
from backend.core.nlu_pipeline import classify as pipeline_classify, example_store

log = get_logger("attosense.multimodal")

TEXT_MODEL     = "llama-3.3-70b-versatile"
VISION_MODEL   = "meta-llama/llama-4-scout-17b-16e-instruct"
AUDIO_MODEL    = "whisper-large-v3"
GROQ_TIMEOUT_S = 45.0
MAX_RETRIES    = 3
MIN_AUDIO_BYTES      = 1_000
MIN_AUDIO_DURATION_S = 0.3
MAX_NO_SPEECH_PROB   = 0.95
MIN_AVG_LOGPROB      = -2.00
MAX_IMAGE_DIMENSION  = 1200
LONG_INPUT_THRESHOLD = 800
ERROR_TYPES = [e.value for e in ErrorType]

# ── Circuit Breaker ────────────────────────────────────────────────────────────
class _CircuitBreaker:
    FAILURE_THRESHOLD = 5; RECOVERY_SECONDS = 30
    def __init__(self): self._failures = 0; self._opened_at: Optional[float] = None
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
        _groq_client = Groq(api_key=api_key, timeout=httpx.Timeout(GROQ_TIMEOUT_S, connect=10.0), max_retries=0)
        log.info("groq_client_initialized")
    return _groq_client

# ── Retry policy ───────────────────────────────────────────────────────────────
_RETRYABLE = (RateLimitError, APIStatusError, APITimeoutError, httpx.TimeoutException)
_retry = retry(retry=retry_if_exception_type(_RETRYABLE), stop=stop_after_attempt(MAX_RETRIES),
               wait=wait_exponential(multiplier=1, min=2, max=20), reraise=True)

@_retry
def _sync_chat(messages: list, max_tokens: int = 800) -> str:
    _breaker.raise_if_open()
    try:
        r = get_groq_client().chat.completions.create(
            model=TEXT_MODEL, messages=messages, temperature=0.05, max_tokens=max_tokens,
        ).choices[0].message.content
        _breaker.record_success(); return r
    except Exception: _breaker.record_failure(); raise

@_retry
def _sync_chat_temp(messages: list, max_tokens: int = 400, temperature: float = 0.15) -> str:
    _breaker.raise_if_open()
    try:
        r = get_groq_client().chat.completions.create(
            model=TEXT_MODEL, messages=messages, temperature=temperature, max_tokens=max_tokens,
        ).choices[0].message.content
        _breaker.record_success(); return r
    except Exception: _breaker.record_failure(); raise

@_retry
def _sync_whisper(audio_bytes: bytes, mime_type: str, ext: str,
                  language: Optional[str], temperature: float = 0.0):
    _breaker.raise_if_open()
    try:
        r = get_groq_client().audio.transcriptions.create(
            model=AUDIO_MODEL, file=(f"audio.{ext}", audio_bytes, mime_type),
            language=language, response_format="verbose_json", temperature=temperature,
        )
        _breaker.record_success(); return r
    except Exception: _breaker.record_failure(); raise

# ── Text helpers ───────────────────────────────────────────────────────────────
_FILLER_RE = re.compile(
    r"\b(um+|uh+|er+|ah+|hmm+|like,?|you know,?|so,?|basically,?|literally,?|right,?)\b",
    re.IGNORECASE)

def _clean_transcript(text: str) -> str:
    return re.sub(r"\s{2,}", " ", _FILLER_RE.sub(" ", text)).strip()

_SIGNAL_KW = re.compile(
    r"\b(invoice|charge|charged|bill|refund|payment|paid|amount|error|crash|bug|"
    r"broken|fail|issue|problem|account|password|login|email|reset|plan|upgrade|"
    r"pricing|quote|complaint|unacceptable|furious|frustrated|manager|escalate|"
    r"legal|bank|dispute|order|ticket|case|reference)\b", re.IGNORECASE)

def _condense(text: str) -> str:
    if len(text) <= LONG_INPUT_THRESHOLD: return text
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]
    if len(sentences) <= 6: return text
    first_3 = sentences[:3]; last_3 = sentences[-3:]
    signal  = [s for s in sentences[3:-3] if _SIGNAL_KW.search(s)]
    seen, unique = set(), []
    for s in first_3 + signal + last_3:
        if s not in seen: seen.add(s); unique.append(s)
    condensed = " ".join(unique)
    log.debug("input_condensed", extra={"original": len(text), "condensed": len(condensed)})
    return f"[Long input — key sentences extracted]\n{condensed}"

def _detect_language(text: str, whisper_lang: Optional[str] = None) -> Optional[str]:
    if whisper_lang and whisper_lang.lower() not in ("en","english",""):
        return whisper_lang
    try:
        from langdetect import detect
        lang = detect(text[:500])
        return lang if lang != "en" else None
    except Exception: return None

_LANG_NAMES = {"es":"Spanish","fr":"French","de":"German","ar":"Arabic","pt":"Portuguese",
               "zh-cn":"Chinese","ja":"Japanese","hi":"Hindi","it":"Italian","ru":"Russian","ko":"Korean","tr":"Turkish"}
def _language_prefix(lang: Optional[str]) -> str:
    if not lang: return ""
    name = _LANG_NAMES.get(lang, lang.upper())
    return f"NOTE: Input is in {name}. Classify intent regardless. Respond JSON only.\n\n"

# ── Audio quality gate ─────────────────────────────────────────────────────────
class AudioQualityError(ValueError): pass

def _check_audio_quality(audio_bytes: bytes, resp=None) -> None:
    if len(audio_bytes) < MIN_AUDIO_BYTES:
        raise AudioQualityError(f"Audio too small ({len(audio_bytes)} bytes). Record at least 2 seconds.")
    if resp is None: return
    dur = getattr(resp, "duration", None)
    # Note: some browser webm clips report 0 duration — skip that check
    if dur is not None and float(dur) > 0 and float(dur) < MIN_AUDIO_DURATION_S:
        raise AudioQualityError(
            f"Audio too short ({float(dur):.1f}s). Please record a longer message.")
    segs = getattr(resp, "segments", None) or []
    # Only apply segment-level checks when Whisper returned rich verbose_json data
    if segs and len(segs) > 0:
        avg_nsp = sum(s.get("no_speech_prob", 0) for s in segs) / len(segs)
        avg_lp  = sum(s.get("avg_logprob",    0) for s in segs) / len(segs)
        log.debug("audio_quality_metrics",
                  extra={"avg_nsp": round(avg_nsp,3), "avg_lp": round(avg_lp,3)})
        if avg_nsp > MAX_NO_SPEECH_PROB:
            raise AudioQualityError(
                f"Audio appears to be silence or background noise "
                f"(no_speech_prob={avg_nsp:.2f}). Please speak clearly into the microphone.")
        if avg_lp < MIN_AVG_LOGPROB:
            raise AudioQualityError(
                f"Audio quality too low to transcribe accurately "
                f"(avg_logprob={avg_lp:.2f}). Please record in a quieter environment.")

# ── Image preprocessing ────────────────────────────────────────────────────────
def _preprocess_image(image_b64: str, mime_type: str) -> tuple[str, str]:
    try:
        from PIL import Image
        img = Image.open(io.BytesIO(base64.b64decode(image_b64)))
        w, h = img.size
        if max(w, h) <= MAX_IMAGE_DIMENSION: return image_b64, mime_type
        scale = MAX_IMAGE_DIMENSION / max(w, h)
        img   = img.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
        buf   = io.BytesIO()
        fmt   = "JPEG" if "jpeg" in mime_type or "jpg" in mime_type else "PNG"
        img.save(buf, format=fmt, quality=90)
        return base64.b64encode(buf.getvalue()).decode(), f"image/{'jpeg' if fmt=='JPEG' else 'png'}"
    except Exception as e:
        log.warning("image_preprocess_failed", extra={"error": str(e)})
        return image_b64, mime_type

# ── OCR fallback ───────────────────────────────────────────────────────────────
def _ocr_extract(image_b64: str) -> Optional[str]:
    try:
        import pytesseract; from PIL import Image
        text = pytesseract.image_to_string(Image.open(io.BytesIO(base64.b64decode(image_b64)))).strip()
        return text if len(text) > 20 else None
    except Exception: return None

# ── JSON parser ────────────────────────────────────────────────────────────────
def _parse_json(raw: str) -> dict:
    cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()
    try: return json.loads(cleaned)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if m: return json.loads(m.group())
        raise ValueError(f"Cannot parse JSON: {raw[:200]}")

# Vision system prompt — no em-dashes, clean string literals
_VISION_SYSTEM = (
    "You are AttoSense Vision. Analyse the screenshot or photo.\n"
    "Extract ALL visible text. Identify errors, dialogs, broken UI.\n\n"
    "Return ONLY a JSON object with these fields:\n"
    "  intent_primary: 2-5 word action phrase starting with a verb\n"
    "  confidence: float 0.0-1.0\n"
    "  sentiment: positive|neutral|negative|frustrated\n"
    "  frustration_score: float 0.0-1.0\n"
    "  error_type: one of " + str(ERROR_TYPES) + "\n"
    "  error_detail: exact error text visible or null\n"
    "  visual_summary: one sentence describing what is visible\n"
    "  screen_type: error_page|login|dashboard|payment|settings|chat|other\n\n"
    "Frustration: 0.0-0.2 calm  0.3-0.5 minor  0.6-0.8 error  0.9-1.0 critical\n"
    "Intent phrase examples: Report HTTP 403 error, Fix login failure, Resolve payment decline,\n"
    "  Debug broken dashboard, Request account access, Troubleshoot app crash\n"
    "Return ONLY the JSON object."
)


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC ASYNC PIPELINES
# ══════════════════════════════════════════════════════════════════════════════

async def classify_text_async(
    message: str,
    context: Optional[list[dict]] = None,
) -> tuple[NLUResult, str, float]:
    t0           = time.perf_counter()
    text_in      = _condense(message)
    lang         = _detect_language(message)
    lang_pfx     = _language_prefix(lang)
    classify_text = lang_pfx + text_in if lang_pfx else text_in

    result = await pipeline_classify(
        text=classify_text, modality="text", language=lang,
        sync_chat=_sync_chat, sync_chat_temp=_sync_chat_temp,
    )
    return result, TEXT_MODEL, (time.perf_counter() - t0) * 1000


async def classify_audio_async(
    audio_b64: str,
    mime_type: str = "audio/wav",
    language: Optional[str] = None,
) -> tuple[NLUResult, str, float]:
    ext_map     = {"audio/wav":"wav","audio/mp3":"mp3","audio/ogg":"ogg","audio/webm":"webm"}
    ext         = ext_map.get(mime_type,"wav")
    audio_bytes = base64.b64decode(audio_b64)
    _check_audio_quality(audio_bytes)
    t0           = time.perf_counter()
    # WebM from browser uses ~8 KB/s; .wav uses ~32 KB/s — detect by mime type
    bytes_per_sec = 8_000 if "webm" in mime_type or "ogg" in mime_type else 32_000
    est_duration  = len(audio_bytes) / bytes_per_sec
    whisper_temp  = 0.2 if est_duration < 10 else 0.0

    try:
        resp = await asyncio.wait_for(
            asyncio.to_thread(_sync_whisper, audio_bytes, mime_type, ext, language, whisper_temp),
            timeout=GROQ_TIMEOUT_S,
        )
    except asyncio.TimeoutError:
        raise TimeoutError(f"Whisper exceeded {GROQ_TIMEOUT_S}s.")

    _check_audio_quality(audio_bytes, resp)
    raw_transcript = getattr(resp, "text", str(resp)).strip()
    whisper_lang   = getattr(resp, "language", None)
    clean          = _clean_transcript(raw_transcript)
    lang           = _detect_language(clean, whisper_lang)
    lang_pfx       = _language_prefix(lang)
    classify_text  = lang_pfx + clean if lang_pfx else clean

    result = await pipeline_classify(
        text=classify_text, modality="audio", language=lang,
        sync_chat=_sync_chat, sync_chat_temp=_sync_chat_temp,
    )
    result = result.model_copy(update={"raw_transcript": raw_transcript})
    return result, f"{AUDIO_MODEL} -> {TEXT_MODEL}", (time.perf_counter() - t0) * 1000


async def classify_vision_async(
    image_b64: str,
    mime_type: str = "image/jpeg",
    caption: Optional[str] = None,
) -> tuple[NLUResult, str, float]:
    image_b64, mime_type = _preprocess_image(image_b64, mime_type)
    t0 = time.perf_counter()

    user_prompt = (
        "Step 1: Extract ALL visible text exactly as shown.\n"
        "Step 2: Identify errors, error codes, dialogs, broken UI elements.\n"
        "Step 3: Customer caption (weigh highly): " + (f'"{caption}"' if caption else "(none)") + "\n"
        "Step 4: Return ONLY the JSON object."
    )
    msgs = [
        {"role":"system","content":_VISION_SYSTEM},
        {"role":"user","content":[
            {"type":"image_url","image_url":{"url":f"data:{mime_type};base64,{image_b64}"}},
            {"type":"text","text":user_prompt},
        ]},
    ]
    try:
        raw = await asyncio.wait_for(
            asyncio.to_thread(lambda: get_groq_client().chat.completions.create(
                model=VISION_MODEL, messages=msgs, temperature=0.05, max_tokens=600,
            ).choices[0].message.content),
            timeout=GROQ_TIMEOUT_S,
        )
        vision_parsed = _parse_json(raw)
    except Exception as exc:
        log.warning("vision_parse_failed", extra={"error": str(exc)})
        vision_parsed = {}

    try:    etype = ErrorType(vision_parsed.get("error_type","none"))
    except: etype = ErrorType.NONE

    vision_analysis = VisionAnalysis(
        frustration_score=max(0.0,min(1.0,float(vision_parsed.get("frustration_score",0.0)))),
        error_type=etype,
        error_detail=vision_parsed.get("error_detail"),
        visual_summary=vision_parsed.get("visual_summary"),
        screen_type=vision_parsed.get("screen_type"),
    )

    nlu_parts = []
    if caption:                          nlu_parts.append(caption)
    if vision_parsed.get("visual_summary"): nlu_parts.append(vision_parsed["visual_summary"])
    if vision_parsed.get("error_detail"):   nlu_parts.append(f"Error: {vision_parsed['error_detail']}")
    nlu_text = ". ".join(nlu_parts) if nlu_parts else "Screenshot provided with no caption."

    # OCR fallback for text-heavy low-confidence images
    vision_conf = float(vision_parsed.get("confidence", 0.65))
    frust_score = vision_analysis.frustration_score
    if vision_conf < 0.70 and frust_score < 0.4:
        ocr_text = _ocr_extract(image_b64)
        if ocr_text:
            nlu_text = f"{nlu_text}. OCR: {ocr_text[:600]}"
            log.info("vision_ocr_appended", extra={"ocr_chars": len(ocr_text)})

    lang   = _detect_language(nlu_text)
    result = await pipeline_classify(
        text=nlu_text, modality="vision", language=lang,
        sync_chat=_sync_chat, sync_chat_temp=_sync_chat_temp,
    )
    result = result.model_copy(update={"vision": vision_analysis})
    latency_ms = (time.perf_counter() - t0) * 1000
    log.info("vision_classified", extra={
        "intent": result.intent, "confidence": result.confidence,
        "frustration": vision_analysis.frustration_score, "latency_ms": round(latency_ms,1),
    })
    return result, VISION_MODEL, latency_ms


async def transcribe_audio_async(
    audio_b64: str,
    mime_type: str = "audio/wav",
    language: Optional[str] = None,
    session_id: Optional[str] = None,
) -> TranscriptionResult:
    ext_map     = {"audio/wav":"wav","audio/mp3":"mp3","audio/ogg":"ogg","audio/webm":"webm"}
    ext         = ext_map.get(mime_type,"wav")
    audio_bytes = base64.b64decode(audio_b64)
    _check_audio_quality(audio_bytes)
    t0 = time.perf_counter()
    try:
        resp = await asyncio.wait_for(
            asyncio.to_thread(_sync_whisper, audio_bytes, mime_type, ext, language, 0.0),
            timeout=GROQ_TIMEOUT_S,
        )
    except asyncio.TimeoutError:
        raise TimeoutError(f"Whisper exceeded {GROQ_TIMEOUT_S}s.")
    _check_audio_quality(audio_bytes, resp)
    transcript = getattr(resp, "text", str(resp)).strip()
    lang_det   = getattr(resp, "language", None)
    dur        = getattr(resp, "duration", None)
    log.info("audio_transcribed", extra={"language": lang_det, "duration_s": dur})
    return TranscriptionResult(
        success=True, transcript=transcript, language_detected=lang_det,
        duration_seconds=float(dur) if dur else None,
        model_used=AUDIO_MODEL, latency_ms=(time.perf_counter()-t0)*1000, session_id=session_id,
    )


async def probe_groq() -> bool:
    try:
        await asyncio.wait_for(asyncio.to_thread(_sync_chat, [{"role":"user","content":"ping"}], 4), timeout=10.0)
        return True
    except Exception: return False


def circuit_breaker_status() -> dict:
    return {
        "open": _breaker.is_open, "failures": _breaker._failures,
        "opened_at": datetime.fromtimestamp(_breaker._opened_at, tz=timezone.utc).isoformat()
                     if _breaker._opened_at else None,
    }
