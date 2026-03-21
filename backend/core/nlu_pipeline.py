"""
AttoSense MK1 — Universal NLU Pipeline
Stage 1: Domain detection (7-way)
Stage 2: Open-ended intent phrase generation with dynamic few-shot
Stage 3: Ensemble confidence (parallel, conditional)
"""
from __future__ import annotations
import asyncio, json, re, threading, time
from typing import Optional
from collections import Counter
from backend.core.logging_config import get_logger
from backend.core.calibration import calibrate_confidence, is_low_confidence
from backend.models.schemas import NLUResult, Entity, IntentDomain

log = get_logger("attosense.pipeline")

ENSEMBLE_LOW             = 0.60
ENSEMBLE_HIGH            = 0.82
MIN_EXAMPLES_FOR_DYNAMIC = 3
MIN_ENTITY_CONFIDENCE    = 0.60
MAX_EXAMPLES_PER_DOMAIN  = 500
ENSEMBLE_SIM_THRESHOLD   = 0.55

_ALL_DOMAINS = [d.value for d in IntentDomain]


def _trigrams(text: str) -> set[str]:
    t = text.lower().strip()
    return {t[i:i+3] for i in range(len(t)-2)} if len(t) >= 3 else set()

def _sim(a: str, b: str) -> float:
    ta, tb = _trigrams(a), _trigrams(b)
    if not ta and not tb: return 1.0
    if not ta or not tb:  return 0.0
    return len(ta & tb) / len(ta | tb)

def _intents_agree(a: str, b: str) -> bool:
    return _sim(a.lower(), b.lower()) >= ENSEMBLE_SIM_THRESHOLD


# ── Example Store ──────────────────────────────────────────────────────────────

class ExampleStore:
    def __init__(self):
        self._store: dict[str, list[tuple[str,str]]] = {d: [] for d in _ALL_DOMAINS}
        self._lock   = threading.Lock()
        self._count  = 0

    def load(self, examples: list[dict]) -> None:
        with self._lock:
            self._store = {d: [] for d in _ALL_DOMAINS}
            for ex in examples:
                domain = (ex.get("intent_domain") or "information").lower()
                text   = (ex.get("text")   or "").strip()
                intent = (ex.get("intent") or "").strip()
                if domain in self._store and text and intent:
                    self._store[domain].append((text, intent))
            self._count = sum(len(v) for v in self._store.values())
        log.info("example_store_loaded", extra={"total": self._count})

    def add(self, domain: str, text: str, intent: str) -> None:
        text = text.strip(); intent = intent.strip(); domain = domain.lower()
        if not text or not intent or domain not in self._store: return
        with self._lock:
            self._store[domain].append((text, intent))
            if len(self._store[domain]) > MAX_EXAMPLES_PER_DOMAIN:
                self._store[domain] = self._store[domain][-MAX_EXAMPLES_PER_DOMAIN:]
            self._count += 1

    def get_similar(self, query: str, domain: str, n: int = 3) -> list[tuple[str,str]]:
        with self._lock:
            candidates = list(self._store.get(domain, []))
        if not candidates: return []
        scored = sorted(((_sim(query, t), t, i) for t,i in candidates), reverse=True)
        return [(t, i) for _, t, i in scored[:n]]

    def total(self) -> int: return self._count

    def per_domain(self) -> dict[str, int]:
        with self._lock:
            return {k: len(v) for k, v in self._store.items()}

    def has_enough(self, domain: str) -> bool:
        with self._lock: return len(self._store.get(domain,[])) >= MIN_EXAMPLES_FOR_DYNAMIC


example_store = ExampleStore()


# ── Prompts ────────────────────────────────────────────────────────────────────

_STAGE1_SYSTEM = """\
You are a fast domain classifier. Classify any input into one domain.
Return ONLY JSON: {"domain": string, "confidence": float}

Domains:
  information  - questions, facts, how-to, research, definitions, news
  action       - requests to DO something: book, schedule, send, calculate, translate
  problem      - something broken/wrong: errors, crashes, complaints, disputes
  transaction  - money, payments, billing, invoices, refunds, purchases
  creative     - writing, design, brainstorming, art, stories, poetry
  personal     - advice, opinions, relationships, health, emotion, conversation
  technical    - code, programming, data analysis, engineering, mathematics

Every input belongs to exactly one domain. Return ONLY the JSON.
"""

_STAGE2_TEMPLATE = """\
You are AttoSense, a universal intent classifier. Identify what this person wants to achieve.
Domain: {domain}

{examples_block}

Return ONLY a JSON object:
{{
  "intent": string,
  "confidence_scores": {{"phrase": score}},
  "entities": [{{"label": string, "value": string, "confidence": float}}],
  "sentiment": "positive"|"neutral"|"negative"|"frustrated",
  "sentiment_score": float -1.0 to +1.0,
  "requires_escalation": boolean,
  "escalation_reason": string or null,
  "reasoning_steps": [string, ...]
}}

INTENT: 2-5 word action phrase starting with a verb. Be specific.
  information  -> "Get weather forecast", "Explain black holes", "Compare phone models"
  action       -> "Book flight to Paris", "Translate email to French", "Calculate mortgage"
  problem      -> "Report app login error", "Fix payment failure", "Resolve billing dispute"
  transaction  -> "Request invoice refund", "Upgrade subscription plan", "Check order status"
  creative     -> "Write birthday poem", "Draft cover letter", "Brainstorm product names"
  personal     -> "Seek relationship advice", "Get fitness recommendation", "Ask for support"
  technical    -> "Debug Python function", "Optimise SQL query", "Explain neural networks"

CONFIDENCE_SCORES: top 3 alternatives as {{phrase: score}}, sum to ~1.0
ENTITIES: PERSON LOCATION DATE TIME AMOUNT PRODUCT LANGUAGE ERROR_CODE EMAIL PHONE URL ORGANISATION ORDER_ID
ESCALATION: true when frustrated + problem/transaction OR manager demand OR legal threat OR confidence < 0.50
SENTIMENT SCORE: -1.0 furious, -0.5 negative, 0.0 neutral, +0.5 positive, +1.0 enthusiastic
REASONING STEPS: 3-5 observations, last one states the intent conclusion

Return ONLY the JSON object.
"""

_STAGE3_SYSTEM = """\
Identify the primary intent of this input as a 2-5 word action phrase starting with a verb.
Return ONLY JSON: {"intent": string, "confidence": float}
Be specific. "Report login error" not "technical issue". Return ONLY the JSON.
"""


# ── Helpers ────────────────────────────────────────────────────────────────────

def _parse_json(raw: str) -> dict:
    cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()
    try: return json.loads(cleaned)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if m: return json.loads(m.group())
        raise ValueError(f"Cannot parse JSON: {raw[:200]}")

_AMOUNT_RE = re.compile(r"[\$£€¥,\s]")

def _normalise_entity(label: str, value: str) -> str:
    try:
        if label == "AMOUNT":
            return f"{float(_AMOUNT_RE.sub('',value)):.2f}"
        if label == "DATE":
            from dateutil import parser as dp
            return dp.parse(value, fuzzy=True).date().isoformat()
        if label in ("ORDER_ID","INVOICE_ID","ACCOUNT_NUMBER"):
            return re.sub(r"\s+","",value).upper()
        if label == "EMAIL": return value.strip().lower()
        if label == "PHONE": return re.sub(r"[^\d+]","",value)
    except Exception: pass
    return value

def _parse_entities(raw: list) -> list[Entity]:
    result = []
    for e in (raw or []):
        conf = float(e.get("confidence", 0.5))
        if conf < MIN_ENTITY_CONFIDENCE: continue
        label = e.get("label","UNKNOWN")
        value = _normalise_entity(label, str(e.get("value","")))
        result.append(Entity(label=label, value=value, confidence=conf))
    return result

def _build_examples_block(query: str, domain: str) -> str:
    if not example_store.has_enough(domain): return ""
    pairs = example_store.get_similar(query, domain, n=3)
    if not pairs: return ""
    lines = ["Examples from this domain:"]
    for text, intent in pairs:
        lines.append(f'\nInput: "{text[:180]}"\nIntent: "{intent}"')
    return "\n".join(lines)


# ── Pipeline Stages ────────────────────────────────────────────────────────────

async def _stage1_domain(text: str, sync_chat) -> tuple[str, float]:
    messages = [{"role":"system","content":_STAGE1_SYSTEM},
                {"role":"user","content":text[:600]}]
    try:
        raw    = await asyncio.wait_for(asyncio.to_thread(sync_chat, messages, 120), timeout=15.0)
        parsed = _parse_json(raw)
        domain = str(parsed.get("domain","information")).lower()
        conf   = float(parsed.get("confidence", 0.85))
        if domain not in _ALL_DOMAINS: domain = "information"
        log.info("stage1_domain", extra={"domain":domain,"confidence":conf})
        return domain, conf
    except Exception as exc:
        log.warning("stage1_failed", extra={"error":str(exc)})
        return "information", 0.70


async def _stage2_intent(text: str, domain: str, sync_chat) -> dict:
    examples = _build_examples_block(text, domain)
    system   = _STAGE2_TEMPLATE.format(domain=domain.upper(), examples_block=examples)
    messages = [{"role":"system","content":system},{"role":"user","content":text}]
    raw = await asyncio.wait_for(asyncio.to_thread(sync_chat, messages, 900), timeout=40.0)
    return _parse_json(raw)


async def _stage3_ensemble(text: str, first_intent: str, sync_chat_temp) -> tuple[str, float]:
    messages = [{"role":"system","content":_STAGE3_SYSTEM},{"role":"user","content":text}]

    async def _one_pass(temp: float) -> str:
        try:
            raw    = await asyncio.wait_for(asyncio.to_thread(sync_chat_temp, messages, 200, temp), timeout=25.0)
            parsed = _parse_json(raw)
            return str(parsed.get("intent", first_intent)).strip()
        except Exception: return first_intent

    r1, r2 = await asyncio.gather(_one_pass(0.15), _one_pass(0.30))
    a12 = _intents_agree(first_intent, r1)
    a13 = _intents_agree(first_intent, r2)
    a23 = _intents_agree(r1, r2)

    if a12 and a13:   return first_intent, 0.92
    if a12 or a13:    return first_intent, 0.78
    if a23:           return r1,           0.72
    return first_intent, 0.60


# ── Public Interface ───────────────────────────────────────────────────────────

async def classify(
    text: str,
    modality:       str            = "text",
    language:       Optional[str]  = None,
    sync_chat                      = None,
    sync_chat_temp                 = None,
) -> NLUResult:
    t0 = time.perf_counter()

    domain, _ = await _stage1_domain(text, sync_chat)

    try:
        s2 = await _stage2_intent(text, domain, sync_chat)
    except Exception as exc:
        log.error("stage2_failed", extra={"error": str(exc)})
        s2 = {"intent":"Process request","confidence_scores":{"Process request":0.51},
              "entities":[],"sentiment":"neutral","sentiment_score":0.0,
              "requires_escalation":False,"escalation_reason":None,
              "reasoning_steps":["Stage 2 failed — degraded fallback."]}

    raw_intent = str(s2.get("intent") or "Process request").strip()
    if len(raw_intent.split()) < 2: raw_intent = "Process request"

    conf_scores_raw = s2.get("confidence_scores") or {}
    if not isinstance(conf_scores_raw, dict): conf_scores_raw = {}

    primary_conf = float(conf_scores_raw.get(raw_intent) or s2.get("confidence", 0.68))
    primary_conf = max(0.0, min(1.0, primary_conf))

    score_total = sum(float(v) for v in conf_scores_raw.values()) if conf_scores_raw else 1.0
    confidence_scores = {k: round(float(v)/max(score_total,1e-6),4) for k,v in conf_scores_raw.items()}
    if raw_intent not in confidence_scores:
        confidence_scores[raw_intent] = round(primary_conf, 4)

    sorted_scores   = sorted(confidence_scores.items(), key=lambda x:-x[1])
    competing_intent: Optional[str] = None
    competing_conf   = 0.0
    for k, v in sorted_scores:
        if k != raw_intent:
            competing_intent = k; competing_conf = v; break

    final_intent = raw_intent; final_conf = primary_conf

    if ENSEMBLE_LOW <= primary_conf <= ENSEMBLE_HIGH and sync_chat_temp:
        ens_intent, ens_conf = await _stage3_ensemble(text, raw_intent, sync_chat_temp)
        if not _intents_agree(ens_intent, raw_intent):
            final_intent = ens_intent; final_conf = ens_conf
        else:
            final_conf = max(primary_conf, ens_conf)

    calibrated = calibrate_confidence(final_conf, modality)
    low_conf   = is_low_confidence(calibrated, modality)

    entities      = _parse_entities(s2.get("entities"))
    sentiment_str = str(s2.get("sentiment","neutral"))
    sent_score    = float(s2.get("sentiment_score", 0.0))
    esc           = bool(s2.get("requires_escalation", calibrated < 0.5))
    esc_reason    = s2.get("escalation_reason") or None
    raw_steps     = s2.get("reasoning_steps") or s2.get("reasoning") or []
    if isinstance(raw_steps, str): raw_steps = [raw_steps]
    steps = [str(r) for r in raw_steps if r]

    if esc and not esc_reason:
        if sentiment_str == "frustrated": esc_reason = "Input signals user frustration."
        elif calibrated < 0.5:           esc_reason = "Confidence too low for automated handling."

    try:    intent_domain = IntentDomain(domain)
    except: intent_domain = IntentDomain.INFORMATION

    latency_ms = (time.perf_counter() - t0) * 1000
    log.info("pipeline_complete", extra={
        "intent": final_intent, "domain": domain,
        "confidence": calibrated, "latency_ms": round(latency_ms, 1),
    })

    return NLUResult(
        intent=final_intent, intent_domain=intent_domain,
        confidence=calibrated, confidence_scores=confidence_scores,
        competing_intent=competing_intent, competing_confidence=competing_conf,
        entities=entities, sentiment=sentiment_str,
        sentiment_score=max(-1.0, min(1.0, sent_score)),
        requires_escalation=esc, escalation_reason=esc_reason,
        reasoning_steps=steps, low_confidence=low_conf,
        raw_transcript=text if modality != "text" else None,
        modality=modality, language_detected=language, vision=None,
    )
