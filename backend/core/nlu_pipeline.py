"""
AttoSense v4 — NLU Pipeline
===========================
Best-in-class intent identification pipeline for text, audio transcript,
and vision-extracted text.

Architecture
------------
Stage 1 — Scope + Family Gate  (1 fast LLM call, ≤150 tokens out)
    Binary: is this customer-support related?
    Coarse: which family? TRANSACTION | ACCOUNT | GENERAL
    → immediately short-circuits out_of_scope for truly OOS content
    → narrows Stage 2 from 8-way to 2-4 way classification

Stage 2 — Fine-Grained Intent  (main LLM call, full schema output)
    Dynamic few-shot: up to 3 real reviewed examples per candidate intent
    Only considers intents in the family from Stage 1
    Returns full NLUResult fields: confidence_scores, reasoning_steps,
    sentiment_score, escalation_reason, competing_intent

Stage 3 — Ensemble Confidence  (2 parallel calls, conditional)
    Fires only when Stage 2 confidence is in the uncertain range [0.60, 0.82]
    asyncio.gather runs both extra passes in parallel (no extra wall-clock time)
    Agreement rate across 3 total passes → calibrated confidence
    Majority vote for the final intent

Dynamic Few-Shot Example Store
-------------------------------
Module-level in-memory cache of verified NLU examples.
Loaded from DB on startup (api.py calls example_store.load()).
Updated in real-time as reviewers approve inbox corrections.
Trigram similarity finds the most relevant examples per intent per query.
Falls back gracefully to static examples when DB has too few.
"""

from __future__ import annotations

import asyncio
import json
import re
import threading
import time
from typing import Optional

from backend.core.logging_config import get_logger
from backend.core.calibration import calibrate_confidence, is_low_confidence, get_threshold
from backend.models.schemas import (
    NLUResult, Entity, IntentCategory, IntentFamily,
    INTENT_TO_FAMILY, FAMILY_INTENTS,
)

log = get_logger("attosense.pipeline")

# ── Constants ──────────────────────────────────────────────────────────────────

ENSEMBLE_LOW  = 0.60   # below this → straight to inbox, skip ensemble
ENSEMBLE_HIGH = 0.82   # above this → confident enough, skip ensemble
MIN_EXAMPLES_FOR_DYNAMIC = 3   # need at least this many DB examples to use dynamic few-shot
MIN_ENTITY_CONFIDENCE    = 0.60
MAX_EXAMPLES_PER_INTENT  = 500  # cap per intent in the store

# Intents in each family (also defined in schemas.py, duplicated here for clarity)
_FAMILY_INTENTS: dict[str, list[str]] = {
    "TRANSACTION": ["billing", "technical_support", "complaint", "escalation"],
    "ACCOUNT":     ["account_management", "sales_inquiry"],
    "GENERAL":     ["general_inquiry", "out_of_scope"],
}
_ALL_INTENTS = [e.value for e in IntentCategory]

# ── Trigram similarity (local copy — no DB import needed) ──────────────────────

def _trigrams(text: str) -> set[str]:
    t = text.lower().strip()
    return {t[i:i+3] for i in range(len(t) - 2)} if len(t) >= 3 else set()

def _sim(a: str, b: str) -> float:
    ta, tb = _trigrams(a), _trigrams(b)
    if not ta and not tb: return 1.0
    if not ta or not tb:  return 0.0
    return len(ta & tb) / len(ta | tb)


# ══════════════════════════════════════════════════════════════════════════════
# DYNAMIC FEW-SHOT EXAMPLE STORE
# ══════════════════════════════════════════════════════════════════════════════

class ExampleStore:
    """
    Thread-safe in-memory store of verified NLU examples.

    api.py calls example_store.load(rows) on startup and each time
    an inbox item is approved. multimodal.py calls example_store.add()
    after each promotion. No DB access needed here.
    """

    def __init__(self) -> None:
        self._store: dict[str, list[str]] = {k: [] for k in _ALL_INTENTS}
        self._lock  = threading.Lock()
        self._count = 0

    def load(self, examples: list[dict]) -> None:
        """
        Bulk-load from DB rows.
        Each row: {"intent": str, "text": str}
        Called from api.py lifespan on startup.
        """
        with self._lock:
            self._store = {k: [] for k in _ALL_INTENTS}
            for ex in examples:
                intent = ex.get("intent", "")
                text   = (ex.get("text") or "").strip()
                if intent in self._store and text:
                    self._store[intent].append(text)
            self._count = sum(len(v) for v in self._store.values())
        log.info("example_store_loaded", extra={
            "total": self._count,
            "per_intent": {k: len(v) for k, v in self._store.items()},
        })

    def add(self, intent: str, text: str) -> None:
        """Add one verified example (called on inbox approval)."""
        text = text.strip()
        if not text or intent not in self._store:
            return
        with self._lock:
            self._store[intent].append(text)
            if len(self._store[intent]) > MAX_EXAMPLES_PER_INTENT:
                self._store[intent] = self._store[intent][-MAX_EXAMPLES_PER_INTENT:]
            self._count += 1

    def get_similar(self, query: str, intent: str, n: int = 3) -> list[str]:
        """
        Return the n most similar examples for this intent,
        ranked by trigram Jaccard similarity.
        """
        with self._lock:
            candidates = list(self._store.get(intent, []))
        if not candidates:
            return []
        scored = sorted(
            ((round(_sim(query, c), 4), c) for c in candidates),
            reverse=True,
        )
        return [c for _, c in scored[:n]]

    def total(self) -> int:
        return self._count

    def per_intent(self) -> dict[str, int]:
        with self._lock:
            return {k: len(v) for k, v in self._store.items()}

    def has_enough(self, intent: str) -> bool:
        with self._lock:
            return len(self._store.get(intent, [])) >= MIN_EXAMPLES_FOR_DYNAMIC


# Module-level singleton — shared across all requests
example_store = ExampleStore()


# ══════════════════════════════════════════════════════════════════════════════
# PROMPT BUILDERS
# ══════════════════════════════════════════════════════════════════════════════

# ── Stage 1: Scope + Family gate ───────────────────────────────────────────────

_STAGE1_SYSTEM = """\
You are a fast pre-classifier for customer support messages.

Return ONLY a JSON object — no markdown, no explanation:
{
  "in_scope": boolean,
  "family": "TRANSACTION" | "ACCOUNT" | "GENERAL" | null,
  "confidence": float 0.0-1.0
}

Families:
  TRANSACTION — billing disputes, invoice issues, technical errors, app crashes,
                service complaints, escalation requests, chargebacks
  ACCOUNT     — account settings, password/login help, adding users, sales
                inquiries, pricing questions, plan upgrades
  GENERAL     — general product questions, support hours, policies, FAQs;
                ALSO used for anything with low clarity between families

in_scope rules:
  true  — ANY message from a customer about a software/SaaS product, even if
           long, multi-topic, emotional, or hard to understand
  false — ONLY for content with zero support relevance:
           cooking recipes, sports scores, creative writing, jokes, weather, math homework

Long emails and multi-paragraph tickets are ALWAYS in_scope.
When uncertain about family, prefer TRANSACTION.
family must be null only when in_scope is false.
"""

# ── Stage 2: Fine-grained intent ───────────────────────────────────────────────

_STAGE2_SYSTEM_TEMPLATE = """\
You are AttoSense, expert NLU classifier for SMB customer support.

This message has been pre-classified as a {family} query.
You must choose ONLY from these intents: {intent_list}

{dynamic_examples_block}

Return ONLY a JSON object:
{{
  "intent": one of {intent_list},
  "confidence_scores": {{intent: score, ...}} (all candidate intents, scores sum ≈ 1.0),
  "entities": [{{"label": string, "value": string, "confidence": float}}],
  "sentiment": one of ["positive","neutral","negative","frustrated"],
  "sentiment_score": float -1.0 to +1.0,
  "requires_escalation": boolean,
  "escalation_reason": string or null,
  "reasoning_steps": [string, ...]  (3-5 ordered observations leading to the intent)
}}

Entity types: ORDER_ID INVOICE_ID PRODUCT DATE ACCOUNT_NUMBER AMOUNT EMAIL PHONE ERROR_CODE PLAN_NAME

Escalation rules:
  requires_escalation = true when:
    sentiment = frustrated OR explicit manager demand OR legal/chargeback threat OR confidence < 0.50
  escalation_reason: one sentence explaining WHY (null if not escalating)

Disambiguation within {family}:
{disambiguation}

Sentiment score guide:
  -1.0 = extremely negative/furious
  -0.5 = clearly negative
   0.0 = neutral
  +0.5 = positive
  +1.0 = very happy/enthusiastic

reasoning_steps: list the strongest signals in order (e.g. "Customer mentions invoice INV-2024-0341",
  "Amount disputed: $299 vs expected $149", "No manager demand present", "Primary intent: billing")

Entities: normalise amounts to decimal (149.00), dates to ISO (2024-03-15), IDs to uppercase.
Filter out any entity with confidence < 0.60.

Return ONLY the JSON object.
"""

_DISAMBIGUATION = {
    "TRANSACTION": (
        "  billing vs complaint   — billing = money/invoice; complaint = service quality\n"
        "  technical vs complaint — technical = 'it's broken'; complaint = 'unacceptable'\n"
        "  escalation vs complaint— escalation demands a manager or threatens legal action"
    ),
    "ACCOUNT": (
        "  account_management — changing settings, adding users, password reset, email update\n"
        "  sales_inquiry      — pricing, plan comparison, upgrade request, enterprise quote"
    ),
    "GENERAL": (
        "  general_inquiry — any question about the product, hours, policies, or how things work\n"
        "  out_of_scope    — ONLY when the message has zero customer support relevance"
    ),
}

# Static fallback examples (used when ExampleStore has too few DB examples)
_STATIC_EXAMPLES: dict[str, list[str]] = {
    "billing": [
        "I was charged $149.99 twice this month. Please refund the duplicate.",
        "Invoice INV-2024-0892 shows charges I never authorised.",
    ],
    "technical_support": [
        "App crashes every time I open the reports tab. Error 403 on login.",
        "The dashboard won't load — just a spinner. Been broken since the update.",
    ],
    "account_management": [
        "Please update my email to john.new@example.com and add a second user.",
        "I need to reset my password — the reset link expired.",
    ],
    "sales_inquiry": [
        "What does the Business plan include? Can I get an enterprise quote for 50 staff?",
        "We want to upgrade from Starter to Pro — what's the price difference?",
    ],
    "complaint": [
        "This is the third time I've contacted you about the same issue. Disgraceful service.",
        "I waited 5 days with no reply. Completely unacceptable.",
    ],
    "escalation": [
        "I want to speak to your manager immediately. I will be contacting my bank.",
        "If this isn't resolved today I am disputing the charge and leaving a public review.",
    ],
    "general_inquiry": [
        "Do you offer a free trial? What are your support hours?",
        "How does automatic renewal work for annual plans?",
    ],
    "out_of_scope": [
        "What is the best recipe for banana bread?",
        "Who won the Champions League last year?",
    ],
}


def _build_dynamic_examples_block(query: str, intents: list[str]) -> str:
    """
    Build few-shot examples block for Stage 2.
    Prefers domain-specific DB examples over static fallbacks.
    """
    lines: list[str] = ["Examples (study the intent signals carefully):"]
    for intent in intents:
        if example_store.has_enough(intent):
            examples = example_store.get_similar(query, intent, n=3)
        else:
            examples = _STATIC_EXAMPLES.get(intent, [])[:2]
        for ex in examples:
            lines.append(f'\nInput ({intent}): "{ex[:200]}"')
    return "\n".join(lines) if len(lines) > 1 else ""


def _build_stage2_prompt(query: str, family: str) -> str:
    intents   = _FAMILY_INTENTS.get(family, _ALL_INTENTS)
    examples  = _build_dynamic_examples_block(query, intents)
    disambig  = _DISAMBIGUATION.get(family, "")
    return _STAGE2_SYSTEM_TEMPLATE.format(
        family=family,
        intent_list=str(intents),
        dynamic_examples_block=examples,
        disambiguation=disambig,
    )


# ══════════════════════════════════════════════════════════════════════════════
# JSON PARSER + ENTITY NORMALISER
# ══════════════════════════════════════════════════════════════════════════════

def _parse_json(raw: str) -> dict:
    cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if m:
            return json.loads(m.group())
        raise ValueError(f"Cannot parse JSON from: {raw[:300]}")


_AMOUNT_RE = re.compile(r"[\$£€¥,\s]")

def _normalise_entity(label: str, value: str) -> str:
    try:
        if label == "AMOUNT":
            return f"{float(_AMOUNT_RE.sub('', value)):.2f}"
        if label == "DATE":
            from dateutil import parser as dp
            return dp.parse(value, fuzzy=True).date().isoformat()
        if label in ("ORDER_ID", "INVOICE_ID", "ACCOUNT_NUMBER"):
            return re.sub(r"\s+", "", value).upper()
        if label == "EMAIL":
            return value.strip().lower()
        if label == "PHONE":
            return re.sub(r"[^\d+]", "", value)
    except Exception:
        pass
    return value


def _parse_entities(raw_entities: list) -> list[Entity]:
    entities = []
    for e in raw_entities:
        conf = float(e.get("confidence", 0.5))
        if conf < MIN_ENTITY_CONFIDENCE:
            continue
        label = e.get("label", "UNKNOWN")
        value = _normalise_entity(label, str(e.get("value", "")))
        entities.append(Entity(label=label, value=value, confidence=conf))
    return entities


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE STAGES
# ══════════════════════════════════════════════════════════════════════════════

async def _stage1_scope_and_family(
    text: str,
    sync_chat,   # injected from multimodal.py to avoid circular import
) -> tuple[bool, str, float]:
    """
    Stage 1: Scope gate + family classification in one call.
    Returns: (in_scope, family, confidence)
    Fast: uses only the first 600 chars for the gate decision.
    """
    snippet = text[:600]
    messages = [
        {"role": "system", "content": _STAGE1_SYSTEM},
        {"role": "user",   "content": snippet},
    ]
    try:
        raw    = await asyncio.wait_for(
            asyncio.to_thread(sync_chat, messages, 120),
            timeout=15.0,
        )
        parsed = _parse_json(raw)
        in_scope = bool(parsed.get("in_scope", True))
        family   = str(parsed.get("family") or "TRANSACTION").upper()
        conf     = float(parsed.get("confidence", 0.85))
        if family not in _FAMILY_INTENTS:
            family = "TRANSACTION"
        log.info("stage1_complete", extra={
            "in_scope": in_scope, "family": family, "confidence": conf
        })
        return in_scope, family, conf
    except Exception as exc:
        log.warning("stage1_failed_fallback", extra={"error": str(exc)})
        # Fail open — assume in-scope TRANSACTION so we don't lose real tickets
        return True, "TRANSACTION", 0.70


async def _stage2_fine_intent(
    text: str,
    family: str,
    sync_chat,
) -> dict:
    """
    Stage 2: Fine-grained intent with dynamic few-shot.
    Returns the raw parsed JSON dict.
    """
    system_prompt = _build_stage2_prompt(text, family)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": text},
    ]
    raw    = await asyncio.wait_for(
        asyncio.to_thread(sync_chat, messages, 900),
        timeout=40.0,
    )
    return _parse_json(raw)


async def _stage3_ensemble(
    text: str,
    family: str,
    first_intent: str,
    sync_chat_temp,  # injected callable(messages, max_tokens, temperature) -> str
) -> tuple[str, float]:
    """
    Stage 3: Two parallel passes at different temperatures.
    Returns (final_intent, calibrated_confidence).
    """
    system_prompt = _build_stage2_prompt(text, family)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": text},
    ]

    async def _one_pass(temp: float) -> str:
        try:
            raw = await asyncio.wait_for(
                asyncio.to_thread(sync_chat_temp, messages, 300, temp),
                timeout=30.0,
            )
            parsed = _parse_json(raw)
            return str(parsed.get("intent", first_intent))
        except Exception:
            return first_intent

    # Two parallel passes
    results = await asyncio.gather(
        _one_pass(0.15),
        _one_pass(0.30),
    )

    votes = [first_intent] + list(results)  # 3 total votes

    # Count agreements
    from collections import Counter
    counts = Counter(votes)
    winner, wins = counts.most_common(1)[0]

    if wins == 3:
        conf = 0.92   # unanimous
    elif wins == 2:
        conf = 0.78   # majority
    else:
        conf = 0.62   # split — stays low, will still go to inbox

    log.info("stage3_ensemble", extra={
        "votes": votes, "winner": winner, "wins": wins, "calibrated_conf": conf
    })
    return winner, conf


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC INTERFACE
# ══════════════════════════════════════════════════════════════════════════════

async def classify(
    text: str,
    modality: str = "text",
    language: Optional[str] = None,
    sync_chat=None,          # injected from multimodal.py
    sync_chat_temp=None,     # injected from multimodal.py (supports temperature arg)
) -> NLUResult:
    """
    Run the full 3-stage NLU pipeline on cleaned text.

    Parameters
    ----------
    text          : cleaned, normalised input text (transcript for audio; caption/OCR for vision)
    modality      : "text" | "audio" | "vision"
    language      : BCP-47 language code if non-English, else None
    sync_chat     : callable(messages, max_tokens) -> str  (Groq chat, temp=0.05)
    sync_chat_temp: callable(messages, max_tokens, temperature) -> str  (Groq chat, variable temp)
    """
    t0 = time.perf_counter()

    # ── Stage 1 ────────────────────────────────────────────────────────────────
    in_scope, family, s1_conf = await _stage1_scope_and_family(text, sync_chat)

    if not in_scope:
        # Short-circuit: truly out of scope — fast return, no Stage 2/3
        oos = IntentCategory.OUT_OF_SCOPE
        fam = INTENT_TO_FAMILY.get(oos.value, IntentFamily.GENERAL)
        conf = calibrate_confidence(s1_conf, modality)
        return NLUResult(
            intent=oos,
            intent_family=fam,
            confidence=conf,
            confidence_scores={"out_of_scope": conf},
            competing_intent=None,
            competing_confidence=0.0,
            entities=[],
            sentiment="neutral",
            sentiment_score=0.0,
            requires_escalation=False,
            escalation_reason=None,
            reasoning_steps=["Stage 1 gate: message has no customer support relevance."],
            low_confidence=is_low_confidence(conf, modality),
            raw_transcript=text if modality != "text" else None,
            modality=modality,
            language_detected=language,
            vision=None,
        )

    # ── Stage 2 ────────────────────────────────────────────────────────────────
    try:
        s2 = await _stage2_fine_intent(text, family, sync_chat)
    except Exception as exc:
        log.error("stage2_failed", extra={"error": str(exc)})
        # Graceful degradation: return general_inquiry with low confidence
        s2 = {
            "intent": "general_inquiry",
            "confidence_scores": {"general_inquiry": 0.51},
            "entities": [],
            "sentiment": "neutral",
            "sentiment_score": 0.0,
            "requires_escalation": False,
            "escalation_reason": None,
            "reasoning_steps": ["Stage 2 classification failed — degraded fallback."],
        }

    # Parse intent + scores
    raw_intent_str   = str(s2.get("intent", "general_inquiry"))
    conf_scores_raw  = s2.get("confidence_scores") or {}
    if not isinstance(conf_scores_raw, dict):
        conf_scores_raw = {}

    # If LLM returned intent outside the expected family, pull from scores instead
    allowed_intents = _FAMILY_INTENTS.get(family, _ALL_INTENTS)
    if raw_intent_str not in allowed_intents and conf_scores_raw:
        # Pick the allowed intent with the highest score
        raw_intent_str = max(
            (i for i in conf_scores_raw if i in allowed_intents),
            key=lambda i: conf_scores_raw[i],
            default=allowed_intents[0],
        )

    try:
        intent = IntentCategory(raw_intent_str)
    except ValueError:
        intent = IntentCategory.GENERAL

    # Primary confidence from scores dict; fall back to top-level if missing
    primary_conf_raw = float(
        conf_scores_raw.get(intent.value)
        or conf_scores_raw.get(raw_intent_str)
        or s2.get("confidence", 0.65)
    )
    primary_conf_raw = max(0.0, min(1.0, primary_conf_raw))

    # Build full confidence_scores dict (normalise to sum ≈ 1.0)
    score_total = sum(float(v) for v in conf_scores_raw.values()) if conf_scores_raw else 1.0
    confidence_scores: dict[str, float] = {
        k: round(float(v) / max(score_total, 1e-6), 4)
        for k, v in conf_scores_raw.items()
    }

    # Competing intent
    sorted_scores = sorted(confidence_scores.items(), key=lambda x: -x[1])
    competing_intent: Optional[IntentCategory] = None
    competing_conf = 0.0
    for k, v in sorted_scores:
        if k != intent.value:
            try:
                competing_intent = IntentCategory(k)
                competing_conf   = v
            except ValueError:
                pass
            break

    # ── Stage 3 — Ensemble (conditional) ────────────────────────────────────────
    final_intent = intent
    final_conf   = primary_conf_raw

    if ENSEMBLE_LOW <= primary_conf_raw <= ENSEMBLE_HIGH and sync_chat_temp is not None:
        ensemble_intent_str, ensemble_conf = await _stage3_ensemble(
            text, family, intent.value, sync_chat_temp
        )
        try:
            final_intent = IntentCategory(ensemble_intent_str)
        except ValueError:
            final_intent = intent
        # Use ensemble confidence if it changed the answer, else blend
        if final_intent != intent:
            final_conf = ensemble_conf
        else:
            final_conf = max(primary_conf_raw, ensemble_conf)

    # ── Calibration ─────────────────────────────────────────────────────────────
    calibrated_conf = calibrate_confidence(final_conf, modality)
    low_conf        = is_low_confidence(calibrated_conf, modality)

    # ── Assemble NLUResult ──────────────────────────────────────────────────────
    entities       = _parse_entities(s2.get("entities") or [])
    sentiment_str  = str(s2.get("sentiment", "neutral"))
    sentiment_score= float(s2.get("sentiment_score", 0.0))
    esc            = bool(s2.get("requires_escalation", calibrated_conf < 0.5))
    esc_reason     = s2.get("escalation_reason") or None
    reasoning_raw  = s2.get("reasoning_steps") or s2.get("reasoning") or []
    if isinstance(reasoning_raw, str):
        reasoning_raw = [reasoning_raw]
    reasoning_steps = [str(r) for r in reasoning_raw if r]

    # Force escalation fields to be consistent
    if esc and not esc_reason:
        if sentiment_str == "frustrated":
            esc_reason = "Customer sentiment is frustrated."
        elif calibrated_conf < 0.5:
            esc_reason = "Confidence too low for automated handling."
        elif final_intent == IntentCategory.ESCALATION:
            esc_reason = "Customer explicitly requested escalation."

    intent_family = INTENT_TO_FAMILY.get(final_intent.value, IntentFamily.GENERAL)

    latency_ms = (time.perf_counter() - t0) * 1000
    log.info("pipeline_complete", extra={
        "intent": final_intent.value,
        "family": family,
        "confidence": calibrated_conf,
        "low_conf": low_conf,
        "ensemble_ran": ENSEMBLE_LOW <= primary_conf_raw <= ENSEMBLE_HIGH,
        "dynamic_examples": example_store.total() > 0,
        "modality": modality,
        "latency_ms": round(latency_ms, 1),
    })

    return NLUResult(
        intent=final_intent,
        intent_family=intent_family,
        confidence=calibrated_conf,
        confidence_scores=confidence_scores,
        competing_intent=competing_intent,
        competing_confidence=competing_conf,
        entities=entities,
        sentiment=sentiment_str,
        sentiment_score=max(-1.0, min(1.0, sentiment_score)),
        requires_escalation=esc,
        escalation_reason=esc_reason,
        reasoning_steps=reasoning_steps,
        low_confidence=low_conf,
        raw_transcript=text if modality != "text" else None,
        modality=modality,
        language_detected=language,
        vision=None,   # set by multimodal.py for vision modality
    )
