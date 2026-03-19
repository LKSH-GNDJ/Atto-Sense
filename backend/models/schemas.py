"""
AttoSense v4 - Pydantic Schemas
NLU result schema extended for hierarchical classification:
  - intent_family   : coarse 3-way grouping (transaction / account / general)
  - confidence_scores: full probability distribution across all 8 intents
  - competing_intent : second-best intent and its score
  - reasoning_steps  : ordered evidence chain (replaces single reasoning string)
  - sentiment_score  : continuous -1.0 → +1.0 alongside the label
  - escalation_reason: why escalation was triggered (null if not triggered)
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, Literal, Any
from enum import Enum


# ── Enums ──────────────────────────────────────────────────────────────────────

class InputModality(str, Enum):
    TEXT   = "text"
    AUDIO  = "audio"
    VISION = "vision"


class IntentFamily(str, Enum):
    """Coarse grouping used in Stage 1 of hierarchical classification."""
    TRANSACTION = "transaction"   # billing, technical_support, complaint, escalation
    ACCOUNT     = "account"       # account_management, sales_inquiry
    GENERAL     = "general"       # general_inquiry, out_of_scope


class IntentCategory(str, Enum):
    BILLING      = "billing"
    TECHNICAL    = "technical_support"
    ACCOUNT      = "account_management"
    SALES        = "sales_inquiry"
    COMPLAINT    = "complaint"
    GENERAL      = "general_inquiry"
    ESCALATION   = "escalation"
    OUT_OF_SCOPE = "out_of_scope"


# Maps each intent to its family
INTENT_TO_FAMILY: dict[str, IntentFamily] = {
    "billing":            IntentFamily.TRANSACTION,
    "technical_support":  IntentFamily.TRANSACTION,
    "complaint":          IntentFamily.TRANSACTION,
    "escalation":         IntentFamily.TRANSACTION,
    "account_management": IntentFamily.ACCOUNT,
    "sales_inquiry":      IntentFamily.ACCOUNT,
    "general_inquiry":    IntentFamily.GENERAL,
    "out_of_scope":       IntentFamily.GENERAL,
}

# Members of each family for Stage 2 targeting
FAMILY_INTENTS: dict[IntentFamily, list[str]] = {
    IntentFamily.TRANSACTION: ["billing", "technical_support", "complaint", "escalation"],
    IntentFamily.ACCOUNT:     ["account_management", "sales_inquiry"],
    IntentFamily.GENERAL:     ["general_inquiry", "out_of_scope"],
}


class ErrorType(str, Enum):
    HTTP_ERROR        = "http_error"
    APP_CRASH         = "app_crash"
    LOGIN_FAILURE     = "login_failure"
    PAYMENT_DECLINE   = "payment_decline"
    BROKEN_UI         = "broken_ui"
    EMPTY_STATE       = "empty_state"
    TIMEOUT           = "timeout"
    PERMISSION_DENIED = "permission_denied"
    DATA_MISMATCH     = "data_mismatch"
    NONE              = "none"


class InboxStatus(str, Enum):
    PENDING  = "pending"
    REVIEWED = "reviewed"
    APPROVED = "approved"
    REJECTED = "rejected"


# ── Inbound Request Models ─────────────────────────────────────────────────────

class TextRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=16000)
    session_id: Optional[str] = Field(None)
    context: Optional[list[dict[str, str]]] = Field(default_factory=list)

    @field_validator("message")
    @classmethod
    def strip_whitespace(cls, v: str) -> str:
        return v.strip()


class AudioRequest(BaseModel):
    audio_b64: str = Field(...)
    mime_type: Literal["audio/wav", "audio/mp3", "audio/ogg", "audio/webm"] = "audio/wav"
    session_id: Optional[str] = None
    language: Optional[str] = Field(None)


class VisionRequest(BaseModel):
    image_b64: str = Field(...)
    mime_type: Literal["image/jpeg", "image/png", "image/webp", "image/gif"] = "image/jpeg"
    caption: Optional[str] = Field(None)
    session_id: Optional[str] = None


# ── NLU Output Models ──────────────────────────────────────────────────────────

class Entity(BaseModel):
    label: str        = Field(..., description="Entity type e.g. ORDER_ID, AMOUNT")
    value: str        = Field(..., description="Normalised extracted value")
    confidence: float = Field(..., ge=0.0, le=1.0)


class VisionAnalysis(BaseModel):
    frustration_score: float      = Field(0.0, ge=0.0, le=1.0)
    error_type: ErrorType         = Field(ErrorType.NONE)
    error_detail: Optional[str]   = Field(None)
    visual_summary: Optional[str] = Field(None)
    screen_type: Optional[str]    = Field(None,
        description="Type of screen visible: error_page|login|dashboard|payment|settings|chat|other")


class NLUResult(BaseModel):
    # ── Core classification ────────────────────────────────────────────────────
    intent: IntentCategory          = Field(..., description="Primary classified intent")
    intent_family: IntentFamily     = Field(..., description="Coarse family (transaction/account/general)")
    confidence: float               = Field(..., ge=0.0, le=1.0,
        description="Calibrated confidence for the primary intent")

    # Full probability distribution — enables calibration and analytics
    confidence_scores: dict[str, float] = Field(
        default_factory=dict,
        description="Score for every intent (values sum to approximately 1.0)"
    )

    # Second-best option — drives CoT triggers and inbox analytics
    competing_intent: Optional[IntentCategory] = Field(None,
        description="Intent with second-highest score")
    competing_confidence: float = Field(0.0, ge=0.0, le=1.0)

    # ── Entities ───────────────────────────────────────────────────────────────
    entities: list[Entity] = Field(default_factory=list)

    # ── Sentiment (enhanced) ───────────────────────────────────────────────────
    sentiment: Literal["positive", "neutral", "negative", "frustrated"] = Field("neutral")
    sentiment_score: float = Field(0.0, ge=-1.0, le=1.0,
        description="Continuous: -1.0 (very negative) → 0.0 (neutral) → +1.0 (very positive)")

    # ── Escalation ─────────────────────────────────────────────────────────────
    requires_escalation: bool       = Field(False)
    escalation_reason: Optional[str] = Field(None,
        description="Why escalation was triggered; null if not triggered")

    # ── Reasoning ─────────────────────────────────────────────────────────────
    reasoning_steps: list[str]      = Field(default_factory=list,
        description="Ordered evidence chain — each step one observation")

    # ── Meta ──────────────────────────────────────────────────────────────────
    low_confidence: bool            = Field(False)
    raw_transcript: Optional[str]  = Field(None)
    modality: Optional[str]        = Field(None)
    language_detected: Optional[str] = Field(None)

    # ── Vision-only ───────────────────────────────────────────────────────────
    vision: Optional[VisionAnalysis] = Field(None)


# ── Transcription-Only Response ────────────────────────────────────────────────

class TranscriptionResult(BaseModel):
    success: bool
    transcript: str
    language_detected: Optional[str]  = None
    duration_seconds: Optional[float] = None
    model_used: str
    latency_ms: float
    session_id: Optional[str]


# ── API Response Wrappers ──────────────────────────────────────────────────────

class ClassifyResponse(BaseModel):
    success: bool
    modality: InputModality
    session_id: Optional[str]
    result: NLUResult
    model_used: str
    latency_ms: float
    inbox_flagged: bool = Field(False)


class ErrorResponse(BaseModel):
    success: bool = False
    error: str
    detail: Optional[Any] = None


# ── Audit / Reporting Models ───────────────────────────────────────────────────

class AuditEntry(BaseModel):
    timestamp: str
    session_id: Optional[str]
    modality: InputModality
    intent: str
    intent_family: str = "unknown"
    confidence: float
    sentiment: str
    sentiment_score: float = 0.0
    requires_escalation: bool
    low_confidence: bool = False
    latency_ms: float
    frustration_score: Optional[float] = None
    error_type: Optional[str]          = None


class MetricsSummary(BaseModel):
    total_requests: int
    avg_confidence: float
    avg_latency_ms: float
    intent_distribution: dict[str, int]
    modality_distribution: dict[str, int]
    escalation_rate: float
    low_confidence_rate: float
    sentiment_distribution: dict[str, int]
    inbox_pending: int = 0


# ── Discovery Inbox ────────────────────────────────────────────────────────────

class InboxItem(BaseModel):
    id: str = Field(...)
    timestamp: str
    session_id: Optional[str]
    modality: InputModality
    raw_input: str
    result: NLUResult
    status: InboxStatus = InboxStatus.PENDING
    reviewer_label: Optional[IntentCategory] = None
    reviewer_note: Optional[str]   = None
    reviewed_at: Optional[str]     = None


class InboxReview(BaseModel):
    status: InboxStatus
    reviewer_label: Optional[IntentCategory] = None
    reviewer_note: Optional[str]             = None


class InboxSummary(BaseModel):
    total: int
    pending: int
    reviewed: int
    approved: int
    rejected: int
    items: list[InboxItem]


# ── Dataset / Training Record ──────────────────────────────────────────────────

class NLUExample(BaseModel):
    text: str
    intent: IntentCategory
    entities: list[Entity]         = Field(default_factory=list)
    source_modality: InputModality = InputModality.TEXT
    verified: bool                 = False
