"""
BotTrain v3.1 - Pydantic Schemas
Defines all request/response models for the API layer.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, Literal, Any
from enum import Enum


# ── Enums ──────────────────────────────────────────────────────────────────────

class InputModality(str, Enum):
    TEXT   = "text"
    AUDIO  = "audio"
    VISION = "vision"


class IntentCategory(str, Enum):
    BILLING      = "billing"
    TECHNICAL    = "technical_support"
    ACCOUNT      = "account_management"
    SALES        = "sales_inquiry"
    COMPLAINT    = "complaint"
    GENERAL      = "general_inquiry"
    ESCALATION   = "escalation"
    OUT_OF_SCOPE = "out_of_scope"


class ErrorType(str, Enum):
    """Detected visual/technical error categories for Vision NLU."""
    HTTP_ERROR       = "http_error"
    APP_CRASH        = "app_crash"
    LOGIN_FAILURE    = "login_failure"
    PAYMENT_DECLINE  = "payment_decline"
    BROKEN_UI        = "broken_ui"
    EMPTY_STATE      = "empty_state"
    TIMEOUT          = "timeout"
    PERMISSION_DENIED= "permission_denied"
    DATA_MISMATCH    = "data_mismatch"
    NONE             = "none"


class InboxStatus(str, Enum):
    PENDING  = "pending"
    REVIEWED = "reviewed"
    APPROVED = "approved"
    REJECTED = "rejected"


# ── Inbound Request Models ─────────────────────────────────────────────────────

class TextRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=16000, description="Raw user message or multi-paragraph text")
    session_id: Optional[str] = Field(None, description="Session tracking ID")
    context: Optional[list[dict[str, str]]] = Field(
        default_factory=list,
        description="Prior conversation turns [{role, content}]"
    )

    @field_validator("message")
    @classmethod
    def strip_whitespace(cls, v: str) -> str:
        return v.strip()


class AudioRequest(BaseModel):
    audio_b64: str = Field(..., description="Base64-encoded audio bytes")
    mime_type: Literal["audio/wav", "audio/mp3", "audio/ogg", "audio/webm"] = "audio/wav"
    session_id: Optional[str] = None
    language: Optional[str] = Field(None, description="BCP-47 language hint e.g. 'en'")


class VisionRequest(BaseModel):
    image_b64: str = Field(..., description="Base64-encoded image bytes")
    mime_type: Literal["image/jpeg", "image/png", "image/webp", "image/gif"] = "image/jpeg"
    caption: Optional[str] = Field(None, description="Optional user text accompanying the image")
    session_id: Optional[str] = None


# ── NLU Output Models ──────────────────────────────────────────────────────────

class Entity(BaseModel):
    label: str        = Field(..., description="Entity type e.g. ORDER_ID, DATE")
    value: str        = Field(..., description="Extracted entity value")
    confidence: float = Field(..., ge=0.0, le=1.0)


class VisionAnalysis(BaseModel):
    """Extended analysis fields returned only by the Vision pipeline."""
    frustration_score: float  = Field(0.0, ge=0.0, le=1.0,
        description="0–1 measure of visible user frustration signals")
    error_type: ErrorType     = Field(ErrorType.NONE,
        description="Detected UI/technical error category")
    error_detail: Optional[str] = Field(None,
        description="Specific error text visible in the image e.g. '403 Forbidden'")
    visual_summary: Optional[str] = Field(None,
        description="One-sentence description of what the image shows")


class NLUResult(BaseModel):
    intent: IntentCategory        = Field(..., description="Classified intent")
    confidence: float             = Field(..., ge=0.0, le=1.0)
    entities: list[Entity]        = Field(default_factory=list)
    sentiment: Literal[
        "positive", "neutral", "negative", "frustrated"
    ]                             = Field("neutral")
    requires_escalation: bool     = Field(False)
    low_confidence: bool          = Field(False,
        description="True when confidence < 0.70 — triggers Discovery Inbox")
    raw_transcript: Optional[str] = Field(None,
        description="For audio/vision: extracted or described text")
    reasoning: Optional[str]      = Field(None,
        description="LLM chain-of-thought summary")
    vision: Optional[VisionAnalysis] = Field(None,
        description="Present only for vision modality responses")


# ── Transcription-Only Response ────────────────────────────────────────────────

class TranscriptionResult(BaseModel):
    """Returned by the standalone /transcribe endpoint."""
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
    inbox_flagged: bool = Field(False,
        description="True if this result was sent to Discovery Inbox")


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
    confidence: float
    sentiment: str
    requires_escalation: bool
    low_confidence: bool = False
    latency_ms: float
    # Vision-specific (optional)
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
    """A low-confidence NLU result queued for human review."""
    id: str                        = Field(..., description="UUID")
    timestamp: str
    session_id: Optional[str]
    modality: InputModality
    raw_input: str                 = Field(...,
        description="Original user text or '[audio]' / '[image]' placeholder")
    result: NLUResult
    status: InboxStatus            = InboxStatus.PENDING
    reviewer_label: Optional[IntentCategory] = Field(None,
        description="Human-corrected intent label")
    reviewer_note: Optional[str]   = None
    reviewed_at: Optional[str]     = None


class InboxReview(BaseModel):
    """Payload for PATCH /inbox/{id} — submit a human review."""
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
