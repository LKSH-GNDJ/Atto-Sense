"""
AttoSense MK1 — Pydantic Schemas
Intent is a free-form 2-5 word action phrase.
IntentDomain is the 7-way coarse grouping for routing and analytics.
"""
from pydantic import BaseModel, Field, field_validator
from typing import Optional, Literal, Any
from enum import Enum


class InputModality(str, Enum):
    TEXT   = "text"
    AUDIO  = "audio"
    VISION = "vision"


class IntentDomain(str, Enum):
    INFORMATION = "information"
    ACTION      = "action"
    PROBLEM     = "problem"
    TRANSACTION = "transaction"
    CREATIVE    = "creative"
    PERSONAL    = "personal"
    TECHNICAL   = "technical"


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


# ── Requests ───────────────────────────────────────────────────────────────────

class TextRequest(BaseModel):
    message:    str = Field(..., min_length=1, max_length=16000)
    session_id: Optional[str]                  = Field(None)
    context:    Optional[list[dict[str, str]]] = Field(default_factory=list)

    @field_validator("message")
    @classmethod
    def strip_whitespace(cls, v: str) -> str:
        return v.strip()


class AudioRequest(BaseModel):
    audio_b64:  str = Field(...)
    mime_type:  Literal["audio/wav","audio/mp3","audio/ogg","audio/webm"] = "audio/wav"
    session_id: Optional[str] = None
    language:   Optional[str] = Field(None)


class VisionRequest(BaseModel):
    image_b64:  str = Field(...)
    mime_type:  Literal["image/jpeg","image/png","image/webp","image/gif"] = "image/jpeg"
    caption:    Optional[str] = Field(None)
    session_id: Optional[str] = None


# ── Output ─────────────────────────────────────────────────────────────────────

class Entity(BaseModel):
    label:      str   = Field(...)
    value:      str   = Field(...)
    confidence: float = Field(..., ge=0.0, le=1.0)


class VisionAnalysis(BaseModel):
    frustration_score: float         = Field(0.0, ge=0.0, le=1.0)
    error_type:        ErrorType     = Field(ErrorType.NONE)
    error_detail:      Optional[str] = Field(None)
    visual_summary:    Optional[str] = Field(None)
    screen_type:       Optional[str] = Field(None)


class NLUResult(BaseModel):
    intent:        str          = Field(..., description="Free-form 2-5 word action phrase")
    intent_domain: IntentDomain = Field(...)
    confidence:    float        = Field(..., ge=0.0, le=1.0)

    confidence_scores:    dict[str, float] = Field(default_factory=dict)
    competing_intent:     Optional[str]    = Field(None)
    competing_confidence: float            = Field(0.0, ge=0.0, le=1.0)

    entities:             list[Entity]     = Field(default_factory=list)

    sentiment:            Literal["positive","neutral","negative","frustrated"] = Field("neutral")
    sentiment_score:      float            = Field(0.0, ge=-1.0, le=1.0)

    requires_escalation:  bool             = Field(False)
    escalation_reason:    Optional[str]    = Field(None)

    reasoning_steps:      list[str]        = Field(default_factory=list)

    low_confidence:       bool             = Field(False)
    raw_transcript:       Optional[str]    = Field(None)
    modality:             Optional[str]    = Field(None)
    language_detected:    Optional[str]    = Field(None)

    vision:               Optional[VisionAnalysis] = Field(None)


class TranscriptionResult(BaseModel):
    success:           bool
    transcript:        str
    language_detected: Optional[str]   = None
    duration_seconds:  Optional[float] = None
    model_used:        str
    latency_ms:        float
    session_id:        Optional[str]


class ClassifyResponse(BaseModel):
    success:       bool
    modality:      InputModality
    session_id:    Optional[str]
    result:        NLUResult
    model_used:    str
    latency_ms:    float
    inbox_flagged: bool = Field(False)


class ErrorResponse(BaseModel):
    success: bool = False
    error:   str
    detail:  Optional[Any] = None


class MetricsSummary(BaseModel):
    total_requests:         int
    avg_confidence:         float
    avg_latency_ms:         float
    intent_distribution:    dict[str, int]
    domain_distribution:    dict[str, int] = Field(default_factory=dict)
    modality_distribution:  dict[str, int]
    escalation_rate:        float
    low_confidence_rate:    float
    sentiment_distribution: dict[str, int]
    inbox_pending:          int = 0


class InboxItem(BaseModel):
    id:             str           = Field(...)
    timestamp:      str
    session_id:     Optional[str]
    modality:       InputModality
    raw_input:      str
    result:         NLUResult
    status:         InboxStatus   = InboxStatus.PENDING
    reviewer_label: Optional[str] = None
    reviewer_note:  Optional[str] = None
    reviewed_at:    Optional[str] = None


class InboxReview(BaseModel):
    status:         InboxStatus
    reviewer_label: Optional[str] = None
    reviewer_note:  Optional[str] = None


class InboxSummary(BaseModel):
    total: int; pending: int; reviewed: int; approved: int; rejected: int
    items: list[InboxItem]


class NLUExample(BaseModel):
    text:            str
    intent:          str
    intent_domain:   str           = "information"
    entities:        list[Entity]  = Field(default_factory=list)
    source_modality: InputModality = InputModality.TEXT
    verified:        bool          = False
