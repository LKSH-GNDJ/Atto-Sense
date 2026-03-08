"""
BotTrain v3.1 - Database Layer
SQLAlchemy + SQLite (dev) / PostgreSQL (prod) via DATABASE_URL env var.
Replaces all JSONL flat files — safe under concurrent async writes.

Tables:
  audit_log    — every classify/transcribe call
  inbox_items  — low-confidence flags for human review
  nlu_examples — verified training examples
"""

import os
import json
from datetime import datetime, timezone
from typing import Optional, AsyncGenerator

from sqlalchemy import (
    Column, String, Float, Boolean, Text, Integer,
    DateTime, Index, text,
)
from sqlalchemy.ext.asyncio import (
    AsyncSession, AsyncEngine, create_async_engine, async_sessionmaker,
)
from sqlalchemy.orm import DeclarativeBase

# ── Engine ─────────────────────────────────────────────────────────────────────

def _make_database_url() -> str:
    raw = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./data/bottrain.db")
    # Heroku / Railway ship postgres:// — SQLAlchemy needs postgresql+asyncpg://
    if raw.startswith("postgres://"):
        raw = raw.replace("postgres://", "postgresql+asyncpg://", 1)
    return raw


DATABASE_URL = _make_database_url()

_engine: Optional[AsyncEngine] = None
_SessionLocal: Optional[async_sessionmaker] = None


def get_engine() -> AsyncEngine:
    global _engine
    if _engine is None:
        connect_args = {}
        if "sqlite" in DATABASE_URL:
            connect_args = {"check_same_thread": False}
        _engine = create_async_engine(
            DATABASE_URL,
            echo=False,                   # set True for SQL debug logs
            pool_pre_ping=True,           # validates connections before use
            pool_recycle=1800,            # recycle after 30 min
            connect_args=connect_args,
        )
    return _engine


def get_session_factory() -> async_sessionmaker:
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = async_sessionmaker(
            get_engine(),
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
            autocommit=False,
        )
    return _SessionLocal


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency — yields a DB session per request."""
    async with get_session_factory()() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


# ── ORM Models ─────────────────────────────────────────────────────────────────

class Base(DeclarativeBase):
    pass


class AuditLog(Base):
    __tablename__ = "audit_log"

    id                = Column(Integer, primary_key=True, autoincrement=True)
    timestamp         = Column(DateTime(timezone=True), nullable=False,
                               default=lambda: datetime.now(timezone.utc))
    session_id        = Column(String(64),  nullable=True,  index=True)
    modality          = Column(String(16),  nullable=False, index=True)
    intent            = Column(String(64),  nullable=False, index=True)
    confidence        = Column(Float,       nullable=False)
    sentiment         = Column(String(20),  nullable=False)
    requires_escalation = Column(Boolean,   nullable=False, default=False)
    low_confidence    = Column(Boolean,     nullable=False, default=False)
    latency_ms        = Column(Float,       nullable=False)
    frustration_score = Column(Float,       nullable=True)
    error_type        = Column(String(32),  nullable=True)

    __table_args__ = (
        Index("ix_audit_timestamp", "timestamp"),
        Index("ix_audit_intent_conf", "intent", "confidence"),
    )


class InboxItem(Base):
    __tablename__ = "inbox_items"

    id               = Column(String(36),  primary_key=True)   # UUID
    timestamp        = Column(DateTime(timezone=True), nullable=False,
                              default=lambda: datetime.now(timezone.utc))
    session_id       = Column(String(64),  nullable=True)
    modality         = Column(String(16),  nullable=False)
    raw_input        = Column(Text,        nullable=False)
    result_json      = Column(Text,        nullable=False)      # JSON-serialised NLUResult
    status           = Column(String(16),  nullable=False, default="pending", index=True)
    reviewer_label   = Column(String(64),  nullable=True)
    reviewer_note    = Column(Text,        nullable=True)
    reviewed_at      = Column(DateTime(timezone=True), nullable=True)


class NLUExample(Base):
    __tablename__ = "nlu_examples"

    id              = Column(Integer,     primary_key=True, autoincrement=True)
    text            = Column(Text,        nullable=False)
    intent          = Column(String(64),  nullable=False, index=True)
    entities_json   = Column(Text,        nullable=False, default="[]")
    source_modality = Column(String(16),  nullable=False, default="text")
    verified        = Column(Boolean,     nullable=False, default=False)
    created_at      = Column(DateTime(timezone=True), nullable=False,
                             default=lambda: datetime.now(timezone.utc))


# ── Table Creation ─────────────────────────────────────────────────────────────

async def create_tables() -> None:
    """Create all tables if they don't exist. Called at app startup."""
    async with get_engine().begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


# ── Aggregate Query Helpers ────────────────────────────────────────────────────

async def get_audit_metrics(db: AsyncSession) -> dict:
    """
    Return pre-aggregated metrics directly from the DB.
    Much faster than reading all rows into Python.
    """
    from sqlalchemy import func, select

    # Totals
    total_q = await db.execute(select(func.count()).select_from(AuditLog))
    total   = total_q.scalar() or 0

    if total == 0:
        return {
            "total_requests": 0, "avg_confidence": 0.0, "avg_latency_ms": 0.0,
            "escalation_rate": 0.0, "low_confidence_rate": 0.0,
            "intent_distribution": {}, "modality_distribution": {},
            "sentiment_distribution": {},
        }

    agg_q = await db.execute(
        select(
            func.avg(AuditLog.confidence).label("avg_conf"),
            func.avg(AuditLog.latency_ms).label("avg_lat"),
            func.sum(AuditLog.requires_escalation.cast(Integer)).label("escalations"),
            func.sum(AuditLog.low_confidence.cast(Integer)).label("low_confs"),
        )
    )
    agg = agg_q.one()

    # Distributions
    intent_q = await db.execute(
        select(AuditLog.intent, func.count().label("n"))
        .group_by(AuditLog.intent)
    )
    modality_q = await db.execute(
        select(AuditLog.modality, func.count().label("n"))
        .group_by(AuditLog.modality)
    )
    sentiment_q = await db.execute(
        select(AuditLog.sentiment, func.count().label("n"))
        .group_by(AuditLog.sentiment)
    )

    return {
        "total_requests":       total,
        "avg_confidence":       round(float(agg.avg_conf or 0), 4),
        "avg_latency_ms":       round(float(agg.avg_lat or 0), 2),
        "escalation_rate":      round(float(agg.escalations or 0) / total, 4),
        "low_confidence_rate":  round(float(agg.low_confs or 0) / total, 4),
        "intent_distribution":   {r.intent:   r.n for r in intent_q.all()},
        "modality_distribution": {r.modality: r.n for r in modality_q.all()},
        "sentiment_distribution":{r.sentiment:r.n for r in sentiment_q.all()},
    }
