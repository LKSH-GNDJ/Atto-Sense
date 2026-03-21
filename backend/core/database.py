"""
AttoSense MK1 — Database Layer
SQLAlchemy async, SQLite dev / PostgreSQL prod.
run_migrations() auto-adds missing columns on every startup.
"""
import os, json, logging
from datetime import datetime, timezone
from typing import Optional, AsyncGenerator
from sqlalchemy import Column, String, Float, Boolean, Text, Integer, DateTime, Index, func
from sqlalchemy.ext.asyncio import AsyncSession, AsyncEngine, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase

log = logging.getLogger("attosense.database")


def _make_url() -> str:
    raw = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./data/attosense.db")
    if raw.startswith("postgres://"):
        raw = raw.replace("postgres://", "postgresql+asyncpg://", 1)
    return raw


DATABASE_URL = _make_url()
_IS_SQLITE   = "sqlite" in DATABASE_URL
_engine: Optional[AsyncEngine] = None
_SessionLocal: Optional[async_sessionmaker] = None


def get_engine() -> AsyncEngine:
    global _engine
    if _engine is None:
        if _IS_SQLITE:
            from sqlalchemy.pool import StaticPool
            _engine = create_async_engine(
                DATABASE_URL, echo=False,
                connect_args={"check_same_thread": False},
                poolclass=StaticPool,
            )
        else:
            _engine = create_async_engine(
                DATABASE_URL, echo=False,
                pool_pre_ping=True, pool_size=5, max_overflow=10,
                pool_recycle=1800, pool_timeout=30,
            )
    return _engine


def get_session_factory() -> async_sessionmaker:
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = async_sessionmaker(
            get_engine(), class_=AsyncSession,
            expire_on_commit=False, autoflush=False, autocommit=False,
        )
    return _SessionLocal


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with get_session_factory()() as session:
        async with session.begin():
            try:
                yield session
            except Exception:
                raise


# ── ORM Models ─────────────────────────────────────────────────────────────────

class Base(DeclarativeBase):
    pass


class AuditLog(Base):
    __tablename__ = "audit_log"
    id                   = Column(Integer, primary_key=True, autoincrement=True)
    timestamp            = Column(DateTime(timezone=True), nullable=False,
                                  default=lambda: datetime.now(timezone.utc))
    session_id           = Column(String(64),   nullable=True,  index=True)
    modality             = Column(String(16),   nullable=False, index=True)
    intent               = Column(String(200),  nullable=False, index=True)   # free-form phrase
    intent_domain        = Column(String(32),   nullable=True,  default="information")
    confidence           = Column(Float,        nullable=False)
    sentiment            = Column(String(20),   nullable=False)
    sentiment_score      = Column(Float,        nullable=True)
    requires_escalation  = Column(Boolean,      nullable=False, default=False)
    low_confidence       = Column(Boolean,      nullable=False, default=False)
    latency_ms           = Column(Float,        nullable=False)
    frustration_score    = Column(Float,        nullable=True)
    error_type           = Column(String(32),   nullable=True)
    language_detected    = Column(String(16),   nullable=True)
    escalation_reason    = Column(Text,         nullable=True)
    competing_intent     = Column(String(200),  nullable=True)
    competing_confidence = Column(Float,        nullable=True)
    __table_args__ = (
        Index("ix_audit_timestamp",  "timestamp"),
        Index("ix_audit_domain",     "intent_domain"),
    )


class InboxItem(Base):
    __tablename__ = "inbox_items"
    id             = Column(String(36),  primary_key=True)
    timestamp      = Column(DateTime(timezone=True), nullable=False,
                            default=lambda: datetime.now(timezone.utc))
    session_id     = Column(String(64),  nullable=True)
    modality       = Column(String(16),  nullable=False)
    raw_input      = Column(Text,        nullable=False)
    result_json    = Column(Text,        nullable=False)
    status         = Column(String(16),  nullable=False, default="pending", index=True)
    reviewer_label = Column(String(200), nullable=True)
    reviewer_note  = Column(Text,        nullable=True)
    reviewed_at    = Column(DateTime(timezone=True), nullable=True)


class NLUExample(Base):
    __tablename__ = "nlu_examples"
    id              = Column(Integer,    primary_key=True, autoincrement=True)
    text            = Column(Text,       nullable=False)
    intent          = Column(String(200),nullable=False, index=True)
    intent_domain   = Column(String(32), nullable=True, default="information")
    entities_json   = Column(Text,       nullable=False, default="[]")
    source_modality = Column(String(16), nullable=False, default="text")
    verified        = Column(Boolean,    nullable=False, default=False)
    created_at      = Column(DateTime(timezone=True), nullable=False,
                             default=lambda: datetime.now(timezone.utc))


class CalibrationSample(Base):
    __tablename__ = "calibration_samples"
    id             = Column(Integer, primary_key=True, autoincrement=True)
    timestamp      = Column(DateTime(timezone=True), nullable=False,
                            default=lambda: datetime.now(timezone.utc))
    modality       = Column(String(16), nullable=False, index=True)
    intent         = Column(String(200),nullable=False)
    raw_confidence = Column(Float,      nullable=False)
    was_correct    = Column(Boolean,    nullable=False)


class LabelDisagreement(Base):
    __tablename__ = "label_disagreements"
    id               = Column(Integer,    primary_key=True, autoincrement=True)
    timestamp        = Column(DateTime(timezone=True), nullable=False,
                              default=lambda: datetime.now(timezone.utc))
    modality         = Column(String(16), nullable=False)
    predicted_intent = Column(String(200),nullable=False, index=True)
    corrected_intent = Column(String(200),nullable=False, index=True)


# ── Table creation + migrations ────────────────────────────────────────────────

async def create_tables() -> None:
    async with get_engine().begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def run_migrations() -> None:
    """Safely add any columns missing from an older DB. Idempotent."""
    from sqlalchemy import text
    MIGRATIONS = {
        "audit_log": [
            ("intent_domain",        "VARCHAR(32)"),
            ("sentiment_score",      "REAL"),
            ("escalation_reason",    "TEXT"),
            ("competing_intent",     "VARCHAR(200)"),
            ("competing_confidence", "REAL"),
            ("language_detected",    "VARCHAR(16)"),
        ],
        "nlu_examples": [
            ("intent_domain", "VARCHAR(32)"),
        ],
    }
    async with get_engine().begin() as conn:
        for table, columns in MIGRATIONS.items():
            if _IS_SQLITE:
                result = await conn.execute(text(f"PRAGMA table_info({table})"))
                existing = {row[1] for row in result.fetchall()}
            else:
                result = await conn.execute(text(
                    "SELECT column_name FROM information_schema.columns "
                    f"WHERE table_name = '{table}'"
                ))
                existing = {row[0] for row in result.fetchall()}

            for col, defn in columns:
                if col not in existing:
                    await conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {col} {defn}"))
                    log.info(f"Migration: added {table}.{col}")


# ── Helpers ─────────────────────────────────────────────────────────────────────

def _trigrams(text: str) -> set[str]:
    t = text.lower().strip()
    return {t[i:i+3] for i in range(len(t)-2)} if len(t) >= 3 else set()

def trigram_similarity(a: str, b: str) -> float:
    ta, tb = _trigrams(a), _trigrams(b)
    if not ta and not tb: return 1.0
    if not ta or not tb:  return 0.0
    return len(ta & tb) / len(ta | tb)


async def get_audit_metrics(db: AsyncSession) -> dict:
    from sqlalchemy import select
    total_q = await db.execute(select(func.count()).select_from(AuditLog))
    total   = total_q.scalar() or 0
    if total == 0:
        return {"total_requests":0,"avg_confidence":0.0,"avg_latency_ms":0.0,
                "escalation_rate":0.0,"low_confidence_rate":0.0,
                "intent_distribution":{},"domain_distribution":{},
                "modality_distribution":{},"sentiment_distribution":{}}
    agg = (await db.execute(select(
        func.avg(AuditLog.confidence).label("avg_conf"),
        func.avg(AuditLog.latency_ms).label("avg_lat"),
        func.sum(AuditLog.requires_escalation.cast(Integer)).label("escalations"),
        func.sum(AuditLog.low_confidence.cast(Integer)).label("low_confs"),
    ))).one()
    intent_rows   = (await db.execute(select(AuditLog.intent,       func.count().label("n")).group_by(AuditLog.intent))).all()
    domain_rows   = (await db.execute(select(AuditLog.intent_domain,func.count().label("n")).group_by(AuditLog.intent_domain))).all()
    modality_rows = (await db.execute(select(AuditLog.modality,     func.count().label("n")).group_by(AuditLog.modality))).all()
    sentiment_rows= (await db.execute(select(AuditLog.sentiment,    func.count().label("n")).group_by(AuditLog.sentiment))).all()
    return {
        "total_requests":        total,
        "avg_confidence":        round(float(agg.avg_conf or 0), 4),
        "avg_latency_ms":        round(float(agg.avg_lat  or 0), 2),
        "escalation_rate":       round(float(agg.escalations or 0) / total, 4),
        "low_confidence_rate":   round(float(agg.low_confs  or 0) / total, 4),
        "intent_distribution":   {r.intent:        r.n for r in intent_rows if r.intent},
        "domain_distribution":   {r.intent_domain: r.n for r in domain_rows  if r.intent_domain},
        "modality_distribution": {r.modality:      r.n for r in modality_rows},
        "sentiment_distribution":{r.sentiment:     r.n for r in sentiment_rows},
    }


async def get_dataset_stats(db: AsyncSession) -> dict:
    from sqlalchemy import select
    total = (await db.execute(select(func.count()).select_from(NLUExample))).scalar() or 0
    if total == 0:
        return {"total":0,"per_intent":{},"per_domain":{},"imbalanced":False,"imbalance_details":[]}
    intent_rows = (await db.execute(select(NLUExample.intent, func.count().label("n")).group_by(NLUExample.intent))).all()
    domain_rows = (await db.execute(select(NLUExample.intent_domain, func.count().label("n")).group_by(NLUExample.intent_domain))).all()
    per_intent  = {r.intent: r.n for r in intent_rows}
    per_domain  = {r.intent_domain: r.n for r in domain_rows}
    max_n = max(per_intent.values()) if per_intent else 1
    min_n = min(per_intent.values()) if per_intent else 0
    ratio = max_n / max(min_n, 1)
    details = [{"intent":k,"count":v,"pct":round(v/total*100,1),"flag":v/total>0.40}
               for k,v in sorted(per_intent.items(), key=lambda x:-x[1])]
    return {"total":total,"per_intent":per_intent,"per_domain":per_domain,
            "imbalanced":ratio>10,"imbalance_ratio":round(ratio,1),"imbalance_details":details}


async def get_disagreement_stats(db: AsyncSession) -> dict:
    from sqlalchemy import select
    rows = (await db.execute(
        select(LabelDisagreement.predicted_intent, LabelDisagreement.corrected_intent,
               func.count().label("n"))
        .group_by(LabelDisagreement.predicted_intent, LabelDisagreement.corrected_intent)
        .order_by(func.count().desc()).limit(20)
    )).all()
    total = (await db.execute(select(func.count()).select_from(LabelDisagreement))).scalar() or 0
    return {"total_corrections":total,
            "top_disagreements":[{"predicted":r.predicted_intent,"corrected":r.corrected_intent,"count":r.n} for r in rows]}
