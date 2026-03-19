"""
AttoSense v4 - Database Layer
Improvements over v3:
  [P] Connection pooling    — explicit pool_size/max_overflow per engine type
  [Q] Session scoping fix   — async with session.begin() guarantees rollback
  [R] CalibrationSample     — stores (raw_conf, correct, modality) per inbox decision
  [S] LabelDisagreement     — tracks reviewer corrections per intent for drift detection
"""
import os
import json
from datetime import datetime, timezone
from typing import Optional, AsyncGenerator
from sqlalchemy import Column, String, Float, Boolean, Text, Integer, DateTime, Index, func
from sqlalchemy.ext.asyncio import AsyncSession, AsyncEngine, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase

def _make_database_url() -> str:
    raw = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./data/attosense.db")
    if raw.startswith("postgres://"):
        raw = raw.replace("postgres://", "postgresql+asyncpg://", 1)
    return raw

DATABASE_URL = _make_database_url()
_IS_SQLITE   = "sqlite" in DATABASE_URL

_engine: Optional[AsyncEngine] = None
_SessionLocal: Optional[async_sessionmaker] = None

def get_engine() -> AsyncEngine:
    global _engine
    if _engine is None:
        # [P] Explicit pool sizing — prevents "too many connections" under load
        if _IS_SQLITE:
            # SQLite doesn't support real connection pooling; use StaticPool
            from sqlalchemy.pool import StaticPool
            _engine = create_async_engine(
                DATABASE_URL, echo=False,
                connect_args={"check_same_thread": False},
                poolclass=StaticPool,
            )
        else:
            # PostgreSQL — tuned pool for production concurrency
            _engine = create_async_engine(
                DATABASE_URL, echo=False,
                pool_pre_ping=True,
                pool_size=5,          # [P] base connections kept alive
                max_overflow=10,      # [P] burst headroom above pool_size
                pool_recycle=1800,    # recycle connections after 30 min
                pool_timeout=30,      # raise after 30s waiting for connection
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
    """
    [Q] FastAPI dependency — scoped session with guaranteed rollback.
    Uses async with session.begin() so any unhandled exception triggers
    an automatic rollback even if the route handler doesn't catch it.
    """
    async with get_session_factory()() as session:
        async with session.begin():
            try:
                yield session
                # commit is automatic at end of `async with session.begin()` block
            except Exception:
                # rollback is automatic on exception exit
                raise

# ── ORM Models ─────────────────────────────────────────────────────────────────
class Base(DeclarativeBase):
    pass

class AuditLog(Base):
    __tablename__ = "audit_log"
    id                  = Column(Integer, primary_key=True, autoincrement=True)
    timestamp           = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    session_id          = Column(String(64),  nullable=True,  index=True)
    modality            = Column(String(16),  nullable=False, index=True)
    intent              = Column(String(64),  nullable=False, index=True)
    confidence          = Column(Float,       nullable=False)
    sentiment           = Column(String(20),  nullable=False)
    requires_escalation = Column(Boolean,     nullable=False, default=False)
    low_confidence      = Column(Boolean,     nullable=False, default=False)
    latency_ms          = Column(Float,       nullable=False)
    frustration_score   = Column(Float,       nullable=True)
    error_type          = Column(String(32),  nullable=True)
    language_detected   = Column(String(16),  nullable=True)
    secondary_intent    = Column(String(64),  nullable=True)
    intent_family       = Column(String(32),  nullable=True)
    sentiment_score     = Column(Float,       nullable=True)
    escalation_reason   = Column(Text,        nullable=True)
    competing_intent    = Column(String(64),  nullable=True)
    competing_confidence= Column(Float,       nullable=True)
    __table_args__ = (
        Index("ix_audit_timestamp", "timestamp"),
        Index("ix_audit_intent_conf", "intent", "confidence"),
    )

class InboxItem(Base):
    __tablename__ = "inbox_items"
    id             = Column(String(36),  primary_key=True)
    timestamp      = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    session_id     = Column(String(64),  nullable=True)
    modality       = Column(String(16),  nullable=False)
    raw_input      = Column(Text,        nullable=False)
    result_json    = Column(Text,        nullable=False)
    status         = Column(String(16),  nullable=False, default="pending", index=True)
    reviewer_label = Column(String(64),  nullable=True)
    reviewer_note  = Column(Text,        nullable=True)
    reviewed_at    = Column(DateTime(timezone=True), nullable=True)

class NLUExample(Base):
    __tablename__ = "nlu_examples"
    id              = Column(Integer,     primary_key=True, autoincrement=True)
    text            = Column(Text,        nullable=False)
    intent          = Column(String(64),  nullable=False, index=True)
    entities_json   = Column(Text,        nullable=False, default="[]")
    source_modality = Column(String(16),  nullable=False, default="text")
    verified        = Column(Boolean,     nullable=False, default=False)
    created_at      = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))

class CalibrationSample(Base):
    """[R] One row per inbox reviewer decision — feeds the confidence calibrator."""
    __tablename__ = "calibration_samples"
    id             = Column(Integer, primary_key=True, autoincrement=True)
    timestamp      = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    modality       = Column(String(16), nullable=False, index=True)
    intent         = Column(String(64), nullable=False, index=True)
    raw_confidence = Column(Float,      nullable=False)
    was_correct    = Column(Boolean,    nullable=False)
    __table_args__ = (Index("ix_cal_modality_intent", "modality", "intent"),)

class LabelDisagreement(Base):
    """[S] Tracks how often each (predicted_intent, corrected_intent) pair occurs."""
    __tablename__ = "label_disagreements"
    id               = Column(Integer,    primary_key=True, autoincrement=True)
    timestamp        = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    modality         = Column(String(16), nullable=False)
    predicted_intent = Column(String(64), nullable=False, index=True)
    corrected_intent = Column(String(64), nullable=False, index=True)
    __table_args__ = (Index("ix_disagreement_pair", "predicted_intent", "corrected_intent"),)

# ── Table creation ─────────────────────────────────────────────────────────────
async def create_tables() -> None:
    async with get_engine().begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def run_migrations() -> None:
    """
    Safe schema migration — adds any columns that exist in the ORM model
    but are missing from the live database (e.g. upgrading from v3 → v4).

    SQLite does not support ALTER TABLE ... ADD COLUMN IF NOT EXISTS, so we
    inspect the existing columns first and only issue ALTER TABLE when needed.
    This function is idempotent — safe to call on every startup.
    """
    # Map: table_name → list of (column_name, column_definition)
    MIGRATIONS: dict[str, list[tuple[str, str]]] = {
        "audit_log": [
            ("language_detected",    "VARCHAR(16)"),
            ("secondary_intent",     "VARCHAR(64)"),
            ("intent_family",        "VARCHAR(32)"),
            ("sentiment_score",      "REAL"),
            ("escalation_reason",    "TEXT"),
            ("competing_intent",     "VARCHAR(64)"),
            ("competing_confidence", "REAL"),
        ],
        # calibration_samples and label_disagreements are brand-new tables
        # created by create_tables() above — no ALTER needed for them.
    }

    async with get_engine().begin() as conn:
        for table, columns in MIGRATIONS.items():
            # Get the current columns for this table
            if _IS_SQLITE:
                result = await conn.execute(
                    __import__("sqlalchemy").text(f"PRAGMA table_info({table})")
                )
                existing = {row[1] for row in result.fetchall()}  # row[1] = column name
            else:
                # PostgreSQL
                result = await conn.execute(
                    __import__("sqlalchemy").text(
                        "SELECT column_name FROM information_schema.columns "
                        f"WHERE table_name = '{table}'"
                    )
                )
                existing = {row[0] for row in result.fetchall()}

            for col_name, col_def in columns:
                if col_name not in existing:
                    await conn.execute(
                        __import__("sqlalchemy").text(
                            f"ALTER TABLE {table} ADD COLUMN {col_name} {col_def}"
                        )
                    )
                    # Import get_logger lazily to avoid circular imports
                    import logging
                    logging.getLogger("attosense.database").info(
                        f"Migration applied: {table}.{col_name} ({col_def})"
                    )

# ── Trigram deduplication helper ───────────────────────────────────────────────
def _trigrams(text: str) -> set[str]:
    text = text.lower().strip()
    return {text[i:i+3] for i in range(len(text)-2)} if len(text) >= 3 else set()

def trigram_similarity(a: str, b: str) -> float:
    """Jaccard similarity between trigram sets of two strings."""
    tA, tB = _trigrams(a), _trigrams(b)
    if not tA and not tB: return 1.0
    if not tA or not tB: return 0.0
    return len(tA & tB) / len(tA | tB)

# ── Aggregate metrics ──────────────────────────────────────────────────────────
async def get_audit_metrics(db: AsyncSession) -> dict:
    from sqlalchemy import select
    total_q = await db.execute(select(func.count()).select_from(AuditLog))
    total   = total_q.scalar() or 0
    if total == 0:
        return {"total_requests":0,"avg_confidence":0.0,"avg_latency_ms":0.0,
                "escalation_rate":0.0,"low_confidence_rate":0.0,
                "intent_distribution":{},"modality_distribution":{},"sentiment_distribution":{}}
    agg = (await db.execute(select(
        func.avg(AuditLog.confidence).label("avg_conf"),
        func.avg(AuditLog.latency_ms).label("avg_lat"),
        func.sum(AuditLog.requires_escalation.cast(Integer)).label("escalations"),
        func.sum(AuditLog.low_confidence.cast(Integer)).label("low_confs"),
    ))).one()
    intent_rows   = (await db.execute(select(AuditLog.intent,   func.count().label("n")).group_by(AuditLog.intent))).all()
    modality_rows = (await db.execute(select(AuditLog.modality, func.count().label("n")).group_by(AuditLog.modality))).all()
    sentiment_rows= (await db.execute(select(AuditLog.sentiment,func.count().label("n")).group_by(AuditLog.sentiment))).all()
    return {
        "total_requests":       total,
        "avg_confidence":       round(float(agg.avg_conf or 0), 4),
        "avg_latency_ms":       round(float(agg.avg_lat  or 0), 2),
        "escalation_rate":      round(float(agg.escalations or 0) / total, 4),
        "low_confidence_rate":  round(float(agg.low_confs  or 0) / total, 4),
        "intent_distribution":  {r.intent:   r.n for r in intent_rows},
        "modality_distribution":{r.modality: r.n for r in modality_rows},
        "sentiment_distribution":{r.sentiment:r.n for r in sentiment_rows},
    }

async def get_dataset_stats(db: AsyncSession) -> dict:
    """Dataset health check — per-intent counts and imbalance flag."""
    from sqlalchemy import select
    total_q = await db.execute(select(func.count()).select_from(NLUExample))
    total   = total_q.scalar() or 0
    if total == 0:
        return {"total":0,"per_intent":{},"imbalanced":False,"imbalance_details":[]}
    rows = (await db.execute(
        select(NLUExample.intent, func.count().label("n")).group_by(NLUExample.intent)
    )).all()
    per_intent = {r.intent: r.n for r in rows}
    max_n = max(per_intent.values()) if per_intent else 1
    min_n = min(per_intent.values()) if per_intent else 0
    imbalance_ratio = max_n / max(min_n, 1)
    imbalanced = imbalance_ratio > 10   # flag if most common > 10× least common
    details = [
        {"intent": k, "count": v, "pct": round(v / total * 100, 1),
         "flag": v / total > 0.40}   # flag any single intent > 40% of dataset
        for k, v in sorted(per_intent.items(), key=lambda x: -x[1])
    ]
    return {
        "total":            total,
        "per_intent":       per_intent,
        "imbalanced":       imbalanced,
        "imbalance_ratio":  round(imbalance_ratio, 1),
        "imbalance_details": details,
    }

async def get_disagreement_stats(db: AsyncSession) -> dict:
    """Correction agreement tracking — surface ambiguous intent boundaries."""
    from sqlalchemy import select
    rows = (await db.execute(
        select(LabelDisagreement.predicted_intent, LabelDisagreement.corrected_intent,
               func.count().label("n"))
        .group_by(LabelDisagreement.predicted_intent, LabelDisagreement.corrected_intent)
        .order_by(func.count().desc())
        .limit(20)
    )).all()
    total_corrections_q = await db.execute(select(func.count()).select_from(LabelDisagreement))
    total = total_corrections_q.scalar() or 0
    pairs = [{"predicted": r.predicted_intent, "corrected": r.corrected_intent, "count": r.n} for r in rows]
    return {"total_corrections": total, "top_disagreements": pairs}
