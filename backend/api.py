"""
BotTrain v3.1 - FastAPI Backend (Production)
Production features implemented:
  [1] Auth         — AuthMiddleware (X-API-Key) on all non-public routes
  [2] Database     — SQLAlchemy async (SQLite dev / Postgres prod)
  [3] Rate limits  — slowapi: 60/min classify, 30/min upload, 120/min read
  [4] Upload caps  — 25 MB hard limit on audio/image uploads
  [5] Retry/backoff— tenacity in multimodal.py (3 attempts, exp backoff)
  [6] Timeouts     — asyncio.wait_for(45s) wraps every Groq call
  [7] Logging      — JSON structured logs via logging_config.py
  [8] Health depth — probes Groq + DB + disk write
  [9] Metrics cache— TTLCache (30s) prevents full-table scan per dashboard poll
  [10] Groq client — singleton, reused across all requests
"""

import os
import uuid
import json
import asyncio
from datetime import datetime, timezone
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional

from dotenv import load_dotenv
from cachetools import TTLCache
import asyncio

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, delete, update

from backend.core.auth import AuthMiddleware
from backend.core.database import (
    create_tables, get_db, get_audit_metrics,
    AuditLog as DBAudit,
    InboxItem as DBInbox,
    NLUExample as DBExample,
)
from backend.core.logging_config import configure_logging, get_logger, RequestLoggingMiddleware
from backend.models.schemas import (
    TextRequest, AudioRequest, VisionRequest,
    ClassifyResponse, TranscriptionResult,
    InputModality, MetricsSummary,
    NLUExample, InboxItem, InboxReview, InboxSummary, InboxStatus,
)
from backend.core.multimodal import (
    classify_text_async, classify_audio_async, classify_vision_async,
    transcribe_audio_async, probe_groq,
)

load_dotenv()
configure_logging()
log = get_logger("bottrain.api")

# ── Config ─────────────────────────────────────────────────────────────────────

MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(25 * 1024 * 1024)))  # 25 MB
INBOX_THRESHOLD  = 0.70
DATA_DIR         = Path("data")

# ── Metrics TTL cache (avoids full-table scan on every sidebar poll) ───────────
# maxsize=1 — we cache one result; TTL=30s — refreshes every 30 seconds
_metrics_cache: TTLCache = TTLCache(maxsize=1, ttl=30)


# ── App Lifespan ───────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    DATA_DIR.mkdir(exist_ok=True)
    await create_tables()
    log.info("app_startup", extra={"version": "3.1.1", "db": os.getenv("DATABASE_URL", "sqlite")})
    yield
    log.info("app_shutdown")


# ── Rate Limiter ───────────────────────────────────────────────────────────────

limiter = Limiter(key_func=get_remote_address, default_limits=["200/minute"])


# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="BotTrain v3.1 API",
    description="Multimodal Zero-Shot NLU — Production Build",
    version="3.1.1",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Middleware stack (order matters — outermost runs first)
app.add_middleware(AuthMiddleware)           # [1] Auth — must be before CORS
app.add_middleware(RequestLoggingMiddleware) # [7] Structured request logs
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# ── Upload Size Guard ──────────────────────────────────────────────────────────

async def _read_upload(file: UploadFile) -> bytes:
    """Read upload and enforce MAX_UPLOAD_BYTES."""
    content = await file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        mb = MAX_UPLOAD_BYTES // (1024 * 1024)
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum allowed size is {mb} MB.",
        )
    return content


# ── Audit Logger (writes to DB) ────────────────────────────────────────────────

async def _log_audit(db: AsyncSession, result, modality: InputModality,
                     session_id: str, latency_ms: float) -> None:
    vision = result.vision
    entry  = DBAudit(
        timestamp           = datetime.now(timezone.utc),
        session_id          = session_id,
        modality            = modality.value,
        intent              = result.intent.value,
        confidence          = result.confidence,
        sentiment           = result.sentiment,
        requires_escalation = result.requires_escalation,
        low_confidence      = result.low_confidence,
        latency_ms          = latency_ms,
        frustration_score   = vision.frustration_score if vision else None,
        error_type          = vision.error_type.value  if vision else None,
    )
    db.add(entry)
    _metrics_cache.clear()   # invalidate cache after each new write


# ── Inbox Helper ───────────────────────────────────────────────────────────────

async def _maybe_inbox(db: AsyncSession, session_id: str, modality: InputModality,
                       raw_input: str, result) -> bool:
    if not result.low_confidence:
        return False
    item = DBInbox(
        id          = str(uuid.uuid4()),
        timestamp   = datetime.now(timezone.utc),
        session_id  = session_id,
        modality    = modality.value,
        raw_input   = raw_input[:500],
        result_json = result.model_dump_json(),
        status      = "pending",
    )
    db.add(item)
    log.info("inbox_flagged", extra={
        "session_id": session_id, "confidence": result.confidence,
        "intent": result.intent.value,
    })
    return True


# ── Shared Post-Classify ───────────────────────────────────────────────────────

async def _finalize(
    db: AsyncSession,
    result, model: str, latency: float,
    modality: InputModality, session_id: str, raw_input: str,
) -> ClassifyResponse:
    await _log_audit(db, result, modality, session_id, latency)
    flagged = await _maybe_inbox(db, session_id, modality, raw_input, result)
    return ClassifyResponse(
        success=True, modality=modality, session_id=session_id,
        result=result, model_used=model, latency_ms=latency,
        inbox_flagged=flagged,
    )


# ── Health (deep probe) ────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
async def health_check(db: AsyncSession = Depends(get_db)):
    checks = {}

    # DB connectivity
    try:
        await db.execute(select(func.count()).select_from(DBAudit))
        checks["database"] = "ok"
    except Exception as e:
        checks["database"] = f"error: {e}"

    # Disk writeable
    try:
        probe = DATA_DIR / ".health_probe"
        probe.write_text("ok")
        probe.unlink()
        checks["disk"] = "ok"
    except Exception as e:
        checks["disk"] = f"error: {e}"

    # Groq reachable (skip in test environments)
    if os.getenv("SKIP_GROQ_HEALTH_PROBE", "false").lower() != "true":
        checks["groq"] = "ok" if await probe_groq() else "unreachable"
    else:
        checks["groq"] = "skipped"

    overall = "ok" if all(v == "ok" or v == "skipped" for v in checks.values()) else "degraded"

    return {
        "status":           overall,
        "version":          "3.1.1",
        "timestamp":        datetime.now(timezone.utc).isoformat(),
        "groq_key_set":     bool(os.getenv("GROQ_API_KEY")),
        "inbox_threshold":  INBOX_THRESHOLD,
        "max_upload_mb":    MAX_UPLOAD_BYTES // (1024 * 1024),
        "checks":           checks,
    }


# ══════════════════════════════════════════════════════════════════════════════
# CLASSIFICATION ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/classify/text", response_model=ClassifyResponse, tags=["Classification"])
@limiter.limit("60/minute")
async def classify_text_endpoint(
    request: Request,
    body: TextRequest,
    db: AsyncSession = Depends(get_db),
):
    sid = body.session_id or str(uuid.uuid4())
    try:
        result, model, latency = await classify_text_async(body.message, body.context)
    except TimeoutError as e:
        raise HTTPException(status_code=504, detail=str(e))
    except Exception as e:
        log.error("classify_text_failed", extra={"error": str(e)})
        raise HTTPException(status_code=500, detail=str(e))
    return await _finalize(db, result, model, latency, InputModality.TEXT, sid, body.message)


@app.post("/classify/audio", response_model=ClassifyResponse, tags=["Classification"])
@limiter.limit("30/minute")
async def classify_audio_endpoint(
    request: Request,
    body: AudioRequest,
    db: AsyncSession = Depends(get_db),
):
    sid = body.session_id or str(uuid.uuid4())
    try:
        result, model, latency = await classify_audio_async(body.audio_b64, body.mime_type, body.language)
    except TimeoutError as e:
        raise HTTPException(status_code=504, detail=str(e))
    except Exception as e:
        log.error("classify_audio_failed", extra={"error": str(e)})
        raise HTTPException(status_code=500, detail=str(e))
    raw = result.raw_transcript or "[audio]"
    return await _finalize(db, result, model, latency, InputModality.AUDIO, sid, raw)


@app.post("/classify/audio/upload", response_model=ClassifyResponse, tags=["Classification"])
@limiter.limit("30/minute")
async def classify_audio_upload(
    request: Request,
    file: UploadFile = File(...),
    session_id: str  = Form(default=None),
    language: str    = Form(default=None),
    db: AsyncSession = Depends(get_db),
):
    import base64 as _b64
    content   = await _read_upload(file)       # [4] size cap enforced here
    audio_b64 = _b64.b64encode(content).decode()
    mime_type = file.content_type or "audio/wav"
    sid       = session_id or str(uuid.uuid4())
    try:
        result, model, latency = await classify_audio_async(audio_b64, mime_type, language)
    except TimeoutError as e:
        raise HTTPException(status_code=504, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    raw = result.raw_transcript or "[audio upload]"
    return await _finalize(db, result, model, latency, InputModality.AUDIO, sid, raw)


@app.post("/classify/vision", response_model=ClassifyResponse, tags=["Classification"])
@limiter.limit("30/minute")
async def classify_vision_endpoint(
    request: Request,
    body: VisionRequest,
    db: AsyncSession = Depends(get_db),
):
    sid = body.session_id or str(uuid.uuid4())
    try:
        result, model, latency = await classify_vision_async(body.image_b64, body.mime_type, body.caption)
    except TimeoutError as e:
        raise HTTPException(status_code=504, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return await _finalize(db, result, model, latency, InputModality.VISION, sid, body.caption or "[image]")


@app.post("/classify/vision/upload", response_model=ClassifyResponse, tags=["Classification"])
@limiter.limit("30/minute")
async def classify_vision_upload(
    request: Request,
    file: UploadFile = File(...),
    caption: str     = Form(default=None),
    session_id: str  = Form(default=None),
    db: AsyncSession = Depends(get_db),
):
    import base64 as _b64
    content   = await _read_upload(file)       # [4] size cap
    image_b64 = _b64.b64encode(content).decode()
    mime_type = file.content_type or "image/jpeg"
    sid       = session_id or str(uuid.uuid4())
    try:
        result, model, latency = await classify_vision_async(image_b64, mime_type, caption)
    except TimeoutError as e:
        raise HTTPException(status_code=504, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return await _finalize(db, result, model, latency, InputModality.VISION, sid, caption or "[image upload]")


# ── Batch Text (up to 20 inputs, concurrent) ──────────────────────────────────

@app.post("/classify/batch", tags=["Classification"])
@limiter.limit("10/minute")
async def classify_batch(
    request: Request,
    messages: list[str],
    session_id: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
):
    if not messages:
        raise HTTPException(status_code=422, detail="messages list is empty.")
    if len(messages) > 20:
        raise HTTPException(status_code=422, detail="Maximum 20 messages per batch.")
    sid = session_id or str(uuid.uuid4())
    tasks    = [classify_text_async(m) for m in messages]
    results  = await asyncio.gather(*tasks, return_exceptions=True)
    output   = []
    for msg, res in zip(messages, results):
        if isinstance(res, Exception):
            output.append({"input": msg[:80], "error": str(res)})
        else:
            result, model, latency = res
            resp = await _finalize(db, result, model, latency, InputModality.TEXT, sid, msg)
            output.append(resp.model_dump())
    return {"batch_size": len(messages), "session_id": sid, "results": output}


# ══════════════════════════════════════════════════════════════════════════════
# TRANSCRIPTION
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/transcribe", response_model=TranscriptionResult, tags=["Transcription"])
@limiter.limit("30/minute")
async def transcribe_endpoint(request: Request, body: AudioRequest):
    sid = body.session_id or str(uuid.uuid4())
    try:
        return await transcribe_audio_async(body.audio_b64, body.mime_type, body.language, sid)
    except TimeoutError as e:
        raise HTTPException(status_code=504, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/transcribe/upload", response_model=TranscriptionResult, tags=["Transcription"])
@limiter.limit("30/minute")
async def transcribe_upload(
    request: Request,
    file: UploadFile = File(...),
    session_id: str  = Form(default=None),
    language: str    = Form(default=None),
):
    import base64 as _b64
    content   = await _read_upload(file)
    audio_b64 = _b64.b64encode(content).decode()
    mime_type = file.content_type or "audio/wav"
    sid       = session_id or str(uuid.uuid4())
    try:
        return await transcribe_audio_async(audio_b64, mime_type, language, sid)
    except TimeoutError as e:
        raise HTTPException(status_code=504, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ══════════════════════════════════════════════════════════════════════════════
# DISCOVERY INBOX
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/inbox", response_model=InboxSummary, tags=["Discovery Inbox"])
@limiter.limit("120/minute")
async def get_inbox(
    request: Request,
    status: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
):
    q = select(DBInbox).order_by(DBInbox.timestamp.desc())
    if status:
        q = q.where(DBInbox.status == status)
    total_q = await db.execute(select(func.count()).select_from(DBInbox).where(
        DBInbox.status == status if status else True
    ))
    total = total_q.scalar() or 0

    rows = (await db.execute(q.offset(offset).limit(limit))).scalars().all()

    counts_q = await db.execute(
        select(DBInbox.status, func.count().label("n")).group_by(DBInbox.status)
    )
    counts = {r.status: r.n for r in counts_q.all()}

    items = []
    for row in rows:
        try:
            result_data = json.loads(row.result_json)
            from backend.models.schemas import NLUResult
            items.append(InboxItem(
                id=row.id, timestamp=row.timestamp.isoformat(),
                session_id=row.session_id, modality=row.modality,
                raw_input=row.raw_input, result=NLUResult(**result_data),
                status=row.status,
                reviewer_label=row.reviewer_label,
                reviewer_note=row.reviewer_note,
                reviewed_at=row.reviewed_at.isoformat() if row.reviewed_at else None,
            ))
        except Exception:
            continue

    return InboxSummary(
        total=total,
        pending=counts.get("pending", 0),
        reviewed=counts.get("reviewed", 0),
        approved=counts.get("approved", 0),
        rejected=counts.get("rejected", 0),
        items=items,
    )


@app.get("/inbox/{item_id}", response_model=InboxItem, tags=["Discovery Inbox"])
async def get_inbox_item(item_id: str, db: AsyncSession = Depends(get_db)):
    row = (await db.execute(select(DBInbox).where(DBInbox.id == item_id))).scalar_one_or_none()
    if not row:
        raise HTTPException(404, f"Inbox item {item_id} not found.")
    from backend.models.schemas import NLUResult
    return InboxItem(
        id=row.id, timestamp=row.timestamp.isoformat(),
        session_id=row.session_id, modality=row.modality,
        raw_input=row.raw_input, result=NLUResult(**json.loads(row.result_json)),
        status=row.status, reviewer_label=row.reviewer_label,
        reviewer_note=row.reviewer_note,
        reviewed_at=row.reviewed_at.isoformat() if row.reviewed_at else None,
    )


@app.patch("/inbox/{item_id}", response_model=InboxItem, tags=["Discovery Inbox"])
async def review_inbox_item(item_id: str, review: InboxReview,
                             db: AsyncSession = Depends(get_db)):
    row = (await db.execute(select(DBInbox).where(DBInbox.id == item_id))).scalar_one_or_none()
    if not row:
        raise HTTPException(404, f"Inbox item {item_id} not found.")

    row.status      = review.status.value
    row.reviewed_at = datetime.now(timezone.utc)
    if review.reviewer_label:
        row.reviewer_label = review.reviewer_label.value
    if review.reviewer_note:
        row.reviewer_note  = review.reviewer_note

    # Auto-promote approved + corrected → NLU dataset
    if review.status == InboxStatus.APPROVED and review.reviewer_label:
        result_data = json.loads(row.result_json)
        db.add(DBExample(
            text=row.raw_input,
            intent=review.reviewer_label.value,
            entities_json=json.dumps(result_data.get("entities", [])),
            source_modality=row.modality,
            verified=True,
        ))
        log.info("example_promoted", extra={
            "inbox_id": item_id, "intent": review.reviewer_label.value
        })

    from backend.models.schemas import NLUResult
    return InboxItem(
        id=row.id, timestamp=row.timestamp.isoformat(),
        session_id=row.session_id, modality=row.modality,
        raw_input=row.raw_input, result=NLUResult(**json.loads(row.result_json)),
        status=row.status, reviewer_label=row.reviewer_label,
        reviewer_note=row.reviewer_note,
        reviewed_at=row.reviewed_at.isoformat() if row.reviewed_at else None,
    )


@app.delete("/inbox/{item_id}", tags=["Discovery Inbox"])
async def delete_inbox_item(item_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(delete(DBInbox).where(DBInbox.id == item_id))
    if result.rowcount == 0:
        raise HTTPException(404, f"Inbox item {item_id} not found.")
    return {"success": True, "deleted": item_id}


@app.delete("/inbox", tags=["Discovery Inbox"])
async def clear_inbox(status: Optional[str] = None, db: AsyncSession = Depends(get_db)):
    q = delete(DBInbox)
    if status:
        q = q.where(DBInbox.status == status)
    await db.execute(q)
    return {"success": True, "cleared_status": status or "all"}


# ══════════════════════════════════════════════════════════════════════════════
# METRICS & AUDIT
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/metrics", response_model=MetricsSummary, tags=["Analytics"])
@limiter.limit("120/minute")
async def get_metrics(request: Request, db: AsyncSession = Depends(get_db)):
    # [9] TTL cache — avoids full-table scan on every poll
    if "metrics" in _metrics_cache:
        return MetricsSummary(**_metrics_cache["metrics"])

    metrics = await get_audit_metrics(db)

    pending_q = await db.execute(
        select(func.count()).select_from(DBInbox).where(DBInbox.status == "pending")
    )
    metrics["inbox_pending"] = pending_q.scalar() or 0

    _metrics_cache["metrics"] = metrics
    return MetricsSummary(**metrics)


@app.get("/audit", tags=["Analytics"])
@limiter.limit("60/minute")
async def get_audit_log(
    request: Request,
    limit: int = 100,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
):
    total_q = await db.execute(select(func.count()).select_from(DBAudit))
    total   = total_q.scalar() or 0
    rows    = (await db.execute(
        select(DBAudit).order_by(DBAudit.timestamp.desc()).offset(offset).limit(limit)
    )).scalars().all()
    entries = [
        {c.name: getattr(r, c.name) for c in DBAudit.__table__.columns}
        for r in rows
    ]
    # Serialise datetime objects
    for e in entries:
        if e.get("timestamp"):
            e["timestamp"] = e["timestamp"].isoformat()
    return {"total": total, "offset": offset, "limit": limit, "entries": entries}


@app.delete("/audit", tags=["Analytics"])
async def clear_audit_log(db: AsyncSession = Depends(get_db)):
    await db.execute(delete(DBAudit))
    _metrics_cache.clear()
    return {"success": True, "message": "Audit log cleared."}


# ══════════════════════════════════════════════════════════════════════════════
# NLU DATASET
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/dataset", tags=["Dataset"])
@limiter.limit("60/minute")
async def get_dataset(request: Request, limit: int = 200, db: AsyncSession = Depends(get_db)):
    rows = (await db.execute(
        select(DBExample).order_by(DBExample.created_at.desc()).limit(limit)
    )).scalars().all()
    total_q = await db.execute(select(func.count()).select_from(DBExample))
    examples = [
        {
            "id": r.id, "text": r.text, "intent": r.intent,
            "entities": json.loads(r.entities_json),
            "source_modality": r.source_modality, "verified": r.verified,
        }
        for r in rows
    ]
    return {"total": total_q.scalar() or 0, "examples": examples}


@app.post("/dataset/add", tags=["Dataset"])
async def add_to_dataset(example: NLUExample, db: AsyncSession = Depends(get_db)):
    db.add(DBExample(
        text=example.text,
        intent=example.intent.value,
        entities_json=json.dumps([e.model_dump() for e in example.entities]),
        source_modality=example.source_modality.value,
        verified=example.verified,
    ))
    return {"success": True, "added": example.text[:80]}
