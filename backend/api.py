"""
AttoSense MK1 — FastAPI Backend
Production features: auth, async DB, rate limiting, deduplication,
circuit breaker, calibration, migrations, dataset health, disagreements.
"""
import os, uuid, json, hashlib, asyncio
from datetime import datetime, timezone
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional

from dotenv import load_dotenv
from cachetools import TTLCache
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, delete

from backend.core.auth import AuthMiddleware
from backend.core.database import (
    create_tables, run_migrations, get_db, get_session_factory,
    get_audit_metrics, get_dataset_stats, get_disagreement_stats,
    trigram_similarity,
    AuditLog as DBAudit, InboxItem as DBInbox,
    NLUExample as DBExample, CalibrationSample as DBCalSample,
    LabelDisagreement as DBDisagreement,
)
from backend.core.logging_config import configure_logging, get_logger, RequestLoggingMiddleware
from backend.core.calibration import record_correction, calibration_status
from backend.core.multimodal import (
    AudioQualityError,
    classify_text_async, classify_audio_async, classify_vision_async,
    transcribe_audio_async, probe_groq, circuit_breaker_status,
)
from backend.core.nlu_pipeline import example_store
from backend.models.schemas import (
    TextRequest, AudioRequest, VisionRequest,
    ClassifyResponse, TranscriptionResult,
    InputModality, MetricsSummary,
    NLUExample, InboxItem, InboxReview, InboxSummary, InboxStatus,
)

load_dotenv()
configure_logging()
log = get_logger("attosense.api")

MAX_UPLOAD_BYTES           = int(os.getenv("MAX_UPLOAD_BYTES", str(25*1024*1024)))
DATA_DIR                   = Path("data")
DEDUP_SIMILARITY_THRESHOLD = 0.85

_dedup_cache:   TTLCache = TTLCache(maxsize=500, ttl=5)
_metrics_cache: TTLCache = TTLCache(maxsize=1,   ttl=30)


@asynccontextmanager
async def lifespan(app: FastAPI):
    DATA_DIR.mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)
    Path("temp_uploads").mkdir(exist_ok=True)
    await create_tables()
    await run_migrations()
    # Load verified examples into dynamic few-shot store
    async with get_session_factory()() as _db:
        async with _db.begin():
            _rows = (await _db.execute(
                select(DBExample.text, DBExample.intent, DBExample.intent_domain)
                .where(DBExample.verified == True)
            )).all()
            example_store.load([
                {"text": r.text, "intent": r.intent,
                 "intent_domain": r.intent_domain or "information"}
                for r in _rows
            ])
    log.info("app_startup", extra={"version":"MK1","example_store":example_store.total()})
    yield
    log.info("app_shutdown")


limiter = Limiter(key_func=get_remote_address, default_limits=["200/minute"])

app = FastAPI(
    title="AttoSense MK1 API",
    description="Universal Multimodal Intent Intelligence",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs", redoc_url="/redoc",
)

app.add_middleware(AuthMiddleware)
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS","*").split(","),
    allow_methods=["GET","POST","PATCH","DELETE","OPTIONS"],
    allow_headers=["*"],
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


async def _read_upload(file: UploadFile) -> bytes:
    content = await file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"File too large. Max {MAX_UPLOAD_BYTES//(1024*1024)} MB.")
    return content

def _dedup_key(session_id: str, raw_input: str) -> str:
    return hashlib.sha256(f"{session_id}::{raw_input[:100]}".encode()).hexdigest()


async def _log_audit(db: AsyncSession, result, modality: InputModality,
                     session_id: str, latency_ms: float, language: Optional[str] = None) -> None:
    vision = result.vision
    db.add(DBAudit(
        timestamp            = datetime.now(timezone.utc),
        session_id           = session_id,
        modality             = modality.value,
        intent               = result.intent,                        # plain str
        intent_domain        = result.intent_domain.value if result.intent_domain else "information",
        confidence           = result.confidence,
        sentiment            = result.sentiment,
        sentiment_score      = result.sentiment_score,
        requires_escalation  = result.requires_escalation,
        escalation_reason    = result.escalation_reason,
        low_confidence       = result.low_confidence,
        latency_ms           = latency_ms,
        frustration_score    = vision.frustration_score if vision else None,
        error_type           = vision.error_type.value  if vision else None,
        language_detected    = language or result.language_detected,
        competing_intent     = result.competing_intent,              # plain str or None
        competing_confidence = result.competing_confidence,
    ))
    _metrics_cache.clear()


async def _maybe_inbox(db: AsyncSession, session_id: str, modality: InputModality,
                       raw_input: str, result) -> bool:
    if not result.low_confidence: return False
    db.add(DBInbox(
        id=str(uuid.uuid4()), timestamp=datetime.now(timezone.utc),
        session_id=session_id, modality=modality.value,
        raw_input=raw_input[:500], result_json=result.model_dump_json(),
        status="pending",
    ))
    log.info("inbox_flagged", extra={
        "session_id": session_id, "confidence": result.confidence,
        "intent": result.intent,                                     # plain str
    })
    return True


async def _finalize(db, result, model, latency, modality, session_id, raw_input, language=None):
    await _log_audit(db, result, modality, session_id, latency, language)
    flagged = await _maybe_inbox(db, session_id, modality, raw_input, result)
    return ClassifyResponse(
        success=True, modality=modality, session_id=session_id,
        result=result, model_used=model, latency_ms=latency, inbox_flagged=flagged,
    )


# ── Health ─────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
async def health_check(db: AsyncSession = Depends(get_db)):
    checks = {}
    try:
        await db.execute(select(func.count()).select_from(DBAudit))
        checks["database"] = "ok"
    except Exception as e:
        checks["database"] = f"error: {e}"
    try:
        probe = DATA_DIR / ".health_probe"; probe.write_text("ok"); probe.unlink()
        checks["disk"] = "ok"
    except Exception as e:
        checks["disk"] = f"error: {e}"
    if os.getenv("SKIP_GROQ_HEALTH_PROBE","false").lower() != "true":
        checks["groq"] = "ok" if await probe_groq() else "unreachable"
    else:
        checks["groq"] = "skipped"
    overall = "ok" if all(v in ("ok","skipped") for v in checks.values()) else "degraded"
    return {
        "status": overall, "version": "MK1",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "groq_key_set": bool(os.getenv("GROQ_API_KEY")),
        "checks": checks,
        "circuit_breaker": circuit_breaker_status(),
        "calibration": calibration_status(),
        "example_store": example_store.per_domain(),
    }


# ── Classification ─────────────────────────────────────────────────────────────

@app.post("/classify/text", response_model=ClassifyResponse, tags=["Classify"])
@limiter.limit("60/minute")
async def classify_text_endpoint(request: Request, body: TextRequest, db: AsyncSession = Depends(get_db)):
    sid = body.session_id or str(uuid.uuid4())
    key = _dedup_key(sid, body.message)
    if key in _dedup_cache: return _dedup_cache[key]
    try:
        result, model, latency = await classify_text_async(body.message, body.context)
    except TimeoutError as e: raise HTTPException(504, str(e))
    except Exception as e:
        log.error("classify_text_failed", extra={"error":str(e)}); raise HTTPException(500, str(e))
    response = await _finalize(db, result, model, latency, InputModality.TEXT, sid, body.message)
    _dedup_cache[key] = response
    return response


@app.post("/classify/audio", response_model=ClassifyResponse, tags=["Classify"])
@limiter.limit("30/minute")
async def classify_audio_endpoint(request: Request, body: AudioRequest, db: AsyncSession = Depends(get_db)):
    sid = body.session_id or str(uuid.uuid4())
    try:
        result, model, latency = await classify_audio_async(body.audio_b64, body.mime_type, body.language)
    except AudioQualityError as e: raise HTTPException(422, str(e))
    except TimeoutError as e:      raise HTTPException(504, str(e))
    except Exception as e:
        log.error("classify_audio_failed", extra={"error":str(e)}); raise HTTPException(500, str(e))
    return await _finalize(db, result, model, latency, InputModality.AUDIO, sid, result.raw_transcript or "[audio]")


@app.post("/classify/audio/upload", response_model=ClassifyResponse, tags=["Classify"])
@limiter.limit("30/minute")
async def classify_audio_upload(
    request: Request, file: UploadFile = File(...),
    session_id: str = Form(default=None), language: str = Form(default=None),
    db: AsyncSession = Depends(get_db),
):
    import base64 as _b64
    content   = await _read_upload(file)
    audio_b64 = _b64.b64encode(content).decode()
    mime_type = file.content_type or "audio/wav"
    sid       = session_id or str(uuid.uuid4())
    key = _dedup_key(sid, f"{len(content)}:{mime_type}")
    if key in _dedup_cache: return _dedup_cache[key]
    try:
        result, model, latency = await classify_audio_async(audio_b64, mime_type, language)
    except AudioQualityError as e: raise HTTPException(422, str(e))
    except TimeoutError as e:      raise HTTPException(504, str(e))
    except Exception as e:         raise HTTPException(500, str(e))
    response = await _finalize(db, result, model, latency, InputModality.AUDIO, sid, result.raw_transcript or "[audio]")
    _dedup_cache[key] = response
    return response


@app.post("/classify/vision", response_model=ClassifyResponse, tags=["Classify"])
@limiter.limit("30/minute")
async def classify_vision_endpoint(request: Request, body: VisionRequest, db: AsyncSession = Depends(get_db)):
    sid = body.session_id or str(uuid.uuid4())
    try:
        result, model, latency = await classify_vision_async(body.image_b64, body.mime_type, body.caption)
    except TimeoutError as e: raise HTTPException(504, str(e))
    except Exception as e:    raise HTTPException(500, str(e))
    return await _finalize(db, result, model, latency, InputModality.VISION, sid, body.caption or "[image]")


@app.post("/classify/vision/upload", response_model=ClassifyResponse, tags=["Classify"])
@limiter.limit("30/minute")
async def classify_vision_upload(
    request: Request, file: UploadFile = File(...),
    caption: str = Form(default=None), session_id: str = Form(default=None),
    db: AsyncSession = Depends(get_db),
):
    import base64 as _b64
    content   = await _read_upload(file)
    image_b64 = _b64.b64encode(content).decode()
    mime_type = file.content_type or "image/jpeg"
    sid       = session_id or str(uuid.uuid4())
    try:
        result, model, latency = await classify_vision_async(image_b64, mime_type, caption)
    except TimeoutError as e: raise HTTPException(504, str(e))
    except Exception as e:    raise HTTPException(500, str(e))
    return await _finalize(db, result, model, latency, InputModality.VISION, sid, caption or "[image]")


@app.post("/transcribe/upload", response_model=TranscriptionResult, tags=["Classify"])
@limiter.limit("20/minute")
async def transcribe_upload(
    request: Request, file: UploadFile = File(...),
    language: str = Form(default=None), session_id: str = Form(default=None),
):
    import base64 as _b64
    content   = await _read_upload(file)
    audio_b64 = _b64.b64encode(content).decode()
    mime_type = file.content_type or "audio/wav"
    sid       = session_id or str(uuid.uuid4())
    try:
        return await transcribe_audio_async(audio_b64, mime_type, language, sid)
    except AudioQualityError as e: raise HTTPException(422, str(e))
    except TimeoutError as e:      raise HTTPException(504, str(e))
    except Exception as e:         raise HTTPException(500, str(e))


# ── Inbox ──────────────────────────────────────────────────────────────────────

@app.get("/inbox", response_model=InboxSummary, tags=["Inbox"])
@limiter.limit("120/minute")
async def get_inbox(request: Request, status: Optional[str] = None,
                    limit: int = 100, offset: int = 0, db: AsyncSession = Depends(get_db)):
    q = select(DBInbox).order_by(DBInbox.timestamp.desc())
    if status: q = q.where(DBInbox.status == status)
    total = (await db.execute(select(func.count()).select_from(DBInbox).where(
        DBInbox.status == status if status else True))).scalar() or 0
    rows   = (await db.execute(q.offset(offset).limit(limit))).scalars().all()
    counts = {r.status: r.n for r in (await db.execute(
        select(DBInbox.status, func.count().label("n")).group_by(DBInbox.status))).all()}
    items  = []
    for row in rows:
        try:
            from backend.models.schemas import NLUResult
            items.append(InboxItem(
                id=row.id, timestamp=row.timestamp.isoformat(),
                session_id=row.session_id, modality=row.modality,
                raw_input=row.raw_input, result=NLUResult(**json.loads(row.result_json)),
                status=row.status, reviewer_label=row.reviewer_label,
                reviewer_note=row.reviewer_note,
                reviewed_at=row.reviewed_at.isoformat() if row.reviewed_at else None,
            ))
        except Exception: continue
    return InboxSummary(total=total, pending=counts.get("pending",0), reviewed=counts.get("reviewed",0),
                        approved=counts.get("approved",0), rejected=counts.get("rejected",0), items=items)


@app.patch("/inbox/{item_id}", response_model=InboxItem, tags=["Inbox"])
async def review_inbox_item(item_id: str, review: InboxReview, db: AsyncSession = Depends(get_db)):
    row = (await db.execute(select(DBInbox).where(DBInbox.id == item_id))).scalar_one_or_none()
    if not row: raise HTTPException(404, f"Item {item_id} not found.")

    result_data  = json.loads(row.result_json)
    raw_conf     = result_data.get("confidence", 0.5)
    modality     = row.modality
    predicted    = result_data.get("intent", "Process request")
    was_approved = review.status == InboxStatus.APPROVED
    label_changed = bool(review.reviewer_label and review.reviewer_label != predicted)

    # Calibration sample
    was_correct = was_approved and not label_changed
    db.add(DBCalSample(timestamp=datetime.now(timezone.utc), modality=modality,
                       intent=predicted, raw_confidence=raw_conf, was_correct=was_correct))
    record_correction(raw_conf, was_correct, modality)

    # Label disagreement
    if label_changed:
        db.add(DBDisagreement(timestamp=datetime.now(timezone.utc), modality=modality,
                              predicted_intent=predicted, corrected_intent=review.reviewer_label))

    row.status = review.status.value; row.reviewed_at = datetime.now(timezone.utc)
    if review.reviewer_label: row.reviewer_label = review.reviewer_label
    if review.reviewer_note:  row.reviewer_note  = review.reviewer_note

    # Auto-promote approved + labelled → dataset (with dedup)
    if was_approved and review.reviewer_label:
        new_text   = row.raw_input
        new_intent = review.reviewer_label
        new_domain = result_data.get("intent_domain","information")
        recent     = [(r.text,) for r in (await db.execute(
            select(DBExample.text).where(DBExample.intent == new_intent).limit(200)
        )).all()]
        is_dup = any(trigram_similarity(new_text, t) >= DEDUP_SIMILARITY_THRESHOLD for (t,) in recent)
        if not is_dup:
            db.add(DBExample(text=new_text, intent=new_intent, intent_domain=new_domain,
                             entities_json=json.dumps(result_data.get("entities",[])),
                             source_modality=modality, verified=True))
            example_store.add(new_domain, new_text, new_intent)
            log.info("example_promoted", extra={"intent":new_intent})
        else:
            log.info("dataset_dedup_skipped", extra={"intent":new_intent})

    from backend.models.schemas import NLUResult
    return InboxItem(id=row.id, timestamp=row.timestamp.isoformat(),
                     session_id=row.session_id, modality=row.modality, raw_input=row.raw_input,
                     result=NLUResult(**result_data), status=row.status,
                     reviewer_label=row.reviewer_label, reviewer_note=row.reviewer_note,
                     reviewed_at=row.reviewed_at.isoformat() if row.reviewed_at else None)


@app.delete("/inbox/{item_id}", tags=["Inbox"])
async def delete_inbox_item(item_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(delete(DBInbox).where(DBInbox.id == item_id))
    if result.rowcount == 0: raise HTTPException(404)
    return {"success": True}


@app.delete("/inbox", tags=["Inbox"])
async def clear_inbox(status: Optional[str] = None, db: AsyncSession = Depends(get_db)):
    q = delete(DBInbox)
    if status: q = q.where(DBInbox.status == status)
    await db.execute(q)
    return {"success": True}


# ── Metrics ────────────────────────────────────────────────────────────────────

@app.get("/metrics", response_model=MetricsSummary, tags=["Analytics"])
@limiter.limit("120/minute")
async def get_metrics(request: Request, db: AsyncSession = Depends(get_db)):
    if "metrics" in _metrics_cache: return MetricsSummary(**_metrics_cache["metrics"])
    metrics = await get_audit_metrics(db)
    metrics["inbox_pending"] = (await db.execute(
        select(func.count()).select_from(DBInbox).where(DBInbox.status=="pending")
    )).scalar() or 0
    _metrics_cache["metrics"] = metrics
    return MetricsSummary(**metrics)


@app.get("/audit", tags=["Analytics"])
@limiter.limit("60/minute")
async def get_audit_log(request: Request, limit: int = 100, offset: int = 0,
                        db: AsyncSession = Depends(get_db)):
    total = (await db.execute(select(func.count()).select_from(DBAudit))).scalar() or 0
    rows  = (await db.execute(select(DBAudit).order_by(DBAudit.timestamp.desc()).offset(offset).limit(limit))).scalars().all()
    entries = [{c.name: getattr(r,c.name) for c in DBAudit.__table__.columns} for r in rows]
    for e in entries:
        if e.get("timestamp"): e["timestamp"] = e["timestamp"].isoformat()
    return {"total": total, "offset": offset, "limit": limit, "entries": entries}


@app.delete("/audit", tags=["Analytics"])
async def clear_audit_log(db: AsyncSession = Depends(get_db)):
    await db.execute(delete(DBAudit)); _metrics_cache.clear()
    return {"success": True}


# ── Dataset ────────────────────────────────────────────────────────────────────

@app.get("/dataset", tags=["Dataset"])
@limiter.limit("60/minute")
async def get_dataset(request: Request, limit: int = 200, db: AsyncSession = Depends(get_db)):
    rows    = (await db.execute(select(DBExample).order_by(DBExample.created_at.desc()).limit(limit))).scalars().all()
    total_q = await db.execute(select(func.count()).select_from(DBExample))
    return {"total": total_q.scalar() or 0,
            "examples": [{"id":r.id,"text":r.text,"intent":r.intent,"intent_domain":r.intent_domain,
                          "entities":json.loads(r.entities_json),"source_modality":r.source_modality,
                          "verified":r.verified} for r in rows]}


@app.get("/dataset/stats", tags=["Dataset"])
async def dataset_stats(db: AsyncSession = Depends(get_db)):
    return await get_dataset_stats(db)


@app.post("/dataset/add", tags=["Dataset"])
async def add_to_dataset(example: NLUExample, db: AsyncSession = Depends(get_db)):
    db.add(DBExample(
        text=example.text, intent=example.intent, intent_domain=example.intent_domain,
        entities_json=json.dumps([e.model_dump() for e in example.entities]),
        source_modality=example.source_modality.value, verified=example.verified,
    ))
    return {"success": True, "added": example.text[:80]}


@app.get("/disagreements", tags=["Analytics"])
async def get_disagreements(db: AsyncSession = Depends(get_db)):
    return await get_disagreement_stats(db)
