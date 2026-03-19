# AttoSense

**Multimodal Intent Intelligence for SMB Customer Support**

AttoSense is a production-grade Natural Language Understanding (NLU) middleware that classifies customer support intent from text, audio, and screenshots in real time. Built on Groq's inference platform with Llama 3.3 70B, Whisper large-v3, and Llama 4 Scout Vision, it delivers sub-second classifications with a full active learning loop that improves accuracy over time.

---

## Features

- **Three input modalities** — typed messages, voice recordings, and customer screenshots all feed the same NLU pipeline
- **3-stage hierarchical pipeline** — scope gate → family classification → fine-grained intent with dynamic few-shot examples drawn from your own reviewed data
- **8 intent categories** — billing, technical support, account management, sales inquiry, complaint, escalation, general inquiry, out of scope
- **Ensemble confidence** — uncertain results trigger two parallel passes at different temperatures; agreement rate replaces self-reported model confidence
- **Active learning loop** — every human correction in the Review Inbox is automatically promoted to the training dataset and used in future classifications
- **Confidence calibration** — per-modality isotonic regression maps raw model scores to calibrated probabilities using reviewer decisions as ground truth
- **Rich NLU output** — intent family, full confidence distribution, competing intent, sentiment score (−1 to +1), escalation reason, ordered reasoning steps, extracted and normalised entities
- **Audio quality gating** — Whisper metadata (no_speech_prob, avg_logprob) rejects silence and garbled recordings before they reach the NLU stage
- **Image preprocessing** — automatic resize to ≤ 1200px before vision inference; OCR fallback via pytesseract for text-heavy screenshots
- **Language detection** — non-English inputs are identified and flagged before classification
- **Circuit breaker** — stops hammering Groq after 5 consecutive failures, recovers automatically after 30 seconds
- **Request deduplication** — 5-second hash cache prevents double-classify on double-click
- **Dataset health monitoring** — per-intent counts, imbalance detection, and label disagreement tracking surface prompt refinement opportunities
- **Two frontends** — React (recommended, port 3000) and Streamlit (legacy, port 8501) both consume the same FastAPI backend

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         INPUT LAYER                             │
│   💬 Text / Email    🎙 Audio (Whisper)    🖼 Screenshot (Vision) │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│                    3-STAGE NLU PIPELINE                         │
│                                                                 │
│  Stage 1  Scope + Family Gate                                   │
│           Is this support-related? TRANSACTION / ACCOUNT /      │
│           GENERAL — one fast call, narrows Stage 2 to 2–4 intents│
│                                                                 │
│  Stage 2  Fine-Grained Intent (dynamic few-shot)                │
│           Real reviewed examples from your dataset injected     │
│           as context. Full schema output: confidence_scores,    │
│           reasoning_steps, sentiment_score, escalation_reason   │
│                                                                 │
│  Stage 3  Ensemble Confidence (conditional, 0.60–0.82 range)    │
│           Two parallel passes via asyncio.gather. Agreement     │
│           rate = calibrated confidence.                         │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│                       CONFIDENCE ROUTING                        │
│                                                                 │
│   ≥ threshold  →  Result returned immediately to agent          │
│   < threshold  →  Flagged to Review Inbox for human correction  │
│   Escalation   →  requires_escalation flag + escalation_reason  │
│                                                                 │
│   Thresholds: text 0.72 · audio 0.75 · vision 0.78             │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│                         OUTPUT LAYER                            │
│                                                                 │
│   Audit Log     Every call logged to SQLite / PostgreSQL        │
│   Review Inbox  Low-confidence queue with approve / reject      │
│   Training Dataset  Corrections auto-promoted (with dedup)      │
│   Analytics     Intent distribution, latency, calibration       │
└─────────────────────────────────────────────────────────────────┘
```

---

## Technology Stack

| Layer | Technology |
|---|---|
| LLM Inference | Groq API — Llama 3.3 70B Versatile |
| Audio Transcription | Groq API — Whisper large-v3 |
| Vision Understanding | Groq API — Llama 4 Scout 17B |
| Backend Framework | FastAPI (async) |
| Database ORM | SQLAlchemy 2.0 async — SQLite (dev) / PostgreSQL (prod) |
| Reliability | tenacity (retry), slowapi (rate limiting), cachetools (TTL cache) |
| Observability | python-json-logger (structured JSON logs) |
| React Frontend | Vite + React 18 |
| Streamlit Frontend | Streamlit 1.37 |
| Charts | Plotly (Streamlit) / CSS bars (React) |
| Image Processing | Pillow |
| Language Detection | langdetect |
| Entity Date Parsing | python-dateutil |
| PDF Export | ReportLab |

---

## Project Structure

```
AttoSense_v4/
│
├── LICENSE
├── README.md
├── requirements.txt
├── .env.example
├── start_backend.bat          Windows launcher — FastAPI backend
├── start_frontend.bat         Windows launcher — Streamlit frontend
│
├── backend/
│   ├── api.py                 FastAPI routes, middleware, lifespan
│   ├── core/
│   │   ├── auth.py            X-API-Key middleware (timing-safe)
│   │   ├── calibration.py     Isotonic regression confidence calibrator
│   │   ├── database.py        SQLAlchemy async engine, ORM models, migrations
│   │   ├── logging_config.py  Structured JSON logging
│   │   ├── multimodal.py      Audio/vision preprocessing + Groq calls
│   │   └── nlu_pipeline.py    3-stage hierarchical NLU pipeline + ExampleStore
│   └── models/
│       └── schemas.py         Pydantic v2 request/response schemas
│
├── frontend/                  Streamlit frontend (legacy, fully functional)
│   ├── app.py
│   ├── components/sidebar.py
│   ├── pages/1_Discovery_Inbox.py
│   └── utils/
│       ├── api_client.py
│       └── visualizer.py
│
└── frontend_react/            React frontend (recommended)
    ├── index.html
    ├── package.json
    ├── vite.config.js
    ├── start_react.bat        Windows launcher — React dev server
    └── src/
        ├── App.jsx
        ├── main.jsx
        ├── api/client.js      Fetch wrapper for all API endpoints
        ├── components/
        │   ├── Layout.jsx         Sidebar navigation shell
        │   ├── ResultCard.jsx     Full intent result with all context panels
        │   └── AudioRecorder.jsx  MediaRecorder-based in-browser recording
        ├── pages/
        │   ├── Classify.jsx   Text / Audio / Vision classifier
        │   ├── Inbox.jsx      Review and label correction workflow
        │   └── Analytics.jsx  Metrics, dataset health, disagreements
        └── styles/
            └── globals.css    Full design system (Liberation Serif, parchment palette)
```

---

## Installation

### Prerequisites

- Python 3.11+
- Node.js 18+ (for the React frontend only)
- A [Groq API key](https://console.groq.com)

### 1 — Clone or extract

```bash
cd C:\Users\MHKL
# Extract AttoSense_v4_complete.zip here
cd AttoSense_v4
```

### 2 — Python environment

```bash
python -m venv venv
venv\Scripts\activate          # Windows CMD
pip install -r requirements.txt
```

### 3 — Environment variables

```bash
copy .env.example .env
```

Open `.env` and set at minimum:

```env
GROQ_API_KEY=your_groq_api_key_here
API_KEY_DISABLED=true
```

Full `.env` reference:

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | — | **Required.** Groq Console API key |
| `API_KEY_DISABLED` | `false` | Set `true` to skip auth in local dev |
| `API_KEY` | — | Secret for X-API-Key header (prod) |
| `DATABASE_URL` | `sqlite+aiosqlite:///./data/attosense.db` | SQLite (dev) or `postgresql+asyncpg://…` (prod) |
| `LOG_LEVEL` | `info` | `debug` / `info` / `warning` / `error` |
| `ENVIRONMENT` | `development` | Included in structured log output |
| `MAX_UPLOAD_BYTES` | `26214400` | 25 MB upload cap |
| `SKIP_GROQ_HEALTH_PROBE` | `false` | Set `true` to skip Groq check on `/health` |

---

## Running

Two terminals are required — the backend must be running before opening either frontend.

### Terminal 1 — Backend (always required)

```bash
cd AttoSense_v4
start_backend.bat
```

Wait for: `Application startup complete.`

### Terminal 2 — React frontend (recommended)

```bash
cd AttoSense_v4\frontend_react
start_react.bat          # installs npm packages on first run
```

Open **http://localhost:3000**

### Terminal 2 — Streamlit frontend (alternative)

```bash
cd AttoSense_v4
start_frontend.bat
```

Open **http://localhost:8501**

---

## API Reference

The full interactive API documentation is available at **http://localhost:8000/docs** when the backend is running.

### Classification

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/classify/text` | Classify a text message |
| `POST` | `/classify/audio` | Classify audio (base64) |
| `POST` | `/classify/audio/upload` | Classify audio file upload |
| `POST` | `/classify/vision` | Classify image (base64) |
| `POST` | `/classify/vision/upload` | Classify image file upload |
| `POST` | `/transcribe/upload` | Transcribe audio without classifying |

### Review Inbox

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/inbox` | List inbox items |
| `PATCH` | `/inbox/{id}` | Approve or reject with optional label correction |
| `DELETE` | `/inbox/{id}` | Delete one item |
| `DELETE` | `/inbox` | Clear inbox (all or by status) |

### Analytics

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/metrics` | Aggregate KPIs (30 s TTL cache) |
| `GET` | `/audit` | Full audit log with pagination |
| `GET` | `/dataset` | Verified NLU training examples |
| `GET` | `/dataset/stats` | Per-intent counts + imbalance flag |
| `GET` | `/disagreements` | Top label correction pairs |
| `GET` | `/health` | Deep health probe (DB + disk + Groq) |

### Example — classify text

```bash
curl -X POST http://localhost:8000/classify/text \
  -H "Content-Type: application/json" \
  -d '{"message": "My invoice shows a duplicate charge of $149", "session_id": "demo-01"}'
```

### Example response

```json
{
  "success": true,
  "modality": "text",
  "session_id": "demo-01",
  "result": {
    "intent": "billing",
    "intent_family": "transaction",
    "confidence": 0.942,
    "confidence_scores": {
      "billing": 0.942,
      "complaint": 0.038,
      "technical_support": 0.012,
      "escalation": 0.008
    },
    "competing_intent": "complaint",
    "competing_confidence": 0.038,
    "entities": [
      { "label": "AMOUNT", "value": "149.00", "confidence": 0.98 }
    ],
    "sentiment": "negative",
    "sentiment_score": -0.62,
    "requires_escalation": false,
    "escalation_reason": null,
    "reasoning_steps": [
      "Customer references a specific invoice charge",
      "Amount $149 explicitly stated",
      "Duplicate charge indicates billing dispute",
      "No manager demand or legal threat present",
      "Primary intent: billing"
    ],
    "low_confidence": false,
    "language_detected": null
  },
  "model_used": "llama-3.3-70b-versatile",
  "latency_ms": 1187.4,
  "inbox_flagged": false
}
```

---

## Intent Categories

| Intent | Trigger signals |
|---|---|
| `billing` | invoices, charges, refunds, payments, amounts |
| `technical_support` | errors, crashes, bugs, HTTP codes, broken features |
| `account_management` | password reset, email update, adding users, login |
| `sales_inquiry` | pricing, plan comparison, upgrade, enterprise quotes |
| `complaint` | service quality dissatisfaction, repeated contact |
| `escalation` | manager demand, legal/chargeback threats |
| `general_inquiry` | FAQs, policies, support hours, how-to questions |
| `out_of_scope` | zero customer support relevance |

---

## The Active Learning Loop

Every low-confidence classification is held in the **Review Inbox**. When a reviewer approves it with a corrected label:

1. A `CalibrationSample` row is written — feeds the per-modality isotonic regression calibrator
2. A `LabelDisagreement` row is written — surfaces ambiguous intent boundaries in `/disagreements`
3. A near-duplicate check runs (trigram Jaccard similarity ≥ 0.85) — skips promotion if too similar to an existing example
4. The corrected example is added to `nlu_examples` as a verified training record
5. The in-memory `ExampleStore` is updated immediately — the next classification call uses the new example as dynamic few-shot context

Over time, this loop shifts the system from generic zero-shot classification toward domain-specific accuracy tuned to your business's exact language.

---

## Production Deployment

For production use, configure these additional settings:

```env
API_KEY_DISABLED=false
API_KEY=<generate with: openssl rand -hex 32>
DATABASE_URL=postgresql+asyncpg://user:password@host:5432/attosense
ENVIRONMENT=production
LOG_LEVEL=info
```

The project includes a `Dockerfile`, `Dockerfile.frontend`, `docker-compose.yml`, and `nginx/nginx.conf` for containerised deployment with TLS termination.

---

## License

MIT License — Copyright (c) 2026 LKSH-GNDJ

See [LICENSE](LICENSE) for full terms.
