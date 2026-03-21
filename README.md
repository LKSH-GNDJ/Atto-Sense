<div align="center">
  <img src="attosense_logo.svg" alt="AttoSense — Intent Intelligence" width="360" />
  <br/><br/>

  <strong>Universal Multimodal Intent Intelligence</strong>

  <br/><br/>

  [![License: MIT](https://img.shields.io/badge/License-MIT-9B3D12.svg?style=flat-square)](LICENSE)
  [![Python 3.11+](https://img.shields.io/badge/Python-3.11+-1B1710.svg?style=flat-square)](https://python.org)
  [![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688.svg?style=flat-square)](https://fastapi.tiangolo.com)
  [![React 18](https://img.shields.io/badge/React-18-61DAFB.svg?style=flat-square)](https://react.dev)
  [![Groq](https://img.shields.io/badge/Powered%20by-Groq-F55036.svg?style=flat-square)](https://groq.com)
</div>

---

AttoSense classifies the intent of **any** input — a support ticket, a coding question, a travel request, a creative brief, a casual conversation. It returns a precise 2–5 word action phrase alongside full context: domain, confidence distribution, sentiment, extracted entities, and step-by-step reasoning.

Built on **Llama 3.3 70B · Whisper large-v3 · Llama 4 Scout Vision** via Groq.

---

## What it classifies

Everything. No topic is out of scope.

| Input | Intent | Domain |
|---|---|---|
| `"My invoice shows a double charge of $149"` | Report duplicate invoice charge | transaction |
| `"What is the capital of France?"` | Get capital city fact | information |
| `"Book a flight to Tokyo next Friday"` | Book flight to destination | action |
| `"Write a haiku about autumn rain"` | Write seasonal haiku | creative |
| `"App crashes with error 500 on login"` | Report app login crash | problem |
| `"Debug this Python function"` | Debug Python code | technical |
| `"I need advice on changing careers"` | Seek career guidance | personal |
| `"Translate this email to Spanish"` | Translate email to Spanish | action |

---

## Project Structure

```
AttoSense/
│
├── attosense_logo.svg          Primary logo — wordmark + Signal Diamond mark
├── attosense_mark.svg          Mark only — favicon, icon, small contexts
│
├── backend/                    FastAPI backend
│   ├── api.py                  All routes, middleware, startup lifecycle
│   ├── core/
│   │   ├── auth.py             X-API-Key middleware (timing-safe)
│   │   ├── calibration.py      Isotonic regression confidence calibrator
│   │   ├── database.py         SQLAlchemy async ORM + auto-migrations
│   │   ├── logging_config.py   Structured JSON logging
│   │   ├── multimodal.py       Audio/vision preprocessing + Groq calls
│   │   └── nlu_pipeline.py     3-stage universal NLU pipeline
│   └── models/
│       └── schemas.py          Pydantic v2 — free-form intent schema
│
├── frontend/                   Streamlit UI (alternative)
│   ├── app.py
│   ├── components/sidebar.py
│   ├── pages/1_Discovery_Inbox.py
│   └── utils/api_client.py · visualizer.py
│
├── frontend_react/             React UI (recommended)
│   ├── index.html
│   ├── package.json
│   ├── vite.config.js
│   ├── start_react.bat
│   └── src/
│       ├── App.jsx · main.jsx
│       ├── api/client.js
│       ├── components/
│       │   ├── AudioRecorder.jsx    Direct MediaRecorder, no third-party widget
│       │   ├── Layout.jsx           Sidebar navigation with live logo
│       │   ├── Logo.jsx             Signal Diamond as inline React component
│       │   └── ResultCard.jsx       Full result — reasoning, distribution, sentiment
│       ├── pages/
│       │   ├── Classify.jsx         Text / Audio / Vision classifier
│       │   ├── Inbox.jsx            Review and label correction workflow
│       │   └── Analytics.jsx        Metrics, domain distribution, disagreements
│       └── styles/globals.css       Liberation Serif design system
│
├── data/                       SQLite database (auto-created)
├── logs/                       Structured JSON logs
├── reports/                    PDF exports
├── temp_uploads/               Temporary upload storage
│
├── bot.py                      CLI intent classifier
├── requirements.txt
├── .env.example
├── .gitignore
├── LICENSE
├── start_backend.bat
├── start_frontend.bat
├── start_bot.bat
└── frontend_react/start_react.bat
```

---

## Installation

### Prerequisites

- Python 3.11+
- Node.js 18+ *(React UI only)*
- [Groq API key](https://console.groq.com)

### Setup

```bash
# Clone the repo
git clone https://github.com/LKSH-GNDJ/attosense.git
cd attosense

# Python environment
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt

# Environment variables
copy .env.example .env
# Open .env — set GROQ_API_KEY
```

---

## Running

Two terminals are required. The backend must start first.

**Terminal 1 — Backend**
```cmd
start_backend.bat
```
Wait for: `Application startup complete.`

**Terminal 2 — Choose your UI**

| Interface | Command | URL |
|---|---|---|
| React *(recommended)* | `cd frontend_react && start_react.bat` | http://localhost:3000 |
| Streamlit | `start_frontend.bat` | http://localhost:8501 |
| CLI Bot | `start_bot.bat` | terminal |
| API Docs | — | http://localhost:8000/docs |

### CLI Bot

```cmd
# Interactive mode
start_bot.bat

# Single message
start_bot.bat "book a flight to Paris"

# Batch file (one message per line)
start_bot.bat --file messages.txt
```

---

## NLU Pipeline

```
Any input — text, audio, image
           │
           ▼
┌──────────────────────────────────────────────┐
│  Stage 1 — Domain Detection   (~150ms)        │
│                                               │
│  information  action  problem  transaction    │
│  creative     personal         technical      │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│  Stage 2 — Open-Ended Intent Generation       │
│                                               │
│  Generates: "Report duplicate invoice charge" │
│  Returns full schema:                         │
│    confidence_scores  reasoning_steps         │
│    sentiment_score    escalation_reason       │
│    entities           competing_intent        │
│                                               │
│  Dynamic few-shot from your reviewed data     │
└──────────────────────┬───────────────────────┘
                       │ (only if 0.60 ≤ conf ≤ 0.82)
                       ▼
┌──────────────────────────────────────────────┐
│  Stage 3 — Ensemble Confidence               │
│  Two parallel passes via asyncio.gather       │
│  Agreement rate → calibrated confidence       │
└──────────────────────┬───────────────────────┘
                       │
            ┌──────────┴──────────┐
            ▼                     ▼
     Confident → Route       Uncertain → Review Inbox
```

---

## API

Full interactive docs at **http://localhost:8000/docs**.

### Classification endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/classify/text` | Classify text |
| `POST` | `/classify/audio/upload` | Upload audio file |
| `POST` | `/classify/vision/upload` | Upload image |
| `POST` | `/transcribe/upload` | Transcribe audio only |

### Example

```bash
curl -X POST http://localhost:8000/classify/text \
  -H "Content-Type: application/json" \
  -d '{"message": "My invoice shows a double charge of $149"}'
```

```json
{
  "result": {
    "intent": "Report duplicate invoice charge",
    "intent_domain": "transaction",
    "confidence": 0.94,
    "confidence_scores": {
      "Report duplicate invoice charge": 0.94,
      "Request billing refund": 0.04,
      "Dispute payment amount": 0.02
    },
    "competing_intent": "Request billing refund",
    "entities": [{ "label": "AMOUNT", "value": "149.00", "confidence": 0.98 }],
    "sentiment": "negative",
    "sentiment_score": -0.62,
    "requires_escalation": false,
    "reasoning_steps": [
      "Customer references an invoice",
      "Amount $149 explicitly stated",
      "Phrase 'double charge' signals duplicate billing",
      "No manager demand or legal threat present",
      "Primary intent: Report duplicate invoice charge"
    ]
  },
  "latency_ms": 1182,
  "inbox_flagged": false
}
```

---

## Active Learning Loop

Every low-confidence classification is held in the **Review Inbox**. When a reviewer approves it with a corrected label:

1. A calibration sample is recorded — feeds the per-modality isotonic regression curve
2. A label disagreement row is written — surfaces ambiguous intent boundaries in `/disagreements`
3. A near-duplicate check runs (trigram Jaccard ≥ 0.85) — prevents duplicate training data
4. The corrected example is added to the training dataset
5. The in-memory `ExampleStore` is updated immediately — the **next classification** uses it as dynamic few-shot context

Over time the system shifts from generic zero-shot classification toward accuracy calibrated to your exact data.

---

## Logo

<div align="center">
  <img src="attosense_mark.svg" alt="AttoSense mark" width="80" />
</div>

The **Signal Diamond** — a precision diamond bisected by a signal thread. Noisy input enters from the left. Through the diamond (the intelligence moment) it emerges as a resolved classified intent, marked by a sienna dot. The name *Atto* (10⁻¹⁸, the smallest named SI prefix) signifies atomic-level precision.

| File | Use |
|---|---|
| `attosense_logo.svg` | Primary — wordmark + mark, light backgrounds |
| `attosense_mark.svg` | Icon — favicon, app icon, small contexts (≥ 16 px) |

---

## License

MIT License — Copyright (c) 2026 LKSH-GNDJ — see [LICENSE](LICENSE)
