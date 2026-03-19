"""
AttoSense v3.1 - Frontend API Client
Handles all communication between Streamlit UI and FastAPI backend.
"""

import base64
import requests
from typing import Optional

BASE_URL = "http://localhost:8000"
TIMEOUT  = 60


def _post(endpoint: str, payload: dict) -> dict:
    try:
        r = requests.post(f"{BASE_URL}{endpoint}", json=payload, timeout=TIMEOUT)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        return {"success": False, "error": "Backend offline. Run: uvicorn backend.api:app --reload"}
    except requests.exceptions.Timeout:
        return {"success": False, "error": "Request timed out."}
    except requests.exceptions.HTTPError as e:
        try:
            detail = e.response.json().get("detail", str(e))
        except Exception:
            detail = str(e)
        return {"success": False, "error": detail}
    except Exception as e:
        return {"success": False, "error": str(e)}


def _get(endpoint: str, params: dict = None, timeout: int = 10) -> dict:
    try:
        r = requests.get(f"{BASE_URL}{endpoint}", params=params, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


# ── Classify ───────────────────────────────────────────────────────────────────

def classify_text(message: str, context: Optional[list] = None, session_id: Optional[str] = None) -> dict:
    return _post("/classify/text", {"message": message, "context": context or [], "session_id": session_id})


def classify_audio_file(file_bytes: bytes, mime_type: str = "audio/wav", session_id: Optional[str] = None) -> dict:
    try:
        r = requests.post(f"{BASE_URL}/classify/audio/upload",
                          files={"file": ("audio", file_bytes, mime_type)},
                          data={"session_id": session_id or ""}, timeout=TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"success": False, "error": str(e)}


def classify_image_file(file_bytes: bytes, mime_type: str = "image/jpeg",
                        caption: Optional[str] = None, session_id: Optional[str] = None) -> dict:
    try:
        r = requests.post(f"{BASE_URL}/classify/vision/upload",
                          files={"file": ("image", file_bytes, mime_type)},
                          data={"caption": caption or "", "session_id": session_id or ""},
                          timeout=TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"success": False, "error": str(e)}


# ── Transcription ──────────────────────────────────────────────────────────────

def transcribe_file(file_bytes: bytes, mime_type: str = "audio/wav",
                    language: Optional[str] = None, session_id: Optional[str] = None) -> dict:
    try:
        r = requests.post(f"{BASE_URL}/transcribe/upload",
                          files={"file": ("audio", file_bytes, mime_type)},
                          data={"language": language or "", "session_id": session_id or ""},
                          timeout=TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"success": False, "error": str(e)}


# ── Discovery Inbox ────────────────────────────────────────────────────────────

def get_inbox(status: Optional[str] = None, limit: int = 200) -> dict:
    params = {"limit": limit}
    if status:
        params["status"] = status
    return _get("/inbox", params=params)


def review_inbox_item(item_id: str, status: str,
                      reviewer_label: Optional[str] = None,
                      reviewer_note: Optional[str] = None) -> dict:
    payload = {"status": status}
    if reviewer_label:
        payload["reviewer_label"] = reviewer_label
    if reviewer_note:
        payload["reviewer_note"] = reviewer_note
    try:
        r = requests.patch(f"{BASE_URL}/inbox/{item_id}", json=payload, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def delete_inbox_item(item_id: str) -> dict:
    try:
        r = requests.delete(f"{BASE_URL}/inbox/{item_id}", timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def clear_inbox(status: Optional[str] = None) -> dict:
    try:
        params = {"status": status} if status else {}
        r = requests.delete(f"{BASE_URL}/inbox", params=params, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


# ── Analytics ──────────────────────────────────────────────────────────────────

def get_metrics() -> dict:
    return _get("/metrics")


def get_audit_log(limit: int = 200) -> dict:
    return _get("/audit", params={"limit": limit})


def clear_audit_log() -> dict:
    try:
        r = requests.delete(f"{BASE_URL}/audit", timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"success": False, "error": str(e)}


def health_check() -> dict:
    try:
        r = requests.get(f"{BASE_URL}/health", timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {"status": "offline"}
