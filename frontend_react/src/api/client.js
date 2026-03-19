/* AttoSense React — API Client
   All calls go to /api/* which Vite proxies to http://localhost:8000
   (configured in vite.config.js → server.proxy)                    */

const BASE = '/api'

async function req(path, opts = {}) {
  try {
    const r = await fetch(BASE + path, {
      headers: { 'Content-Type': 'application/json', ...opts.headers },
      ...opts,
    })
    if (!r.ok) {
      const err = await r.json().catch(() => ({ detail: r.statusText }))
      return { success: false, error: err.detail || r.statusText }
    }
    return r.json()
  } catch (e) {
    return { success: false, error: 'Backend offline — start uvicorn in Terminal 1' }
  }
}

// ── System ─────────────────────────────────────────────────────
export const health = () => req('/health')

// ── Classification ─────────────────────────────────────────────
export const classifyText = (message, context = [], sessionId) =>
  req('/classify/text', {
    method:  'POST',
    body:    JSON.stringify({ message, context, session_id: sessionId }),
  })

export const classifyAudio = async (audioBlob, mimeType = 'audio/webm', sessionId) => {
  const form = new FormData()
  form.append('file', audioBlob, 'audio.webm')
  if (sessionId) form.append('session_id', sessionId)
  return fetch(BASE + '/classify/audio/upload', { method: 'POST', body: form })
    .then(r => r.json())
    .catch(() => ({ success: false, error: 'Backend offline' }))
}

export const classifyImage = async (imageFile, caption, sessionId) => {
  const form = new FormData()
  form.append('file', imageFile)
  if (caption)   form.append('caption',    caption)
  if (sessionId) form.append('session_id', sessionId)
  return fetch(BASE + '/classify/vision/upload', { method: 'POST', body: form })
    .then(r => r.json())
    .catch(() => ({ success: false, error: 'Backend offline' }))
}

export const transcribeAudio = async (audioBlob, sessionId) => {
  const form = new FormData()
  form.append('file', audioBlob, 'audio.webm')
  if (sessionId) form.append('session_id', sessionId)
  return fetch(BASE + '/transcribe/upload', { method: 'POST', body: form })
    .then(r => r.json())
    .catch(() => ({ success: false, error: 'Backend offline' }))
}

// ── Inbox ───────────────────────────────────────────────────────
export const getInbox = (status, limit = 200) => {
  const p = new URLSearchParams({ limit })
  if (status) p.set('status', status)
  return req(`/inbox?${p}`)
}

export const reviewInboxItem = (id, status, reviewerLabel, reviewerNote) =>
  req(`/inbox/${id}`, {
    method: 'PATCH',
    body:   JSON.stringify({ status, reviewer_label: reviewerLabel, reviewer_note: reviewerNote }),
  })

export const deleteInboxItem = id =>
  req(`/inbox/${id}`, { method: 'DELETE' })

export const clearInbox = status => {
  const p = status ? `?status=${status}` : ''
  return req(`/inbox${p}`, { method: 'DELETE' })
}

// ── Metrics & Audit ────────────────────────────────────────────
export const getMetrics      = ()            => req('/metrics')
export const getAuditLog     = (limit = 200) => req(`/audit?limit=${limit}`)
export const getDatasetStats = ()            => req('/dataset/stats')
export const getDisagreements= ()            => req('/disagreements')
