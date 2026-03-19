import { useState, useRef } from 'react'
import * as api from '../api/client'
import ResultCard from '../components/ResultCard'
import AudioRecorder from '../components/AudioRecorder'

const EXAMPLES = [
  "My invoice shows a duplicate charge of $149 from last month",
  "Error 403 appears every time I try to log in",
  "I'd like a quote for 50 users on the Enterprise plan",
  "The app keeps crashing when I open the reports tab",
  "I've been waiting 3 days — this is completely unacceptable",
  "How do I add a second user to my account?",
]

export default function Classify({ sessionId }) {
  const [tab,        setTab]        = useState('text')
  const [textInput,  setTextInput]  = useState('')
  const [loading,    setLoading]    = useState(false)
  const [result,     setResult]     = useState(null)
  const [history,    setHistory]    = useState([])
  const [imageFile,  setImageFile]  = useState(null)
  const [imagePreview, setImagePreview] = useState(null)
  const [caption,    setCaption]    = useState('')
  const [context,    setContext]    = useState(true)
  const imageRef = useRef()

  async function classifyText(e) {
    e.preventDefault()
    if (!textInput.trim() || loading) return
    setLoading(true)
    const ctx = context
      ? history.filter(h => h.role === 'user').slice(-6).map(h => ({ role: 'user', content: h.content }))
      : []
    const r = await api.classifyText(textInput.trim(), ctx, sessionId)
    setLoading(false)
    setResult(r)
    setHistory(prev => [
      ...prev,
      { role: 'user', content: textInput.trim() },
      { role: 'bot',  result: r },
    ])
    setTextInput('')
  }

  async function classifyAudio(blob) {
    setLoading(true)
    const r = await api.classifyAudio(blob, 'audio/webm', sessionId)
    setLoading(false)
    setResult(r)
    setHistory(prev => [...prev, { role: 'user', content: '♪ Voice recording' }, { role: 'bot', result: r }])
  }

  async function classifyImage() {
    if (!imageFile || loading) return
    setLoading(true)
    const r = await api.classifyImage(imageFile, caption || undefined, sessionId)
    setLoading(false)
    setResult(r)
    setHistory(prev => [...prev, { role: 'user', content: `◉ ${imageFile.name}${caption ? ` — "${caption}"` : ''}` }, { role: 'bot', result: r }])
    setImageFile(null)
    setImagePreview(null)
    setCaption('')
  }

  function onImageChange(e) {
    const f = e.target.files[0]
    if (!f) return
    setImageFile(f)
    setImagePreview(URL.createObjectURL(f))
  }

  const wordCount = textInput.trim() ? textInput.trim().split(/\s+/).length : 0

  return (
    <>
      <div className="page-header">
        <div>
          <div className="page-title">Intent Classifier</div>
          <div className="page-sub">Text · Audio · Vision — all three modalities in one view</div>
        </div>
        {loading && <div className="spinner" />}
      </div>

      <div className="page-body" style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 24, alignItems: 'start' }}>
        {/* ── Left: input ───────────────────────────────────── */}
        <div>
          {/* Tabs */}
          <div className="card" style={{ overflow: 'hidden' }}>
            <div className="tabs" style={{ padding: '0 24px', background: 'var(--surface)' }}>
              {['text', 'audio', 'vision'].map(t => (
                <button key={t} className={`tab${tab === t ? ' active' : ''}`} onClick={() => setTab(t)}>
                  {t === 'text' ? '↩ Text' : t === 'audio' ? '♪ Audio' : '◉ Vision'}
                </button>
              ))}
            </div>

            <div className="card-pad" style={{ borderTop: '1px solid var(--border)' }}>
              {/* TEXT TAB */}
              {tab === 'text' && (
                <>
                  <div className="section-label">Quick Examples</div>
                  <div className="example-chips mb-sm">
                    {EXAMPLES.map((ex, i) => (
                      <button key={i} className="example-chip" onClick={() => setTextInput(ex)}>
                        {ex.length > 42 ? ex.slice(0, 42) + '…' : ex}
                      </button>
                    ))}
                  </div>

                  <form onSubmit={classifyText}>
                    <textarea
                      className="textarea"
                      value={textInput}
                      onChange={e => setTextInput(e.target.value)}
                      placeholder="Type or paste a customer message, email, or support ticket…"
                      style={{ minHeight: 140 }}
                    />
                    <div className="flex items-center justify-between mt-sm">
                      <label className="flex items-center gap-sm text-sm" style={{ cursor: 'pointer', color: 'var(--ink-3)' }}>
                        <input
                          type="checkbox"
                          checked={context}
                          onChange={e => setContext(e.target.checked)}
                          style={{ accentColor: 'var(--accent)' }}
                        />
                        Include conversation context
                      </label>
                      <span className="text-xs mono muted">{wordCount} words</span>
                    </div>
                    <button
                      type="submit"
                      className="btn btn-primary btn-full mt-sm"
                      disabled={!textInput.trim() || loading}
                    >
                      {loading ? 'Classifying…' : 'Classify Intent →'}
                    </button>
                  </form>
                </>
              )}

              {/* AUDIO TAB */}
              {tab === 'audio' && (
                <>
                  <p className="text-sm italic muted mb">
                    Record a voice message directly in the browser. The audio is transcribed by Whisper large-v3, then classified.
                  </p>
                  <AudioRecorder onRecorded={classifyAudio} disabled={loading} />

                  <div className="divider" />

                  <div className="section-label">Or upload a file</div>
                  <p className="text-xs muted mb-sm">Supports .wav · .mp3 · .ogg · .webm — up to 25 MB</p>
                  <input
                    type="file"
                    accept=".wav,.mp3,.ogg,.webm"
                    style={{ display: 'none' }}
                    id="audio-upload"
                    onChange={async e => {
                      const f = e.target.files[0]
                      if (!f) return
                      setLoading(true)
                      const r = await api.classifyAudio(f, f.type, sessionId)
                      setLoading(false)
                      setResult(r)
                      setHistory(prev => [...prev, { role: 'user', content: `♪ ${f.name}` }, { role: 'bot', result: r }])
                    }}
                  />
                  <label htmlFor="audio-upload" className="btn btn-outline btn-full" style={{ cursor: 'pointer', display: 'flex', justifyContent: 'center' }}>
                    Upload Audio File
                  </label>
                </>
              )}

              {/* VISION TAB */}
              {tab === 'vision' && (
                <>
                  <p className="text-sm italic muted mb">
                    Upload a screenshot or photo. Llama 4 Scout Vision extracts all visible text and error codes before classifying.
                  </p>

                  {imagePreview ? (
                    <div style={{ position: 'relative', marginBottom: 14 }}>
                      <img
                        src={imagePreview}
                        alt="preview"
                        style={{ width: '100%', borderRadius: 'var(--r)', border: '1px solid var(--border)', maxHeight: 220, objectFit: 'contain', background: 'var(--bg-alt)' }}
                      />
                      <button
                        onClick={() => { setImageFile(null); setImagePreview(null) }}
                        style={{
                          position: 'absolute', top: 8, right: 8,
                          background: 'var(--nav-bg)', color: 'var(--nav-text)',
                          border: 'none', borderRadius: '50%',
                          width: 26, height: 26, cursor: 'pointer', fontSize: 14,
                          display: 'flex', alignItems: 'center', justifyContent: 'center',
                        }}
                      >
                        ×
                      </button>
                    </div>
                  ) : (
                    <div
                      onClick={() => imageRef.current.click()}
                      style={{
                        border: '1px dashed var(--border-2)',
                        borderRadius: 'var(--r)',
                        padding: '36px 24px',
                        textAlign: 'center',
                        cursor: 'pointer',
                        marginBottom: 14,
                        background: 'var(--surface-2)',
                        transition: 'background var(--dur) var(--ease)',
                      }}
                      onMouseEnter={e => e.currentTarget.style.background = 'var(--accent-light)'}
                      onMouseLeave={e => e.currentTarget.style.background = 'var(--surface-2)'}
                    >
                      <div style={{ fontSize: 28, marginBottom: 8, opacity: 0.4 }}>◉</div>
                      <div className="text-sm muted italic">Click to upload image</div>
                      <div className="text-xs muted">.jpg · .png · .webp · up to 25 MB</div>
                    </div>
                  )}

                  <input ref={imageRef} type="file" accept="image/*" style={{ display: 'none' }} onChange={onImageChange} />

                  <input
                    className="input mb-sm"
                    value={caption}
                    onChange={e => setCaption(e.target.value)}
                    placeholder="Optional context — e.g. Customer sent this error screenshot"
                  />

                  <button
                    className="btn btn-primary btn-full"
                    disabled={!imageFile || loading}
                    onClick={classifyImage}
                  >
                    {loading ? 'Analysing…' : 'Analyse & Classify →'}
                  </button>
                </>
              )}
            </div>
          </div>

          {/* History */}
          {history.length > 0 && (
            <div style={{ marginTop: 20 }}>
              <div className="flex items-center justify-between mb-sm">
                <div className="section-label" style={{ marginBottom: 0 }}>History</div>
                <button className="btn btn-ghost btn-sm" onClick={() => setHistory([])}>Clear</button>
              </div>
              {[...history].reverse().slice(0, 6).map((h, i) => (
                <div key={i} style={{ marginBottom: 6 }}>
                  {h.role === 'user' ? (
                    <div className="history-user-bubble">{h.content}</div>
                  ) : (
                    <div className="text-xs muted mono" style={{ padding: '4px 0', color: 'var(--ink-4)' }}>
                      {h.result?.result?.intent?.replace(/_/g, ' ')} — {Math.round((h.result?.result?.confidence ?? 0) * 100)}%
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>

        {/* ── Right: result ─────────────────────────────────── */}
        <div>
          {result ? (
            <ResultCard response={result} />
          ) : (
            <div className="card card-pad" style={{ textAlign: 'center', padding: '60px 24px' }}>
              <div style={{ fontSize: 36, marginBottom: 14, opacity: 0.2 }}>◎</div>
              <div style={{ fontWeight: 700, fontSize: 16, color: 'var(--ink-3)', marginBottom: 6 }}>
                Ready to classify
              </div>
              <p className="text-sm italic muted">
                Submit a message, recording, or screenshot on the left.<br />
                The full analysis will appear here.
              </p>
            </div>
          )}
        </div>
      </div>
    </>
  )
}
