import { useState } from 'react'

// ── Domain metadata — drives colour and symbol ──────────────────────────────
const DOMAIN_META = {
  information: { label: 'Information', color: 'var(--c-account)',    sym: '?' },
  action:      { label: 'Action',       color: 'var(--c-sales)',      sym: '→' },
  problem:     { label: 'Problem',      color: 'var(--c-complaint)',  sym: '!' },
  transaction: { label: 'Transaction',  color: 'var(--c-billing)',    sym: '§' },
  creative:    { label: 'Creative',     color: '#7C3AED',             sym: '✦' },
  personal:    { label: 'Personal',     color: '#0D9488',             sym: '◉' },
  technical:   { label: 'Technical',    color: 'var(--c-technical)',  sym: '⚙' },
}

const SENTIMENT_LABEL = {
  positive:   { text: 'Positive',   chip: 'chip-green'   },
  neutral:    { text: 'Neutral',    chip: 'chip-neutral' },
  negative:   { text: 'Negative',   chip: 'chip-amber'   },
  frustrated: { text: 'Frustrated', chip: 'chip-red'     },
}

const MODALITY_LABEL = { text: '↩ Text', audio: '♪ Audio', vision: '◉ Vision' }

function confColor(c) {
  if (c >= 0.82) return 'var(--green)'
  if (c >= 0.70) return 'var(--amber)'
  return 'var(--red)'
}

// ── Confidence Distribution ──────────────────────────────────────────────────
function ConfDist({ scores, winner }) {
  if (!scores || !Object.keys(scores).length) return null
  const sorted = Object.entries(scores).sort((a, b) => b[1] - a[1]).slice(0, 3)
  const max    = sorted[0]?.[1] || 1

  return (
    <div>
      <div className="section-label">Alternative Intents</div>
      <div className="bar-chart">
        {sorted.map(([intent, score]) => {
          const pct   = Math.round(score * 100)
          const isWin = intent === winner
          return (
            <div key={intent} className="dist-row">
              <span
                className={`dist-name${isWin ? ' winner' : ''}`}
                title={intent}
              >
                {intent.length > 24 ? intent.slice(0, 24) + '…' : intent}
              </span>
              <div className="dist-track">
                <div
                  className="dist-fill"
                  style={{ width: `${(score / max) * 100}%`, background: 'var(--accent-mid)' }}
                />
              </div>
              <span className={`dist-pct${isWin ? ' winner' : ''}`}>{pct}%</span>
            </div>
          )
        })}
      </div>
    </div>
  )
}

// ── Sentiment Meter ──────────────────────────────────────────────────────────
function SentimentMeter({ score, label }) {
  const pct     = Math.round(((score ?? 0) + 1) / 2 * 100)
  const pinLeft = Math.max(4, Math.min(96, pct))
  const sentMeta = SENTIMENT_LABEL[label] || { text: label, chip: 'chip-neutral' }
  const col     = score > 0.2 ? 'var(--green)' : score < -0.2 ? 'var(--red)' : 'var(--amber)'
  return (
    <div>
      <div className="section-label">Sentiment</div>
      <div className="flex items-center gap-sm mb-sm">
        <span className={`chip ${sentMeta.chip}`}>{sentMeta.text}</span>
        <span className="text-xs mono muted" style={{ marginLeft: 'auto' }}>
          {score != null ? `${score >= 0 ? '+' : ''}${score.toFixed(2)}` : ''}
        </span>
      </div>
      <div className="sentiment-track">
        <div className="sentiment-pin" style={{ left: `${pinLeft}%` }} />
      </div>
      <div className="sentiment-labels">
        <span>Negative</span><span>Neutral</span><span>Positive</span>
      </div>
    </div>
  )
}

// ── Vision Panel ─────────────────────────────────────────────────────────────
function VisionPanel({ vision }) {
  if (!vision) return null
  const frust  = vision.frustration_score ?? 0
  const fp     = Math.round(frust * 100)
  const fCol   = frust < 0.4 ? 'var(--green)' : frust < 0.7 ? 'var(--amber)' : 'var(--red)'
  const pinPct = Math.max(3, Math.min(97, fp))
  const etype  = (vision.error_type || 'none').replace(/_/g, ' ')
  return (
    <div className="result-section">
      <div className="section-label">
        Image Analysis{vision.screen_type ? ` · ${vision.screen_type.replace(/_/g, ' ')}` : ''}
      </div>
      <div className="vision-panel">
        {vision.visual_summary && (
          <p className="text-sm italic mb-sm" style={{
            color: 'var(--ink-2)', borderBottom: '1px solid var(--border)',
            paddingBottom: '10px', marginBottom: '10px',
          }}>
            {vision.visual_summary}
          </p>
        )}
        <div className="flex items-center justify-between mb-sm">
          <span className="text-xs muted" style={{ textTransform: 'uppercase', letterSpacing: '0.08em' }}>
            Frustration
          </span>
          <span className="bold mono" style={{ fontSize: '13px', color: fCol }}>{fp}%</span>
        </div>
        <div className="frustration-track">
          <div className="frustration-pin" style={{ left: `${pinPct}%` }} />
        </div>
        <div className="mt-sm text-xs" style={{ color: 'var(--ink-3)' }}>
          Error: <span className="bold" style={{ color: 'var(--ink-2)', textTransform: 'capitalize' }}>
            {etype}
          </span>
          {vision.error_detail && vision.error_detail !== 'null' && (
            <span style={{ color: 'var(--red)' }}> — {vision.error_detail}</span>
          )}
        </div>
      </div>
    </div>
  )
}

// ── Main ResultCard ──────────────────────────────────────────────────────────
export default function ResultCard({ response }) {
  const [showAll, setShowAll] = useState(false)

  if (!response?.success) {
    return (
      <div className="card card-pad" style={{ borderColor: 'var(--red-border)', background: 'var(--red-bg)' }}>
        <span style={{ color: 'var(--red)', fontWeight: 700 }}>Error:</span>{' '}
        <span style={{ color: 'var(--red)' }}>{response?.error || 'Unknown error'}</span>
      </div>
    )
  }

  const result   = response.result || {}
  const intent   = result.intent   || 'Understand input'
  const domain   = (result.intent_domain || 'information').toLowerCase()
  const meta     = DOMAIN_META[domain] || DOMAIN_META.information
  const conf     = result.confidence ?? 0
  const sentScore= result.sentiment_score ?? 0
  const sentiment= result.sentiment || 'neutral'
  const esc      = result.requires_escalation
  const escWhy   = result.escalation_reason
  const lowC     = result.low_confidence
  const inbox    = response.inbox_flagged
  const entities = result.entities  || []
  const steps    = result.reasoning_steps || []
  const scores   = result.confidence_scores || {}
  const compI    = result.competing_intent
  const compC    = result.competing_confidence ?? 0
  const trans    = result.raw_transcript || ''
  const vision   = result.vision
  const modality = result.modality || response.modality || 'text'
  const lang     = result.language_detected
  const latency  = response.latency_ms ?? 0

  const confPct = Math.round(conf * 100)
  const confCol = confColor(conf)

  return (
    <div className="result-card">
      {/* ── Header ──────────────────────────────────────────────── */}
      <div className="result-header">
        <div style={{ flex: 1 }}>
          {/* Domain accent bar */}
          <div style={{
            width: 28, height: 3, borderRadius: 2,
            background: meta.color, marginBottom: 10,
          }} />

          {/* Intent phrase — the main result */}
          <div className="result-intent-name">
            <span style={{ color: meta.color, marginRight: 10, fontSize: 20 }}>
              {meta.sym}
            </span>
            {intent}
          </div>

          {/* Meta row */}
          <div className="result-meta-row">
            <span className="chip chip-neutral" style={{ fontSize: 11 }}>
              {meta.label}
            </span>
            <span className="chip chip-neutral" style={{ fontSize: 11 }}>
              {MODALITY_LABEL[modality] || modality}
            </span>
            {lang && (
              <span className="chip chip-blue" style={{ fontSize: 11 }}>
                ⊕ {lang.toUpperCase()}
              </span>
            )}
            {esc  && <span className="chip chip-red">▲ Urgent</span>}
            {lowC && <span className="chip chip-amber">⚠ Low confidence</span>}
            {inbox && <span className="chip chip-accent">↓ Sent to Inbox</span>}
          </div>
        </div>

        {/* Confidence block */}
        <div className="result-conf-block">
          <div className="result-conf-number" style={{ color: confCol }}>
            {confPct}<span style={{ fontSize: 16, color: 'var(--ink-4)' }}>%</span>
          </div>
          <div className="result-conf-label">confidence</div>
          <div className="result-conf-bar">
            <div
              className="result-conf-fill"
              style={{ width: `${confPct}%`, background: confCol }}
            />
          </div>
          <div className="text-xs mono muted mt-sm" style={{ textAlign: 'right' }}>
            {latency.toFixed(0)} ms
          </div>
        </div>
      </div>

      {/* ── Transcript ──────────────────────────────────────────── */}
      {trans && !['[audio]','[image]','[audio upload]','[image upload]'].includes(trans) && (
        <div style={{ padding: '14px 24px', borderBottom: '1px solid var(--border)' }}>
          <div className="section-label">Transcript</div>
          <div className="transcript-box">
            {trans.length > 500 && !showAll ? (
              <>
                {trans.slice(0, 500)}…{' '}
                <button
                  onClick={() => setShowAll(true)}
                  style={{
                    color: 'var(--accent)', background: 'none',
                    border: 'none', cursor: 'pointer',
                    fontFamily: 'var(--font)', fontSize: 13,
                  }}
                >
                  Show all
                </button>
              </>
            ) : trans}
          </div>
        </div>
      )}

      {/* ── Body — two columns ──────────────────────────────────── */}
      <div className="result-body">
        {/* Left: reasoning + escalation + entities + vision */}
        <div className="result-col">
          {steps.length > 0 && (
            <div className="result-section">
              <div className="section-label">How the model decided</div>
              <ol className="reasoning-list">
                {steps.map((step, i) => (
                  <li key={i} className="reasoning-item"
                    style={{ animationDelay: `${i * 0.06}s` }}>
                    <span className="reasoning-num">{String(i + 1).padStart(2, '0')}</span>
                    <span>{step}</span>
                  </li>
                ))}
              </ol>
            </div>
          )}

          {esc && escWhy && (
            <div className="result-section">
              <div className="escalation-box">
                <span style={{ fontWeight: 700, flexShrink: 0 }}>▲</span>
                <span>{escWhy}</span>
              </div>
            </div>
          )}

          {entities.length > 0 && (
            <div className="result-section">
              <div className="section-label">Extracted Entities</div>
              <div className="entity-row">
                {entities.map((e, i) => (
                  <div key={i} className="entity-tag">
                    <span className="entity-label">{e.label}</span>
                    <span className="entity-value">{e.value}</span>
                    <span style={{ fontSize: 10, color: 'var(--ink-4)', marginLeft: 2 }}>
                      {Math.round(e.confidence * 100)}%
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}

          <VisionPanel vision={vision} />
        </div>

        {/* Right: sentiment + distribution + runner-up */}
        <div className="result-col">
          <div className="result-section">
            <SentimentMeter score={sentScore} label={sentiment} />
          </div>

          <div className="result-section">
            <ConfDist scores={scores} winner={intent} />
          </div>

          {compI && compC > 0.04 && (
            <div className="result-section">
              <div className="section-label">Runner-up</div>
              <div
                className="chip chip-neutral"
                style={{
                  fontSize: 12, padding: '5px 12px',
                  display: 'inline-flex', alignItems: 'center', gap: 8,
                }}
              >
                {compI.length > 26 ? compI.slice(0, 26) + '…' : compI}
                <span className="mono" style={{ color: 'var(--ink-3)', marginLeft: 4 }}>
                  {Math.round(compC * 100)}%
                </span>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
