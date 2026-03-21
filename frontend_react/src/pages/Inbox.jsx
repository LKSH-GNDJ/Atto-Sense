import { useState, useEffect, useCallback } from 'react'
import * as api from '../api/client'

const INTENT_COLORS = {
  billing:'var(--c-billing)', technical_support:'var(--c-technical)',
  account_management:'var(--c-account)', sales_inquiry:'var(--c-sales)',
  complaint:'var(--c-complaint)', general_inquiry:'var(--c-general)',
  escalation:'var(--c-escalation)', out_of_scope:'var(--c-oos)',
}

const STATUS_CHIP = {
  pending:  { cls: 'chip-amber',   label: 'Pending'  },
  approved: { cls: 'chip-green',   label: 'Approved' },
  rejected: { cls: 'chip-red',     label: 'Rejected' },
  reviewed: { cls: 'chip-neutral', label: 'Reviewed' },
}

function fmtTs(ts) {
  if (!ts) return ''
  const d = new Date(ts)
  return `${d.toLocaleDateString()} ${d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}`
}

function InboxRow({ item, onRefresh }) {
  const [open,  setOpen]  = useState(false)
  const [label, setLabel] = useState('')
  const [note,  setNote]  = useState('')
  const [busy,  setBusy]  = useState(false)

  const result  = item.result    || {}
  const intent  = result.intent  || 'Understand input'
  const conf    = Math.round((result.confidence ?? 0) * 100)
  const sc      = STATUS_CHIP[item.status] || STATUS_CHIP.pending
  const dotCol  = INTENT_COLORS[intent] || 'var(--c-general)'
  const steps   = result.reasoning_steps || []
  const entities= result.entities || []
  const isPending = item.status === 'pending'

  async function submit(status) {
    setBusy(true)
    await api.reviewInboxItem(item.id, status, label || undefined, note || undefined)
    setBusy(false)
    onRefresh()
  }

  return (
    <>
      <div
        className={`inbox-row${open ? ' expanded' : ''}`}
        onClick={() => setOpen(o => !o)}
      >
        <div className="inbox-intent-dot" style={{ background: dotCol }} />
        <div>
          <div className="inbox-text">
            {(item.raw_input || '').slice(0, 120)}{(item.raw_input || '').length > 120 ? '…' : ''}
          </div>
          <div className="inbox-meta">
            {intent.replace(/_/g, ' ')} · {item.modality} · {conf}% confidence · {fmtTs(item.timestamp)}
          </div>
        </div>
        <span className={`chip ${sc.cls}`} style={{ fontSize: 11 }}>{sc.label}</span>
        <span style={{ color: 'var(--ink-4)', fontSize: 14 }}>{open ? '▲' : '▼'}</span>
      </div>

      {open && (
        <div style={{ padding: '16px 20px', background: 'var(--surface-2)', borderBottom: '1px solid var(--border)' }}>
          <div className="grid-2" style={{ gap: 20, marginBottom: 16 }}>
            {/* Reasoning */}
            {steps.length > 0 && (
              <div>
                <div className="section-label">How it was classified</div>
                <ol style={{ listStyle: 'none', padding: 0 }}>
                  {steps.map((s, i) => (
                    <li key={i} style={{ display: 'flex', gap: 10, padding: '4px 0', fontSize: 13, color: 'var(--ink-2)', borderBottom: '1px solid var(--border)' }}>
                      <span style={{ color: 'var(--ink-4)', fontFamily: 'var(--font-mono)', fontSize: 11, minWidth: 18, paddingTop: 2 }}>
                        {String(i+1).padStart(2,'0')}
                      </span>
                      {s}
                    </li>
                  ))}
                </ol>
              </div>
            )}

            {/* Entities */}
            <div>
              {entities.length > 0 && (
                <>
                  <div className="section-label">Entities</div>
                  <div className="entity-row">
                    {entities.map((e, i) => (
                      <div key={i} className="entity-tag">
                        <span className="entity-label">{e.label}</span>
                        <span className="entity-value">{e.value}</span>
                      </div>
                    ))}
                  </div>
                </>
              )}
              {item.reviewer_label && (
                <div className="mt" style={{
                  background: 'var(--green-bg)', border: '1px solid var(--green-border)',
                  borderRadius: 'var(--r)', padding: '10px 14px', fontSize: 13,
                }}>
                  <span style={{ color: 'var(--green)', fontWeight: 700 }}>✓ Corrected to:</span>{' '}
                  <span style={{ color: 'var(--ink-2)' }}>{item.reviewer_label.replace(/_/g,' ')}</span>
                  {item.reviewer_note && (
                    <div style={{ color: 'var(--ink-3)', fontStyle: 'italic', marginTop: 4, fontSize: 12 }}>
                      "{item.reviewer_note}"
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>

          {/* Review form */}
          {isPending && (
            <div className="review-panel">
              <div>
                <label className="form-label">Correct intent label (optional)</label>
                <input
                  className="input"
                  value={label}
                  onChange={e => setLabel(e.target.value)}
                  placeholder={`e.g. "${intent}" or any action phrase…`}
                />
              </div>
              <div>
                <label className="form-label">Note (optional)</label>
                <input
                  className="input"
                  value={note}
                  onChange={e => setNote(e.target.value)}
                  placeholder="e.g. Customer mentioned invoice number"
                />
              </div>
              <button className="btn btn-primary" disabled={busy} onClick={() => submit('approved')}>
                {busy ? '…' : '✓ Approve'}
              </button>
              <button className="btn btn-outline" disabled={busy} onClick={() => submit('rejected')}>
                {busy ? '…' : '✗ Reject'}
              </button>
            </div>
          )}
        </div>
      )}
    </>
  )
}

export default function Inbox({ onInboxCount }) {
  const [data,    setData]    = useState(null)
  const [filter,  setFilter]  = useState('pending')
  const [loading, setLoading] = useState(true)

  const load = useCallback(async () => {
    setLoading(true)
    const r = await api.getInbox(null, 300)
    setData(r)
    onInboxCount(r.pending || 0)
    setLoading(false)
  }, [onInboxCount])

  useEffect(() => { load() }, [load])

  const items = (data?.items || []).filter(i => filter === 'all' || i.status === filter)

  return (
    <>
      <div className="page-header">
        <div>
          <div className="page-title">Review Inbox</div>
          <div className="page-sub">
            Low-confidence classifications awaiting human review. Approving a correction adds it to the training dataset.
          </div>
        </div>
        <div className="flex gap-sm items-center">
          {loading && <div className="spinner" />}
          <button className="btn btn-outline btn-sm" onClick={load}>↺ Refresh</button>
        </div>
      </div>

      <div className="page-body">
        {/* Stats */}
        {data && (
          <div className="stat-grid" style={{ gridTemplateColumns: 'repeat(5,1fr)' }}>
            {[
              { label: 'Total',    value: data.total,    color: 'var(--ink)'  },
              { label: 'Pending',  value: data.pending,  color: 'var(--amber)'},
              { label: 'Approved', value: data.approved, color: 'var(--green)'},
              { label: 'Rejected', value: data.rejected, color: 'var(--red)'  },
              { label: 'Reviewed', value: data.reviewed, color: 'var(--ink-3)'},
            ].map(s => (
              <div key={s.label} className="stat-card">
                <div className="stat-value" style={{ color: s.color }}>{s.value ?? 0}</div>
                <div className="stat-label">{s.label}</div>
              </div>
            ))}
          </div>
        )}

        {/* Filter + actions */}
        <div className="card" style={{ marginBottom: 16 }}>
          <div className="flex items-center justify-between" style={{ padding: '12px 20px', borderBottom: '1px solid var(--border)' }}>
            <div className="tabs" style={{ borderBottom: 'none', marginBottom: 0 }}>
              {['pending','approved','rejected','all'].map(f => (
                <button key={f} className={`tab${filter === f ? ' active' : ''}`} onClick={() => setFilter(f)}>
                  {f.charAt(0).toUpperCase() + f.slice(1)}
                  {f === 'pending' && data?.pending > 0 && (
                    <span style={{ marginLeft: 6, background: 'var(--amber)', color: '#fff', fontSize: 10, padding: '1px 6px', borderRadius: 10, fontFamily: 'var(--font-mono)' }}>
                      {data.pending}
                    </span>
                  )}
                </button>
              ))}
            </div>
            <div className="flex gap-sm">
              <button className="btn btn-ghost btn-sm" onClick={() => api.clearInbox('rejected').then(load)}>
                Clear Rejected
              </button>
              <button className="btn btn-ghost btn-sm" onClick={() => api.clearInbox('approved').then(load)}>
                Clear Approved
              </button>
            </div>
          </div>

          {/* List */}
          {items.length === 0 ? (
            <div className="empty">
              <div className="empty-icon">{filter === 'pending' ? '▣' : '✓'}</div>
              <div className="empty-title">
                {filter === 'pending' ? 'Nothing to review' : `No ${filter} items`}
              </div>
              <p className="empty-body italic">
                {filter === 'pending'
                  ? 'Items appear here when the classifier returns confidence below the modality threshold.'
                  : 'Change the filter above to see other items.'}
              </p>
            </div>
          ) : (
            items.map(item => (
              <InboxRow key={item.id} item={item} onRefresh={load} />
            ))
          )}
        </div>
      </div>
    </>
  )
}
