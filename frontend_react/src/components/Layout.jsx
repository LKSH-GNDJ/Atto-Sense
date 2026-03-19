import { useEffect, useState } from 'react'
import { health } from '../api/client'

const NAV = [
  { id: 'classify',  label: 'Classify',      icon: '◎' },
  { id: 'inbox',     label: 'Review Inbox',  icon: '▣' },
  { id: 'analytics', label: 'Analytics',     icon: '◈' },
]

export default function Layout({ page, setPage, inboxCount, sessionId, onNewSession, children }) {
  const [status, setStatus] = useState('checking')

  useEffect(() => {
    const check = () =>
      health()
        .then(r => setStatus(r.status === 'ok' || r.status === 'degraded' ? 'online' : 'offline'))
        .catch(() => setStatus('offline'))
    check()
    const t = setInterval(check, 30_000)
    return () => clearInterval(t)
  }, [])

  return (
    <div className="app-shell">
      {/* ── Sidebar ───────────────────────────────────────────── */}
      <nav className="nav">
        <div className="nav-logo">
          <div className="nav-logo-mark">Atto<span>Sense</span></div>
          <div className="nav-logo-sub">Intent Intelligence · v4</div>
        </div>

        <div className="nav-section">
          <div className="nav-section-label">Navigation</div>
          {NAV.map(item => (
            <button
              key={item.id}
              className={`nav-item${page === item.id ? ' active' : ''}`}
              onClick={() => setPage(item.id)}
            >
              <span className="nav-icon">{item.icon}</span>
              {item.label}
              {item.id === 'inbox' && inboxCount > 0 && (
                <span className="nav-badge">{inboxCount}</span>
              )}
            </button>
          ))}
        </div>

        <div className="nav-footer">
          <div className="nav-status">
            <div className={`status-dot ${status}`} />
            <span className="nav-status-text">
              {status === 'online'   ? 'Connected'    :
               status === 'offline'  ? 'Backend offline' :
               'Checking…'}
            </span>
          </div>
          <div className="nav-session">
            Session {sessionId}
            <button
              onClick={onNewSession}
              style={{
                marginLeft: 8, background: 'none', border: 'none',
                color: 'var(--nav-accent)', cursor: 'pointer',
                fontFamily: 'var(--font)', fontSize: 10,
                textDecoration: 'underline', padding: 0,
              }}
            >
              New
            </button>
          </div>
        </div>
      </nav>

      {/* ── Main ──────────────────────────────────────────────── */}
      <main className="main">
        {children}
      </main>
    </div>
  )
}
