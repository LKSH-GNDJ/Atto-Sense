import { useState, useEffect } from 'react'
import * as api from '../api/client'

const INTENT_COLORS = {
  billing:'var(--c-billing)', technical_support:'var(--c-technical)',
  account_management:'var(--c-account)', sales_inquiry:'var(--c-sales)',
  complaint:'var(--c-complaint)', general_inquiry:'var(--c-general)',
  escalation:'var(--c-escalation)', out_of_scope:'var(--c-oos)',
}

function pct(n) { return `${(n * 100).toFixed(1)}%` }

function BarChart({ data, colorMap, title }) {
  if (!data || !Object.keys(data).length) return <p className="text-sm italic muted">No data yet.</p>
  const max  = Math.max(...Object.values(data))
  const rows = Object.entries(data).sort((a,b) => b[1] - a[1])
  return (
    <div>
      <div className="section-label">{title}</div>
      <div className="bar-chart">
        {rows.map(([k, v]) => (
          <div key={k} className="bar-row">
            <span className="bar-row-label">{k.replace(/_/g,' ')}</span>
            <div className="bar-track">
              <div
                className="bar-fill"
                style={{ width: `${max ? v/max*100 : 0}%`, background: colorMap?.[k] || 'var(--accent)' }}
              />
            </div>
            <span className="bar-count">{v}</span>
          </div>
        ))}
      </div>
    </div>
  )
}

export default function Analytics() {
  const [metrics,  setMetrics]  = useState(null)
  const [dataset,  setDataset]  = useState(null)
  const [disagree, setDisagree] = useState(null)
  const [loading,  setLoading]  = useState(true)

  async function load() {
    setLoading(true)
    const [m, d, g] = await Promise.all([api.getMetrics(), api.getDatasetStats(), api.getDisagreements()])
    setMetrics(m)
    setDataset(d)
    setDisagree(g)
    setLoading(false)
  }

  useEffect(() => { load() }, [])

  return (
    <>
      <div className="page-header">
        <div>
          <div className="page-title">Analytics</div>
          <div className="page-sub">Classification metrics, dataset health, and label disagreement tracking</div>
        </div>
        <div className="flex gap-sm items-center">
          {loading && <div className="spinner" />}
          <button className="btn btn-outline btn-sm" onClick={load}>↺ Refresh</button>
        </div>
      </div>

      <div className="page-body">
        {/* KPI row */}
        {metrics && (
          <div className="stat-grid">
            {[
              { label: 'Total Requests',    value: metrics.total_requests,                         color: 'var(--ink)' },
              { label: 'Avg Confidence',    value: pct(metrics.avg_confidence),                    color: metrics.avg_confidence >= 0.82 ? 'var(--green)' : metrics.avg_confidence >= 0.70 ? 'var(--amber)' : 'var(--red)' },
              { label: 'Avg Latency',       value: `${metrics.avg_latency_ms?.toFixed(0) ?? 0} ms`, color: 'var(--ink-2)' },
              { label: 'Escalation Rate',   value: pct(metrics.escalation_rate),                   color: 'var(--red)' },
              { label: 'Low-Conf Rate',     value: pct(metrics.low_confidence_rate),               color: 'var(--amber)' },
              { label: 'Pending Reviews',   value: metrics.inbox_pending ?? 0,                     color: metrics.inbox_pending > 0 ? 'var(--amber)' : 'var(--green)' },
            ].map(s => (
              <div key={s.label} className="stat-card">
                <div className="stat-value" style={{ color: s.color, fontSize: 24 }}>{s.value}</div>
                <div className="stat-label">{s.label}</div>
              </div>
            ))}
          </div>
        )}

        <div className="grid-2" style={{ alignItems: 'start', gap: 24 }}>
          {/* Intent distribution */}
          <div className="card card-pad">
            <BarChart
              data={metrics?.intent_distribution}
              colorMap={INTENT_COLORS}
              title="Intent Distribution"
            />
          </div>

          {/* Modality + Sentiment */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
            <div className="card card-pad">
              <BarChart data={metrics?.modality_distribution} title="Modality Breakdown" />
            </div>
            <div className="card card-pad">
              <BarChart data={metrics?.sentiment_distribution} title="Sentiment Breakdown" />
            </div>
          </div>
        </div>

        {/* Dataset health */}
        {dataset && (
          <div className="card card-pad mt-lg">
            <div className="flex items-center justify-between mb">
              <div>
                <div className="page-title" style={{ fontSize: 17 }}>Training Dataset</div>
                <div className="text-sm muted italic">{dataset.total ?? 0} verified examples</div>
              </div>
              {dataset.imbalanced && (
                <span className="chip chip-amber">⚠ Imbalanced — ratio {dataset.imbalance_ratio}×</span>
              )}
            </div>
            {(dataset.imbalance_details || []).map(row => (
              <div key={row.intent} className="bar-row">
                <span className="bar-row-label">
                  {row.intent.replace(/_/g,' ')}
                  {row.flag && <span style={{ color: 'var(--red)', marginLeft: 4 }}>●</span>}
                </span>
                <div className="bar-track">
                  <div
                    className="bar-fill"
                    style={{
                      width: `${row.pct}%`,
                      background: row.flag ? 'var(--red)' : INTENT_COLORS[row.intent] || 'var(--accent)',
                    }}
                  />
                </div>
                <span className="bar-count">{row.count}</span>
              </div>
            ))}
          </div>
        )}

        {/* Disagreements */}
        {disagree && disagree.top_disagreements?.length > 0 && (
          <div className="card card-pad mt-lg">
            <div className="mb">
              <div className="page-title" style={{ fontSize: 17 }}>Label Disagreements</div>
              <div className="text-sm muted italic">
                {disagree.total_corrections} total corrections · Most frequent intent boundary confusion
              </div>
            </div>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ borderBottom: '2px solid var(--border)', color: 'var(--ink-3)', textTransform: 'uppercase', fontSize: 11, letterSpacing: '0.08em' }}>
                  <th style={{ textAlign: 'left', padding: '6px 8px', fontWeight: 700 }}>Model predicted</th>
                  <th style={{ textAlign: 'left', padding: '6px 8px', fontWeight: 700 }}>Reviewer corrected to</th>
                  <th style={{ textAlign: 'right', padding: '6px 8px', fontWeight: 700 }}>Count</th>
                </tr>
              </thead>
              <tbody>
                {disagree.top_disagreements.map((d, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid var(--border)', background: i % 2 === 0 ? 'transparent' : 'var(--bg-alt)' }}>
                    <td style={{ padding: '8px', color: 'var(--red)' }}>
                      <span className="intent-accent" style={{ background: INTENT_COLORS[d.predicted] || 'var(--c-general)' }} />
                      {d.predicted.replace(/_/g,' ')}
                    </td>
                    <td style={{ padding: '8px', color: 'var(--green)', fontWeight: 600 }}>
                      <span className="intent-accent" style={{ background: INTENT_COLORS[d.corrected] || 'var(--c-general)' }} />
                      {d.corrected.replace(/_/g,' ')}
                    </td>
                    <td style={{ padding: '8px', textAlign: 'right', fontFamily: 'var(--font-mono)', color: 'var(--ink-3)' }}>
                      {d.count}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </>
  )
}
