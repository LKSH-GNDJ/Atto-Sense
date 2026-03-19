"""
AttoSense v3.1 - Sidebar
Clean, light-theme analytics panel for non-technical support staff.
"""

import datetime
import streamlit as st
from frontend.utils import api_client, visualizer


# ── Shared design tokens (must match app.py) ───────────────────────────────────
SIDEBAR_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

[data-testid="stSidebar"] * { font-family: 'DM Sans', sans-serif; }
[data-testid="stSidebar"] { background: #FFFFFF !important; border-right: 1px solid #E8ECF2 !important; }

.sb-logo {
  display: flex; align-items: center; gap: 10px;
  padding: 4px 0 16px 0; border-bottom: 1px solid #F1F5F9;
  margin-bottom: 16px;
}
.sb-logo-icon {
  width: 36px; height: 36px; border-radius: 10px;
  background: linear-gradient(135deg,#0EA5E9,#0284C7);
  display:flex;align-items:center;justify-content:center;
  font-size:18px; flex-shrink:0;
}
.sb-logo-name {
  font-family:'Outfit',sans-serif; font-size:1.1rem;
  font-weight:800; color:#1A2332;
}
.sb-logo-ver { font-size:11px; color:#94A3B8; }

.sb-section {
  font-family:'Outfit',sans-serif; font-size:10px;
  font-weight:700; letter-spacing:1.5px;
  text-transform:uppercase; color:#94A3B8;
  margin: 16px 0 10px 0;
}

/* Status pill */
.status-pill {
  display:inline-flex; align-items:center; gap:7px;
  padding:6px 14px; border-radius:20px;
  font-size:13px; font-weight:500; width:100%;
  margin-bottom:12px;
}
.pill-ok  { background:#F0FDF4; color:#166534; border:1px solid #BBF7D0; }
.pill-err { background:#FEF2F2; color:#991B1B; border:1px solid #FECACA; }
.pill-dot { width:8px;height:8px;border-radius:50%;flex-shrink:0; }
.dot-ok   { background:#10B981; }
.dot-err  { background:#EF4444; }

/* KPI cards */
.kpi-grid { display:grid; grid-template-columns:1fr 1fr; gap:8px; margin-bottom:8px; }
.kpi-card {
  background:#F8FAFC; border:1px solid #E8ECF2;
  border-radius:12px; padding:12px 14px;
  text-align:center;
}
.kpi-value {
  font-family:'Outfit',sans-serif; font-size:1.3rem;
  font-weight:800; color:#1A2332; line-height:1.1;
}
.kpi-label { font-size:11px; color:#94A3B8; margin-top:3px; }
.kpi-good  { color:#059669; }
.kpi-warn  { color:#D97706; }
.kpi-bad   { color:#DC2626; }

/* Inbox alert card */
.inbox-alert {
  background:#FFFBEB; border:1px solid #FDE68A;
  border-radius:12px; padding:12px 16px;
  display:flex; align-items:center; gap:12px;
  margin:8px 0;
}
.inbox-count {
  font-family:'Outfit',sans-serif; font-size:1.8rem;
  font-weight:800; color:#B45309; min-width:36px;
  text-align:center;
}
.inbox-text { font-size:12px; color:#92400E; line-height:1.4; }
.inbox-text b { display:block; font-size:13px; color:#B45309; margin-bottom:2px; }

/* Offline command hint */
.cmd-hint {
  background:#F8FAFC; border:1px dashed #CBD5E1;
  border-radius:8px; padding:10px 14px;
  font-family:'DM Mono',monospace; font-size:12px;
  color:#475569; margin-top:8px;
}

/* Action buttons */
.stButton > button {
  font-family:'Outfit',sans-serif !important;
  font-size:13px !important; font-weight:600 !important;
  border-radius:9px !important;
}
</style>
"""


def _kpi(value: str, label: str, color_class: str = "") -> str:
    return (
        f'<div class="kpi-card">'
        f'<div class="kpi-value {color_class}">{value}</div>'
        f'<div class="kpi-label">{label}</div>'
        f'</div>'
    )


def render_sidebar():
    with st.sidebar:
        st.markdown(SIDEBAR_CSS, unsafe_allow_html=True)

        # ── Logo ──────────────────────────────────────────────────────────────
        st.markdown("""
        <div class="sb-logo">
          <div class="sb-logo-icon">🎯</div>
          <div>
            <div class="sb-logo-name">AttoSense</div>
            <div class="sb-logo-ver">v3.1 · Intent Classifier</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Connection status ─────────────────────────────────────────────────
        health = api_client.health_check()
        online = health.get("status") in ("ok", "degraded")

        if online:
            st.markdown(
                '<div class="status-pill pill-ok">'
                '<span class="pill-dot dot-ok"></span>'
                'API connected &amp; ready'
                '</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="status-pill pill-err">'
                '<span class="pill-dot dot-err"></span>'
                'API offline'
                '</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<div class="cmd-hint">uvicorn backend.api:app --reload</div>',
                unsafe_allow_html=True,
            )
            st.caption("Start the backend server, then refresh this page.")
            return

        # ── Live metrics ──────────────────────────────────────────────────────
        st.markdown('<div class="sb-section">Live Stats</div>', unsafe_allow_html=True)
        metrics = api_client.get_metrics()

        if "error" not in metrics:
            total   = metrics.get("total_requests", 0)
            conf    = metrics.get("avg_confidence", 0)
            lat     = metrics.get("avg_latency_ms", 0)
            esc     = metrics.get("escalation_rate", 0)
            low_c   = metrics.get("low_confidence_rate", 0)
            pending = metrics.get("inbox_pending", 0)

            conf_cls = "kpi-good" if conf >= 0.75 else "kpi-warn" if conf >= 0.5 else "kpi-bad"
            esc_cls  = "kpi-bad"  if esc  >= 0.3  else "kpi-warn" if esc  >= 0.1 else "kpi-good"
            lat_cls  = "kpi-warn" if lat  > 3000  else ""

            st.markdown(
                f'<div class="kpi-grid">'
                f'{_kpi(f"{total:,}", "Total Requests")}'
                f'{_kpi(f"{conf*100:.0f}%", "Avg Confidence", conf_cls)}'
                f'{_kpi(f"{lat:.0f}ms", "Avg Response", lat_cls)}'
                f'{_kpi(f"{esc*100:.1f}%", "Escalation Rate", esc_cls)}'
                f'</div>',
                unsafe_allow_html=True,
            )

            # Inbox alert
            if pending > 0:
                st.markdown(
                    f'<div class="inbox-alert">'
                    f'<div class="inbox-count">{pending}</div>'
                    f'<div class="inbox-text">'
                    f'<b>Reviews waiting</b>'
                    f'Low-confidence results need your attention in the Inbox.'
                    f'</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            # Gauge only when there's data
            if total > 0:
                fig = visualizer.confidence_gauge(conf, "Average Confidence")
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        # ── Charts ────────────────────────────────────────────────────────────
        audit   = api_client.get_audit_log(limit=500)
        entries = audit.get("entries", [])

        if entries and "error" not in metrics:
            st.markdown('<div class="sb-section">Analytics</div>', unsafe_allow_html=True)

            intent_dist = metrics.get("intent_distribution", {})
            if intent_dist:
                st.plotly_chart(
                    visualizer.intent_bar(intent_dist),
                    use_container_width=True,
                    config={"displayModeBar": False},
                )

            c1, c2 = st.columns(2)
            with c1:
                sent_dist = metrics.get("sentiment_distribution", {})
                if sent_dist:
                    st.plotly_chart(
                        visualizer.sentiment_donut(sent_dist),
                        use_container_width=True,
                        config={"displayModeBar": False},
                    )
            with c2:
                mod_dist = metrics.get("modality_distribution", {})
                if mod_dist:
                    st.plotly_chart(
                        visualizer.modality_donut(mod_dist),
                        use_container_width=True,
                        config={"displayModeBar": False},
                    )

            st.plotly_chart(
                visualizer.confidence_timeline(entries),
                use_container_width=True,
                config={"displayModeBar": False},
            )

            # ── Export & maintenance ──────────────────────────────────────────
            st.markdown('<div class="sb-section">Reports</div>', unsafe_allow_html=True)

            if st.button("⬇  Export PDF Report", use_container_width=True):
                with st.spinner("Generating report…"):
                    pdf_bytes = visualizer.export_pdf_report(metrics, entries)
                ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                st.download_button(
                    label="📥  Download Report",
                    data=pdf_bytes,
                    file_name=f"attosense_report_{ts}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )

            if st.button("🗑  Clear Audit Log", use_container_width=True, type="secondary"):
                res = api_client.clear_audit_log()
                if res.get("success"):
                    st.success("Audit log cleared.")
                    st.rerun()
                else:
                    st.error(res.get("error", "Failed to clear log."))

        elif not entries:
            st.markdown(
                "<div style='text-align:center;padding:24px 0;color:#CBD5E1;font-size:13px;'>"
                "No data yet.<br>Send a few requests to see analytics here."
                "</div>",
                unsafe_allow_html=True,
            )
