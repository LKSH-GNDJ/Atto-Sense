"""
AttoSense v4 - Review Inbox
sys.path fixed at top so this works regardless of launch method.
"""

# ── Path fix (MUST be first) ───────────────────────────────────────────────────
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# ── Standard imports ───────────────────────────────────────────────────────────
import streamlit as st
from frontend.utils import api_client
from frontend.components.sidebar import render_sidebar

st.set_page_config(
    page_title="Review Inbox — AttoSense",
    page_icon="📬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family:'DM Sans',sans-serif; background:#F7F8FA; color:#1A2332; }
.stApp { background:#F7F8FA; }
[data-testid="stSidebar"] { background:#FFFFFF !important; border-right:1px solid #E8ECF2 !important; }

.page-header {
  background:#FFFFFF; border:1px solid #E8ECF2; border-left:4px solid #F59E0B;
  border-radius:16px; padding:20px 28px; margin-bottom:20px;
  box-shadow:0 1px 4px rgba(26,35,50,0.06); display:flex; align-items:center; gap:16px;
}
.page-icon {
  width:44px; height:44px; border-radius:12px;
  background:linear-gradient(135deg,#F59E0B,#D97706);
  display:flex; align-items:center; justify-content:center; font-size:22px; flex-shrink:0;
}
.page-title { font-family:'Outfit',sans-serif; font-size:1.4rem; font-weight:800; color:#1A2332; margin:0; }
.page-subtitle { font-size:13px; color:#64748B; margin:2px 0 0 0; }

.stat-strip { display:grid; grid-template-columns:repeat(5,1fr); gap:10px; margin-bottom:16px; }
.stat-box {
  background:#FFFFFF; border:1px solid #E8ECF2; border-radius:14px; padding:14px 16px;
  text-align:center; box-shadow:0 1px 3px rgba(26,35,50,0.05);
}
.stat-num { font-family:'Outfit',sans-serif; font-size:1.6rem; font-weight:800; color:#1A2332; line-height:1; }
.stat-lbl { font-size:11px; color:#94A3B8; margin-top:4px; }
.stat-pending  .stat-num { color:#D97706; }
.stat-approved .stat-num { color:#059669; }
.stat-rejected .stat-num { color:#DC2626; }

.review-card {
  background:#FFFFFF; border:1px solid #E8ECF2; border-radius:16px; padding:20px 24px;
  margin-bottom:12px; box-shadow:0 1px 4px rgba(26,35,50,0.05);
  transition:box-shadow 0.15s;
}
.review-card:hover { box-shadow:0 4px 12px rgba(26,35,50,0.1); }
.review-card-pending  { border-left:4px solid #F59E0B; }
.review-card-approved { border-left:4px solid #10B981; }
.review-card-rejected { border-left:4px solid #EF4444; }
.review-card-reviewed { border-left:4px solid #0EA5E9; }

.badge {
  display:inline-flex; align-items:center; gap:5px; padding:3px 10px; border-radius:20px;
  font-size:11px; font-weight:600; font-family:'Outfit',sans-serif;
}
.badge-pending  { background:#FFFBEB; color:#B45309; border:1px solid #FDE68A; }
.badge-approved { background:#F0FDF4; color:#166534; border:1px solid #BBF7D0; }
.badge-rejected { background:#FEF2F2; color:#DC2626; border:1px solid #FECACA; }
.badge-reviewed { background:#EFF6FF; color:#1D4ED8; border:1px solid #BFDBFE; }

.intent-chip {
  display:inline-flex; align-items:center; gap:5px; padding:4px 12px; border-radius:8px;
  font-size:13px; font-weight:600; font-family:'Outfit',sans-serif;
  background:#F1F5F9; color:#475569; border:1px solid #E2E8F0;
}
.mini-conf-track { height:5px; border-radius:3px; background:#F1F5F9; overflow:hidden; margin:4px 0 2px; }
.mini-conf-fill  { height:100%; border-radius:3px; }
.mc-high { background:#10B981; } .mc-mid { background:#F59E0B; } .mc-low { background:#EF4444; }

.msg-bubble {
  background:#F8FAFC; border:1px solid #E2E8F0; border-radius:10px; padding:10px 14px;
  font-size:13px; color:#475569; margin:10px 0; line-height:1.5; font-style:italic;
}
.reasoning-line { font-size:12px; color:#94A3B8; margin-top:6px; line-height:1.5; }
.entity-row { display:flex; flex-wrap:wrap; gap:5px; margin-top:8px; }
.entity-tag {
  display:inline-flex; align-items:center; gap:4px; padding:3px 9px; border-radius:7px;
  background:#EFF6FF; border:1px solid #BFDBFE; font-size:11px; color:#1D4ED8;
}
.vision-insight {
  background:#FDF4FF; border:1px solid #E9D5FF; border-radius:10px; padding:12px 16px;
  margin-top:10px; font-size:12px;
}
.vision-insight-title {
  font-family:'Outfit',sans-serif; font-size:10px; font-weight:700; letter-spacing:1px;
  text-transform:uppercase; color:#7C3AED; margin-bottom:6px;
}
.frust-track {
  height:6px; border-radius:3px; margin:4px 0;
  background:linear-gradient(to right,#10B981,#F59E0B,#EF4444); position:relative;
}
.frust-dot {
  position:absolute; top:-4px; width:14px; height:14px; border-radius:50%;
  background:white; border:2px solid #7C3AED; transform:translateX(-50%);
  box-shadow:0 1px 3px rgba(0,0,0,0.15);
}
.review-note {
  background:#F0FDF4; border:1px solid #BBF7D0; border-radius:8px;
  padding:8px 12px; font-size:12px; color:#166534; margin-top:8px;
}
.empty-inbox { text-align:center; padding:80px 20px; color:#94A3B8; }
.empty-inbox-icon { font-size:56px; margin-bottom:16px; }
.empty-inbox-title { font-family:'Outfit',sans-serif; font-size:1.2rem; font-weight:700; color:#64748B; margin-bottom:8px; }
.empty-inbox-body  { font-size:14px; color:#94A3B8; line-height:1.6; }

.stButton > button {
  font-family:'Outfit',sans-serif !important; font-weight:600 !important;
  border-radius:9px !important; font-size:13px !important;
}
.stSelectbox > div > div,
.stTextInput > div > input {
  background:#F8FAFC !important; border:1.5px solid #E2E8F0 !important;
  border-radius:9px !important; font-family:'DM Sans',sans-serif !important;
  font-size:13px !important; color:#1A2332 !important;
}
hr { border-color:#F1F5F9 !important; }
</style>
""", unsafe_allow_html=True)

render_sidebar()

# ── Constants ──────────────────────────────────────────────────────────────────
INTENT_OPTIONS = [
    "billing", "technical_support", "account_management", "sales_inquiry",
    "complaint", "general_inquiry", "escalation", "out_of_scope",
]
INTENT_META = {
    "billing":            ("💳",), "technical_support":  ("🔧",),
    "account_management": ("👤",), "sales_inquiry":      ("📈",),
    "complaint":          ("⚠️",),  "general_inquiry":    ("💬",),
    "escalation":         ("🚨",), "out_of_scope":       ("❓",),
}
MODALITY_ICON    = {"text": "💬", "audio": "🎙", "vision": "🖼"}
SENTIMENT_COLOR  = {"positive":"#059669","neutral":"#94A3B8","negative":"#D97706","frustrated":"#DC2626"}

# ── Page header ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="page-header">
  <div class="page-icon">📬</div>
  <div>
    <div class="page-title">Review Inbox</div>
    <div class="page-subtitle">
      Low-confidence classifications waiting for your review.
      Approving a corrected label automatically adds it to the training dataset.
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Load data ──────────────────────────────────────────────────────────────────
inbox_data = api_client.get_inbox(status=None, limit=500)
if "error" in inbox_data:
    st.error(f"Could not reach the API: {inbox_data['error']}")
    st.info("Make sure the backend is running: `uvicorn backend.api:app --reload`")
    st.stop()

total    = inbox_data.get("total", 0)
pending  = inbox_data.get("pending", 0)
approved = inbox_data.get("approved", 0)
rejected = inbox_data.get("rejected", 0)
reviewed = inbox_data.get("reviewed", 0)

# ── Stat strip ─────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="stat-strip">
  <div class="stat-box"><div class="stat-num">{total}</div><div class="stat-lbl">Total Flagged</div></div>
  <div class="stat-box stat-pending"><div class="stat-num">{pending}</div><div class="stat-lbl">⏳ Awaiting Review</div></div>
  <div class="stat-box stat-approved"><div class="stat-num">{approved}</div><div class="stat-lbl">✅ Approved</div></div>
  <div class="stat-box stat-rejected"><div class="stat-num">{rejected}</div><div class="stat-lbl">❌ Rejected</div></div>
  <div class="stat-box"><div class="stat-num">{reviewed}</div><div class="stat-lbl">👁 Reviewed</div></div>
</div>
""", unsafe_allow_html=True)

# ── Filters ────────────────────────────────────────────────────────────────────
fc1, fc2, fc3, fc4 = st.columns([2, 2, 2, 1])
with fc1:
    status_sel = st.selectbox("Status",
        ["All statuses","Pending review","Approved","Rejected","Reviewed"],
        label_visibility="collapsed", key="status_f")
with fc2:
    mod_sel = st.selectbox("Modality",
        ["All types","💬 Text","🎙 Audio","🖼 Image"],
        label_visibility="collapsed", key="mod_f")
with fc3:
    conf_max = st.slider("Max confidence", 0.0, 1.0, 0.70, 0.01, key="conf_f")
with fc4:
    if st.button("↺ Refresh", use_container_width=True):
        st.rerun()

STATUS_MAP = {"All statuses":None,"Pending review":"pending","Approved":"approved","Rejected":"rejected","Reviewed":"reviewed"}
MOD_MAP    = {"All types":None,"💬 Text":"text","🎙 Audio":"audio","🖼 Image":"vision"}
status_val = STATUS_MAP[status_sel]
mod_val    = MOD_MAP[mod_sel]

items = inbox_data.get("items", [])
if status_val:
    items = [i for i in items if i.get("status") == status_val]
if mod_val:
    items = [i for i in items if i.get("modality") == mod_val]
items = [i for i in items if i.get("result", {}).get("confidence", 1.0) <= conf_max]

# ── Empty state ────────────────────────────────────────────────────────────────
if not items:
    msg = ("Your inbox is clear — great work! 🎉<br><br>"
           "<span style='font-size:13px'>Items appear here when the AI classifies a message<br>"
           "with less than 70% confidence and needs a human check.</span>"
           if total == 0 else
           "No items match the current filters.<br><br>"
           "<span style='font-size:13px'>Try changing the status or modality filter above.</span>")
    st.markdown(f"""
    <div class="empty-inbox">
      <div class="empty-inbox-icon">{"📭" if total == 0 else "🔍"}</div>
      <div class="empty-inbox-title">{"Nothing to review" if total == 0 else "No matches"}</div>
      <div class="empty-inbox-body">{msg}</div>
    </div>""", unsafe_allow_html=True)
    st.stop()

# ── Bulk actions ───────────────────────────────────────────────────────────────
st.markdown(f"<div style='font-size:13px;color:#64748B;margin-bottom:8px;'>"
            f"Showing <b>{len(items)}</b> item{'s' if len(items)!=1 else ''}</div>",
            unsafe_allow_html=True)
ba1, ba2, ba3, _ = st.columns([2, 2, 2, 6])
with ba1:
    if st.button("Clear Pending",  use_container_width=True, key="cl_p"):
        api_client.clear_inbox(status="pending");  st.rerun()
with ba2:
    if st.button("Clear Approved", use_container_width=True, key="cl_a"):
        api_client.clear_inbox(status="approved"); st.rerun()
with ba3:
    if st.button("Clear All", use_container_width=True, type="secondary", key="cl_all"):
        api_client.clear_inbox(); st.rerun()

st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

# ── Item cards ─────────────────────────────────────────────────────────────────
for item in items:
    result    = item.get("result", {})
    conf      = result.get("confidence", 0.0)
    intent_k  = result.get("intent", "general_inquiry")
    sentiment = result.get("sentiment", "neutral")
    entities  = result.get("entities", [])
    reasoning = result.get("reasoning", "")
    vision    = result.get("vision")
    raw_input = item.get("raw_input", "")
    modality  = item.get("modality", "text")
    status    = item.get("status", "pending")
    item_id   = item.get("id", "")
    ts        = str(item.get("timestamp",""))[:16].replace("T"," ")
    rev_label = item.get("reviewer_label")
    rev_note  = item.get("reviewer_note")

    icon      = INTENT_META.get(intent_k, ("💬",))[0]
    conf_pct  = int(conf * 100)
    conf_cls  = "mc-high" if conf >= 0.75 else "mc-mid" if conf >= 0.5 else "mc-low"
    conf_col  = "#059669" if conf >= 0.75 else "#D97706" if conf >= 0.5 else "#DC2626"
    sent_col  = SENTIMENT_COLOR.get(sentiment, "#94A3B8")

    STATUS_TEXT = {"pending":"⏳ Pending","approved":"✅ Approved","rejected":"❌ Rejected","reviewed":"👁 Reviewed"}

    with st.container():
        st.markdown(f'<div class="review-card review-card-{status}">', unsafe_allow_html=True)

        left, right = st.columns([5, 3])
        with left:
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;">'
                f'<span class="intent-chip">{icon} {intent_k.replace("_"," ").title()}</span>'
                f'<span class="badge badge-{status}">{STATUS_TEXT.get(status,status)}</span>'
                f'</div>'
                f'<div style="font-size:11px;color:#CBD5E1;margin-top:6px;">'
                f'{MODALITY_ICON.get(modality,"•")} {modality.title()} · {ts} · ID: {item_id[:8]}'
                f'</div>', unsafe_allow_html=True)
        with right:
            st.markdown(
                f'<div style="text-align:right;">'
                f'<div style="font-family:Outfit,sans-serif;font-size:1.3rem;font-weight:800;color:{conf_col};">{conf_pct}%</div>'
                f'<div style="font-size:11px;color:#94A3B8;">confidence</div>'
                f'<div class="mini-conf-track" style="width:80px;margin-left:auto;">'
                f'<div class="mini-conf-fill {conf_cls}" style="width:{conf_pct}%"></div></div>'
                f'<span style="font-size:11px;color:{sent_col};">{sentiment.title()}</span>'
                + (' &nbsp;·&nbsp; <span style="color:#DC2626;font-size:11px;">🚨 Escalation</span>'
                   if result.get("requires_escalation") else "")
                + '</div>', unsafe_allow_html=True)

        if raw_input and raw_input not in ("[audio]","[image]","[audio upload]","[image upload]"):
            disp = raw_input[:400] + ("…" if len(raw_input) > 400 else "")
            st.markdown(f'<div class="msg-bubble">"{disp}"</div>', unsafe_allow_html=True)

        if reasoning:
            st.markdown(f'<div class="reasoning-line">💡 {reasoning}</div>', unsafe_allow_html=True)

        if entities:
            tags = "".join(
                f'<span class="entity-tag"><b>{e.get("label","?")}:</b> {e.get("value","?")}</span>'
                for e in entities)
            st.markdown(f'<div class="entity-row">{tags}</div>', unsafe_allow_html=True)

        if vision and modality == "vision":
            frust    = vision.get("frustration_score", 0.0)
            etype    = vision.get("error_type", "none")
            edetail  = vision.get("error_detail") or ""
            vsummary = vision.get("visual_summary") or ""
            fp       = int(frust * 100)
            fcol     = "#059669" if frust < 0.4 else "#D97706" if frust < 0.7 else "#DC2626"
            st.markdown(
                f'<div class="vision-insight">'
                f'<div class="vision-insight-title">🖼 Image Analysis</div>'
                + (f'<div style="color:#475569;margin-bottom:8px;">{vsummary}</div>' if vsummary else "")
                + f'<div style="display:flex;justify-content:space-between;font-size:12px;color:#7C3AED;">'
                  f'<span>Frustration</span><span style="font-weight:700;color:{fcol}">{fp}%</span></div>'
                  f'<div class="frust-track"><div class="frust-dot" style="left:{fp}%"></div></div>'
                  f'<div style="font-size:12px;color:#6D28D9;margin-top:6px;">'
                  f'Error: <b>{etype.replace("_"," ").title()}</b>'
                + (f' — {edetail}' if edetail not in ("","null","None") else "")
                + '</div></div>', unsafe_allow_html=True)

        if rev_label:
            st.markdown(
                f'<div class="review-note">✅ Corrected to <b>{rev_label.replace("_"," ").title()}</b>'
                + (f' &nbsp;·&nbsp; "{rev_note}"' if rev_note else "")
                + '</div>', unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        # ── Review form ────────────────────────────────────────────────────────
        if status == "pending":
            with st.expander("✏️  Review this item", expanded=False):
                rc1, rc2 = st.columns([3, 3])
                with rc1:
                    label_choice = st.selectbox(
                        "Correct intent label",
                        options=["Keep original"] + [i.replace("_"," ").title() for i in INTENT_OPTIONS],
                        key=f"lbl_{item_id}")
                with rc2:
                    note = st.text_input("Add a note (optional)",
                                         placeholder="e.g. Customer mentioned invoice",
                                         key=f"note_{item_id}")
                st.markdown(
                    "<div style='font-size:12px;color:#94A3B8;margin:8px 0;'>"
                    "Approving with a corrected label adds this example to the training dataset."
                    "</div>", unsafe_allow_html=True)

                b1, b2, b3 = st.columns([3, 3, 1])
                with b1:
                    if st.button("✅  Approve", key=f"app_{item_id}",
                                  use_container_width=True, type="primary"):
                        raw_lbl = None
                        if label_choice != "Keep original":
                            raw_lbl = INTENT_OPTIONS[
                                [i.replace("_"," ").title() for i in INTENT_OPTIONS].index(label_choice)]
                        res = api_client.review_inbox_item(item_id, "approved", raw_lbl, note or None)
                        if "error" not in res:
                            st.success("Approved!" + (f" Labelled as **{label_choice}**." if raw_lbl else ""))
                            st.rerun()
                        else:
                            st.error(res["error"])
                with b2:
                    if st.button("❌  Reject", key=f"rej_{item_id}", use_container_width=True):
                        res = api_client.review_inbox_item(item_id, "rejected", None, note or None)
                        if "error" not in res:
                            st.warning("Marked as rejected.")
                            st.rerun()
                        else:
                            st.error(res["error"])
                with b3:
                    if st.button("🗑", key=f"del_{item_id}", use_container_width=True,
                                  help="Delete this item"):
                        api_client.delete_inbox_item(item_id)
                        st.rerun()

        st.markdown("<div style='height:2px'></div>", unsafe_allow_html=True)
