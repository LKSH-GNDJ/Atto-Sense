"""
BotTrain v4 - Main Streamlit App
sys.path is fixed at the very top so this file works regardless of
which folder or method Streamlit is launched from.
"""

# ── Path fix (MUST be first) ───────────────────────────────────────────────────
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ── Standard imports ───────────────────────────────────────────────────────────
import uuid
import streamlit as st
from frontend.components.sidebar import render_sidebar
from frontend.utils import api_client

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BotTrain · Intent Classifier",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,400&display=swap');

html, body, [class*="css"] { font-family:'DM Sans',sans-serif; background:#F7F8FA; color:#1A2332; }
.stApp { background:#F7F8FA; }
[data-testid="stSidebar"] { background:#FFFFFF; border-right:1px solid #E8ECF2; }

.app-header {
  display:flex; align-items:center; gap:16px; padding:20px 28px;
  background:#FFFFFF; border-radius:16px; border:1px solid #E8ECF2;
  box-shadow:0 1px 4px rgba(26,35,50,0.06); margin-bottom:20px;
}
.app-logo {
  width:44px; height:44px; background:linear-gradient(135deg,#0EA5E9,#0284C7);
  border-radius:12px; display:flex; align-items:center; justify-content:center;
  font-size:22px; flex-shrink:0;
}
.app-title { font-family:'Outfit',sans-serif; font-size:1.5rem; font-weight:800; color:#1A2332; margin:0; }
.app-subtitle { font-size:13px; color:#64748B; margin:2px 0 0 0; }
.status-dot { width:8px; height:8px; border-radius:50%; display:inline-block; margin-right:6px; }
.status-online  { background:#10B981; box-shadow:0 0 0 3px #D1FAE5; }
.status-offline { background:#EF4444; box-shadow:0 0 0 3px #FEE2E2; }

.input-panel {
  background:#FFFFFF; border:1px solid #E8ECF2; border-radius:16px; padding:24px;
  box-shadow:0 1px 4px rgba(26,35,50,0.05);
}
.panel-label {
  font-family:'Outfit',sans-serif; font-size:11px; font-weight:700;
  letter-spacing:1.5px; text-transform:uppercase; color:#94A3B8; margin-bottom:12px;
}
.mode-description { font-size:13px; color:#64748B; margin:0 0 16px 0; line-height:1.5; }

.result-card {
  background:#FFFFFF; border:1px solid #E8ECF2; border-radius:16px; padding:20px 24px;
  box-shadow:0 2px 8px rgba(26,35,50,0.07); margin:12px 0;
  animation:slideUp 0.25s ease;
}
@keyframes slideUp { from{opacity:0;transform:translateY(8px)} to{opacity:1;transform:translateY(0)} }

.intent-card { border-left:4px solid; padding-left:20px; }
.intent-billing       { border-color:#3B82F6; }
.intent-technical     { border-color:#F97316; }
.intent-account       { border-color:#0EA5E9; }
.intent-sales         { border-color:#10B981; }
.intent-complaint     { border-color:#EF4444; }
.intent-general       { border-color:#94A3B8; }
.intent-escalation    { border-color:#DC2626; }
.intent-out_of_scope  { border-color:#CBD5E1; }

.intent-label { font-family:'Outfit',sans-serif; font-size:1.15rem; font-weight:700; color:#1A2332; margin:0; }
.intent-sublabel { font-size:12px; color:#94A3B8; margin:2px 0 14px 0; }

.conf-track { height:6px; border-radius:3px; background:#F1F5F9; overflow:hidden; margin:6px 0; }
.conf-fill  { height:100%; border-radius:3px; transition:width 0.4s ease; }
.conf-high   { background:#10B981; }
.conf-medium { background:#F59E0B; }
.conf-low    { background:#EF4444; }
.conf-label  { font-size:12px; color:#64748B; }

.badge { display:inline-flex; align-items:center; gap:5px; padding:4px 12px; border-radius:20px; font-size:12px; font-weight:600; font-family:'Outfit',sans-serif; }
.badge-escalate { background:#FEF2F2; color:#DC2626; border:1px solid #FECACA; }
.badge-inbox    { background:#FFFBEB; color:#B45309; border:1px solid #FDE68A; }
.badge-pos  { background:#F0FDF4; color:#166534; border:1px solid #BBF7D0; }
.badge-neg  { background:#FFF7ED; color:#C2410C; border:1px solid #FED7AA; }
.badge-frus { background:#FEF2F2; color:#DC2626; border:1px solid #FECACA; }
.badge-neu  { background:#F8FAFC; color:#64748B; border:1px solid #E2E8F0; }

.entity-row { display:flex; flex-wrap:wrap; gap:6px; margin-top:10px; }
.entity-tag {
  display:inline-flex; align-items:center; gap:5px; padding:4px 10px; border-radius:8px;
  background:#EFF6FF; border:1px solid #BFDBFE; font-size:12px; color:#1D4ED8;
}
.entity-label { font-weight:600; }

.vision-insight {
  background:#FDF4FF; border:1px solid #E9D5FF; border-radius:12px;
  padding:14px 18px; margin-top:12px; font-size:13px;
}
.vision-insight-title {
  font-family:'Outfit',sans-serif; font-weight:700; font-size:11px;
  letter-spacing:1px; text-transform:uppercase; color:#7C3AED; margin-bottom:8px;
}
.frustration-track {
  height:8px; border-radius:4px;
  background:linear-gradient(to right,#10B981,#F59E0B,#EF4444);
  position:relative; margin:8px 0;
}
.frustration-marker {
  position:absolute; top:-4px; width:16px; height:16px; border-radius:50%;
  background:white; border:2px solid #7C3AED; transform:translateX(-50%);
  box-shadow:0 1px 4px rgba(0,0,0,0.2);
}
.reasoning-box {
  background:#F8FAFC; border:1px solid #E2E8F0; border-radius:10px;
  padding:12px 16px; font-size:13px; color:#475569; margin-top:12px;
  line-height:1.5; font-style:italic;
}
.transcript-box {
  background:#F0FDF4; border:1px solid #BBF7D0; border-radius:10px;
  padding:10px 16px; font-size:13px; color:#166534; margin-top:10px; line-height:1.5;
}
.bubble-user {
  background:#EFF6FF; border:1px solid #BFDBFE; border-radius:14px 14px 4px 14px;
  padding:12px 16px; font-size:14px; color:#1E40AF;
  display:inline-block; max-width:80%; margin-bottom:4px;
}
.empty-state { text-align:center; padding:60px 20px; color:#94A3B8; }
.empty-icon  { font-size:48px; margin-bottom:16px; }
.empty-title { font-family:'Outfit',sans-serif; font-size:1.1rem; font-weight:700; color:#64748B; margin-bottom:8px; }
.empty-body  { font-size:14px; color:#94A3B8; line-height:1.6; }

.stButton > button {
  font-family:'Outfit',sans-serif !important; font-weight:600 !important;
  border-radius:10px !important; font-size:14px !important;
}
[data-testid="stForm"] { border:none !important; padding:0 !important; }
.stTextInput > div > input,
.stTextArea > div > textarea {
  background:#F8FAFC !important; border:1.5px solid #E2E8F0 !important;
  border-radius:10px !important; font-family:'DM Sans',sans-serif !important;
  font-size:14px !important; color:#1A2332 !important;
}
.stTabs [data-baseweb="tab-list"] {
  background:#F1F5F9 !important; border-radius:12px !important; padding:4px !important;
}
.stTabs [data-baseweb="tab"] {
  font-family:'Outfit',sans-serif !important; font-size:13px !important;
  font-weight:600 !important; border-radius:9px !important; color:#64748B !important;
}
.stTabs [aria-selected="true"] {
  background:#FFFFFF !important; color:#0EA5E9 !important;
  box-shadow:0 1px 3px rgba(26,35,50,0.12) !important;
}
.stRadio label { font-size:13px !important; font-family:'DM Sans',sans-serif !important; }
hr { border-color:#E8ECF2 !important; }
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────────
for k, v in [
    ("session_id",       str(uuid.uuid4())[:8]),
    ("history",          []),
    ("last_audio_bytes", None),
    ("last_audio_mime",  "audio/wav"),
    ("example_text",     ""),
]:
    if k not in st.session_state:
        st.session_state[k] = v

# ── Sidebar ────────────────────────────────────────────────────────────────────
render_sidebar()

# ── Header ─────────────────────────────────────────────────────────────────────
health  = api_client.health_check()
online  = health.get("status") in ("ok", "degraded")
dot_cls = "status-dot status-online" if online else "status-dot status-offline"
dot_txt = "Connected" if online else "Backend offline — start uvicorn in Terminal 1"

st.markdown(f"""
<div class="app-header">
  <div class="app-logo">🎯</div>
  <div style="flex:1">
    <div class="app-title">BotTrain</div>
    <div class="app-subtitle">Multimodal Intent Classifier · v4</div>
  </div>
  <div style="font-size:13px;color:#64748B;display:flex;align-items:center;">
    <span class="{dot_cls}"></span>{dot_txt}
  </div>
  <div style="font-size:11px;color:#CBD5E1;padding-left:16px;border-left:1px solid #E8ECF2;">
    Session<br><b style="color:#94A3B8">{st.session_state.session_id}</b>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Metadata ───────────────────────────────────────────────────────────────────
INTENT_META = {
    "billing":            ("💳", "#3B82F6", "Billing & Payments",  "intent-billing"),
    "technical_support":  ("🔧", "#F97316", "Technical Support",   "intent-technical"),
    "account_management": ("👤", "#0EA5E9", "Account Management",  "intent-account"),
    "sales_inquiry":      ("📈", "#10B981", "Sales Inquiry",       "intent-sales"),
    "complaint":          ("⚠️",  "#EF4444", "Complaint",           "intent-complaint"),
    "general_inquiry":    ("💬", "#94A3B8", "General Inquiry",     "intent-general"),
    "escalation":         ("🚨", "#DC2626", "Escalation Required", "intent-escalation"),
    "out_of_scope":       ("❓", "#CBD5E1", "Out of Scope",        "intent-out_of_scope"),
}
SENTIMENT_BADGE = {
    "positive":   ("😊 Positive",  "badge-pos"),
    "neutral":    ("😐 Neutral",    "badge-neu"),
    "negative":   ("😞 Negative",   "badge-neg"),
    "frustrated": ("😤 Frustrated", "badge-frus"),
}
MODALITY_LABEL = {"text": "💬 Text", "audio": "🎙 Audio", "vision": "🖼 Image"}


# ── Result renderer ────────────────────────────────────────────────────────────
def render_result(response: dict):
    if not response.get("success"):
        st.error(f"Error: {response.get('error', 'Unknown error')}")
        return

    result   = response["result"]
    intent   = result.get("intent", "general_inquiry")
    conf     = result.get("confidence", 0)
    sent     = result.get("sentiment", "neutral")
    esc      = result.get("requires_escalation", False)
    low_c    = result.get("low_confidence", False)
    inbox    = response.get("inbox_flagged", False)
    entities = result.get("entities", [])
    reason   = result.get("reasoning", "")
    trans    = result.get("raw_transcript", "")
    vision   = result.get("vision")
    latency  = response.get("latency_ms", 0)
    modality = response.get("modality", "text")

    icon, _, label, css_class = INTENT_META.get(
        intent, ("💬", "#94A3B8", intent.replace("_", " ").title(), "intent-general")
    )
    conf_cls = "conf-high" if conf >= 0.75 else "conf-medium" if conf >= 0.5 else "conf-low"
    conf_pct = int(conf * 100)
    sent_txt, sent_cls = SENTIMENT_BADGE.get(sent, (sent, "badge-neu"))

    st.markdown(f"""
    <div class="result-card intent-card {css_class}">
      <div style="display:flex;align-items:flex-start;justify-content:space-between;flex-wrap:wrap;gap:8px;">
        <div>
          <div class="intent-label">{icon}&nbsp;&nbsp;{label}</div>
          <div class="intent-sublabel">{MODALITY_LABEL.get(modality,modality)} · {latency:.0f}ms</div>
        </div>
        <div style="display:flex;flex-wrap:wrap;gap:6px;align-items:center;">
          <span class="badge {sent_cls}">{sent_txt}</span>
          {"<span class='badge badge-escalate'>🚨 Needs Escalation</span>" if esc else ""}
          {"<span class='badge badge-inbox'>📥 Sent to Review Inbox</span>" if inbox else ""}
        </div>
      </div>
      <div style="margin-top:14px;">
        <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
          <span class="conf-label">Confidence</span>
          <span class="conf-label" style="font-weight:600;color:#1A2332;">{conf_pct}%</span>
        </div>
        <div class="conf-track"><div class="conf-fill {conf_cls}" style="width:{conf_pct}%"></div></div>
        {"<div style='font-size:12px;color:#F59E0B;margin-top:4px;'>⚠️ Low confidence — added to Review Inbox</div>" if low_c else ""}
      </div>
    """, unsafe_allow_html=True)

    if trans and trans not in ("[audio]", "[image]", "[audio upload]", "[image upload]"):
        st.markdown(
            f'<div class="transcript-box">🎙 <b>Transcript:</b> {trans[:400]}{"…" if len(trans)>400 else ""}</div>',
            unsafe_allow_html=True,
        )

    if entities:
        tags = "".join(
            f'<span class="entity-tag"><span class="entity-label">{e["label"]}</span>&nbsp;{e["value"]}</span>'
            for e in entities
        )
        st.markdown(f'<div class="entity-row">{tags}</div>', unsafe_allow_html=True)

    if vision:
        frust     = vision.get("frustration_score", 0.0)
        etype     = vision.get("error_type", "none")
        edetail   = vision.get("error_detail") or ""
        vsummary  = vision.get("visual_summary") or ""
        fp        = int(frust * 100)
        fcol      = "#10B981" if frust < 0.4 else "#F59E0B" if frust < 0.7 else "#EF4444"
        estr      = f" — <b>{edetail}</b>" if edetail not in ("", "null", "None") else ""
        st.markdown(f"""
        <div class="vision-insight">
          <div class="vision-insight-title">🧠 Image Analysis</div>
          {f'<div style="font-size:13px;color:#475569;margin-bottom:10px;">{vsummary}</div>' if vsummary else ''}
          <div style="display:flex;justify-content:space-between;font-size:12px;color:#7C3AED;margin-bottom:6px;">
            <span>User Frustration Level</span>
            <span style="font-weight:700;color:{fcol}">{fp}%</span>
          </div>
          <div class="frustration-track"><div class="frustration-marker" style="left:{fp}%"></div></div>
          <div style="font-size:12px;color:#6D28D9;margin-top:8px;">
            Error detected: <b>{etype.replace("_"," ").title()}</b>{estr}
          </div>
        </div>""", unsafe_allow_html=True)

    if reason:
        st.markdown(f'<div class="reasoning-box">💡 {reason}</div>', unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ── History ────────────────────────────────────────────────────────────────────
if not st.session_state.history:
    st.markdown("""
    <div class="empty-state">
      <div class="empty-icon">🎯</div>
      <div class="empty-title">Ready to classify</div>
      <div class="empty-body">
        Type a customer message, upload a voice recording,<br>
        or send a screenshot below to get started.
      </div>
    </div>""", unsafe_allow_html=True)
else:
    for msg in st.session_state.history:
        if msg["role"] == "user":
            preview = msg["content"][:200] + ("…" if len(msg["content"]) > 200 else "")
            st.markdown(
                f'<div style="margin:8px 0"><div class="bubble-user">👤 &nbsp;{preview}</div></div>',
                unsafe_allow_html=True,
            )
        else:
            render_result(msg["result"])

# ── Tabs ───────────────────────────────────────────────────────────────────────
st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
tab_text, tab_audio, tab_vision = st.tabs([
    "💬   Type a Message",
    "🎙   Voice Input",
    "🖼   Send an Image",
])

EXAMPLES = [
    "My invoice has a double charge from last month",
    "I can't log in — it says my account is locked",
    "I'd like to upgrade to the Business plan",
    "The app keeps crashing when I open reports",
    "I've been waiting 3 days, this is unacceptable",
]

# ── TEXT TAB ───────────────────────────────────────────────────────────────────
with tab_text:
    st.markdown('<div class="input-panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-label">Input Mode</div>', unsafe_allow_html=True)

    text_mode = st.radio(
        "mode", ["Single message", "Paste long text / email"],
        horizontal=True, label_visibility="collapsed", key="text_mode",
    )
    is_para = text_mode == "Paste long text / email"

    if not is_para:
        st.markdown('<div style="font-size:13px;color:#64748B;margin-bottom:8px;">Try an example:</div>',
                    unsafe_allow_html=True)
        cols = st.columns(len(EXAMPLES))
        for i, (col, ex) in enumerate(zip(cols, EXAMPLES)):
            with col:
                if st.button(ex[:28] + "…", key=f"ex_{i}", use_container_width=True):
                    st.session_state.example_text = ex

    with st.form("text_form", clear_on_submit=True):
        user_input = st.text_area(
            "message",
            value=st.session_state.get("example_text", ""),
            placeholder="Type a customer support message here…" if not is_para
                        else "Paste a full email or support ticket…",
            height=180 if is_para else 110,
            max_chars=16000,
            label_visibility="collapsed",
        )
        char_count = len(user_input)
        word_count = len(user_input.split()) if user_input.strip() else 0
        cc = "#EF4444" if char_count > 15000 else "#F59E0B" if char_count > 10000 else "#CBD5E1"
        c1, c2 = st.columns([3, 1])
        with c1:
            include_ctx = st.checkbox("Include conversation context", value=not is_para)
        with c2:
            st.markdown(
                f"<div style='text-align:right;font-size:11px;color:{cc};padding-top:6px;'>"
                f"{word_count} words · {char_count:,} chars</div>",
                unsafe_allow_html=True,
            )
        submitted = st.form_submit_button("Classify Intent →", use_container_width=True, type="primary")

    st.markdown("</div>", unsafe_allow_html=True)

    if submitted and user_input.strip():
        st.session_state.example_text = ""
        context = None
        if include_ctx:
            context = [{"role": m["role"], "content": m["content"][:400]}
                       for m in st.session_state.history if m["role"] == "user"][-6:]
        with st.spinner("Analysing message…"):
            response = api_client.classify_text(user_input.strip(), context, st.session_state.session_id)
        st.session_state.history.append({"role": "user", "content": user_input.strip()})
        st.session_state.history.append({"role": "bot", "content": "", "result": response})
        st.rerun()

# ── AUDIO TAB ──────────────────────────────────────────────────────────────────
with tab_audio:
    st.markdown('<div class="input-panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-label">Audio Source</div>', unsafe_allow_html=True)

    audio_mode = st.radio(
        "amode", ["🎙 Record from microphone", "📁 Upload an audio file"],
        horizontal=True, label_visibility="collapsed", key="audio_mode",
    )
    lang_hint = st.text_input("Language (optional)", placeholder="e.g.  en · es · fr · de",
                               max_chars=10, key="audio_lang",
                               help="Leave blank for automatic detection")

    audio_bytes_out, audio_mime_out, audio_label = None, "audio/wav", ""

    if audio_mode == "🎙 Record from microphone":
        st.markdown('<div class="mode-description">Click the microphone to start/stop recording.</div>',
                    unsafe_allow_html=True)
        try:
            from audio_recorder_streamlit import audio_recorder
            recorded = audio_recorder(
                text="", recording_color="#EF4444", neutral_color="#94A3B8",
                icon_name="microphone", icon_size="2x", pause_threshold=3.0, sample_rate=16000,
            )
            if recorded:
                st.session_state.last_audio_bytes = recorded
                st.session_state.last_audio_mime  = "audio/wav"
                st.audio(recorded, format="audio/wav")
                st.caption(f"✅ {len(recorded):,} bytes captured")
        except ImportError:
            st.info("Run `pip install audio-recorder-streamlit` to enable live recording.")
        if st.session_state.last_audio_bytes:
            audio_bytes_out = st.session_state.last_audio_bytes
            audio_mime_out  = st.session_state.last_audio_mime
            audio_label     = "🎙 Live recording"
    else:
        st.markdown('<div class="mode-description">Supports .wav · .mp3 · .ogg · .webm — up to 25 MB</div>',
                    unsafe_allow_html=True)
        af = st.file_uploader("audio", type=["wav","mp3","ogg","webm"],
                              label_visibility="collapsed", key="audio_up")
        if af:
            st.audio(af)
            audio_bytes_out = af.read()
            audio_mime_out  = af.type or "audio/wav"
            audio_label     = f"📁 {af.name}"

    btn_col, txt_col = st.columns([2, 1])
    with btn_col:
        go_classify = st.button("Transcribe & Classify →", type="primary",
                                use_container_width=True,
                                disabled=audio_bytes_out is None, key="audio_go")
    with txt_col:
        go_transcribe = st.button("Transcribe only", use_container_width=True,
                                  disabled=audio_bytes_out is None, key="audio_txt")

    if go_classify and audio_bytes_out:
        with st.spinner("Transcribing and classifying…"):
            response = api_client.classify_audio_file(
                audio_bytes_out, mime_type=audio_mime_out, session_id=st.session_state.session_id)
        st.session_state.history.append({"role": "user", "content": audio_label})
        st.session_state.history.append({"role": "bot", "content": "", "result": response})
        if audio_mode == "🎙 Record from microphone":
            st.session_state.last_audio_bytes = None
        render_result(response)

    if go_transcribe and audio_bytes_out:
        with st.spinner("Transcribing…"):
            res = api_client.transcribe_file(audio_bytes_out, mime_type=audio_mime_out,
                                             language=lang_hint.strip() or None,
                                             session_id=st.session_state.session_id)
        if res.get("success"):
            lang = res.get("language_detected", "—")
            dur  = res.get("duration_seconds")
            st.success(f"**Transcript** · language: {lang}{f' · {dur:.1f}s' if dur else ''}")
            st.markdown(
                f"<div style='background:#F0FDF4;border:1px solid #BBF7D0;border-radius:10px;"
                f"padding:14px 18px;font-size:14px;color:#166534;line-height:1.6;'>"
                f"{res.get('transcript','')}</div>", unsafe_allow_html=True)
        else:
            st.error(res.get("error", "Transcription failed."))

    st.markdown("</div>", unsafe_allow_html=True)

# ── VISION TAB ─────────────────────────────────────────────────────────────────
with tab_vision:
    st.markdown('<div class="input-panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-label">Image Source</div>', unsafe_allow_html=True)

    vision_mode = st.radio(
        "vmode", ["📷 Take a photo / screenshot", "📁 Upload an image file"],
        horizontal=True, label_visibility="collapsed", key="vis_mode",
    )
    caption = st.text_input("Add context (optional)",
                            placeholder="e.g.  Customer sent this error screenshot",
                            key="vis_cap")

    img_bytes_out, img_mime_out, img_label = None, "image/jpeg", ""

    if vision_mode == "📷 Take a photo / screenshot":
        st.markdown('<div class="mode-description">Allow camera access then click capture.</div>',
                    unsafe_allow_html=True)
        cam = st.camera_input("cam", label_visibility="collapsed", key="webcam")
        if cam:
            img_bytes_out = cam.getvalue()
            img_mime_out  = "image/jpeg"
            img_label     = "📷 Webcam snapshot"
            st.caption(f"✅ {len(img_bytes_out):,} bytes captured")
    else:
        st.markdown('<div class="mode-description">Supports .jpg · .png · .webp — up to 25 MB</div>',
                    unsafe_allow_html=True)
        imgf = st.file_uploader("img", type=["jpg","jpeg","png","webp"],
                                label_visibility="collapsed", key="img_up")
        if imgf:
            st.image(imgf, width=360, caption=imgf.name)
            img_bytes_out = imgf.read()
            img_mime_out  = imgf.type or "image/jpeg"
            img_label     = f"📁 {imgf.name}"

    go_vision = st.button("Analyse & Classify →", type="primary",
                          use_container_width=True,
                          disabled=img_bytes_out is None, key="vis_go")
    if go_vision and img_bytes_out:
        with st.spinner("Analysing with Llama 4 Scout Vision…"):
            response = api_client.classify_image_file(
                img_bytes_out, mime_type=img_mime_out,
                caption=caption.strip() or None, session_id=st.session_state.session_id)
        full_label = img_label + (f' — "{caption}"' if caption.strip() else "")
        st.session_state.history.append({"role": "user", "content": full_label})
        st.session_state.history.append({"role": "bot", "content": "", "result": response})
        render_result(response)

    st.markdown("</div>", unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
_, c1, c2 = st.columns([5, 1, 1])
with c1:
    if st.button("New Session", use_container_width=True):
        st.session_state.session_id      = str(uuid.uuid4())[:8]
        st.session_state.history         = []
        st.session_state.last_audio_bytes = None
        st.rerun()
with c2:
    if st.button("Clear History", use_container_width=True):
        st.session_state.history = []
        st.rerun()
