"""
BotTrain v3.1 - Visualizer
Light-theme Plotly charts and PDF report generation.
"""

import io
from datetime import datetime
from typing import Optional

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd


# ── Light palette (matches app.py / sidebar.py) ────────────────────────────────
P = {
    "blue":    "#0EA5E9",
    "indigo":  "#6366F1",
    "emerald": "#10B981",
    "amber":   "#F59E0B",
    "red":     "#EF4444",
    "violet":  "#8B5CF6",
    "sky":     "#38BDF8",
    "slate":   "#64748B",
    "bg":      "#FFFFFF",
    "surface": "#F8FAFC",
    "border":  "#E8ECF2",
    "text":    "#1A2332",
    "muted":   "#94A3B8",
}

INTENT_COLORS = {
    "billing":            "#3B82F6",
    "technical_support":  "#F97316",
    "account_management": "#0EA5E9",
    "sales_inquiry":      "#10B981",
    "complaint":          "#EF4444",
    "general_inquiry":    "#94A3B8",
    "escalation":         "#DC2626",
    "out_of_scope":       "#CBD5E1",
}
SENTIMENT_COLORS = {
    "positive":   "#10B981",
    "neutral":    "#94A3B8",
    "negative":   "#F59E0B",
    "frustrated": "#EF4444",
}
MODALITY_COLORS = {
    "text":   "#0EA5E9",
    "audio":  "#F97316",
    "vision": "#8B5CF6",
}

_LAYOUT = dict(
    paper_bgcolor=P["bg"],
    plot_bgcolor=P["bg"],
    font=dict(color=P["text"], family="DM Sans, sans-serif", size=12),
    margin=dict(l=12, r=12, t=36, b=12),
)


# ── Confidence Gauge ───────────────────────────────────────────────────────────

def confidence_gauge(confidence: float, title: str = "Confidence") -> go.Figure:
    pct   = round(confidence * 100, 1)
    color = P["emerald"] if confidence >= 0.75 else P["amber"] if confidence >= 0.5 else P["red"]
    fig   = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        number={"suffix": "%", "font": {"size": 26, "color": color, "family": "Outfit, sans-serif"}},
        title={"text": title, "font": {"size": 13, "color": P["muted"]}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": P["border"],
                     "tickfont": {"color": P["muted"], "size": 11}},
            "bar":  {"color": color, "thickness": 0.7},
            "bgcolor": P["surface"],
            "bordercolor": P["border"],
            "steps": [
                {"range": [0,  50],  "color": "#FEF2F2"},
                {"range": [50, 75],  "color": "#FFFBEB"},
                {"range": [75, 100], "color": "#F0FDF4"},
            ],
            "threshold": {
                "line": {"color": P["slate"], "width": 2},
                "thickness": 0.9,
                "value": 70,
            },
        },
    ))
    fig.update_layout(**_LAYOUT, height=180)
    return fig


# ── Intent Bar Chart ───────────────────────────────────────────────────────────

def intent_bar(intent_dist: dict) -> go.Figure:
    labels = [k.replace("_", " ").title() for k in intent_dist.keys()]
    values = list(intent_dist.values())
    colors = [INTENT_COLORS.get(k, P["slate"]) for k in intent_dist.keys()]

    fig = go.Figure(go.Bar(
        x=values, y=labels,
        orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=values, textposition="outside",
        textfont=dict(size=11, color=P["slate"]),
        hovertemplate="%{y}: %{x}<extra></extra>",
    ))
    fig.update_layout(
        **_LAYOUT,
        title=dict(text="Intent Breakdown", font=dict(size=13, color=P["slate"])),
        height=max(140, len(labels) * 30 + 60),
        xaxis=dict(showgrid=True, gridcolor=P["border"], zeroline=False,
                   tickfont=dict(color=P["muted"])),
        yaxis=dict(showgrid=False, tickfont=dict(color=P["text"], size=12)),
        bargap=0.35,
    )
    return fig


# ── Sentiment Donut ────────────────────────────────────────────────────────────

def sentiment_donut(sent_dist: dict) -> go.Figure:
    labels = [k.title() for k in sent_dist.keys()]
    values = list(sent_dist.values())
    colors = [SENTIMENT_COLORS.get(k, P["slate"]) for k in sent_dist.keys()]

    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        hole=0.6,
        marker=dict(colors=colors, line=dict(color="#FFFFFF", width=2)),
        textinfo="percent",
        textfont=dict(size=11),
        hovertemplate="%{label}: %{value} (%{percent})<extra></extra>",
    ))
    fig.update_layout(
        **_LAYOUT,
        title=dict(text="Sentiment", font=dict(size=13, color=P["slate"])),
        height=180,
        showlegend=True,
        legend=dict(
            orientation="v", font=dict(size=10, color=P["muted"]),
            itemsizing="constant", x=1.0,
        ),
    )
    return fig


# ── Modality Donut ─────────────────────────────────────────────────────────────

def modality_donut(mod_dist: dict) -> go.Figure:
    labels = [k.title() for k in mod_dist.keys()]
    values = list(mod_dist.values())
    colors = [MODALITY_COLORS.get(k, P["slate"]) for k in mod_dist.keys()]

    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        hole=0.6,
        marker=dict(colors=colors, line=dict(color="#FFFFFF", width=2)),
        textinfo="percent",
        textfont=dict(size=11),
        hovertemplate="%{label}: %{value}<extra></extra>",
    ))
    fig.update_layout(
        **_LAYOUT,
        title=dict(text="Modality", font=dict(size=13, color=P["slate"])),
        height=180,
        showlegend=True,
        legend=dict(
            orientation="v", font=dict(size=10, color=P["muted"]),
            itemsizing="constant", x=1.0,
        ),
    )
    return fig


# ── Confidence Timeline ────────────────────────────────────────────────────────

def confidence_timeline(entries: list) -> go.Figure:
    if not entries:
        return go.Figure()

    df = pd.DataFrame(entries)
    if "timestamp" not in df.columns or "confidence" not in df.columns:
        return go.Figure()

    df["ts"]   = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df["conf"] = pd.to_numeric(df["confidence"], errors="coerce")
    df = df.dropna(subset=["ts", "conf"]).sort_values("ts").tail(100)

    colors_line = df["conf"].apply(
        lambda c: P["emerald"] if c >= 0.75 else P["amber"] if c >= 0.5 else P["red"]
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["ts"], y=df["conf"] * 100,
        mode="lines+markers",
        line=dict(color=P["blue"], width=2),
        marker=dict(color=colors_line, size=6, line=dict(color="#FFFFFF", width=1)),
        fill="tozeroy",
        fillcolor="rgba(14,165,233,0.08)",
        hovertemplate="%{y:.1f}%<extra></extra>",
        name="Confidence",
    ))

    for lvl, col, label in [(70, P["amber"], "70% threshold"), (50, P["red"], "50%")]:
        fig.add_hline(
            y=lvl, line_dash="dot", line_color=col, line_width=1,
            annotation_text=label,
            annotation_position="bottom right",
            annotation_font=dict(size=10, color=col),
        )

    fig.update_layout(
        **_LAYOUT,
        title=dict(text="Confidence Over Time", font=dict(size=13, color=P["slate"])),
        height=200,
        xaxis=dict(showgrid=False, tickfont=dict(color=P["muted"], size=10)),
        yaxis=dict(range=[0, 105], showgrid=True, gridcolor=P["border"],
                   ticksuffix="%", tickfont=dict(color=P["muted"], size=10)),
        showlegend=False,
    )
    return fig


# ── Latency Timeline ───────────────────────────────────────────────────────────

def latency_timeline(entries: list) -> go.Figure:
    if not entries:
        return go.Figure()

    df = pd.DataFrame(entries)
    if "timestamp" not in df.columns or "latency_ms" not in df.columns:
        return go.Figure()

    df["ts"]  = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df["lat"] = pd.to_numeric(df["latency_ms"], errors="coerce")
    df = df.dropna(subset=["ts", "lat"]).sort_values("ts").tail(100)

    modalities = df.get("modality", pd.Series(["text"] * len(df)))
    traces = {}
    for mod in modalities.unique():
        sub = df[df["modality"] == mod] if "modality" in df.columns else df
        traces[mod] = sub

    fig = go.Figure()
    for mod, sub in traces.items():
        fig.add_trace(go.Scatter(
            x=sub["ts"], y=sub["lat"],
            mode="lines+markers",
            name=mod.title(),
            line=dict(color=MODALITY_COLORS.get(mod, P["slate"]), width=2),
            marker=dict(size=5),
            hovertemplate=f"{mod}: %{{y:.0f}}ms<extra></extra>",
        ))

    fig.update_layout(
        **_LAYOUT,
        title=dict(text="Response Latency", font=dict(size=13, color=P["slate"])),
        height=180,
        xaxis=dict(showgrid=False, tickfont=dict(color=P["muted"], size=10)),
        yaxis=dict(showgrid=True, gridcolor=P["border"],
                   ticksuffix="ms", tickfont=dict(color=P["muted"], size=10)),
        legend=dict(font=dict(size=10), orientation="h", y=-0.2),
    )
    return fig


# ── PDF Report ─────────────────────────────────────────────────────────────────

def export_pdf_report(metrics: dict, entries: list) -> bytes:
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
        )

        buf   = io.BytesIO()
        doc   = SimpleDocTemplate(buf, pagesize=A4,
                                  leftMargin=2*cm, rightMargin=2*cm,
                                  topMargin=2*cm, bottomMargin=2*cm)
        styles = getSampleStyleSheet()
        story  = []

        BLUE = colors.HexColor("#0EA5E9")
        DARK = colors.HexColor("#1A2332")
        GRAY = colors.HexColor("#64748B")
        LIGHT= colors.HexColor("#F8FAFC")

        title_style = ParagraphStyle("title", parent=styles["Title"],
                                      fontSize=22, textColor=DARK, spaceAfter=4)
        sub_style   = ParagraphStyle("sub", parent=styles["Normal"],
                                      fontSize=11, textColor=GRAY, spaceAfter=16)
        head_style  = ParagraphStyle("head", parent=styles["Heading2"],
                                      fontSize=13, textColor=DARK, spaceAfter=8, spaceBefore=16)
        body_style  = ParagraphStyle("body", parent=styles["Normal"],
                                      fontSize=10, textColor=GRAY, spaceAfter=4)

        story.append(Paragraph("BotTrain Audit Report", title_style))
        story.append(Paragraph(
            f"Generated {datetime.now().strftime('%d %B %Y, %H:%M UTC')}",
            sub_style,
        ))
        story.append(HRFlowable(width="100%", color=colors.HexColor("#E8ECF2"), thickness=1))

        # Summary table
        story.append(Paragraph("Summary", head_style))
        total = metrics.get("total_requests", 0)
        rows  = [
            ["Metric", "Value"],
            ["Total Requests", str(total)],
            ["Average Confidence", f"{metrics.get('avg_confidence',0)*100:.1f}%"],
            ["Average Latency", f"{metrics.get('avg_latency_ms',0):.0f} ms"],
            ["Escalation Rate", f"{metrics.get('escalation_rate',0)*100:.1f}%"],
            ["Low Confidence Rate", f"{metrics.get('low_confidence_rate',0)*100:.1f}%"],
            ["Inbox Pending Reviews", str(metrics.get('inbox_pending', 0))],
        ]
        t = Table(rows, colWidths=[8*cm, 8*cm])
        t.setStyle(TableStyle([
            ("BACKGROUND",  (0, 0), (-1, 0),  BLUE),
            ("TEXTCOLOR",   (0, 0), (-1, 0),  colors.white),
            ("FONTNAME",    (0, 0), (-1, 0),  "Helvetica-Bold"),
            ("FONTSIZE",    (0, 0), (-1, -1), 10),
            ("BACKGROUND",  (0, 1), (-1, -1), LIGHT),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [LIGHT, colors.white]),
            ("GRID",        (0, 0), (-1, -1), 0.5, colors.HexColor("#E8ECF2")),
            ("TOPPADDING",  (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING",(0, 0), (-1, -1), 6),
            ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ]))
        story.append(t)

        # Intent distribution
        intent_dist = metrics.get("intent_distribution", {})
        if intent_dist:
            story.append(Paragraph("Intent Distribution", head_style))
            rows = [["Intent", "Count", "% of Total"]]
            for intent, count in sorted(intent_dist.items(), key=lambda x: -x[1]):
                pct = (count / total * 100) if total else 0
                rows.append([intent.replace("_"," ").title(), str(count), f"{pct:.1f}%"])
            t2 = Table(rows, colWidths=[9*cm, 4*cm, 4*cm])
            t2.setStyle(TableStyle([
                ("BACKGROUND",  (0, 0), (-1, 0),  BLUE),
                ("TEXTCOLOR",   (0, 0), (-1, 0),  colors.white),
                ("FONTNAME",    (0, 0), (-1, 0),  "Helvetica-Bold"),
                ("FONTSIZE",    (0, 0), (-1, -1), 10),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [LIGHT, colors.white]),
                ("GRID",        (0, 0), (-1, -1), 0.5, colors.HexColor("#E8ECF2")),
                ("TOPPADDING",  (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING",(0,0), (-1, -1), 6),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
            ]))
            story.append(t2)

        # Recent log
        if entries:
            story.append(Paragraph("Recent Activity (last 30)", head_style))
            rows = [["Time", "Modality", "Intent", "Confidence", "Sentiment"]]
            for e in entries[-30:]:
                ts  = str(e.get("timestamp",""))[:16].replace("T"," ")
                mod = str(e.get("modality","")).title()
                intent = str(e.get("intent","")).replace("_"," ").title()
                conf   = f"{float(e.get('confidence',0))*100:.0f}%"
                sent   = str(e.get("sentiment","")).title()
                rows.append([ts, mod, intent, conf, sent])
            t3 = Table(rows, colWidths=[4.2*cm, 2.8*cm, 4.5*cm, 2.5*cm, 3*cm])
            t3.setStyle(TableStyle([
                ("BACKGROUND",  (0, 0), (-1, 0),  BLUE),
                ("TEXTCOLOR",   (0, 0), (-1, 0),  colors.white),
                ("FONTNAME",    (0, 0), (-1, 0),  "Helvetica-Bold"),
                ("FONTSIZE",    (0, 0), (-1, -1), 8),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [LIGHT, colors.white]),
                ("GRID",        (0, 0), (-1, -1), 0.4, colors.HexColor("#E8ECF2")),
                ("TOPPADDING",  (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING",(0,0), (-1, -1), 5),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ]))
            story.append(t3)

        doc.build(story)
        return buf.getvalue()

    except ImportError:
        # Fallback plain-text report
        lines = [
            "BotTrain Audit Report",
            f"Generated: {datetime.now().strftime('%d %B %Y %H:%M')}",
            "",
            f"Total Requests:      {metrics.get('total_requests', 0)}",
            f"Avg Confidence:      {metrics.get('avg_confidence', 0)*100:.1f}%",
            f"Avg Latency:         {metrics.get('avg_latency_ms', 0):.0f}ms",
            f"Escalation Rate:     {metrics.get('escalation_rate', 0)*100:.1f}%",
        ]
        return "\n".join(lines).encode()
