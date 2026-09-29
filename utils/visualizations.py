"""
utils/visualizations.py
-----------------------
Reusable, theme-consistent Plotly chart builders plus Streamlit KPI-card
helpers. Every figure shares the dark FinTech palette defined in config so the
whole app feels like one product.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from config.settings import COLS, PALETTE


# --------------------------------------------------------------------------- #
# Layout theming                                                              #
# --------------------------------------------------------------------------- #
def _apply_theme(fig: go.Figure, height: int = 420, title: str = "") -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        title=dict(text=title, font=dict(size=18, color=PALETTE.text)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=PALETTE.text, family="Inter, sans-serif"),
        height=height,
        margin=dict(l=40, r=20, t=50, b=40),
        colorway=PALETTE.sequence,
        hoverlabel=dict(bgcolor=PALETTE.panel_2, font_size=12),
        legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", y=1.02, x=0),
    )
    fig.update_xaxes(gridcolor=PALETTE.grid, zeroline=False)
    fig.update_yaxes(gridcolor=PALETTE.grid, zeroline=False)
    return fig


# --------------------------------------------------------------------------- #
# KPI cards                                                                    #
# --------------------------------------------------------------------------- #
def kpi_card(
    label: str, value: str, delta: Optional[str] = None,
    icon: str = "📊", accent: str = PALETTE.accent,
) -> str:
    """Return HTML for a single glassmorphism KPI card."""
    delta_html = f"<div class='kpi-delta'>{delta}</div>" if delta else ""
    # IMPORTANT: keep this on a single line with NO leading whitespace. Markdown
    # renders any line indented 4+ spaces as a code block, which would print the
    # HTML literally instead of rendering the card.
    return (
        f"<div class='kpi-card' style='--accent:{accent}'>"
        f"<div class='kpi-icon'>{icon}</div>"
        f"<div class='kpi-body'>"
        f"<div class='kpi-label'>{label}</div>"
        f"<div class='kpi-value'>{value}</div>"
        f"{delta_html}"
        f"</div></div>"
    )


def render_kpi_row(cards: Sequence[str]) -> None:
    """Render a responsive row of KPI cards (expects raw HTML strings)."""
    import streamlit as st

    # Strip each card defensively so no fragment can start indented (which would
    # otherwise be parsed as a markdown code block and shown as raw text).
    body = "".join(c.strip() for c in cards)
    html = f"<div class='kpi-row'>{body}</div>"
    st.markdown(html, unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Price / market charts                                                       #
# --------------------------------------------------------------------------- #
def candlestick_chart(df: pd.DataFrame, title: str = "Price Action") -> go.Figure:
    fig = go.Figure(
        go.Candlestick(
            x=df[COLS.date],
            open=df[COLS.stock_open], high=df[COLS.stock_high],
            low=df[COLS.stock_low], close=df[COLS.stock_close],
            increasing_line_color=PALETTE.up, decreasing_line_color=PALETTE.down,
            name="OHLC",
        )
    )
    fig.update_layout(xaxis_rangeslider_visible=False)
    return _apply_theme(fig, height=460, title=title)


def line_chart(
    df: pd.DataFrame, x: str, ys: List[str], title: str = ""
) -> go.Figure:
    fig = go.Figure()
    for y in ys:
        if y in df.columns:
            fig.add_trace(go.Scatter(x=df[x], y=df[y], mode="lines", name=y))
    return _apply_theme(fig, title=title)


def area_chart(df: pd.DataFrame, x: str, y: str, title: str = "") -> go.Figure:
    fig = go.Figure(
        go.Scatter(
            x=df[x], y=df[y], fill="tozeroy", mode="lines",
            line=dict(color=PALETTE.accent), name=y,
        )
    )
    return _apply_theme(fig, title=title)


def scatter_chart(
    df: pd.DataFrame, x: str, y: str, color: Optional[str] = None, title: str = ""
) -> go.Figure:
    fig = px.scatter(df, x=x, y=y, color=color,
                     color_discrete_sequence=PALETTE.sequence)
    return _apply_theme(fig, title=title)


def correlation_heatmap(df: pd.DataFrame, cols: List[str], title: str = "Correlation Matrix") -> go.Figure:
    valid = [c for c in cols if c in df.columns]
    corr = df[valid].corr()
    fig = go.Figure(
        go.Heatmap(
            z=corr.values, x=corr.columns, y=corr.columns,
            colorscale="Viridis", zmid=0,
            text=np.round(corr.values, 2), texttemplate="%{text}",
            textfont=dict(size=9),
        )
    )
    return _apply_theme(fig, height=520, title=title)


def heatmap(z, x, y, title: str = "", colorscale: str = "Viridis") -> go.Figure:
    fig = go.Figure(go.Heatmap(z=z, x=x, y=y, colorscale=colorscale))
    return _apply_theme(fig, title=title)


def radar_chart(categories: List[str], values: List[float], title: str = "") -> go.Figure:
    fig = go.Figure(
        go.Scatterpolar(
            r=values + values[:1], theta=categories + categories[:1],
            fill="toself", line=dict(color=PALETTE.accent_2),
        )
    )
    fig.update_layout(polar=dict(bgcolor="rgba(0,0,0,0)"))
    return _apply_theme(fig, title=title)


def treemap(df: pd.DataFrame, path: List[str], values: str, title: str = "") -> go.Figure:
    fig = px.treemap(df, path=path, values=values,
                     color_discrete_sequence=PALETTE.sequence)
    return _apply_theme(fig, title=title)


def sunburst(df: pd.DataFrame, path: List[str], values: str, title: str = "") -> go.Figure:
    fig = px.sunburst(df, path=path, values=values,
                      color_discrete_sequence=PALETTE.sequence)
    return _apply_theme(fig, title=title)


def gauge_chart(
    value: float, title: str = "", min_v: float = 0, max_v: float = 100,
    suffix: str = "", thresholds: Optional[Sequence[float]] = None,
) -> go.Figure:
    thresholds = thresholds or [max_v * 0.33, max_v * 0.66]
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            number=dict(suffix=suffix, font=dict(size=34)),
            gauge=dict(
                axis=dict(range=[min_v, max_v], tickcolor=PALETTE.muted),
                bar=dict(color=PALETTE.accent),
                bgcolor=PALETTE.panel,
                steps=[
                    {"range": [min_v, thresholds[0]], "color": PALETTE.up},
                    {"range": [thresholds[0], thresholds[1]], "color": PALETTE.warn},
                    {"range": [thresholds[1], max_v], "color": PALETTE.down},
                ],
            ),
        )
    )
    return _apply_theme(fig, height=300, title=title)


def bar_chart(
    df: pd.DataFrame, x: str, y: str, color: Optional[str] = None,
    orientation: str = "v", title: str = "",
) -> go.Figure:
    fig = px.bar(df, x=x, y=y, color=color, orientation=orientation,
                 color_continuous_scale="Viridis",
                 color_discrete_sequence=PALETTE.sequence)
    return _apply_theme(fig, title=title)


def cluster_scatter(
    df: pd.DataFrame, x: str, y: str, labels: Sequence[int], title: str = "Clusters"
) -> go.Figure:
    fig = px.scatter(
        df, x=x, y=y, color=[f"Cluster {l}" for l in labels],
        color_discrete_sequence=PALETTE.sequence,
    )
    return _apply_theme(fig, title=title)


def forecast_chart(
    hist_x, hist_y, fcst_x, fcst_y,
    lower: Optional[Sequence[float]] = None, upper: Optional[Sequence[float]] = None,
    title: str = "Forecast",
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist_x, y=hist_y, mode="lines",
                             name="History", line=dict(color=PALETTE.accent)))
    fig.add_trace(go.Scatter(x=fcst_x, y=fcst_y, mode="lines+markers",
                             name="Forecast", line=dict(color=PALETTE.warn, dash="dash")))
    if lower is not None and upper is not None:
        fig.add_trace(go.Scatter(
            x=list(fcst_x) + list(fcst_x)[::-1],
            y=list(upper) + list(lower)[::-1],
            fill="toself", fillcolor="rgba(255,176,32,0.15)",
            line=dict(color="rgba(0,0,0,0)"), name="Confidence", showlegend=True,
        ))
    return _apply_theme(fig, title=title)


def confusion_matrix_fig(cm: np.ndarray, labels: List[str], title: str = "Confusion Matrix") -> go.Figure:
    fig = go.Figure(
        go.Heatmap(
            z=cm, x=labels, y=labels, colorscale="Blues",
            text=cm, texttemplate="%{text}", textfont=dict(size=16),
        )
    )
    fig.update_yaxes(autorange="reversed")
    return _apply_theme(fig, height=360, title=title)


def roc_curve_fig(fpr, tpr, auc_score: float, title: str = "ROC Curve") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                             name=f"ROC (AUC={auc_score:.3f})",
                             line=dict(color=PALETTE.accent, width=3)))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                             name="Random", line=dict(color=PALETTE.muted, dash="dash")))
    return _apply_theme(fig, title=title)