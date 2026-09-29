"""
modules/news_sentiment_impact.py
---------------------------------
Model 3 — How does financial-news sentiment relate to returns?

Pipeline: Headline → FinBERT (or lexicon fallback) → Sentiment Score → analysis.

NOTE: All sentiment scoring, correlation maths, caching, data processing and the
export are UNCHANGED. `_daily_sentiment()` is byte-for-byte the original. Only the
presentation layer was redesigned into an investor-friendly "Investor Assistant"
(plain-language cards, quick answers, an impact meter, headline labels, a
historical-impact table and an AI summary). The original analytics charts are
preserved verbatim inside an "Advanced Analysis" expander.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from config.settings import COLS, PALETTE
from utils import visualizations as viz
from utils.helper_functions import (
    download_buttons,
    has_package,
    page_header,
    score_sentiment,
)
from utils.preprocessing import get_engineered_data


# --------------------------------------------------------------------------- #
# UNCHANGED data/calculation layer                                            #
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def _daily_sentiment(symbol: str, use_finbert: bool) -> pd.DataFrame:
    df = get_engineered_data()
    sdf = df[df[COLS.symbol] == symbol].sort_values(COLS.date).copy()
    scores, engine = score_sentiment(sdf[COLS.headlines].tolist(),
                                     use_finbert=use_finbert)
    sdf["Sentiment"] = scores
    sdf.attrs["engine"] = engine
    return sdf


# --------------------------------------------------------------------------- #
# Presentation-only helpers (translate existing metrics → plain language)     #
# These add NO new model logic — they only label values already computed.     #
# --------------------------------------------------------------------------- #
def _mood(avg_sent: float):
    """News environment from the existing average sentiment (±0.02 band)."""
    if avg_sent > 0.02:
        return "🟢", "Positive News Environment", "Recent headlines are mostly positive."
    if avg_sent < -0.02:
        return "🔴", "Negative News Environment", "Recent headlines are mostly negative."
    return "🟡", "Mixed / Neutral News Environment", "Recent headlines are broadly balanced."


def _class_label(score: float):
    """Per-headline label using the module's own ±0.05 classification band."""
    if score > 0.05:
        return "🟢", "Positive"
    if score < -0.05:
        return "🔴", "Negative"
    return "🟡", "Neutral"


def _confidence(pos: int, neg: int, total: int, avg_sent: float) -> str:
    dominance = abs(pos - neg) / max(total, 1)
    if dominance >= 0.25 or abs(avg_sent) >= 0.10:
        return "High"
    if dominance >= 0.10 or abs(avg_sent) >= 0.03:
        return "Moderate"
    return "Low"


def _impact_strength(corr: float) -> str:
    ac = abs(corr) if np.isfinite(corr) else 0.0
    if ac >= 0.30:
        return "Very Strong"
    if ac >= 0.20:
        return "Strong"
    if ac >= 0.10:
        return "Moderate"
    if ac >= 0.05:
        return "Weak"
    return "Very Weak"


def _attention(corr: float) -> str:
    ac = abs(corr) if np.isfinite(corr) else 0.0
    if ac >= 0.20:
        return "Yes"
    if ac >= 0.10:
        return "Sometimes"
    return "Probably not"


def _pct(x: float) -> str:
    return f"{x:+.2f}%" if np.isfinite(x) else "—"


# --------------------------------------------------------------------------- #
# Entry point                                                                 #
# --------------------------------------------------------------------------- #
def run(df: Optional[pd.DataFrame] = None, df_full: Optional[pd.DataFrame] = None) -> None:
    page_header("News Investor Assistant",
                "What the news is saying about this stock — in plain English", "🗞️")

    if df is None:
        df = get_engineered_data()
    if df.empty:
        st.warning("No data available.")
        return

    symbols = sorted(df[COLS.symbol].unique().tolist())
    c1, c2 = st.columns([2, 1])
    with c1:
        symbol = st.selectbox("Select stock", symbols, key="sent_symbol")
    with c2:
        use_finbert = st.toggle("Use FinBERT", value=False,
                                help="Slower but more accurate. Falls back to "
                                     "the lexicon engine if not installed.")

    if use_finbert and not has_package("transformers"):
        st.info("⚙️ `transformers` not installed — using the built-in lexicon "
                "engine. Install with `pip install transformers torch`.")

    with st.spinner("Scoring headlines…"):
        sdf = _daily_sentiment(symbol, use_finbert)
    st.caption(f"Sentiment engine: **{sdf.attrs.get('engine', 'Lexicon')}**")

    # ===================== UNCHANGED CALCULATIONS =========================
    avg_sent = float(sdf["Sentiment"].mean())
    pos = int((sdf["Sentiment"] > 0.05).sum())
    neg = int((sdf["Sentiment"] < -0.05).sum())
    neu = len(sdf) - pos - neg
    total = len(sdf)

    sdf["Next_Return"] = sdf[COLS.stock_close].pct_change().shift(-1)
    valid = sdf.dropna(subset=["Next_Return"])
    corr_today = corr_next = float("nan")
    if len(valid) > 10:
        corr_today = valid["Sentiment"].corr(valid["Daily_Return"])
        corr_next = valid["Sentiment"].corr(valid["Next_Return"])

    # Average next-day return grouped by the existing sentiment classification.
    def _avg_next(mask) -> float:
        v = valid.loc[mask, "Next_Return"]
        return float(v.mean() * 100) if len(v) else float("nan")
    avg_next_pos = _avg_next(valid["Sentiment"] > 0.05)
    avg_next_neu = _avg_next(valid["Sentiment"].between(-0.05, 0.05))
    avg_next_neg = _avg_next(valid["Sentiment"] < -0.05)
    # ======================================================================

    # Pick the most relevant correlation for plain-language scoring.
    corr_main = corr_next if np.isfinite(corr_next) else corr_today

    # ===================================================== SECTION 1 ======
    # News Health Card — the headline summary, no raw numbers up front.
    emoji, mood_title, mood_sub = _mood(avg_sent)
    conf = _confidence(pos, neg, total, avg_sent)
    pos_pct = pos / total * 100 if total else 0
    neu_pct = neu / total * 100 if total else 0
    neg_pct = neg / total * 100 if total else 0
    st.markdown(
        f"<div class='glass'>"
        f"<h2 style='margin:0.1rem 0'>{emoji} {mood_title}</h2>"
        f"<p style='margin:0.25rem 0;font-size:1.05rem'>{mood_sub}</p>"
        f"<p style='margin:0.2rem 0'>Reading confidence: <b>{conf}</b></p>"
        f"<p style='margin:0.2rem 0;opacity:0.9'>"
        f"🟢 Positive {pos_pct:.0f}% &nbsp;·&nbsp; "
        f"🟡 Neutral {neu_pct:.0f}% &nbsp;·&nbsp; "
        f"🔴 Negative {neg_pct:.0f}%</p>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # ===================================================== SECTION 2 ======
    # Investor Quick Answers — four plain-language Q&A cards.
    st.markdown("### ❓ Quick Answers")
    today_sent = float(sdf["Sentiment"].iloc[-1])
    _, today_label = _class_label(today_sent)
    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Is the latest news good or bad?", today_label)
    q2.metric("Does news usually move this stock?", _impact_strength(corr_main))
    q3.metric("Avg next-day return after positive news", _pct(avg_next_pos))
    q4.metric("Should I pay attention to news?", _attention(corr_main))

    # ===================================================== SECTION 3 ======
    # News Impact Meter — correlation translated into a strength label + bar.
    st.markdown("### 📡 News Impact Meter")
    strength = _impact_strength(corr_main)
    ac = abs(corr_main) if np.isfinite(corr_main) else 0.0
    st.markdown(f"**News Impact Strength: {strength}**")
    st.progress(min(ac / 0.30, 1.0))   # scale: 0.30+ correlation = full bar
    st.caption("How closely this stock's moves have historically tracked its news tone. "
               "Most stocks land in the Weak–Moderate range — that's normal.")
    with st.expander("Show exact correlation values (advanced)"):
        cc1, cc2 = st.columns(2)
        cc1.metric("Sentiment ↔ Same-Day Return",
                   f"{corr_today:+.3f}" if np.isfinite(corr_today) else "—")
        cc2.metric("Sentiment ↔ Next-Day Return",
                   f"{corr_next:+.3f}" if np.isfinite(corr_next) else "—")

    # ===================================================== SECTION 4 ======
    # Recent Headlines — most recent first, each labelled by its sentiment.
    st.markdown("### 📰 Recent Headlines")
    recent = sdf.tail(8).iloc[::-1]
    if COLS.headlines in recent.columns:
        for _, row in recent.iterrows():
            dot, _lbl = _class_label(float(row["Sentiment"]))
            date_str = pd.to_datetime(row[COLS.date]).date()
            text = str(row[COLS.headlines]).replace("\n", " ").strip()
            text = (text[:160] + "…") if len(text) > 160 else (text or "—")
            st.markdown(f"{dot} **{date_str}** — {text}")
    else:
        st.info("No headline text available for this stock.")

    # ===================================================== SECTION 5 ======
    # Historical News Impact — investor-friendly outcome table.
    st.markdown("### 📅 What Usually Happens Next")
    impact_tbl = pd.DataFrame({
        "Sentiment Type": ["🟢 Positive News", "🟡 Neutral News", "🔴 Negative News"],
        "Avg Next-Day Return": [_pct(avg_next_pos), _pct(avg_next_neu), _pct(avg_next_neg)],
    })
    st.dataframe(impact_tbl, use_container_width=True, hide_index=True)
    best = max([("positive", avg_next_pos), ("negative", avg_next_neg)],
               key=lambda t: (t[1] if np.isfinite(t[1]) else -1e9))
    st.caption(
        f"On average, days following **{best[0]}** headlines saw the stronger next-day "
        f"move for {symbol}. These are historical averages across all sessions, not a "
        f"prediction — individual outcomes varied widely.")

    # ===================================================== SECTION 6 ======
    # Advanced Analysis — the ORIGINAL charts, preserved verbatim, tucked away.
    with st.expander("Advanced Analysis (charts & statistics)"):
        # ---- Distribution + split (unchanged) ----
        g1, g2 = st.columns(2)
        with g1:
            fig = px.histogram(sdf, x="Sentiment", nbins=40,
                               color_discrete_sequence=[PALETTE.accent])
            fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)", height=400,
                              title="Sentiment Distribution")
            st.plotly_chart(fig, use_container_width=True)
        with g2:
            split = pd.DataFrame({
                "Class": ["Positive", "Neutral", "Negative"],
                "Count": [pos, neu, neg],
            })
            pie = px.pie(split, names="Class", values="Count", hole=0.55,
                         color="Class",
                         color_discrete_map={"Positive": PALETTE.up,
                                             "Neutral": PALETTE.warn,
                                             "Negative": PALETTE.down})
            pie.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                              height=400, title="Positive vs Negative")
            st.plotly_chart(pie, use_container_width=True)

        # ---- Correlation charts (unchanged) ----
        if len(valid) > 10:
            st.markdown("#### 🔗 Correlation Analysis")
            cc1, cc2 = st.columns(2)
            cc1.metric("Sentiment ↔ Same-Day Return", f"{corr_today:+.3f}")
            cc2.metric("Sentiment ↔ Next-Day Return", f"{corr_next:+.3f}")
            st.plotly_chart(
                viz.scatter_chart(valid, "Sentiment", "Daily_Return",
                                  title="Sentiment vs Same-Day Return"),
                use_container_width=True,
            )
            roll = sdf.set_index(COLS.date)[["Sentiment", COLS.stock_close]].copy()
            roll["Sentiment_MA"] = roll["Sentiment"].rolling(21, min_periods=1).mean()
            st.plotly_chart(
                viz.line_chart(roll.reset_index(), COLS.date,
                               ["Sentiment_MA"], "21-Day Rolling Sentiment"),
                use_container_width=True,
            )

    # ===================================================== SECTION 7 ======
    # AI Investor Summary — narrative built from the existing metrics only.
    st.markdown("### 🤖 AI Investor Summary")
    tone = "positive" if avg_sent > 0.02 else ("negative" if avg_sent < -0.02 else "broadly neutral")
    rel = _impact_strength(corr_main).lower()
    gain_txt = (f"produced an average next-day move of {_pct(avg_next_pos)}"
                if np.isfinite(avg_next_pos) else "showed mixed next-day outcomes")
    st.markdown(
        f"<div class='glass'>"
        f"<p>For <b>{symbol}</b>, recent news sentiment is <b>{tone}</b>. "
        f"Historically, news tone has shown a <b>{rel}</b> relationship with this "
        f"stock's future returns. Days following positive headlines {gain_txt}. "
        f"News should be considered as one factor among many when evaluating this "
        f"stock — not a standalone signal.</p>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # ---- Export (UNCHANGED) ----------------------------------------------
    out = sdf[[COLS.date, COLS.stock_close, "Daily_Return", "Sentiment"]].copy()
    download_buttons(out, key="sentiment", label=f"{symbol}_sentiment")