"""
modules/ai_investment_thesis_generator.py
------------------------------------------
AI Investment Thesis Generator.

Answers, with data-driven evidence rather than a raw ML score:
    "Why should an investor BUY, HOLD, or AVOID this stock?"

The module reads the already-engineered dataframe (technical indicators such as
RSI, MACD, moving averages, volatility, returns and momentum are produced by the
feature-engineering step before any module runs), derives a set of transparent
sub-scores, and assembles them into a clear, heuristic-based analytical report.

Architecture: identical to the other modules — a single `run(df, df_full)` entry
point, project helpers (`page_header`, `download_buttons`, `score_sentiment` is
NOT used here because Section 5 requires an in-module keyword sentiment), the
shared `COLS`/`PALETTE` config and the `viz` theme. Everything is defensive: the
page degrades gracefully and never raises on missing columns, NaNs, or tiny
samples.

Libraries: pandas, numpy, plotly, streamlit (sklearn not required).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config.settings import COLS, PALETTE
from utils import visualizations as viz
from utils.helper_functions import download_buttons, page_header
from utils.preprocessing import get_engineered_data

# --------------------------------------------------------------------------- #
# Lightweight, dependency-free keyword sentiment (Section 5 requirement).      #
# These finance-oriented word lists are intentionally small and transparent —  #
# no paid APIs, no model downloads.                                            #
# --------------------------------------------------------------------------- #
POSITIVE_WORDS = {
    "surge", "soar", "jump", "gain", "rally", "rise", "record", "profit", "beat",
    "growth", "strong", "upgrade", "bullish", "outperform", "expand", "boost",
    "win", "approval", "robust", "optimistic", "buy", "high", "rebound", "dividend",
    "raise", "raises", "raised", "positive", "recovery", "demand", "momentum",
}
NEGATIVE_WORDS = {
    "fall", "drop", "plunge", "slump", "loss", "losses", "decline", "weak", "miss",
    "downgrade", "bearish", "underperform", "cut", "cuts", "fraud", "probe", "lawsuit",
    "default", "debt", "warning", "concern", "slowdown", "sell", "low", "crash",
    "negative", "risk", "fear", "scandal", "layoff", "recession", "selloff",
}


# --------------------------------------------------------------------------- #
# Small numeric helpers (all NaN-safe)                                        #
# --------------------------------------------------------------------------- #
def _last(series: pd.Series, default: float = np.nan) -> float:
    """Last finite value of a series, or a default."""
    s = pd.to_numeric(series, errors="coerce").dropna()
    return float(s.iloc[-1]) if len(s) else default


def _clip(x: float, lo: float = 0.0, hi: float = 100.0) -> float:
    if not np.isfinite(x):
        return (lo + hi) / 2
    return float(max(lo, min(hi, x)))


def _keyword_sentiment(text: str) -> float:
    """Per-headline sentiment in [-1, 1] from positive/negative word counts."""
    if not isinstance(text, str) or not text:
        return 0.0
    tokens = "".join(c.lower() if c.isalnum() else " " for c in text).split()
    pos = sum(t in POSITIVE_WORDS for t in tokens)
    neg = sum(t in NEGATIVE_WORDS for t in tokens)
    return (pos - neg) / (pos + neg) if (pos + neg) else 0.0


# --------------------------------------------------------------------------- #
# Sub-score builders (each returns a transparent 0-100 score)                 #
# --------------------------------------------------------------------------- #
def _trend_score(s: pd.DataFrame, close: pd.Series) -> Tuple[float, Dict[str, float]]:
    """Trend strength from price position relative to moving averages."""
    last = _last(close)
    sma20 = _last(s["SMA_20"]) if "SMA_20" in s else _last(close.rolling(20).mean())
    sma50 = _last(s["SMA_50"]) if "SMA_50" in s else _last(close.rolling(50).mean())
    sma200 = _last(close.rolling(200, min_periods=20).mean())
    checks = [last > sma20, last > sma50, last > sma200, sma20 > sma50]
    checks = [bool(c) for c in checks if np.isfinite(last)]
    score = (sum(checks) / len(checks) * 100) if checks else 50.0
    return _clip(score), {"sma20": sma20, "sma50": sma50, "sma200": sma200, "last": last}


def _momentum_score(s: pd.DataFrame, close: pd.Series) -> float:
    """Momentum from the 20-day return tilted by RSI distance from 50."""
    ret20 = (close.iloc[-1] / close.iloc[-21] - 1.0) if len(close) > 21 else 0.0
    rsi = _last(s["RSI"]) if "RSI" in s else 50.0
    score = 50.0 + _clip(ret20 * 400, -40, 40) + (rsi - 50.0) * 0.3
    return _clip(score)


def _volume_score(s: pd.DataFrame) -> float:
    """Recent (20d) volume participation versus the trailing (120d) baseline."""
    if COLS.stock_volume not in s:
        return 50.0
    vol = pd.to_numeric(s[COLS.stock_volume], errors="coerce").dropna()
    if len(vol) < 25:
        return 50.0
    recent, base = vol.tail(20).mean(), vol.tail(120).mean()
    ratio = recent / base if base else 1.0
    return _clip(50.0 + (ratio - 1.0) * 100.0)


def _volatility_score(s: pd.DataFrame) -> Tuple[float, float]:
    """Lower annualised volatility → higher score. Returns (score, ann_vol)."""
    rets = pd.to_numeric(s.get("Daily_Return", pd.Series(dtype=float)),
                         errors="coerce").dropna()
    ann_vol = float(rets.std() * np.sqrt(252)) if len(rets) > 5 else np.nan
    if not np.isfinite(ann_vol):
        return 50.0, np.nan
    return _clip(100.0 - ann_vol * 120.0, 10, 95), ann_vol


def _sentiment_series(s: pd.DataFrame) -> pd.Series:
    """Per-row keyword sentiment for the symbol's headlines."""
    if COLS.headlines not in s:
        return pd.Series(dtype=float)
    return s[COLS.headlines].fillna("").astype(str).map(_keyword_sentiment)


# --------------------------------------------------------------------------- #
# Entry point                                                                 #
# --------------------------------------------------------------------------- #
def run(df: Optional[pd.DataFrame] = None, df_full: Optional[pd.DataFrame] = None) -> None:
    page_header("Equity Thesis Studio",
                "Explore an evidence-based analytical summary built from trend strength, momentum, risk, sentiment, and market factors.")

    # Prefer the full engineered frame so the thesis uses the stock's whole
    # history (the filtered frame may be trimmed by global date/volume filters).
    source = df_full if df_full is not None and not df_full.empty else df
    if source is None or source.empty:
        source = get_engineered_data()
    if source is None or source.empty:
        st.warning("No data available to build a thesis.")
        return

    symbols = sorted(source[COLS.symbol].unique().tolist())
    symbol = st.selectbox("Select stock", symbols, key="thesis_symbol")

    s = source[source[COLS.symbol] == symbol].sort_values(COLS.date).reset_index(drop=True)
    close = pd.to_numeric(s[COLS.stock_close], errors="coerce")
    if len(s) < 60 or close.dropna().empty:
        st.warning("Not enough history for this stock to build a reliable thesis.")
        return

    company = str(s[COLS.company].iloc[-1]) if COLS.company in s else symbol
    industry = str(s[COLS.industry].iloc[-1]) if COLS.industry in s else "—"
    last_close = _last(close)

    # ---- Compute all sub-scores once -------------------------------------
    trend, ma = _trend_score(s, close)
    momentum = _momentum_score(s, close)
    volume = _volume_score(s)
    volatility, ann_vol = _volatility_score(s)
    sent_series = _sentiment_series(s)
    sent_mean = float(sent_series.mean()) if len(sent_series) else 0.0
    sentiment = _clip(50.0 + sent_mean * 50.0)

    # Composite Stock Health Score (weighted blend of the five pillars).
    weights = {"trend": 0.30, "momentum": 0.25, "volume": 0.15,
               "volatility": 0.15, "sentiment": 0.15}
    health = (trend * weights["trend"] + momentum * weights["momentum"]
              + volume * weights["volume"] + volatility * weights["volatility"]
              + sentiment * weights["sentiment"])
    health = _clip(health)

    # Investment stance from the composite score.
    if health >= 65:
        stance, stance_badge = "Bullish", "badge-buy"
    elif health >= 45:
        stance, stance_badge = "Neutral", "badge-hold"
    else:
        stance, stance_badge = "Bearish", "badge-sell"

    # Shared derived values used across sections.
    ret20 = (close.iloc[-1] / close.iloc[-21] - 1.0) * 100 if len(close) > 21 else 0.0
    rsi_now = _last(s["RSI"]) if "RSI" in s else 50.0

    gold = pd.to_numeric(s.get(COLS.gold_close, pd.Series(dtype=float)), errors="coerce")
    gold_corr = float(close.corr(gold)) if gold.notna().sum() > 30 else np.nan

    # ===================================================== SECTION 1 ======
    # Stock Overview (header card)
    with st.container(border=True):
        st.markdown(
            f"<h2 style='margin:0.1rem 0'>{company} "
            f"<span style='font-size:1rem;opacity:0.7'>({symbol} · {industry})</span></h2>"
            f"<p style='margin:0.3rem 0;font-size:1.05rem'>Latest Close "
            f"<b>₹{last_close:,.2f}</b> &nbsp;·&nbsp; Current Stance: "
            f"<span class='badge {stance_badge}'>{stance}</span></p>"
            f"<p style='opacity:0.7;font-size:0.85rem;margin:0.2rem 0 0 0'>"
            f"Stance is derived directly from the heuristic Health Score below — "
            f"it is a descriptive label, not a trade recommendation.</p>",
            unsafe_allow_html=True,
        )

    st.write("")

    # ===================================================== SECTION 2 ======
    # Stock Health Score (0-100) with a gauge.
    with st.container(border=True):
        st.markdown("Stock Health Score")
        st.caption("Composite heuristic score based on trend, momentum, volume, "
                    "volatility, and sentiment.")
        band = "Strong" if health >= 70 else ("Moderate" if health >= 40 else "Weak")
        gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=health,
            number=dict(suffix=" / 100", font=dict(size=34)),
            gauge=dict(
                axis=dict(range=[0, 100], tickcolor=PALETTE.muted),
                bar=dict(color=PALETTE.accent),
                steps=[
                    {"range": [0, 40], "color": PALETTE.down},     # Weak
                    {"range": [40, 70], "color": PALETTE.warn},    # Moderate
                    {"range": [70, 100], "color": PALETTE.up},     # Strong
                ],
            ),
        ))
        gauge.update_layout(template="plotly_dark", height=280,
                            margin=dict(t=50, b=10, l=20, r=20),
                            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                            title=f"Overall Health: {band}")
        st.plotly_chart(gauge, use_container_width=True)

    st.write("")

    # ===================================================== SECTION 3 ======
    # Sub-Score Breakdown — bar chart, no new calculations.
    with st.container(border=True):
        st.markdown("Sub-Score Breakdown")
        st.caption("Each pillar is scored 0–100. Higher is more favourable for "
                    "that specific dimension.")
        sub_scores = pd.DataFrame({
            "Pillar": ["Trend", "Momentum", "Volume", "Stability (low volatility)", "Sentiment"],
            "Score": [trend, momentum, volume, volatility, sentiment],
        })
        bar = go.Figure(go.Bar(
            x=sub_scores["Score"], y=sub_scores["Pillar"], orientation="h",
            marker_color=PALETTE.accent, text=sub_scores["Score"].round(0),
            textposition="outside",
        ))
        bar.update_layout(template="plotly_dark", height=300,
                          margin=dict(t=10, b=10, l=10, r=30),
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          xaxis=dict(range=[0, 100], title="Score (0-100)"),
                          yaxis=dict(title=""))
        st.plotly_chart(bar, use_container_width=True)

    st.write("")

    # ===================================================== SECTION 4 ======
    # Evidence Engine — grouped, automatically generated ✓ / ✗ bullets.
    with st.container(border=True):
        st.markdown("Evidence Engine")
        st.caption("Automatically generated checks based on the indicator values above.")

        trend_signals: List[Tuple[bool, str]] = []
        if np.isfinite(ma["sma50"]):
            trend_signals.append((last_close > ma["sma50"],
                             f"Trading {'above' if last_close > ma['sma50'] else 'below'} the 50-day moving average"))
        if np.isfinite(ma["sma200"]):
            trend_signals.append((last_close > ma["sma200"],
                             f"Trading {'above' if last_close > ma['sma200'] else 'below'} the 200-day moving average"))
        trend_signals.append((ret20 >= 0, f"Recent 20-day momentum is {'positive' if ret20 >= 0 else 'negative'} ({ret20:+.1f}%)"))

        risk_signals: List[Tuple[bool, str]] = []
        risk_signals.append((volume >= 50, "Healthy volume participation" if volume >= 50 else "Soft volume participation"))
        risk_signals.append((volatility >= 50, "Volatility is contained" if volatility >= 50 else "Volatility is elevated"))
        if np.isfinite(rsi_now):
            if rsi_now > 70:
                risk_signals.append((False, f"RSI is overbought ({rsi_now:.0f})"))
            elif rsi_now < 30:
                risk_signals.append((True, f"RSI is oversold ({rsi_now:.0f}) — potential value zone"))

        sentiment_signals: List[Tuple[bool, str]] = []
        sentiment_signals.append((sent_mean >= 0, f"News sentiment is {'positive' if sent_mean >= 0 else 'negative'} ({sent_mean:+.2f})"))

        ec1, ec2, ec3 = st.columns(3)
        for col, title, items in (
            (ec1, "Trend Signals", trend_signals),
            (ec2, "Risk Signals", risk_signals),
            (ec3, "Sentiment Signals", sentiment_signals),
        ):
            with col:
                st.markdown(f"**{title}**")
                if items:
                    for good, text in items:
                        mark = "✅" if good else "❌"
                        st.markdown(f"{mark} {text}")
                else:
                    st.markdown("—")

    st.write("")

    # ===================================================== SECTION 5 ======
    # Risk Panel (simplified dashboard card)
    with st.container(border=True):
        st.markdown("Risk Panel")
        st.caption("Descriptive risk readings derived from existing volatility, "
                    "drawdown, and trend indicators — not a forecast.")

        # Volatility risk
        if np.isfinite(ann_vol):
            vol_risk = "Low" if ann_vol < 0.25 else ("Medium" if ann_vol < 0.45 else "High")
        else:
            vol_risk = "Medium"
        # Drawdown risk (worst peak-to-trough over the series)
        dd = float((close / close.cummax() - 1.0).min()) if close.notna().any() else np.nan
        if np.isfinite(dd):
            draw_risk = "Low" if dd > -0.20 else ("Medium" if dd > -0.40 else "High")
        else:
            draw_risk = "Medium"
        # Trend risk (price below key averages / weak momentum)
        below = sum([np.isfinite(ma["sma50"]) and last_close < ma["sma50"],
                     np.isfinite(ma["sma200"]) and last_close < ma["sma200"],
                     ret20 < 0])
        trend_risk = "Low" if below == 0 else ("Medium" if below <= 1 else "High")
        severity = {"Low": 0, "Medium": 1, "High": 2}
        overall_risk = max([vol_risk, draw_risk, trend_risk], key=lambda r: severity[r])

        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Volatility Risk", vol_risk,
                  f"{ann_vol*100:.0f}% ann." if np.isfinite(ann_vol) else "—")
        r2.metric("Drawdown Risk", draw_risk,
                  f"{dd*100:.0f}% max" if np.isfinite(dd) else "—")
        r3.metric("Trend Risk", trend_risk)
        r4.metric("Overall Risk", overall_risk)

    st.write("")

    # ===================================================== SECTION 6 ======
    # AI Investment Thesis — transparency-first narrative.
    if health >= 75:
        assessment = "STRONGLY BULLISH"
    elif health >= 60:
        assessment = "MODERATELY BULLISH"
    elif health >= 45:
        assessment = "NEUTRAL"
    elif health >= 35:
        assessment = "MODERATELY BEARISH"
    else:
        assessment = "BEARISH"

    # Build the paragraphs conditionally from the computed metrics.
    trend_clause = ("trading above its medium-term trend indicators"
                    if last_close > ma.get("sma50", last_close)
                    else "trading below its medium-term trend indicators")
    mom_clause = ("positive momentum" if ret20 >= 0 else "softening momentum")
    vol_clause = ("healthy" if volume >= 50 else "subdued")
    sent_clause = ("broadly positive" if sent_mean > 0.02 else
                   "broadly negative" if sent_mean < -0.02 else "broadly neutral")
    if np.isfinite(gold_corr) and abs(gold_corr) >= 0.3:
        gold_clause = (f"a {'positive' if gold_corr > 0 else 'negative'} "
                       f"correlation with gold ({gold_corr:+.2f}), so commodity "
                       f"price movements have historically aligned with it")
    else:
        gold_clause = ("only weak correlation with gold prices, so commodity "
                       "moves are unlikely to explain its behaviour")
    risk_clause = {"Low": "low", "Medium": "moderate", "High": "elevated"}[overall_risk]

    with st.container(border=True):
        st.markdown("Heuristic Analytical Summary")
        st.markdown(
            f"<p>{company} currently has a Stock Health Score of "
            f"<b>{health:.0f}/100</b> ({band.lower()}), which places its overall "
            f"reading as <b>{assessment.replace('_', ' ').title()}</b> on the scale "
            f"used by this tool. It is {trend_clause}, with {mom_clause} "
            f"({ret20:+.1f}% over the last 20 days).</p>"
            f"<p>Volume participation has been {vol_clause}, and recent news "
            f"sentiment has been {sent_clause} ({sent_mean:+.2f}). The stock shows "
            f"{gold_clause}. Overall risk is assessed as <b>{risk_clause}</b>, "
            f"reflecting a worst historical drawdown of {dd*100:.0f}% and "
            f"{trend_risk.lower()} trend risk.</p>"
            f"<p style='opacity:0.85;font-size:0.9rem;margin-top:0.6rem'>"
            f"<b>Disclaimer:</b> This output is generated from rule-based, "
            f"heuristic indicators applied to historical data. It is an "
            f"educational summary, not financial advice, and is not a "
            f"recommendation to buy, hold, or sell any security.</p>",
            unsafe_allow_html=True,
        )

    st.write("")

    # ===================================================== SECTION 7 ======
    # What Would Change This View — transparent conditional logic.
    with st.container(border=True):
        st.markdown("What Would Change This View")
        st.caption("These are conditional descriptions of how the underlying "
                    "indicators could shift — not predictions.")

        triggers: List[str] = []
        if np.isfinite(ma["sma50"]):
            if last_close > ma["sma50"]:
                triggers.append("If price falls below the 50-day moving average, "
                                "the Trend score would decrease.")
            else:
                triggers.append("If price rises above the 50-day moving average, "
                                 "the Trend score would increase.")
        if np.isfinite(rsi_now):
            if rsi_now > 70:
                triggers.append(f"RSI is currently elevated ({rsi_now:.0f}); a move "
                                 "back below 70 would change the RSI-based signal "
                                 "in the Evidence Engine.")
            elif rsi_now < 30:
                triggers.append(f"RSI is currently low ({rsi_now:.0f}); a move back "
                                 "above 30 would change the RSI-based signal in the "
                                 "Evidence Engine.")
            else:
                triggers.append("If RSI crosses above 70 or below 30, the "
                                 "Evidence Engine would flag an overbought or "
                                 "oversold condition.")
        triggers.append("A sustained shift in the 20-day average trading volume "
                         "relative to the 120-day average would change the "
                         "Volume score.")
        triggers.append("A change in the average tone of recent headlines "
                         "(more positive or negative keywords) would shift the "
                         "Sentiment score and, with it, the overall Health Score.")

        for t in triggers:
            st.markdown(f"- {t}")

    st.write("")

    # ===================================================== SECTION 8 ======
    # Gold Sensitivity — kept as supporting context.
    with st.container(border=True):
        st.markdown("Gold Sensitivity")
        if np.isfinite(gold_corr):
            a = abs(gold_corr)
            if a >= 0.6:
                label = "Strong Positive" if gold_corr > 0 else "Strong Negative"
            elif a >= 0.3:
                label = "Moderate Positive" if gold_corr > 0 else "Moderate Negative"
            else:
                label = "Weak"
            gc1, gc2 = st.columns([1, 2])
            gc1.metric("Stock ↔ Gold Correlation", f"{gold_corr:+.2f}", label)
            gc2.info(
                f"A **{label.lower()}** correlation means gold-market moves have "
                f"{'historically tracked' if a >= 0.3 else 'had little historical link to'} "
                f"{symbol}'s price. "
                + ("Gold price swings could be relevant context for this stock."
                   if a >= 0.3 else
                   "Gold movements are unlikely to explain this stock's behaviour."))
        else:
            st.info("Not enough overlapping gold data to assess sensitivity.")

    st.write("")

    # ===================================================== SECTION 9 ======
    # News Intelligence — keyword sentiment summary.
    with st.container(border=True):
        st.markdown("News Intelligence")
        st.caption("Keyword-based sentiment over the stock's headline history "
                    "(no external APIs).")
        if len(sent_series):
            pos_days = int((sent_series > 0.05).sum())
            neg_days = int((sent_series < -0.05).sum())
            neu_days = int(len(sent_series) - pos_days - neg_days)
            tone = "positive" if sent_mean > 0.02 else ("negative" if sent_mean < -0.02 else "broadly neutral")
            n1, n2, n3, n4 = st.columns(4)
            n1.metric("Sentiment Score", f"{sent_mean:+.2f}")
            n2.metric("Positive Days", f"{pos_days:,}")
            n3.metric("Neutral Days", f"{neu_days:,}")
            n4.metric("Negative Days", f"{neg_days:,}")
            st.caption(f"Headline tone for {symbol} has been **{tone}** overall.")

            hist = go.Figure(go.Histogram(x=sent_series.values, nbinsx=30,
                                          marker_color=PALETTE.accent))
            hist.update_layout(template="plotly_dark", height=320,
                               margin=dict(t=30, b=10, l=10, r=10),
                               paper_bgcolor="rgba(0,0,0,0)",
                               plot_bgcolor="rgba(0,0,0,0)",
                               title="Headline Sentiment Distribution")
            st.plotly_chart(hist, use_container_width=True)
        else:
            st.info("No headline text available for this stock.")

    st.write("")

    # ===================================================== SECTION 10 =====
    # Supporting charts: price trend and gold co-movement.
    with st.container(border=True):
        st.markdown("Price & Trend Charts")

        trend_df = pd.DataFrame({COLS.date: s[COLS.date], "Close": close.values})
        trend_df["MA20"] = close.rolling(20, min_periods=1).mean().values
        trend_df["MA50"] = close.rolling(50, min_periods=1).mean().values
        trend_df["MA200"] = close.rolling(200, min_periods=1).mean().values
        st.plotly_chart(
            viz.line_chart(trend_df, COLS.date, ["Close", "MA20", "MA50", "MA200"],
                           title=f"{symbol} — Trend Overview"),
            use_container_width=True,
        )

        comp = pd.DataFrame({COLS.date: s[COLS.date]})
        base_close = close.dropna().iloc[0] if close.dropna().size else 1.0
        comp["Stock"] = close.values / base_close * 100
        if gold.notna().any():
            base_gold = gold.dropna().iloc[0]
            comp["Gold"] = gold.values / base_gold * 100
            ys = ["Stock", "Gold"]
        else:
            ys = ["Stock"]
        title = (f"Stock vs Gold (corr {gold_corr:+.2f})"
                 if np.isfinite(gold_corr) else "Stock (indexed to 100)")
        st.plotly_chart(viz.line_chart(comp, COLS.date, ys, title=title),
                        use_container_width=True)

    st.write("")

    # ---- Export the scorecard --------------------------------------------
    summary = pd.DataFrame({
        "Metric": ["Health Score", "Trend", "Momentum", "Volume", "Stability",
                   "News Sentiment", "Gold Correlation", "Overall Risk",
                   "Stance", "Assessment"],
        "Value": [f"{health:.0f}", f"{trend:.0f}", f"{momentum:.0f}", f"{volume:.0f}",
                  f"{volatility:.0f}", f"{sentiment:.0f}",
                  f"{gold_corr:+.2f}" if np.isfinite(gold_corr) else "—",
                  overall_risk, stance, assessment],
    })
    with st.expander("Export thesis scorecard"):
        st.dataframe(summary, use_container_width=True, hide_index=True)
        download_buttons(summary, key="thesis", label=f"{symbol}_investment_thesis")