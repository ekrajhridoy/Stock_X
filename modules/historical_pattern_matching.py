"""
modules/historical_pattern_matching.py
---------------------------------------
Model 8 — Historical Analogues & Outcome Analysis.

Answers the investor question:
    "Historically, when this stock looked like it does now, what usually
     happened next?"

Core similarity engine (UNCHANGED): a z-normalised 30-day query window is
compared against every historical 30-day window using either Dynamic Time
Warping (self-contained numpy DTW, or tslearn when available) or Euclidean KNN.

On top of that engine this module now derives, from the TOP-N closest analogues:
    * 7 / 30 / 90-day forward returns for each analogue,
    * aggregate outcome statistics (avg / median / best / worst),
    * bullish-vs-bearish probability,
    * a match-confidence rating,
    * a "what happened next" future-path chart,
    * an outcome-distribution chart,
    * a plain-language interpretation panel,
    * an investor-friendly matches table.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from config.settings import COLS, PALETTE
from utils.helper_functions import download_buttons, has_package, page_header
from utils.preprocessing import get_engineered_data

WINDOW = 30
TOP_N = 10                       # number of historical analogues to aggregate
FWD_HORIZONS = (7, 30, 90)       # forward-return horizons (trading days)
FUTURE_DAYS = 30                 # days of "what followed" to draw on the path chart


# --------------------------------------------------------------------------- #
# Core similarity helpers (UNCHANGED)                                         #
# --------------------------------------------------------------------------- #
def _znorm(x: np.ndarray) -> np.ndarray:
    sd = x.std()
    return (x - x.mean()) / (sd if sd > 1e-9 else 1.0)


def _dtw_distance(a: np.ndarray, b: np.ndarray, band: int = 8) -> float:
    """Sakoe-Chiba banded DTW distance (pure numpy)."""
    n, m = len(a), len(b)
    inf = float("inf")
    D = np.full((n + 1, m + 1), inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        lo = max(1, i - band)
        hi = min(m, i + band)
        for j in range(lo, hi + 1):
            cost = (a[i - 1] - b[j - 1]) ** 2
            D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return float(np.sqrt(D[n, m]))


# --------------------------------------------------------------------------- #
# New presentation/analysis helpers (do not touch the similarity engine)      #
# --------------------------------------------------------------------------- #
def _znorm_ref(x: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Z-normalise `x` using the mean/std of a reference window `ref` so a
    pattern and its continuation stay on one comparable scale."""
    sd = ref.std()
    return (x - ref.mean()) / (sd if sd > 1e-9 else 1.0)


def _forward_returns(close: np.ndarray, end_idx: int,
                     horizons: Tuple[int, ...] = FWD_HORIZONS) -> Dict[int, float]:
    """Forward % returns measured from the last day of a matched window.

    Returns NaN for any horizon that runs past the end of the series (handles
    the 'insufficient future data' / 'missing forward window' edge cases)."""
    out: Dict[int, float] = {}
    base = close[end_idx] if 0 <= end_idx < len(close) else np.nan
    for h in horizons:
        j = end_idx + h
        out[h] = (close[j] / base - 1.0) * 100 if (np.isfinite(base) and base > 0
                                                   and j < len(close)) else np.nan
    return out


def _confidence(similarity: float) -> str:
    """Map a similarity score (0-100) to an investor-friendly confidence label."""
    if similarity >= 90:
        return "Very High"
    if similarity >= 80:
        return "High"
    if similarity >= 65:
        return "Moderate"
    return "Weak"


def _tendency(pos_pct: float, avg_ret: float) -> str:
    """Rule-based bullish/bearish characterisation for the interpretation panel."""
    if not (np.isfinite(pos_pct) and np.isfinite(avg_ret)):
        return "inconclusive"
    if pos_pct >= 70 and avg_ret >= 5:
        return "strongly bullish"
    if pos_pct >= 55 and avg_ret > 0:
        return "moderately bullish"
    if pos_pct <= 30 and avg_ret <= -5:
        return "strongly bearish"
    if pos_pct <= 45 and avg_ret < 0:
        return "moderately bearish"
    return "broadly neutral / mixed"


def _pct(x: float) -> str:
    return f"{x:+.1f}%" if np.isfinite(x) else "—"


# --------------------------------------------------------------------------- #
# Entry point                                                                 #
# --------------------------------------------------------------------------- #
def run(df: Optional[pd.DataFrame] = None, df_full: Optional[pd.DataFrame] = None) -> None:
    page_header("Historical Analogues & Outcome Analysis",
                "When the stock looked like this before, what happened next?", "🔎")

    if df is None:
        df = get_engineered_data()
    if df.empty:
        st.warning("No data available.")
        return

    symbols = sorted(df[COLS.symbol].unique().tolist())
    c1, c2 = st.columns([2, 1])
    with c1:
        symbol = st.selectbox("Select stock", symbols, key="pattern_symbol")
    with c2:
        method = st.selectbox("Similarity metric", ["DTW", "Euclidean (KNN)"],
                              key="pattern_method")

    sdf = df[df[COLS.symbol] == symbol].sort_values(COLS.date).reset_index(drop=True)
    close = sdf[COLS.stock_close].astype(float).values
    dates = sdf[COLS.date].values
    if len(close) < WINDOW * 3:
        st.warning("Not enough history for pattern search.")
        return

    query = _znorm(close[-WINDOW:])

    # ---- Core sliding-window search (UNCHANGED engine) -------------------
    candidates: List[Tuple[int, float]] = []
    use_tslearn = method == "DTW" and has_package("tslearn")
    if use_tslearn:
        from tslearn.metrics import dtw as ts_dtw  # type: ignore

    with st.spinner("Scanning history…"):
        step = 2  # stride for speed
        for start in range(0, len(close) - 2 * WINDOW, step):
            window = _znorm(close[start:start + WINDOW])
            if method == "DTW":
                d = ts_dtw(query, window) if use_tslearn \
                    else _dtw_distance(query, window)
            else:
                d = float(np.linalg.norm(query - window))
            candidates.append((start, d))

    if not candidates:                       # robustness: tiny dataset
        st.warning("Not enough historical windows to analyse.")
        return

    candidates.sort(key=lambda t: t[1])
    max_d = max(d for _, d in candidates) or 1.0

    # ---- Build the top-N analogue table with forward outcomes ------------
    top = candidates[:min(TOP_N, len(candidates))]
    match_rows = []
    for start, d in top:
        sim = max(0.0, 100.0 * (1 - d / max_d))
        end_idx = start + WINDOW - 1
        fr = _forward_returns(close, end_idx)
        match_rows.append({
            "Start_Date": pd.to_datetime(dates[start]).date(),
            "Similarity_%": round(sim, 1),
            "Ret_7d": fr[7], "Ret_30d": fr[30], "Ret_90d": fr[90],
            "_start": start, "_distance": d,            # kept internally only
        })
    matches_df = pd.DataFrame(match_rows).sort_values("Similarity_%", ascending=False)

    # Primary horizon for bullish/bearish (30d if available, else 7d).
    prim_h = 30 if matches_df["Ret_30d"].notna().any() else 7
    prim = matches_df[f"Ret_{prim_h}d"]
    prim_valid = prim.dropna()
    pos_pct = (prim_valid > 0).mean() * 100 if len(prim_valid) else np.nan
    bear_pct = 100 - pos_pct if np.isfinite(pos_pct) else np.nan

    # Aggregate outcome statistics (NaN-safe).
    def _agg(col, fn):
        v = matches_df[col].dropna().values
        return float(fn(v)) if len(v) else np.nan
    avg7, avg30, avg90 = (_agg("Ret_7d", np.mean), _agg("Ret_30d", np.mean),
                          _agg("Ret_90d", np.mean))
    med30 = _agg("Ret_30d", np.median)
    best30, worst30 = _agg("Ret_30d", np.max), _agg("Ret_30d", np.min)

    best = matches_df.iloc[0]
    best_sim = float(best["Similarity_%"])
    best_start = int(best["_start"])
    best_close = close[best_start:best_start + WINDOW]
    best_date0 = pd.to_datetime(dates[best_start]).date()

    # ===================================================== SECTION 1 ======
    # KPI cards
    st.markdown("### At a Glance")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Similarity Score", f"{best_sim:.1f}%")
    k2.metric("Match Confidence", _confidence(best_sim))
    k3.metric("Bullish Probability", f"{pos_pct:.0f}%" if np.isfinite(pos_pct) else "—")
    k4.metric("Average 30-Day Return", _pct(avg30))

    st.markdown(f"#### Outcome Analysis · top {len(matches_df)} analogues")
    o1, o2, o3 = st.columns(3)
    o1.metric("Avg 7-Day Return", _pct(avg7))
    o2.metric("Avg 30-Day Return", _pct(avg30))
    o3.metric("Avg 90-Day Return", _pct(avg90))
    o4, o5, o6 = st.columns(3)
    o4.metric("Median 30-Day", _pct(med30))
    o5.metric("Best 30-Day", _pct(best30))
    o6.metric("Worst 30-Day", _pct(worst30))
    bb1, bb2 = st.columns(2)
    bb1.metric("Bullish Outcome Probability", f"{pos_pct:.0f}%" if np.isfinite(pos_pct) else "—")
    bb2.metric("Bearish Outcome Probability", f"{bear_pct:.0f}%" if np.isfinite(bear_pct) else "—")

    import plotly.graph_objects as go

    # ===================================================== SECTION 2 ======
    # Pattern comparison chart (kept from the original module)
    st.markdown("### Pattern Comparison")
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=_znorm(close[-WINDOW:]), mode="lines",
                             name="Current (now)",
                             line=dict(color=PALETTE.accent, width=3)))
    fig.add_trace(go.Scatter(y=_znorm(best_close), mode="lines",
                             name=f"Match @ {best_date0}",
                             line=dict(color=PALETTE.warn, dash="dash", width=3)))
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", height=420,
                      title="Pattern Comparison (z-normalised)")
    st.plotly_chart(fig, use_container_width=True)

    # ===================================================== SECTION 3 ======
    # Future-path chart: the matched pattern AND what followed it.
    st.markdown("### What Happened After the Closest Match")
    future_avail = max(0, len(close) - (best_start + WINDOW))
    fut = min(FUTURE_DAYS, future_avail)
    seg = close[best_start: best_start + WINDOW + fut]
    seg_norm = _znorm_ref(seg, best_close)          # normalise by the pattern window
    cur_norm = _znorm(close[-WINDOW:])
    figf = go.Figure()
    figf.add_trace(go.Scatter(x=list(range(WINDOW)), y=cur_norm, mode="lines",
                              name="Current pattern",
                              line=dict(color=PALETTE.accent, width=3)))
    figf.add_trace(go.Scatter(x=list(range(len(seg_norm))), y=seg_norm, mode="lines",
                              name=f"Matched pattern + next {fut}d",
                              line=dict(color=PALETTE.warn, width=3)))
    # Vertical marker where the pattern ends and the "future" begins.
    figf.add_vline(x=WINDOW - 1, line=dict(color=PALETTE.muted, width=1, dash="dot"))
    figf.add_annotation(x=WINDOW - 1, y=max(seg_norm.max(), cur_norm.max()),
                        text="pattern ends →", showarrow=False,
                        font=dict(color=PALETTE.muted, size=11), xanchor="left")
    figf.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                       plot_bgcolor="rgba(0,0,0,0)", height=420,
                       title="Historical Future Path (z-normalised)",
                       xaxis_title="Trading days from start of pattern")
    st.plotly_chart(figf, use_container_width=True)
    if fut == 0:
        st.caption("This match sits at the very end of the available history, so "
                   "there is no recorded future path to show.")

    # ===================================================== SECTION 4 ======
    # Distribution of historical forward returns across the analogues.
    st.markdown("### Distribution of Historical Outcomes")
    figd = go.Figure()
    colors = {7: PALETTE.accent, 30: PALETTE.accent_2, 90: PALETTE.warn}
    any_dist = False
    for h in FWD_HORIZONS:
        vals = matches_df[f"Ret_{h}d"].dropna().values
        if len(vals):
            any_dist = True
            figd.add_trace(go.Box(y=vals, name=f"{h}-day", boxpoints="all",
                                  marker_color=colors[h], jitter=0.4,
                                  pointpos=0))
    if any_dist:
        figd.add_hline(y=0, line=dict(color=PALETTE.muted, width=1, dash="dot"))
        figd.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                           plot_bgcolor="rgba(0,0,0,0)", height=420,
                           title="Forward Returns Across Similar Periods (%)",
                           yaxis_title="Return (%)")
        st.plotly_chart(figd, use_container_width=True)
    else:
        st.info("Not enough forward data among the matches to plot a distribution.")

    # ===================================================== SECTION 5 ======
    # AI-style interpretation panel (generated from the computed statistics).
    st.markdown("### Historical Pattern Insight")
    tend = _tendency(pos_pct, avg30 if np.isfinite(avg30) else avg7)
    pos_txt = f"{pos_pct:.0f}%" if np.isfinite(pos_pct) else "an unknown share of"
    # Built as a single-line HTML string (no leading indentation) so Streamlit
    # renders it as HTML rather than a markdown code block.
    insight = (
        f"<div class='glass'>"
        f"<p>The current <b>{WINDOW}-day</b> price pattern for <b>{symbol}</b> closely "
        f"resembles <b>{len(matches_df)}</b> historical periods. Among those periods:</p>"
        f"<ul>"
        f"<li><b>{pos_txt}</b> resulted in positive {prim_h}-day returns</li>"
        f"<li>Average 30-day return was <b>{_pct(avg30)}</b></li>"
        f"<li>Average 90-day return was <b>{_pct(avg90)}</b></li>"
        f"<li>Outcomes ranged from <b>{_pct(worst30)}</b> to <b>{_pct(best30)}</b> over 30 days</li>"
        f"</ul>"
        f"<p>This suggests a <b>{tend}</b> historical tendency, although outcomes "
        f"varied across periods. Past analogues are not a guarantee of future results.</p>"
        f"</div>"
    )
    st.markdown(insight, unsafe_allow_html=True)

    # ============================================ HOW THE MATCHING WORKS ===
    # Plain-language explainer for general users. Reflects the metric actually
    # selected above (DTW vs Euclidean) so the description always matches.
    with st.expander(" How are these matches found?"):
        if method == "DTW":
            st.markdown(
                "**Dynamic Time Warping (DTW)** is the method used to find these "
                "look-alike periods.\n\n"
                "Think of comparing two songs hummed at slightly different speeds. "
                "If you lined them up note-for-note they'd look different, but your "
                "ear still recognises the same tune because it *stretches* time to "
                "match the rhythm. DTW does exactly that for price charts: it gently "
                "**stretches and compresses the time axis** so two patterns can be "
                "compared by their *shape*, even if one played out a little faster or "
                "slower than the other.\n\n"
                "In plain steps:\n"
                "1. Take today's most recent 30-day price shape (the *current pattern*).\n"
                "2. Slide across the stock's whole history, comparing it against every "
                "past 30-day window.\n"
                "3. For each comparison, DTW finds the best alignment between the two "
                "shapes and measures how much they still differ — a smaller difference "
                "means a closer match.\n"
                "4. Prices are first put on the same scale (z-normalised), so DTW "
                "focuses on the **pattern's shape**, not whether the stock was cheap or "
                "expensive at the time.\n\n"
                "Why it's useful here: a rise-dip-recovery can unfold over 25 days in "
                "one period and 35 in another. A rigid point-by-point comparison would "
                "miss the resemblance; DTW catches it because it tolerates that timing "
                "difference. The closest matches it finds are listed below."
            )
        else:
            st.markdown(
                "**Euclidean (KNN) matching** is the method used to find these "
                "look-alike periods.\n\n"
                "It lines up today's 30-day price shape against every past 30-day "
                "window **point-by-point on the same days**, then measures the total "
                "straight-line difference between them — like laying one chart on top "
                "of another and adding up the gaps. The windows with the **smallest "
                "total gap** are the closest matches (this is the 'nearest neighbours' "
                "idea).\n\n"
                "Prices are first put on the same scale (z-normalised) so the "
                "comparison is about the **pattern's shape**, not the price level. "
                "Unlike Dynamic Time Warping, this method compares the days rigidly and "
                "does not stretch the timeline, so it's faster but less forgiving when a "
                "pattern unfolds a bit faster or slower than today's. The closest "
                "matches it finds are listed below."
            )

    # ===================================================== SECTION 6 ======
    # Improved, investor-friendly top-matches table (+ export).
    st.markdown("### Closest Historical Windows")
    display = matches_df[["Start_Date", "Similarity_%", "Ret_7d", "Ret_30d", "Ret_90d"]].copy()
    display = display.rename(columns={"Ret_7d": "7-Day Return %",
                                      "Ret_30d": "30-Day Return %",
                                      "Ret_90d": "90-Day Return %"})
    for c in ["7-Day Return %", "30-Day Return %", "90-Day Return %"]:
        display[c] = display[c].round(2)
    st.dataframe(display, use_container_width=True, hide_index=True)
    # Distance is retained internally for export but is not the primary metric.
    export = matches_df.drop(columns=["_start"]).rename(columns={"_distance": "Distance"})
    download_buttons(export, key="pattern", label=f"{symbol}_historical_analogues")