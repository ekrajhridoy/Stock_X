"""
modules/company_news_reaction.py
---------------------------------
Model 12 — Company News Sensitivity.

Models : FinBERT (optional, via score_sentiment) + correlation analysis.
Outputs: Sensitivity Ranking, News Impact Dashboard.

For every company we measure how strongly daily news sentiment co-moves with
same-day and next-day returns. The "sensitivity" is the absolute next-day
correlation — companies whose prices react most to news rank highest.

NOTE: This revision changes ONLY the Streamlit UI/UX layer (sidebar filters,
chart interactivity, comparison mode, explain panel, table styling, dashboard
modes). The backend — `_company_daily()`, `_sensitivity_table()`, every
correlation / sentiment / sensitivity computation — is unchanged. All filters
affect display only and never recompute any metric.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px              # UI only (rich hover tooltips)
import plotly.graph_objects as go        # UI only (comparison overlays)
import streamlit as st

from config.settings import APP, COLS, PALETTE
from utils import visualizations as viz
from utils.helper_functions import (
    download_buttons,
    has_package,
    page_header,
    score_sentiment,
)
from utils.preprocessing import get_engineered_data


# --------------------------------------------------------------------------- #
# UNCHANGED backend (do not modify)                                           #
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def _company_daily(symbol: str, use_finbert: bool) -> pd.DataFrame:
    df = get_engineered_data()
    sdf = df[df[COLS.symbol] == symbol].sort_values(COLS.date).copy()
    if COLS.headlines not in sdf.columns:
        return pd.DataFrame()
    texts = sdf[COLS.headlines].fillna("").astype(str).tolist()
    scores, _engine = score_sentiment(texts, use_finbert=use_finbert)
    sdf = sdf.assign(Sentiment=scores)
    sdf["Return"] = sdf[COLS.stock_close].pct_change()
    sdf["Next_Return"] = sdf["Return"].shift(-1)
    return sdf[[COLS.date, COLS.stock_close, "Sentiment", "Return", "Next_Return"]].dropna()


@st.cache_data(show_spinner=False)
def _sensitivity_table(use_finbert: bool) -> pd.DataFrame:
    df = get_engineered_data()
    if COLS.headlines not in df.columns:
        return pd.DataFrame()
    rows = []
    for sym, sdf in df.groupby(COLS.symbol):
        sdf = sdf.sort_values(COLS.date)
        if len(sdf) < 60:
            continue
        texts = sdf[COLS.headlines].fillna("").astype(str).tolist()
        scores, _ = score_sentiment(texts, use_finbert=False)  # fast for the universe
        ret = sdf[COLS.stock_close].pct_change().values
        nxt = pd.Series(ret).shift(-1).values
        sent = pd.Series(scores)
        valid = (~np.isnan(ret)) & (~np.isnan(nxt))
        if valid.sum() < 30 or sent.std() < 1e-9:
            continue
        same = np.corrcoef(sent[valid], pd.Series(ret)[valid])[0, 1]
        nextc = np.corrcoef(sent[valid], pd.Series(nxt)[valid])[0, 1]
        rows.append({
            COLS.symbol: sym,
            COLS.company: sdf[COLS.company].iloc[-1],
            COLS.industry: sdf[COLS.industry].iloc[-1],
            "SameDay_Corr": float(same) if np.isfinite(same) else 0.0,
            "NextDay_Corr": float(nextc) if np.isfinite(nextc) else 0.0,
            "Avg_Sentiment": float(sent.mean()),
        })
    tbl = pd.DataFrame(rows)
    if tbl.empty:
        return tbl
    tbl["Sensitivity"] = tbl["NextDay_Corr"].abs()
    tbl = tbl.sort_values("Sensitivity", ascending=False).reset_index(drop=True)
    tbl.insert(0, "Rank", tbl.index + 1)
    return tbl


# --------------------------------------------------------------------------- #
# UI helpers (presentation only)                                              #
# --------------------------------------------------------------------------- #
def _theme(fig: go.Figure, height: int = 440, title: str = "") -> go.Figure:
    fig.update_layout(template="plotly_dark", height=height, title=title,
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(color=PALETTE.text, size=12),
                      legend=dict(bgcolor="rgba(0,0,0,0)"),
                      margin=dict(l=10, r=10, t=50 if title else 16, b=10))
    return fig


def _strength_word(value: float) -> str:
    """Plain-language label for an absolute correlation (display only)."""
    a = abs(value)
    if a >= 0.30:
        return "very strong"
    if a >= 0.20:
        return "strong"
    if a >= 0.10:
        return "moderate"
    if a >= 0.05:
        return "weak"
    return "very weak"


# --------------------------------------------------------------------------- #
# Entry point                                                                 #
# --------------------------------------------------------------------------- #
def run(df: Optional[pd.DataFrame] = None, df_full: Optional[pd.DataFrame] = None) -> None:
    page_header("News Reaction Analytics",
                "Discover which companies experience the largest price movements following positive or negative news events.")

    use_finbert = st.toggle("Use FinBERT for the selected company",
                            value=False, key="cnr_finbert")
    if use_finbert and not (has_package("transformers") and has_package("torch")):
        st.info("ℹ️ FinBERT stack not installed — using the fast built-in lexicon. "
                "Install `transformers` + `torch` for FinBERT scoring.")
        use_finbert = False

    tbl = _sensitivity_table(False)
    if tbl.empty:
        st.warning("No headline data available to compute sensitivity.")
        return

    # Existing global-filter behaviour (unchanged): restrict to selected symbols.
    if df is not None and not df.empty:
        sel = set(df[COLS.symbol].unique())
        v = tbl[tbl[COLS.symbol].isin(sel)].copy()
        if not v.empty:
            v["Rank"] = range(1, len(v) + 1)
            tbl = v

    # ===================================================== UI: SIDEBAR =====
    # Display-only filters — they never trigger any recomputation.
    st.sidebar.markdown("### 🔎 News Sensitivity Filters")
    industries_all = sorted(tbl[COLS.industry].dropna().unique().tolist())
    industries_sel = st.sidebar.multiselect("Industry", industries_all,
                                             default=industries_all, key="cnr_ind")
    smax = float(tbl["Sensitivity"].max())
    min_sens = st.sidebar.slider("Minimum sensitivity", 0.0, round(smax, 2), 0.0,
                                 0.01, key="cnr_minsens")
    top_n = st.sidebar.slider("Top-N companies", 3, max(3, len(tbl)),
                              min(15, len(tbl)), 1, key="cnr_topn")
    # Date range bounds (for time-series displays only).
    bounds = get_engineered_data()
    dmin = pd.to_datetime(bounds[COLS.date]).min().date()
    dmax = pd.to_datetime(bounds[COLS.date]).max().date()
    date_range = st.sidebar.date_input("Date range (charts)", (dmin, dmax),
                                       min_value=dmin, max_value=dmax, key="cnr_dates")

    # Apply display filters to a VIEW copy (precomputed values only).
    view = tbl.copy()
    if industries_sel:
        view = view[view[COLS.industry].isin(industries_sel)]
    view = view[view["Sensitivity"] >= min_sens].sort_values("Sensitivity", ascending=False)
    if view.empty:
        st.warning("No companies match the current filters. Loosen them in the sidebar.")
        return
    view_top = view.head(top_n)

    # ===================================================== UI: MODE ========
    mode = st.radio("Dashboard mode", ["Overview", "Deep Dive", "Compare Mode"],
                    horizontal=True, key="cnr_mode")

    # ------------------------------------------------------ OVERVIEW -------
    if mode == "Overview":
        top = view.iloc[0]
        k1, k2, k3 = st.columns(3)
        k1.metric("Most News-Sensitive", top[COLS.symbol], f"{top['Sensitivity']:.2f}")
        k2.metric("Companies Shown", f"{len(view)} stocks")
        k3.metric("Avg Sensitivity", f"{view['Sensitivity'].mean():.2f}")

        # Sensitivity ranking with rich hover (company / industry / sensitivity).
        bar = px.bar(
            view_top.sort_values("Sensitivity"), x="Sensitivity", y=COLS.symbol,
            orientation="h", color="Sensitivity", color_continuous_scale="Tealgrn",
            hover_data={COLS.company: True, COLS.industry: True,
                        "SameDay_Corr": ":.3f", "NextDay_Corr": ":.3f",
                        "Sensitivity": ":.3f", COLS.symbol: False},
        )
        bar.update_layout(coloraxis_showscale=False)
        st.plotly_chart(_theme(bar, height=480,
                        title="News Sensitivity Ranking (|next-day correlation|)"),
                        use_container_width=True)

        # Same-day vs next-day scatter with hover detail.
        sc = px.scatter(
            view, x="SameDay_Corr", y="NextDay_Corr", color=COLS.industry,
            size="Sensitivity", hover_name=COLS.company,
            color_discrete_sequence=PALETTE.sequence,
            hover_data={COLS.symbol: True, "Sensitivity": ":.3f",
                        "SameDay_Corr": ":.3f", "NextDay_Corr": ":.3f"},
        )
        sc.add_hline(y=0, line=dict(color=PALETTE.muted, width=1, dash="dot"))
        sc.add_vline(x=0, line=dict(color=PALETTE.muted, width=1, dash="dot"))
        st.plotly_chart(_theme(sc, title="Same-Day vs Next-Day News Correlation"),
                        use_container_width=True)

        # ---- Styled data table (display formatting only) -----------------
        st.markdown("#### Sensitivity Table")
        show = view_top[["Rank", COLS.symbol, COLS.company, COLS.industry,
                         "Sensitivity", "SameDay_Corr", "NextDay_Corr",
                         "Avg_Sentiment"]].copy()
        for c in ["Sensitivity", "SameDay_Corr", "NextDay_Corr", "Avg_Sentiment"]:
            show[c] = show[c].round(3)
        try:
            styled = (show.style
                      .format({c: "{:.3f}" for c in ["Sensitivity", "SameDay_Corr",
                                                     "NextDay_Corr", "Avg_Sentiment"]})
                      .background_gradient(subset=["Sensitivity"], cmap="Greens")
                      .background_gradient(subset=["Avg_Sentiment"], cmap="RdYlGn"))
            st.dataframe(styled, use_container_width=True, height=380)
        except Exception:  # noqa: BLE001 (graceful fallback if styler unavailable)
            st.dataframe(show, use_container_width=True, height=380)
        download_buttons(show, key="cnr", label="news_sensitivity_ranking")

    # ------------------------------------------------------ DEEP DIVE ------
    elif mode == "Deep Dive":
        st.markdown("### 🔬 Company Deep-Dive")
        symbol = st.selectbox("Select company", view[COLS.symbol].tolist(),
                              key="cnr_symbol")
        cd = _company_daily(symbol, use_finbert)
        if cd.empty:
            st.info("No news rows for this company.")
            return

        # UNCHANGED correlation metrics (computed on the full series).
        same = cd["Sentiment"].corr(cd["Return"])
        nextc = cd["Sentiment"].corr(cd["Next_Return"])
        c1, c2, c3 = st.columns(3)
        c1.metric("Same-Day Corr", f"{same:.3f}")
        c2.metric("Next-Day Corr", f"{nextc:.3f}")
        c3.metric("Avg Sentiment", f"{cd['Sentiment'].mean():+.3f}")

        # Date filter trims DISPLAYED rows only (no recompute).
        cd_view = cd
        if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
            d = pd.to_datetime(cd[COLS.date]).dt.date
            cd_view = cd[(d >= date_range[0]) & (d <= date_range[1])]
        if cd_view.empty:
            cd_view = cd

        g1, g2 = st.columns(2)
        with g1:
            st.plotly_chart(viz.scatter_chart(cd_view, "Sentiment", "Next_Return",
                            title=f"{symbol}: Sentiment vs Next-Day Return"),
                            use_container_width=True)
        with g2:
            roll = cd.copy()
            roll["Sentiment_MA"] = roll["Sentiment"].rolling(21, min_periods=1).mean()
            if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
                d = pd.to_datetime(roll[COLS.date]).dt.date
                roll = roll[(d >= date_range[0]) & (d <= date_range[1])]
            st.plotly_chart(viz.line_chart(roll, COLS.date, ["Sentiment_MA"],
                            title=f"{symbol}: 21-Day Sentiment Trend"),
                            use_container_width=True)

        # ---- "Explain Sensitivity" panel (existing metrics only) ---------
        if st.button("🧠 Explain Sensitivity", key="cnr_explain"):
            row = tbl[tbl[COLS.symbol] == symbol]
            rank_txt = f"#{int(row['Rank'].iloc[0])} of {len(tbl)}" if not row.empty else "n/a"
            sens_val = float(row["Sensitivity"].iloc[0]) if not row.empty else abs(nextc)
            ma = cd["Sentiment"].rolling(21, min_periods=1).mean()
            trend = "rising" if ma.iloc[-1] > ma.iloc[0] else "falling" if ma.iloc[-1] < ma.iloc[0] else "flat"
            st.markdown(
                f"<div class='glass'>"
                f"<h4 style='margin:0.1rem 0'>Why {symbol} ranks where it does</h4>"
                f"<ul>"
                f"<li>Sensitivity score <b>{sens_val:.3f}</b> — a <b>{_strength_word(nextc)}</b> "
                f"link between news tone and the <i>next</i> day's return (rank {rank_txt}).</li>"
                f"<li>Same-day correlation is <b>{same:+.3f}</b> ({_strength_word(same)}); "
                f"next-day correlation is <b>{nextc:+.3f}</b>.</li>"
                f"<li>Average news tone is <b>{cd['Sentiment'].mean():+.3f}</b>, and the "
                f"21-day sentiment trend is currently <b>{trend}</b>.</li>"
                f"</ul>"
                f"<p style='opacity:0.85'>Higher sensitivity means this stock's moves have "
                f"historically tracked its news tone more closely. It is a descriptive "
                f"relationship, not a prediction.</p>"
                f"</div>",
                unsafe_allow_html=True,
            )

    # ------------------------------------------------------ COMPARE --------
    else:
        st.markdown("### ⚖️ Compare Companies")
        choices = view[COLS.symbol].tolist()
        picks = st.multiselect("Select 2–5 companies to compare", choices,
                               default=choices[:min(3, len(choices))],
                               max_selections=5, key="cnr_compare")
        if len(picks) < 2:
            st.info("Pick at least 2 companies to compare.")
            return

        # Sensitivity comparison (precomputed values).
        comp = view[view[COLS.symbol].isin(picks)][
            [COLS.symbol, "Sensitivity", "SameDay_Corr", "NextDay_Corr"]]
        long = comp.melt(id_vars=COLS.symbol,
                         value_vars=["Sensitivity", "SameDay_Corr", "NextDay_Corr"],
                         var_name="Metric", value_name="Value")
        cbar = px.bar(long, x=COLS.symbol, y="Value", color="Metric", barmode="group",
                      color_discrete_sequence=PALETTE.sequence,
                      hover_data={"Value": ":.3f"})
        st.plotly_chart(_theme(cbar, title="Sensitivity Comparison"),
                        use_container_width=True)

        # Sentiment-trend comparison (reuses the existing per-company function).
        trend_fig = go.Figure()
        for sym in picks:
            cdc = _company_daily(sym, False)
            if cdc.empty:
                continue
            ma = cdc.assign(MA=cdc["Sentiment"].rolling(21, min_periods=1).mean())
            if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
                d = pd.to_datetime(ma[COLS.date]).dt.date
                ma = ma[(d >= date_range[0]) & (d <= date_range[1])]
            trend_fig.add_trace(go.Scatter(x=ma[COLS.date], y=ma["MA"],
                                           mode="lines", name=sym))
        st.plotly_chart(_theme(trend_fig, title="21-Day Sentiment Trend Comparison"),
                        use_container_width=True)

        st.markdown("#### Comparison Table")
        ct = comp.copy()
        for c in ["Sensitivity", "SameDay_Corr", "NextDay_Corr"]:
            ct[c] = ct[c].round(3)
        st.dataframe(ct, use_container_width=True, hide_index=True)
        download_buttons(ct, key="cnr", label="news_sensitivity_comparison")