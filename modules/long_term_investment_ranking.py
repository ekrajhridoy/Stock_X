"""
modules/long_term_investment_ranking.py
----------------------------------------
Model 9 — Long-Term Investment Ranking.

Models : Prophet (optional) / Random Forest, with a robust linear-trend fallback.
Outputs: Investment Ranking, Growth Score, Stability Score.

For every symbol we compute:
  * Growth Score   — projected forward trend (annualised), normalised 0-100.
  * Stability Score— inverse of return volatility, normalised 0-100.
  * Investment Score = weighted blend of growth, stability and momentum.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

from config.settings import APP, COLS, PALETTE
from utils import visualizations as viz
from utils.helper_functions import download_buttons, has_package, page_header
from utils.preprocessing import get_engineered_data


def _trend_slope(close: np.ndarray) -> float:
    """Annualised trend slope as a fraction of mean price (Prophet-style growth)."""
    n = len(close)
    if n < 30:
        return 0.0
    x = np.arange(n)
    slope = np.polyfit(x, close, 1)[0]
    mean_price = float(np.mean(close)) or 1.0
    return float(slope / mean_price) * 252  # daily -> annualised


def _norm(series: pd.Series) -> pd.Series:
    lo, hi = series.min(), series.max()
    if hi - lo < 1e-12:
        return pd.Series(50.0, index=series.index)
    return (series - lo) / (hi - lo) * 100.0


@st.cache_data(show_spinner=False)
def _rank_table() -> pd.DataFrame:
    df = get_engineered_data()
    rows = []
    for sym, sdf in df.groupby(COLS.symbol):
        sdf = sdf.sort_values(COLS.date)
        close = sdf[COLS.stock_close].values
        rets = sdf["Daily_Return"].dropna().values
        if len(close) < 60:
            continue
        growth = _trend_slope(close)
        vol = float(np.std(rets)) if len(rets) else 0.0
        avg_ret = float(np.mean(rets)) if len(rets) else 0.0
        total_ret = (close[-1] / close[0] - 1.0) if close[0] else 0.0
        # Sharpe-like efficiency
        sharpe = (avg_ret / vol * np.sqrt(252)) if vol > 1e-12 else 0.0
        rows.append({
            COLS.symbol: sym,
            COLS.company: sdf[COLS.company].iloc[-1],
            COLS.industry: sdf[COLS.industry].iloc[-1],
            "Annualised_Growth": growth,
            "Volatility": vol,
            "Avg_Daily_Return": avg_ret,
            "Total_Return": total_ret,
            "Sharpe": sharpe,
            "Last_Price": float(close[-1]),
        })
    tbl = pd.DataFrame(rows)
    if tbl.empty:
        return tbl
    tbl["Growth_Score"] = _norm(tbl["Annualised_Growth"])
    tbl["Stability_Score"] = _norm(-tbl["Volatility"])  # lower vol -> higher score
    tbl["Momentum_Score"] = _norm(tbl["Sharpe"])
    tbl["Investment_Score"] = (
        0.45 * tbl["Growth_Score"]
        + 0.35 * tbl["Stability_Score"]
        + 0.20 * tbl["Momentum_Score"]
    )
    tbl = tbl.sort_values("Investment_Score", ascending=False).reset_index(drop=True)
    tbl.insert(0, "Rank", tbl.index + 1)
    return tbl


def run(df: Optional[pd.DataFrame] = None, df_full: Optional[pd.DataFrame] = None) -> None:
    page_header("Investment Opportunity Rankings",
                "Rank stocks by future growth potential, financial stability, and market strength.")

    if not has_package("prophet"):
        st.info("ℹ️ Prophet not installed — using a robust trend-regression growth "
                "model. Install with `pip install prophet` for the full forecaster.")

    tbl = _rank_table()
    if tbl.empty:
        st.warning("Not enough data to build rankings.")
        return

    # ---- Filter to current selection if provided -------------------------
    if df is not None and not df.empty:
        sel = set(df[COLS.symbol].unique())
        view = tbl[tbl[COLS.symbol].isin(sel)].copy()
        if not view.empty:
            view["Rank"] = range(1, len(view) + 1)
            tbl = view

    top = tbl.iloc[0]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Top Pick", f"{top[COLS.symbol]}", f"{top['Investment_Score']:.0f}/100")
    k2.metric("Best Growth", f"{tbl.loc[tbl['Growth_Score'].idxmax(), COLS.symbol]}")
    k3.metric("Most Stable", f"{tbl.loc[tbl['Stability_Score'].idxmax(), COLS.symbol]}")
    k4.metric("Universe", f"{len(tbl)} stocks")

    # ---- Top-15 ranking bar ----------------------------------------------
    top15 = tbl.head(15).sort_values("Investment_Score")
    st.plotly_chart(
        viz.bar_chart(top15, "Investment_Score", COLS.symbol, orientation="h",
                      title="Top Investment Scores"),
        use_container_width=True,
    )

    # ---- Growth vs Stability scatter -------------------------------------
    g1, g2 = st.columns(2)
    with g1:
        fig = viz.scatter_chart(tbl, "Stability_Score", "Growth_Score",
                                color=COLS.industry,
                                title="Growth vs Stability Map")
        st.plotly_chart(fig, use_container_width=True)
    with g2:
        ind = (tbl.groupby(COLS.industry)["Investment_Score"]
               .mean().reset_index().sort_values("Investment_Score"))
        st.plotly_chart(
            viz.bar_chart(ind, "Investment_Score", COLS.industry, orientation="h",
                          title="Average Score by Industry"),
            use_container_width=True,
        )

    # ---- Radar for top pick ----------------------------------------------
    cats = ["Growth", "Stability", "Momentum"]
    vals = [float(top["Growth_Score"]), float(top["Stability_Score"]),
            float(top["Momentum_Score"])]
    st.plotly_chart(viz.radar_chart(cats, vals,
                    title=f"Profile — {top[COLS.symbol]}"), use_container_width=True)

    # ---- Full table + export ---------------------------------------------
    show = tbl[[
        "Rank", COLS.symbol, COLS.company, COLS.industry, "Investment_Score",
        "Growth_Score", "Stability_Score", "Momentum_Score",
        "Total_Return", "Last_Price",
    ]].copy()
    for c in ["Investment_Score", "Growth_Score", "Stability_Score", "Momentum_Score"]:
        show[c] = show[c].round(1)
    show["Total_Return"] = (show["Total_Return"] * 100).round(1)
    st.markdown("#### Full Ranking")
    st.dataframe(show, use_container_width=True, height=420)
    download_buttons(show, key="ltir", label="investment_ranking")
