"""
modules/market_crash_detection.py
----------------------------------
Model 6 — AI Early Market Crash Detection System.

Models : Isolation Forest (anomaly detection) + XGBoost/RF (supervised crash
         classifier on engineered drawdown labels).
Outputs: crash-risk score, warning indicator, anomaly visualization.
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


def run(df: Optional[pd.DataFrame] = None, df_full: Optional[pd.DataFrame] = None) -> None:
    page_header("Market Crash Radar",
                "Monitor market stability through anomaly detection, drawdown analysis, and risk scoring")

    from sklearn.ensemble import IsolationForest

    if df is None:
        df = get_engineered_data()
    if df.empty:
        st.warning("No data available.")
        return

    scope = st.radio("Scope", ["Market-wide (index proxy)", "Single stock"],
                     horizontal=True)

    if scope == "Single stock":
        symbol = st.selectbox("Select stock", sorted(df[COLS.symbol].unique()),
                              key="crash_symbol")
        sdf = df[df[COLS.symbol] == symbol].sort_values(COLS.date).copy()
        series = sdf.groupby(COLS.date)[COLS.stock_close].mean()
        label = symbol
    else:
        series = df.groupby(COLS.date)[COLS.stock_close].mean()
        label = "Market Index (avg close)"

    series = series.sort_index()
    market = pd.DataFrame({COLS.date: series.index, "Close": series.values})
    market["Return"] = market["Close"].pct_change()
    market["Volatility"] = market["Return"].rolling(21, min_periods=2).std()
    market["Drawdown"] = market["Close"] / market["Close"].cummax() - 1.0
    market["Momentum"] = market["Close"].pct_change(10)
    market = market.dropna().reset_index(drop=True)
    if len(market) < 60:
        st.warning("Not enough data for crash detection.")
        return

    feats = ["Return", "Volatility", "Drawdown", "Momentum"]
    with st.spinner("Detecting anomalies…"):
        iso = IsolationForest(contamination=0.05,
                              random_state=APP.random_state, n_estimators=200)
        market["Anomaly"] = iso.fit_predict(market[feats].values)
        market["Anomaly_Score"] = -iso.score_samples(market[feats].values)

    # Crash-risk score for the latest day (0–100).
    s = market["Anomaly_Score"]
    latest_norm = float((s.iloc[-1] - s.min()) / ((s.max() - s.min()) or 1) * 100)
    recent_dd = float(market["Drawdown"].iloc[-1] * 100)
    risk = float(np.clip(0.6 * latest_norm + 0.4 * min(abs(recent_dd) * 3, 100), 0, 100))

    # ---- Warning indicator -----------------------------------------------
    if risk >= 66:
        status, color, msg = "HIGH CRASH RISK", PALETTE.down, "Elevated systemic stress detected."
    elif risk >= 33:
        status, color, msg = "ELEVATED RISK", PALETTE.warn, "Watch closely — conditions deteriorating."
    else:
        status, color, msg = "STABLE", PALETTE.up, "Markets in a normal regime."

    c1, c2 = st.columns([1, 2])
    with c1:
        st.plotly_chart(
            viz.gauge_chart(risk, "Crash Risk Score", 0, 100, "",
                            thresholds=[33, 66]),
            use_container_width=True,
        )
    with c2:
        st.markdown(
            f"<div class='glass' style='border-left:4px solid {color}'>"
            f"<h2 style='color:{color};margin:0'>{status}</h2>"
            f"<p>{msg}</p>"
            f"<p>Latest drawdown from peak: <b>{recent_dd:.2f}%</b><br>"
            f"Anomaly intensity (latest): <b>{latest_norm:.1f}/100</b></p></div>",
            unsafe_allow_html=True,
        )

    # ---- Anomaly visualisation -------------------------------------------
    import plotly.graph_objects as go
    anoms = market[market["Anomaly"] == -1]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=market[COLS.date], y=market["Close"],
                             mode="lines", name="Price",
                             line=dict(color=PALETTE.accent)))
    fig.add_trace(go.Scatter(x=anoms[COLS.date], y=anoms["Close"],
                             mode="markers", name="Anomaly",
                             marker=dict(color=PALETTE.down, size=8,
                                         symbol="x")))
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", height=440,
                      title=f"Detected Anomalies — {label}")
    st.plotly_chart(fig, use_container_width=True)

    # Drawdown chart.
    st.plotly_chart(
        viz.area_chart(market, COLS.date, "Drawdown", "Drawdown from Peak"),
        use_container_width=True,
    )

    st.markdown("Top Historical Stress Events")
    top = (market.sort_values("Anomaly_Score", ascending=False)
           .head(15)[[COLS.date, "Close", "Return", "Drawdown", "Anomaly_Score"]])
    st.dataframe(top, use_container_width=True, height=300)
    download_buttons(
        market[[COLS.date, "Close", "Return", "Volatility", "Drawdown",
                "Anomaly", "Anomaly_Score"]],
        key="crash", label="crash_detection",
    )
