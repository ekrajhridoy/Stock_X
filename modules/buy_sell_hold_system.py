"""
modules/buy_sell_hold_system.py
-------------------------------
Model 11 — BUY / SELL / HOLD signal generator (professional trading assistant).

Models : XGBoost / LSTM (with a reliable RandomForest fallback).
Features: RSI, MACD, Moving Averages, Momentum, Volume.
Output : BUY / SELL / HOLD recommendation + a trading-assistant dashboard.

The target is a 3-class label built from forward returns:
    BUY  -> next-period return >  +threshold
    SELL -> next-period return <  -threshold
    HOLD -> otherwise
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from config.settings import APP, COLS, PALETTE
from utils import visualizations as viz
from utils.helper_functions import (
    classification_metrics,
    download_buttons,
    has_package,
    page_header,
)
from utils.preprocessing import get_engineered_data

SIGNAL_FEATURES: List[str] = [
    "RSI", "MACD", "MACD_Signal", "MACD_Hist", "Momentum",
    "SMA_5", "SMA_20", "EMA_12", "EMA_26", "Volume_Change", "Daily_Return",
]
LABELS = ["SELL", "HOLD", "BUY"]  # 0, 1, 2


def _build_model(name: str):
    rs = APP.random_state
    if name == "XGBoost" and has_package("xgboost"):
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, eval_metric="mlogloss",
            random_state=rs, n_jobs=-1,
        ), "XGBoost"
    from sklearn.ensemble import RandomForestClassifier
    label = "Random Forest (fallback)" if name != "Random Forest" else "Random Forest"
    return RandomForestClassifier(
        n_estimators=300, max_depth=12, random_state=rs, n_jobs=-1,
        class_weight="balanced",
    ), label


def _make_labels(returns: pd.Series, threshold: float) -> np.ndarray:
    lab = np.ones(len(returns), dtype=int)  # HOLD
    lab[returns.values > threshold] = 2     # BUY
    lab[returns.values < -threshold] = 0    # SELL
    return lab


@st.cache_data(show_spinner=False)
def _prepare(symbol: str, threshold: float) -> Tuple[pd.DataFrame, List[str]]:
    df = get_engineered_data()
    sdf = df[df[COLS.symbol] == symbol].sort_values(COLS.date).copy()
    sdf["Fwd_Return"] = sdf[COLS.stock_close].pct_change().shift(-1)
    sdf = sdf.dropna(subset=["Fwd_Return"])
    sdf["Signal"] = _make_labels(sdf["Fwd_Return"], threshold)
    feats = [c for c in SIGNAL_FEATURES if c in sdf.columns]
    return sdf, feats


def run(df: Optional[pd.DataFrame] = None, df_full: Optional[pd.DataFrame] = None) -> None:
    page_header("Personal Trading Partner",
                "Generate an evidence-based BUY, HOLD, or AVOID investment recommendation using technical, sentiment, and risk indicators.")

    if df is None:
        df = get_engineered_data()
    if df.empty:
        st.warning("No data available.")
        return

    symbols = sorted(df[COLS.symbol].unique().tolist())
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        symbol = st.selectbox("Select stock", symbols, key="bsh_symbol")
    with c2:
        algo = st.selectbox("Model", ["XGBoost", "Random Forest"], key="bsh_algo")
    with c3:
        thr_pct = st.slider("Signal threshold (%)", 0.5, 5.0, 1.5, 0.5, key="bsh_thr")

    threshold = thr_pct / 100.0
    sdf, feats = _prepare(symbol, threshold)
    if len(sdf) < 120 or not feats:
        st.warning("Not enough history for this stock to train reliably.")
        return

    X = sdf[feats].values
    y = sdf["Signal"].values
    cut = int(len(sdf) * 0.8)
    X_tr, X_te = X[:cut], X[cut:]
    y_tr, y_te = y[:cut], y[cut:]

    with st.spinner("Training trading assistant…"):
        model, used = _build_model(algo)
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)

    if used.endswith("(fallback)"):
        st.info(f"ℹ️ `{algo}` not installed — using **{used}**. "
                f"Install with `pip install {algo.lower()}` for the full engine.")

    # ---- Current recommendation ------------------------------------------
    last_row = sdf[feats].iloc[[-1]].values
    sig = int(model.predict(last_row)[0])
    try:
        proba = model.predict_proba(last_row)[0]
        conf = float(proba[sig]) * 100
    except Exception:  # noqa: BLE001
        conf = 100.0
    call = LABELS[sig]
    badge = {"BUY": "badge-buy", "SELL": "badge-sell", "HOLD": "badge-hold"}[call]
    arrow = {"BUY": "▲", "SELL": "▼", "HOLD": "■"}[call]

    last = sdf.iloc[-1]
    st.markdown(
        f"<div class='glass'><h2>Recommendation for {symbol}: "
        f"<span class='badge {badge}'>{call} {arrow}</span></h2>"
        f"<p>Model confidence: <b>{conf:.1f}%</b> &nbsp;·&nbsp; "
        f"RSI <b>{last.get('RSI', float('nan')):.1f}</b> &nbsp;·&nbsp; "
        f"MACD <b>{last.get('MACD', float('nan')):.3f}</b> &nbsp;·&nbsp; "
        f"Close <b>₹{last[COLS.stock_close]:,.2f}</b></p></div>",
        unsafe_allow_html=True,
    )

    # ---- Signal distribution + accuracy ----------------------------------
    m = classification_metrics(y_te, y_pred)
    counts = pd.Series(sdf["Signal"]).value_counts().sort_index()
    dist = pd.DataFrame({
        "Signal": [LABELS[i] for i in counts.index],
        "Days": counts.values,
    })

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Accuracy", f"{m['Accuracy']*100:.2f}%")
    k2.metric("Precision", f"{m['Precision']*100:.2f}%")
    k3.metric("Recall", f"{m['Recall']*100:.2f}%")
    k4.metric("F1 Score", f"{m['F1']*100:.2f}%")

    g1, g2 = st.columns(2)
    with g1:
        fig = viz.bar_chart(dist, "Signal", "Days", color="Signal",
                            title="Historical Signal Distribution")
        st.plotly_chart(fig, use_container_width=True)
    with g2:
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_te, y_pred, labels=[0, 1, 2])
        st.plotly_chart(viz.confusion_matrix_fig(cm, LABELS,
                        title="Signal Confusion Matrix"), use_container_width=True)

    # ---- Feature importance ----------------------------------------------
    imp = getattr(model, "feature_importances_", None)
    if imp is not None:
        idf = (pd.DataFrame({"Feature": feats, "Importance": imp})
               .sort_values("Importance", ascending=True))
        st.plotly_chart(viz.bar_chart(idf, "Importance", "Feature",
                        orientation="h", title="Indicator Importance"),
                        use_container_width=True)

    # ---- Price with signal markers ---------------------------------------
    recent = sdf.tail(250).copy()
    recent["Signal_Label"] = [LABELS[s] for s in recent["Signal"]]
    fig = viz.line_chart(recent, COLS.date, [COLS.stock_close],
                         title=f"{symbol} — Price with Recent Signals")
    st.plotly_chart(fig, use_container_width=True)

    # ---- Export ----------------------------------------------------------
    out = sdf.iloc[cut:][[COLS.date, COLS.stock_close, "RSI", "MACD", "Momentum"]].copy()
    out["Actual_Signal"] = [LABELS[s] for s in y_te]
    out["Predicted_Signal"] = [LABELS[s] for s in y_pred]
    st.markdown("#### Signal Log")
    st.dataframe(out.tail(200), use_container_width=True, height=300)
    download_buttons(out, key="bsh", label=f"{symbol}_buy_sell_hold")
