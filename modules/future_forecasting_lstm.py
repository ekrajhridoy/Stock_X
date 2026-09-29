"""
modules/future_forecasting_lstm.py
-----------------------------------
Model 4 — multi-day ahead price forecasting.

Models : LSTM / GRU (TensorFlow). When TensorFlow is unavailable a windowed
         Ridge-regression forecaster is used so the page always returns a
         genuine multi-step forecast.
Outputs: next N-day forecast (N chosen by the user), forecast chart,
         RMSE / MAE / MAPE.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from config.settings import APP, COLS
from utils import visualizations as viz
from utils.helper_functions import (
    has_package,
    page_header,
    regression_metrics,
)
from utils.preprocessing import get_engineered_data

WINDOW = 30
DEFAULT_HORIZON = 7      # default number of days to forecast
MAX_HORIZON = 30         # upper bound for the in-page day selector


def _make_windows(series: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for i in range(len(series) - window):
        X.append(series[i:i + window])
        y.append(series[i + window])
    return np.array(X), np.array(y)


def _forecast_dl(series: np.ndarray, kind: str, horizon: int):
    """LSTM/GRU forecaster. Returns (test_pred, test_true, future, name)."""
    import tensorflow as tf
    from tensorflow.keras.layers import GRU, LSTM, Dense
    from tensorflow.keras.models import Sequential

    tf.random.set_seed(APP.random_state)
    mn, mx = series.min(), series.max()
    rng = (mx - mn) or 1.0
    scaled = (series - mn) / rng

    X, y = _make_windows(scaled, WINDOW)
    X = X.reshape((X.shape[0], WINDOW, 1))
    cut = int(len(X) * 0.85)
    Xtr, Xte, ytr, yte = X[:cut], X[cut:], y[:cut], y[cut:]

    layer = LSTM if kind == "LSTM" else GRU
    model = Sequential([
        layer(48, input_shape=(WINDOW, 1)),
        Dense(24, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(Xtr, ytr, epochs=12, batch_size=32, verbose=0,
              validation_split=0.1)

    test_pred = model.predict(Xte, verbose=0).ravel() * rng + mn
    test_true = yte * rng + mn

    # Iterative multi-step future forecast for `horizon` days.
    window = scaled[-WINDOW:].tolist()
    future = []
    for _ in range(horizon):
        x = np.array(window[-WINDOW:]).reshape(1, WINDOW, 1)
        nxt = float(model.predict(x, verbose=0).ravel()[0])
        future.append(nxt)
        window.append(nxt)
    future = np.array(future) * rng + mn
    return test_pred, test_true, future, kind


def _forecast_ridge(series: np.ndarray, horizon: int):
    """Windowed Ridge fallback. Same return contract as _forecast_dl."""
    from sklearn.linear_model import Ridge

    mn, mx = series.min(), series.max()
    rng = (mx - mn) or 1.0
    scaled = (series - mn) / rng
    X, y = _make_windows(scaled, WINDOW)
    cut = int(len(X) * 0.85)
    Xtr, Xte, ytr, yte = X[:cut], X[cut:], y[:cut], y[cut:]

    model = Ridge(alpha=1.0)
    model.fit(Xtr, ytr)
    test_pred = model.predict(Xte) * rng + mn
    test_true = yte * rng + mn

    window = scaled[-WINDOW:].tolist()
    future = []
    for _ in range(horizon):
        x = np.array(window[-WINDOW:]).reshape(1, -1)
        nxt = float(model.predict(x)[0])
        future.append(nxt)
        window.append(nxt)
    future = np.array(future) * rng + mn
    return test_pred, test_true, future, "Ridge (fallback)"


def run(df: Optional[pd.DataFrame] = None, df_full: Optional[pd.DataFrame] = None) -> None:
    page_header("Predictive Price Intelligence",
                "Forecast future stock prices and market trends for the coming trading days.")

    if df is None:
        df = get_engineered_data()
    if df.empty:
        st.warning("No data available.")
        return

    symbols = sorted(df[COLS.symbol].unique().tolist())

    # ---- In-page controls (NOT the sidebar) ------------------------------
    c1, c2 = st.columns(2)
    with c1:
        symbol = st.selectbox("Select stock", symbols, key="lstm_symbol")
    with c2:
        kind = st.selectbox("Architecture", ["LSTM", "GRU"], key="lstm_kind")

    # Forecast-horizon controls. Seed the default ONCE, then let the quick-pick
    # buttons update it *before* the number_input is instantiated — Streamlit
    # forbids writing to a widget's key after that widget has been created.
    if "lstm_horizon" not in st.session_state:
        st.session_state["lstm_horizon"] = DEFAULT_HORIZON

    st.markdown("**Forecast horizon** — choose how many trading days to project:")
    b1, b2, b3, b4, b5 = st.columns([1, 1, 1, 1, 2])
    for col, days in zip((b1, b2, b3, b4), (7, 14, 21, 30)):
        if col.button(f"{days} days", key=f"lstm_qp_{days}",
                      use_container_width=True):
            st.session_state["lstm_horizon"] = days        # set before widget below
    with b5:
        # No `value=` here: the widget reads its value from session_state,
        # which the buttons above may have just updated.
        horizon = st.number_input(
            "Custom", min_value=1, max_value=MAX_HORIZON, step=1,
            key="lstm_horizon", help="Type any value from 1 to 30.",
        )
    horizon = int(horizon)

    sdf = df[df[COLS.symbol] == symbol].sort_values(COLS.date)
    series = sdf[COLS.stock_close].astype(float).values
    dates = sdf[COLS.date].values
    if len(series) < WINDOW + 40:
        st.warning("Not enough history for this stock.")
        return

    has_tf = has_package("tensorflow")
    if not has_tf:
        st.info("⚙️ TensorFlow not installed — using a reliable windowed Ridge "
                "forecaster. Install with `pip install tensorflow` for true "
                f"{kind}.")

    with st.spinner(f"Training {kind if has_tf else 'fallback'} model for a "
                    f"{horizon}-day forecast…"):
        if has_tf:
            try:
                test_pred, test_true, future, used = _forecast_dl(series, kind, horizon)
            except Exception as exc:  # noqa: BLE001
                st.warning(f"Deep model failed ({exc}); using Ridge fallback.")
                test_pred, test_true, future, used = _forecast_ridge(series, horizon)
        else:
            test_pred, test_true, future, used = _forecast_ridge(series, horizon)

    st.caption(f"Model: **{used}** · Horizon: **{horizon} trading day(s)**")

    # ---- Metrics ----------------------------------------------------------
    m = regression_metrics(test_true, test_pred)
    k1, k2, k3 = st.columns(3)
    k1.metric("RMSE", f"{m['RMSE']:.2f}")
    k2.metric("MAE", f"{m['MAE']:.2f}")
    k3.metric("MAPE", f"{m['MAPE']:.2f}%")

    # ---- Forecast chart ---------------------------------------------------
    last_date = pd.to_datetime(dates[-1])
    future_dates = pd.bdate_range(last_date, periods=horizon + 1)[1:]
    hist_n = min(120, len(series))
    band = np.std(test_true - test_pred) if len(test_pred) else future.std()
    st.plotly_chart(
        viz.forecast_chart(
            dates[-hist_n:], series[-hist_n:], future_dates, future,
            lower=future - 1.96 * band, upper=future + 1.96 * band,
            title=f"{symbol} — Next {horizon}-Day Forecast",
        ),
        use_container_width=True,
    )

    # ---- Forecast table ---------------------------------------------------
    fc = pd.DataFrame({
        "Date": future_dates.date,
        "Forecast_Close": np.round(future, 2),
        "Change_%": np.round(
            np.r_[(future[0] / series[-1] - 1) * 100,
                  (future[1:] / future[:-1] - 1) * 100], 2),
    })
    st.markdown(f"#### {horizon}-Day Forecast")
    st.dataframe(fc, use_container_width=True)

    from utils.helper_functions import download_buttons
    download_buttons(fc, key="forecast", label=f"{symbol}_{horizon}d_forecast")