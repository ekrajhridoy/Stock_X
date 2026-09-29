"""
utils/feature_engineering.py
----------------------------
Pure-pandas/numpy technical-indicator engineering. Every indicator is computed
**per symbol** (group-wise) so signals never leak across different stocks.

Indicators
----------
* Daily_Return, Log_Return
* SMA / EMA moving averages
* RSI (Wilder)
* MACD + signal + histogram
* Rolling volatility
* Momentum
* Lag features

All functions are cached-friendly (no Streamlit dependency) and return new
dataframes — inputs are never mutated.
"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from config.settings import COLS


# --------------------------------------------------------------------------- #
# Individual indicators (operate on a single price Series)                    #
# --------------------------------------------------------------------------- #
def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)


def compute_macd(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> pd.DataFrame:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - macd_signal
    return pd.DataFrame({"MACD": macd, "MACD_Signal": macd_signal, "MACD_Hist": hist})


def compute_volatility(close: pd.Series, window: int = 21) -> pd.Series:
    ret = close.pct_change()
    return ret.rolling(window, min_periods=2).std() * np.sqrt(252)


# --------------------------------------------------------------------------- #
# Group-wise engineering over the whole dataset                               #
# --------------------------------------------------------------------------- #
def add_technical_indicators(
    df: pd.DataFrame,
    sma_windows: List[int] = (5, 10, 20, 50),
    lags: List[int] = (1, 2, 3, 5),
) -> pd.DataFrame:
    """
    Add the full indicator suite, grouped by Symbol.

    Resulting new columns include:
        Daily_Return, Log_Return, SMA_*, EMA_12, RSI, MACD, MACD_Signal,
        MACD_Hist, Volatility, Momentum, Volume_Change, Close_lag_*
    """
    if df.empty or COLS.stock_close not in df.columns:
        return df.copy()

    out = df.sort_values([COLS.symbol, COLS.date]).copy()
    g = out.groupby(COLS.symbol, group_keys=False)
    close = COLS.stock_close

    out["Daily_Return"] = g[close].pct_change()
    out["Log_Return"] = g[close].transform(lambda s: np.log(s / s.shift(1)))

    for w in sma_windows:
        out[f"SMA_{w}"] = g[close].transform(
            lambda s, _w=w: s.rolling(_w, min_periods=1).mean()
        )
    out["EMA_12"] = g[close].transform(lambda s: s.ewm(span=12, adjust=False).mean())
    out["EMA_26"] = g[close].transform(lambda s: s.ewm(span=26, adjust=False).mean())

    out["RSI"] = g[close].transform(compute_rsi)

    macd_parts = g[close].apply(compute_macd)
    if isinstance(macd_parts.index, pd.MultiIndex):
        macd_parts = macd_parts.reset_index(level=0, drop=True)
    out = out.join(macd_parts)

    out["Volatility"] = g[close].transform(compute_volatility)
    out["Momentum"] = g[close].transform(lambda s: s - s.shift(10))

    if COLS.stock_volume in out.columns:
        out["Volume_Change"] = g[COLS.stock_volume].pct_change()

    for lag in lags:
        out[f"Close_lag_{lag}"] = g[close].shift(lag)

    # Clean up infinities / NaNs introduced by diffs and divisions.
    num_cols = out.select_dtypes(include=[np.number]).columns
    out[num_cols] = out[num_cols].replace([np.inf, -np.inf], np.nan)
    out[num_cols] = (
        out.groupby(COLS.symbol)[num_cols].transform(lambda s: s.ffill().bfill())
    )
    out[num_cols] = out[num_cols].fillna(0.0)
    return out


def make_direction_target(df: pd.DataFrame) -> pd.DataFrame:
    """Add `Target_Up` = 1 if next day's close > today's close (per symbol)."""
    out = df.copy()
    nxt = out.groupby(COLS.symbol)[COLS.stock_close].shift(-1)
    out["Tomorrow_Close"] = nxt
    out["Target_Up"] = (nxt > out[COLS.stock_close]).astype(int)
    return out


DEFAULT_FEATURES: List[str] = [
    COLS.stock_open, COLS.stock_high, COLS.stock_low, COLS.stock_close,
    COLS.stock_volume, COLS.stock_vwap, "Daily_Return", COLS.gold_close,
    "Volatility", "RSI", "MACD", "MACD_Signal", "Momentum",
    "SMA_5", "SMA_20",
]


def available_features(df: pd.DataFrame, candidates: List[str] = None) -> List[str]:
    """Return the subset of feature columns that actually exist in `df`."""
    candidates = candidates or DEFAULT_FEATURES
    return [c for c in candidates if c in df.columns]
