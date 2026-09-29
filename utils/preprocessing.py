"""
utils/preprocessing.py
----------------------
Data ingestion and preparation:

* Cached, encoding-robust CSV loading (`load_data`).
* Missing-value handling.
* Categorical encoding.
* Feature scaling.
* Chronological & random train/test splits.

`load_data` is decorated with Streamlit's cache so the 120k-row file is parsed
once per session, keeping every page responsive.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from config.settings import COLS, DATA_ENCODING, DATA_FILE
from utils.helper_functions import LOGGER

# Streamlit is optional at import time so the module is unit-testable headless.
try:
    import streamlit as st

    _cache = st.cache_data(show_spinner=False)
except Exception:  # noqa: BLE001
    def _cache(func):  # type: ignore
        return func


def _read_csv_robust(path: Path) -> pd.DataFrame:
    """Read the CSV trying a couple of encodings before giving up."""
    for enc in (DATA_ENCODING, "utf-8", "utf-8-sig", "cp1252"):
        try:
            return pd.read_csv(path, encoding=enc)
        except (UnicodeDecodeError, UnicodeError):
            continue
    # Last resort: replace undecodable bytes so we never crash.
    return pd.read_csv(path, encoding="latin-1", encoding_errors="replace")


@_cache
def load_data(path: Optional[str] = None) -> pd.DataFrame:
    """
    Load, type-normalise and sort the merged market dataset.

    Returns a clean dataframe with a parsed `Date` column, sorted by
    (Symbol, Date) which is the assumption every downstream module relies on.
    """
    file_path = Path(path) if path else DATA_FILE
    if not file_path.exists():
        LOGGER.error("Data file not found at %s", file_path)
        return pd.DataFrame()

    df = _read_csv_robust(file_path)

    # Parse dates.
    if COLS.date in df.columns:
        df[COLS.date] = pd.to_datetime(df[COLS.date], errors="coerce")

    # Coerce numeric columns (everything except known text columns).
    text_cols = {COLS.date, COLS.symbol, COLS.company, COLS.industry,
                 COLS.isin, COLS.headlines}
    for col in df.columns:
        if col not in text_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Tidy text.
    if COLS.headlines in df.columns:
        df[COLS.headlines] = df[COLS.headlines].fillna("").astype(str)

    df = df.dropna(subset=[COLS.date, COLS.symbol]).copy()
    df = df.sort_values([COLS.symbol, COLS.date]).reset_index(drop=True)
    LOGGER.info("Loaded dataset: %s rows x %s cols", *df.shape)
    return df


def handle_missing_values(df: pd.DataFrame, strategy: str = "ffill") -> pd.DataFrame:
    """Fill missing numeric values per-symbol then globally."""
    out = df.copy()
    num_cols = out.select_dtypes(include=[np.number]).columns
    if COLS.symbol in out.columns:
        out[num_cols] = (
            out.groupby(COLS.symbol)[num_cols]
            .transform(lambda s: s.ffill().bfill() if strategy == "ffill"
                       else s.fillna(s.median()))
        )
    out[num_cols] = out[num_cols].fillna(out[num_cols].median())
    out[num_cols] = out[num_cols].fillna(0.0)
    return out


def encode_categoricals(
    df: pd.DataFrame, columns: List[str]
) -> Tuple[pd.DataFrame, dict]:
    """Integer-encode categorical columns. Returns (df, mapping)."""
    out = df.copy()
    mappings: dict = {}
    for col in columns:
        if col in out.columns:
            cats = out[col].astype("category")
            mappings[col] = dict(enumerate(cats.cat.categories))
            out[col + "_enc"] = cats.cat.codes
    return out, mappings


def scale_features(
    df: pd.DataFrame, columns: List[str], method: str = "standard"
) -> Tuple[pd.DataFrame, object]:
    """Scale selected numeric columns. Returns (scaled_df_copy, fitted_scaler)."""
    from sklearn.preprocessing import MinMaxScaler, StandardScaler

    scaler = StandardScaler() if method == "standard" else MinMaxScaler()
    out = df.copy()
    valid = [c for c in columns if c in out.columns]
    if valid:
        out[valid] = scaler.fit_transform(out[valid].fillna(0.0))
    return out, scaler


def chronological_split(
    df: pd.DataFrame, test_ratio: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split keeping time order (no shuffling) — correct for time-series."""
    n = len(df)
    cut = int(n * (1 - test_ratio))
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()


def train_test_split_xy(
    X: np.ndarray, y: np.ndarray, test_ratio: float = 0.2,
    shuffle: bool = False, random_state: int = 42,
):
    """Thin wrapper around sklearn's split with time-series-safe defaults."""
    from sklearn.model_selection import train_test_split

    return train_test_split(
        X, y, test_size=test_ratio, shuffle=shuffle, random_state=random_state
    )


def get_engineered_data() -> pd.DataFrame:
    """
    Load + feature-engineer the dataset (cached). Used as a fallback inside
    feature modules so each `run()` works even when called with no arguments.
    """
    df = load_data()
    if df.empty:
        return df
    from utils.feature_engineering import (
        add_technical_indicators,
        make_direction_target,
    )

    df = add_technical_indicators(df)
    df = make_direction_target(df)
    return df


#def filter_dataframe(
 #   df: pd.DataFrame,
  #  companies: Optional[List[str]] = None,
   # industries: Optional[List[str]] = None,
    #date_range: Optional[Tuple] = None,
    #gold_range: Optional[Tuple[float, float]] = None,
    #volume_range: Optional[Tuple[float, float]] = None,
#) -> pd.DataFrame:
 #   """Apply the global dashboard filters. Empty / None filters are ignored."""
  #  out = df
   # if companies:
    #    out = out[out[COLS.company].isin(companies)]
    #if industries:
     #   out = out[out[COLS.industry].isin(industries)]
    #if date_range and len(date_range) == 2 and all(date_range):
     #   start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
      #  out = out[(out[COLS.date] >= start) & (out[COLS.date] <= end)]
    #if gold_range:
     #   out = out[(out[COLS.gold_close] >= gold_range[0]) &
      #            (out[COLS.gold_close] <= gold_range[1])]
    #if volume_range:
    #    out = out[(out[COLS.stock_volume] >= volume_range[0]) &
     #             (out[COLS.stock_volume] <= volume_range[1])]
    #return out.reset_index(drop=True)
def filter_dataframe(
    df: pd.DataFrame,
    companies: Optional[List[str]] = None,
    industries: Optional[List[str]] = None,
    date_range: Optional[Tuple] = None,
    gold_range: Optional[Tuple[float, float]] = None,
    volume_range: Optional[Tuple[float, float]] = None,
) -> pd.DataFrame:
    """Apply the global dashboard filters. Empty / None filters are ignored.

    Builds one combined boolean mask and materialises the result once, instead
    of re-copying the whole frame per condition (which could exhaust RAM).
    """
    if df.empty:
        return df

    mask = pd.Series(True, index=df.index)
    applied = False
    if companies:
        mask &= df[COLS.company].isin(companies); applied = True
    if industries:
        mask &= df[COLS.industry].isin(industries); applied = True
    if date_range and len(date_range) == 2 and all(date_range):
        start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        mask &= (df[COLS.date] >= start) & (df[COLS.date] <= end); applied = True
    if gold_range:
        mask &= ((df[COLS.gold_close] >= gold_range[0]) &
                 (df[COLS.gold_close] <= gold_range[1])); applied = True
    if volume_range:
        mask &= ((df[COLS.stock_volume] >= volume_range[0]) &
                 (df[COLS.stock_volume] <= volume_range[1])); applied = True

    if not applied:
        out = df.copy(deep=False)
        out.index = pd.RangeIndex(len(out))
        return out

    out = df.loc[mask]
    out.index = pd.RangeIndex(len(out))
    return out
