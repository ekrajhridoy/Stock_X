"""
utils/helper_functions.py
-------------------------
Cross-cutting helpers used throughout the platform:

* Optional-dependency detection (the whole app degrades gracefully when a heavy
  library such as TensorFlow / FinBERT / Prophet is not installed).
* Logging setup.
* Model save / load.
* Metric helpers.
* CSV / Excel export utilities.
* Small Streamlit UI helpers (notices, downloads).

Nothing here ever raises on import: missing optional libraries are reported as
flags, never exceptions, which is what keeps `streamlit run main.py` crash-free
regardless of the user's environment.
"""
from __future__ import annotations

import importlib
import io
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Logging                                                                     #
# --------------------------------------------------------------------------- #
_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def get_logger(name: str = "ai_fip") -> logging.Logger:
    """Return a configured module-level logger (idempotent)."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


LOGGER = get_logger()


# --------------------------------------------------------------------------- #
# Optional dependency detection                                               #
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=None)
def has_package(name: str) -> bool:
    """True if `name` can be imported. Cached, never raises."""
    try:
        importlib.import_module(name)
        return True
    except Exception:  # noqa: BLE001 - any import failure means "unavailable"
        return False


# Convenience flags (evaluated lazily through the function above).
def deps() -> Dict[str, bool]:
    """Snapshot of which optional engines are available in this environment."""
    return {
        "xgboost": has_package("xgboost"),
        "lightgbm": has_package("lightgbm"),
        "catboost": has_package("catboost"),
        "shap": has_package("shap"),
        "lime": has_package("lime"),
        "tensorflow": has_package("tensorflow"),
        "torch": has_package("torch"),
        "transformers": has_package("transformers"),
        "prophet": has_package("prophet"),
        "statsmodels": has_package("statsmodels"),
        "tslearn": has_package("tslearn"),
        "stable_baselines3": has_package("stable_baselines3"),
    }


def require_notice(label: str, pip_name: Optional[str] = None) -> str:
    """Human-friendly message shown in the UI when an optional engine is absent."""
    pip_name = pip_name or label
    return (
        f"⚙️ Optional engine **{label}** is not installed, so a reliable "
        f"built-in fallback is being used instead. "
        f"To enable the full engine run: `pip install {pip_name}`"
    )


# --------------------------------------------------------------------------- #
# Model persistence                                                           #
# --------------------------------------------------------------------------- #
def save_model(model: Any, name: str, models_dir: Path) -> Path:
    """Persist a model with joblib. Returns the path written."""
    import joblib

    models_dir.mkdir(parents=True, exist_ok=True)
    path = models_dir / f"{name}.joblib"
    joblib.dump(model, path)
    LOGGER.info("Saved model -> %s", path)
    return path


def load_model(name: str, models_dir: Path) -> Optional[Any]:
    """Load a previously saved model, or None if it does not exist."""
    import joblib

    path = models_dir / f"{name}.joblib"
    if path.exists():
        try:
            return joblib.load(path)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Could not load %s: %s", path, exc)
    return None


# --------------------------------------------------------------------------- #
# Metrics                                                                     #
# --------------------------------------------------------------------------- #
def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """RMSE / MAE / MAPE for forecasting tasks. Robust to zeros and NaNs."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]
    if y_true.size == 0:
        return {"RMSE": float("nan"), "MAE": float("nan"), "MAPE": float("nan")}
    err = y_pred - y_true
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    denom = np.where(np.abs(y_true) < 1e-8, np.nan, y_true)
    mape = float(np.nanmean(np.abs(err / denom)) * 100.0)
    return {"RMSE": rmse, "MAE": mae, "MAPE": mape}


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Accuracy / precision / recall / F1 (binary or multiclass)."""
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
    )

    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


# --------------------------------------------------------------------------- #
# Export utilities                                                            #
# --------------------------------------------------------------------------- #
def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def df_to_excel_bytes(df: pd.DataFrame, sheet_name: str = "data") -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name[:31])
    return buffer.getvalue()


def download_buttons(df: pd.DataFrame, key: str, label: str = "data") -> None:
    """Render CSV + Excel download buttons for a dataframe (Streamlit)."""
    import streamlit as st

    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "⬇️ Export CSV",
            data=df_to_csv_bytes(df),
            file_name=f"{label}.csv",
            mime="text/csv",
            use_container_width=True,
            key=f"csv_{key}",
        )
    with c2:
        st.download_button(
            "⬇️ Export Excel",
            data=df_to_excel_bytes(df, sheet_name=label),
            file_name=f"{label}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
            key=f"xlsx_{key}",
        )


def get_engineered_data_placeholder() -> None:  # pragma: no cover
    """Reserved hook (kept for symmetry / future use)."""
    return None


# --------------------------------------------------------------------------- #
# Sentiment engine (FinBERT when available, lexicon fallback otherwise)       #
# --------------------------------------------------------------------------- #
_POS_WORDS = {
    "gain", "gains", "rise", "rises", "rose", "up", "surge", "surges", "growth",
    "grow", "profit", "profits", "boost", "rally", "rallies", "high", "higher",
    "beat", "beats", "strong", "stronger", "optimistic", "recovery", "approve",
    "approved", "upgrade", "bullish", "soar", "soars", "jump", "jumps", "win",
    "wins", "record", "expand", "expands", "outperform", "positive", "rebound",
}
_NEG_WORDS = {
    "fall", "falls", "fell", "drop", "drops", "down", "loss", "losses", "slump",
    "slumps", "weak", "weaker", "fear", "fears", "crisis", "cut", "cuts",
    "decline", "declines", "warn", "warns", "risk", "risks", "slip", "slips",
    "fails", "fail", "blast", "plunge", "plunges", "downgrade", "bearish",
    "crash", "crashes", "default", "fraud", "probe", "lawsuit", "negative",
}


@lru_cache(maxsize=1)
def _finbert_pipeline():
    """Load FinBERT once. Returns the HF pipeline or None on any failure."""
    if not (has_package("transformers") and (has_package("torch")
                                              or has_package("tensorflow"))):
        return None
    try:
        from transformers import pipeline

        return pipeline("sentiment-analysis", model="ProsusAI/finbert",
                        truncation=True)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("FinBERT unavailable, using lexicon fallback: %s", exc)
        return None


def lexicon_sentiment(text: str) -> float:
    """Return sentiment in [-1, 1] using a financial word lexicon."""
    if not text:
        return 0.0
    words = [w.strip(".,/:;()[]\"'").lower() for w in str(text).split()]
    p = sum(w in _POS_WORDS for w in words)
    n = sum(w in _NEG_WORDS for w in words)
    return (p - n) / (p + n) if (p + n) else 0.0


def score_sentiment(texts, use_finbert: bool = False, max_finbert: int = 300):
    """
    Score an iterable of texts to floats in [-1, 1].

    When `use_finbert` is True and FinBERT is importable, the first
    `max_finbert` texts are scored with FinBERT; everything else (and the whole
    series when FinBERT is unavailable) uses the fast lexicon scorer.

    Returns (scores: np.ndarray, engine_name: str).
    """
    texts = list(texts)
    if use_finbert:
        pipe = _finbert_pipeline()
        if pipe is not None:
            scores = []
            for i, t in enumerate(texts):
                if i >= max_finbert:
                    scores.append(lexicon_sentiment(t))
                    continue
                snippet = (str(t) or "")[:512]
                if not snippet.strip():
                    scores.append(0.0)
                    continue
                try:
                    res = pipe(snippet)[0]
                    lab = res["label"].lower()
                    sc = res["score"]
                    scores.append(sc if lab == "positive"
                                  else (-sc if lab == "negative" else 0.0))
                except Exception:  # noqa: BLE001
                    scores.append(lexicon_sentiment(t))
            return np.array(scores), "FinBERT"
    return np.array([lexicon_sentiment(t) for t in texts]), "Lexicon (built-in)"


def page_header(title: str, subtitle: str = "", icon: str = "📈") -> None:
    """Render a consistent gradient page header."""
    import streamlit as st

    st.markdown(
        f"""
        <div class="page-header">
            <div class="page-header-icon">{icon}</div>
            <div>
                <div class="page-header-title">{title}</div>
                <div class="page-header-sub">{subtitle}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def info_banner(message: str, kind: str = "info") -> None:
    """Small inline notice (info / success / warning)."""
    import streamlit as st

    {"info": st.info, "success": st.success, "warning": st.warning,
     "error": st.error}.get(kind, st.info)(message)


def safe_sample(df: pd.DataFrame, max_rows: int, random_state: int = 42) -> pd.DataFrame:
    """Return at most `max_rows` rows, sampling only when necessary."""
    if len(df) > max_rows:
        return df.sample(max_rows, random_state=random_state).sort_index()
    return df
