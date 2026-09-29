"""
modules/llm_explainer.py
------------------------
Local-LLM explanation engine (Ollama) for the AI Financial Intelligence Platform.

Purpose
    Aggregate the platform's analytics into one unified context and use a locally
    running Ollama model (e.g. ``llama3.2:1b``) to turn the technical outputs into
    plain-English, investor-friendly explanations.

Design notes (important & honest about this codebase)
    * The other modules' ``run(df, df_full)`` functions render Streamlit UI and
      return ``None`` — they do NOT return result dictionaries. This explainer
      therefore does **not** scrape their return values. Instead it offers two
      supported ways to obtain a unified ``results`` context:
        1. ``build_results_from_data(df)`` — a READ-ONLY collector that derives
           summary metrics (trend, risk, sentiment, drivers, stance, crash flag,
           long-term suitability) directly from the engineered dataframe using the
           project's existing utilities. No module logic is touched or modified.
        2. Pass your own pre-built ``results`` dict (the documented contract,
           e.g. ``{"prediction": {"prediction": "UP", "confidence": 0.81}, ...}``)
           and the explainer will validate, normalise and explain it.
    * Ollama is an OPTIONAL dependency. If the package or server is unavailable,
      every method degrades gracefully to a deterministic, non-LLM template
      summary so the page never crashes.

Clean architecture
    Collection  -> build_results_from_data() / collect_results()
    Aggregation -> build_context()
    Prompting   -> _build_prompt() with 4 templates (Exec / Detailed / Beginner / Risk)
    Inference   -> _chat() (Ollama client, fully guarded)
    Public API  -> generate_explanation / generate_short_summary / generate_detailed_report
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st

from config.settings import COLS
from utils.helper_functions import get_logger, score_sentiment
from utils.preprocessing import get_engineered_data

LOGGER = get_logger("ai_fip.llm_explainer")

# Ollama is optional — import lazily so the module always loads.
try:
    from ollama import Client as _OllamaClient  # type: ignore
    _HAS_OLLAMA = True
except Exception:  # noqa: BLE001
    _OllamaClient = None  # type: ignore
    _HAS_OLLAMA = False


# --------------------------------------------------------------------------- #
# Custom errors                                                               #
# --------------------------------------------------------------------------- #
class ExplainerError(Exception):
    """Base error for the explanation engine."""


class OllamaUnavailableError(ExplainerError):
    """Raised when the Ollama package or server cannot be reached."""


# --------------------------------------------------------------------------- #
# Report-type definitions                                                     #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ReportType:
    EXECUTIVE: str = "executive"
    DETAILED: str = "detailed"
    BEGINNER: str = "beginner"
    RISK: str = "risk"


REPORTS = ReportType()


# --------------------------------------------------------------------------- #
# READ-ONLY results collector (derives a unified context from the data)       #
# --------------------------------------------------------------------------- #
def _safe_last(series: pd.Series, default: float = float("nan")) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    return float(s.iloc[-1]) if len(s) else default


def build_results_from_data(df: Optional[pd.DataFrame] = None,
                            symbol: Optional[str] = None) -> Dict[str, Any]:
    """Build a unified analytics context for one symbol from engineered data.

    This is a READ-ONLY summariser: it computes descriptive statistics with the
    project's own utilities and does NOT call, modify or replicate any module's
    model logic. It exists so the explainer is usable without each module having
    to return a dictionary.

    Parameters
    ----------
    df : optional engineered dataframe; falls back to ``get_engineered_data()``.
    symbol : optional ticker; defaults to the first available symbol.

    Returns
    -------
    dict keyed by analytics area (eda, price_trend, forecast, risk, sentiment,
    feature_drivers, buy_sell_hold, crash_risk, long_term, sector_gold).
    """
    if df is None or df.empty:
        try:
            df = get_engineered_data()
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Could not load engineered data: %s", exc)
            return {}
    if df is None or df.empty or COLS.symbol not in df.columns:
        return {}

    symbols = sorted(df[COLS.symbol].dropna().unique().tolist())
    if not symbols:
        return {}
    symbol = symbol if symbol in symbols else symbols[0]

    s = df[df[COLS.symbol] == symbol].sort_values(COLS.date).reset_index(drop=True)
    close = pd.to_numeric(s[COLS.stock_close], errors="coerce")
    if close.dropna().empty:
        return {}

    last = _safe_last(close)
    company = str(s[COLS.company].iloc[-1]) if COLS.company in s else symbol
    industry = str(s[COLS.industry].iloc[-1]) if COLS.industry in s else "—"

    # ---- EDA snapshot ----
    eda = {
        "symbol": symbol, "company": company, "industry": industry,
        "rows": int(len(s)),
        "date_start": str(pd.to_datetime(s[COLS.date]).min().date()),
        "date_end": str(pd.to_datetime(s[COLS.date]).max().date()),
        "universe_symbols": int(df[COLS.symbol].nunique()),
        "industries": int(df[COLS.industry].nunique()) if COLS.industry in df else None,
    }

    # ---- Price trend ----
    sma20 = _safe_last(close.rolling(20, min_periods=1).mean())
    sma50 = _safe_last(close.rolling(50, min_periods=1).mean())
    sma200 = _safe_last(close.rolling(200, min_periods=20).mean())
    ret20 = float(close.iloc[-1] / close.iloc[-21] - 1.0) * 100 if len(close) > 21 else 0.0
    trend_dir = ("uptrend" if (np.isfinite(sma50) and last > sma50 and ret20 >= 0)
                 else "downtrend" if (np.isfinite(sma50) and last < sma50 and ret20 < 0)
                 else "sideways")
    price_trend = {
        "last_close": round(last, 2), "sma20": round(sma20, 2),
        "sma50": round(sma50, 2), "sma200": round(sma200, 2),
        "return_20d_pct": round(ret20, 2),
        "above_sma50": bool(np.isfinite(sma50) and last > sma50),
        "above_sma200": bool(np.isfinite(sma200) and last > sma200),
        "trend_direction": trend_dir,
    }

    # ---- Forecast outlook (descriptive drift, NOT the LSTM module) ----
    rets = pd.to_numeric(s.get("Daily_Return", close.pct_change()), errors="coerce").dropna()
    drift = float(rets.tail(60).mean()) if len(rets) else 0.0
    forecast = {
        "method": "recent-drift heuristic (descriptive summary, not the LSTM module)",
        "near_term_bias": "upward" if drift > 0 else "downward" if drift < 0 else "flat",
        "avg_daily_drift_pct": round(drift * 100, 3),
    }

    # ---- Risk ----
    ann_vol = float(rets.std() * np.sqrt(252)) if len(rets) > 5 else float("nan")
    max_dd = float((close / close.cummax() - 1.0).min()) if close.notna().any() else float("nan")
    risk_level = ("Low" if (np.isfinite(ann_vol) and ann_vol < 0.25)
                  else "High" if (np.isfinite(ann_vol) and ann_vol >= 0.45) else "Medium")
    risk = {
        "annualised_volatility_pct": round(ann_vol * 100, 1) if np.isfinite(ann_vol) else None,
        "max_drawdown_pct": round(max_dd * 100, 1) if np.isfinite(max_dd) else None,
        "risk_level": risk_level,
    }

    # ---- Sentiment (uses the project's existing lexicon scorer) ----
    sentiment: Dict[str, Any] = {"available": False}
    if COLS.headlines in s.columns:
        try:
            scores, engine = score_sentiment(
                s[COLS.headlines].fillna("").astype(str).tolist(), use_finbert=False)
            sc = pd.Series(scores)
            sentiment = {
                "available": True, "engine": engine,
                "mean_sentiment": round(float(sc.mean()), 3),
                "positive_days": int((sc > 0.05).sum()),
                "negative_days": int((sc < -0.05).sum()),
                "tone": ("positive" if sc.mean() > 0.02 else
                         "negative" if sc.mean() < -0.02 else "neutral"),
            }
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Sentiment scoring failed: %s", exc)

    # ---- Feature drivers (descriptive correlations with next-day return) ----
    drivers: Dict[str, float] = {}
    nxt = close.pct_change().shift(-1)
    for col in ["RSI", "MACD", "Momentum", "Volatility", "Volume_Change"]:
        if col in s.columns:
            try:
                c = float(pd.to_numeric(s[col], errors="coerce").corr(nxt))
                if np.isfinite(c):
                    drivers[col] = round(c, 3)
            except Exception:  # noqa: BLE001
                pass
    feature_drivers = {"corr_with_next_day_return": drivers}

    # ---- Buy/Sell/Hold (transparent heuristic summary) ----
    score = (int(price_trend["above_sma50"]) + int(price_trend["above_sma200"])
             + int(ret20 >= 0) + int(sentiment.get("mean_sentiment", 0) >= 0))
    stance = "BUY" if score >= 3 else "SELL" if score <= 1 else "HOLD"
    buy_sell_hold = {
        "signal": stance,
        "basis": "heuristic blend of MA position, 20-day momentum and news tone",
        "supporting_score": f"{score}/4",
    }

    # ---- Crash-risk indicators ----
    recent_vol = float(rets.tail(20).std() * np.sqrt(252)) if len(rets) > 20 else float("nan")
    vol_spike = bool(np.isfinite(recent_vol) and np.isfinite(ann_vol) and recent_vol > 1.5 * ann_vol)
    crash_risk = {
        "recent_vol_spike": vol_spike,
        "deep_drawdown": bool(np.isfinite(max_dd) and max_dd < -0.40),
        "flag": "elevated" if (vol_spike or (np.isfinite(max_dd) and max_dd < -0.40)) else "normal",
    }

    # ---- Long-term suitability (trend-slope sign over history) ----
    x = np.arange(len(close.dropna()))
    slope = float(np.polyfit(x, close.dropna().values, 1)[0]) if len(x) > 2 else 0.0
    long_term = {
        "long_run_slope_sign": "positive" if slope > 0 else "negative" if slope < 0 else "flat",
        "suitability": ("suitable for long-term holding" if slope > 0 and risk_level != "High"
                        else "requires caution for long-term holding"),
    }

    # ---- Sector / gold link ----
    sector_gold: Dict[str, Any] = {"available": False}
    if COLS.gold_close in s.columns:
        gold = pd.to_numeric(s[COLS.gold_close], errors="coerce")
        if gold.notna().sum() > 30:
            gc = float(close.corr(gold))
            sector_gold = {
                "available": True,
                "stock_gold_correlation": round(gc, 2) if np.isfinite(gc) else None,
                "interpretation": ("moves with gold" if np.isfinite(gc) and gc >= 0.3
                                   else "moves against gold" if np.isfinite(gc) and gc <= -0.3
                                   else "little link to gold"),
            }

    return {
        "eda": eda, "price_trend": price_trend, "forecast": forecast, "risk": risk,
        "sentiment": sentiment, "feature_drivers": feature_drivers,
        "buy_sell_hold": buy_sell_hold, "crash_risk": crash_risk,
        "long_term": long_term, "sector_gold": sector_gold,
    }


# --------------------------------------------------------------------------- #
# The explainer                                                               #
# --------------------------------------------------------------------------- #
@dataclass
class StockAnalysisExplainer:
    """Turn aggregated analytics into investor-friendly explanations via Ollama.

    Parameters
    ----------
    model_name : Ollama model tag, e.g. ``"llama3.2:1b"``.
    host : Ollama server URL.
    timeout : seconds before a generation attempt is abandoned.
    temperature : sampling temperature for the model.
    """

    model_name: str = "llama3.2:1b"
    host: str = "http://localhost:11434"
    timeout: int = 120
    temperature: float = 0.3
    _client: Any = field(default=None, init=False, repr=False)

    # ---------------------- collection & aggregation ----------------------
    def collect_results(self, raw: Optional[Dict[str, Any]] = None,
                        df: Optional[pd.DataFrame] = None,
                        symbol: Optional[str] = None) -> Dict[str, Any]:
        """Return a unified results dict.

        If ``raw`` (a pre-built results dict) is supplied it is validated and
        returned; otherwise a read-only context is built from the data.
        """
        if raw:
            if not isinstance(raw, dict):
                raise ExplainerError("`raw` results must be a dictionary.")
            cleaned = {k: v for k, v in raw.items() if v not in (None, {}, [])}
            if not cleaned:
                LOGGER.warning("Provided results were empty; falling back to data.")
                return build_results_from_data(df, symbol)
            return cleaned
        return build_results_from_data(df, symbol)

    def build_context(self, results: Dict[str, Any]) -> str:
        """Serialise the unified results into a compact JSON context string."""
        if not results:
            raise ExplainerError("No analytics results available to explain.")
        return json.dumps(results, default=str, ensure_ascii=False, indent=2)

    # ---------------------------- prompting -------------------------------
    _SYSTEM = (
        "You are a professional financial analyst assistant writing for ordinary, "
        "non-technical investors. Explain findings in simple, plain language. Avoid "
        "jargon (or define it in one short phrase). Always explain risks and "
        "uncertainty, reference the supporting evidence from the analytics provided, "
        "and give an actionable interpretation. Never guarantee future returns and "
        "never invent numbers that are not in the data. End with a one-line reminder "
        "that this is educational information, not financial advice."
    )

    _TEMPLATES = {
        REPORTS.EXECUTIVE: (
            "Write a concise EXECUTIVE SUMMARY (about 120-160 words) covering the "
            "overall market condition, price trend, forecast bias, risk level, news "
            "sentiment and the buy/hold/sell stance with a one-sentence rationale."
        ),
        REPORTS.DETAILED: (
            "Write a DETAILED EXPLANATION with short labelled sections: Market "
            "Condition, Price Trend, Forecast Outlook, Risk Level, Sentiment Impact, "
            "Key Drivers, Buy/Sell/Hold Rationale, Crash-Risk Indicators, and "
            "Long-Term Suitability. Keep each section to 2-3 plain sentences."
        ),
        REPORTS.BEGINNER: (
            "Write a BEGINNER-FRIENDLY explanation as if to someone new to investing. "
            "Use everyday analogies, no technical terms, short paragraphs, and explain "
            "what the numbers mean for a regular person deciding what to do."
        ),
        REPORTS.RISK: (
            "Write an INVESTMENT RISK ASSESSMENT focused on what could go wrong: "
            "volatility, drawdown, crash-risk flags and trend risk. Explain the "
            "uncertainty clearly and who this stock is or is not suitable for."
        ),
    }

    def _build_prompt(self, results: Dict[str, Any], kind: str) -> str:
        """Compose the full prompt for a given report type."""
        kind = kind if kind in self._TEMPLATES else REPORTS.DETAILED
        context = self.build_context(results)
        return (
            f"{self._SYSTEM}\n\n"
            f"TASK:\n{self._TEMPLATES[kind]}\n\n"
            f"ANALYTICS CONTEXT (JSON — use only these facts):\n{context}\n\n"
            f"Begin the {kind} explanation now:"
        )

    # ---------------------------- inference -------------------------------
    def _get_client(self):
        if not _HAS_OLLAMA:
            raise OllamaUnavailableError(
                "The `ollama` Python package is not installed. Run "
                "`pip install ollama` and ensure the Ollama app is running.")
        if self._client is None:
            self._client = _OllamaClient(host=self.host, timeout=self.timeout)
        return self._client

    def _chat(self, prompt: str) -> str:
        """Send the prompt to Ollama with full exception handling."""
        client = self._get_client()
        t0 = time.time()
        try:
            resp = client.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": self.temperature},
            )
        except Exception as exc:  # noqa: BLE001
            msg = str(exc).lower()
            if "model" in msg and ("not found" in msg or "no such" in msg or "pull" in msg):
                raise OllamaUnavailableError(
                    f"Model '{self.model_name}' is not available. Pull it first with "
                    f"`ollama pull {self.model_name}`.") from exc
            if "connect" in msg or "connection" in msg or "refused" in msg or "11434" in msg:
                raise OllamaUnavailableError(
                    "Could not reach the Ollama server at "
                    f"{self.host}. Start it with `ollama serve`.") from exc
            raise ExplainerError(f"Ollama generation failed: {exc}") from exc

        LOGGER.info("Ollama generation took %.1fs", time.time() - t0)
        text = (resp or {}).get("message", {}).get("content", "").strip()
        if not text:
            raise ExplainerError("Ollama returned an empty response.")
        return text

    # ---------------------------- fallback --------------------------------
    @staticmethod
    def _fallback(results: Dict[str, Any], kind: str) -> str:
        """Deterministic, non-LLM summary used when Ollama is unavailable."""
        pt = results.get("price_trend", {})
        rk = results.get("risk", {})
        se = results.get("sentiment", {})
        bsh = results.get("buy_sell_hold", {})
        cr = results.get("crash_risk", {})
        eda = results.get("eda", {})
        name = eda.get("company") or eda.get("symbol") or "This stock"
        lines = [
            f"{name} is currently in a {pt.get('trend_direction', 'n/a')} "
            f"(last close {pt.get('last_close', 'n/a')}, 20-day move "
            f"{pt.get('return_20d_pct', 'n/a')}%).",
            f"Risk level is {rk.get('risk_level', 'n/a')} "
            f"(volatility {rk.get('annualised_volatility_pct', 'n/a')}%, "
            f"max drawdown {rk.get('max_drawdown_pct', 'n/a')}%).",
        ]
        if se.get("available"):
            lines.append(f"Recent news tone is {se.get('tone', 'neutral')} "
                         f"(score {se.get('mean_sentiment', 0)}).")
        lines.append(f"Heuristic stance: {bsh.get('signal', 'HOLD')} "
                     f"({bsh.get('supporting_score', '')}).")
        if cr.get("flag") == "elevated":
            lines.append("Crash-risk indicators are elevated — treat with caution.")
        lines.append("Note: Ollama was unavailable, so this is a basic templated "
                     "summary. This is educational information, not financial advice.")
        return " ".join(lines)

    # ---------------------------- public API ------------------------------
    def generate_explanation(self, results: Dict[str, Any],
                             kind: str = REPORTS.DETAILED) -> str:
        """Generate an explanation of the given type, with graceful fallback."""
        if not results:
            return "No analytics results were available to explain."
        try:
            return self._chat(self._build_prompt(results, kind))
        except OllamaUnavailableError as exc:
            LOGGER.warning("Ollama unavailable: %s", exc)
            return f"⚠️ {exc}\n\n{self._fallback(results, kind)}"
        except ExplainerError as exc:
            LOGGER.error("Explanation error: %s", exc)
            return f"⚠️ {exc}\n\n{self._fallback(results, kind)}"

    def generate_short_summary(self, results: Dict[str, Any]) -> str:
        """Executive-style short summary."""
        return self.generate_explanation(results, REPORTS.EXECUTIVE)

    def generate_detailed_report(self, results: Dict[str, Any]) -> str:
        """Full detailed report."""
        return self.generate_explanation(results, REPORTS.DETAILED)


# --------------------------------------------------------------------------- #
# Streamlit page (so the feature is usable inside the app immediately)        #
# --------------------------------------------------------------------------- #
def run(df: Optional[pd.DataFrame] = None, df_full: Optional[pd.DataFrame] = None) -> None:
    """Streamlit entry point: pick a stock, choose a report style, generate."""
    try:
        from utils.helper_functions import page_header
        page_header("AI Explanation Engine",
                    "Plain-English investor reports from your analytics, via a local Llama model", "💬")
    except Exception:  # noqa: BLE001
        st.title("💬 AI Explanation Engine")

    source = df_full if df_full is not None and not df_full.empty else df
    if source is None or source.empty:
        try:
            source = get_engineered_data()
        except Exception:  # noqa: BLE001
            source = pd.DataFrame()
    if source is None or source.empty:
        st.warning("No data available to explain.")
        return

    symbols = sorted(source[COLS.symbol].unique().tolist())
    c1, c2, c3 = st.columns([2, 1.4, 1.2])
    with c1:
        symbol = st.selectbox("Stock", symbols, key="llm_symbol")
    with c2:
        report_label = st.selectbox("Report style",
                                    ["Executive Summary", "Detailed Explanation",
                                     "Beginner-Friendly", "Risk Assessment"],
                                    key="llm_report")
    with c3:
        model_name = st.text_input("Ollama model", value="llama3.2:1b", key="llm_model")

    kind = {"Executive Summary": REPORTS.EXECUTIVE,
            "Detailed Explanation": REPORTS.DETAILED,
            "Beginner-Friendly": REPORTS.BEGINNER,
            "Risk Assessment": REPORTS.RISK}[report_label]

    if not _HAS_OLLAMA:
        st.info("ℹ️ The `ollama` package isn't installed, so a basic templated "
                "summary will be shown. Install with `pip install ollama` and run "
                "`ollama serve` + `ollama pull llama3.2:1b` for full AI explanations.")

    results = build_results_from_data(source, symbol)
    if not results:
        st.warning("Could not build an analytics context for this stock.")
        return

    if st.button("Generate explanation", key="llm_go", type="primary"):
        explainer = StockAnalysisExplainer(model_name=model_name)
        with st.spinner("Generating explanation…"):
            text = explainer.generate_explanation(results, kind)
        st.markdown(f"<div class='glass'>{text}</div>", unsafe_allow_html=True)

    with st.expander("Aggregated analytics context (JSON)"):
        st.json(results)