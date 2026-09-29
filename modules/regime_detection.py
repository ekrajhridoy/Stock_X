"""
modules/regime_detection.py
---------------------------
Market Regime Detection.

Answers:
    "What hidden market regime is this stock in right now — a calm uptrend, a
     volatile/bearish phase, or a sideways drift — and what has that historically
     implied?"

Model
    A Hidden Markov Model (``hmmlearn.GaussianHMM``) is trained UNSUPERVISED on
    each stock's [Daily_Return, Volatility] sequence. The HMM learns hidden
    states, their Gaussian emission means/variances, and the transition
    probabilities between them. States are then mapped to human labels
    (Calm Bull / Volatile-Bearish / Sideways) by their return & volatility
    signatures.

    If ``hmmlearn`` is not installed, the module falls back to ``KMeans`` over the
    same two features (a transparent, dependency-light substitute) and estimates
    an empirical transition matrix from the resulting state sequence. The active
    engine is shown to the user.

Outputs
    * Current regime + confidence badge (reuses badge-buy/hold/sell styling)
    * Regime-coloured price timeline
    * Transition-probability matrix heatmap
    * Per-regime statistics (avg return, volatility, frequency, avg duration)
    * Inline "Explanation of the Results" via the local Llama explainer

Architecture: identical to the other modules — a single ``run(df, df_full)``
entry point, project helpers, shared COLS/PALETTE config, themed viz, defensive
imports and graceful fallbacks so the page never crashes.

Libraries: pandas, numpy, plotly, streamlit, scikit-learn (hmmlearn optional).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config.settings import COLS, PALETTE
from utils import visualizations as viz
from utils.helper_functions import download_buttons, has_package, page_header
from utils.preprocessing import get_engineered_data

# Optional inline LLM explanation (graceful if unavailable).
try:
    from modules.llm_explainer import render_llm_explanation
    _HAS_LLM = True
except Exception:  # noqa: BLE001
    _HAS_LLM = False

_HAS_HMM = has_package("hmmlearn")
_HAS_SKLEARN = has_package("sklearn")

N_REGIMES = 3
_MIN_ROWS = 120
_RANDOM_STATE = 42


# --------------------------------------------------------------------------- #
# Feature preparation                                                         #
# --------------------------------------------------------------------------- #
def _prepare_features(s: pd.DataFrame, close: pd.Series) -> Tuple[np.ndarray, pd.Series]:
    """Build the [Daily_Return, Volatility] feature matrix for a single stock.

    Falls back to computed values when the engineered columns are missing, and
    standardises each column so neither feature dominates the model.
    """
    if "Daily_Return" in s.columns:
        ret = pd.to_numeric(s["Daily_Return"], errors="coerce")
    else:
        ret = close.pct_change()

    if "Volatility" in s.columns:
        vol = pd.to_numeric(s["Volatility"], errors="coerce")
    else:
        vol = ret.rolling(20, min_periods=5).std()

    feat = pd.DataFrame({"ret": ret, "vol": vol})
    valid = feat.replace([np.inf, -np.inf], np.nan).dropna()
    if valid.empty:
        return np.empty((0, 2)), valid.index.to_series()

    X = valid.to_numpy(dtype=float)
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd < 1e-9] = 1.0
    X_std = (X - mu) / sd
    return X_std, valid.index.to_series()


# --------------------------------------------------------------------------- #
# Model fitting (HMM with KMeans fallback)                                     #
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def _fit_regimes(symbol: str, X: np.ndarray, n_states: int) -> Dict[str, object]:
    """Fit a regime model and return states, transition matrix and engine name.

    Returns a dict with:
        states     : int array of regime ids per observation
        trans      : (n,n) transition-probability matrix (rows sum to 1)
        engine     : "Hidden Markov Model" or "KMeans (fallback)"
        confidence : posterior/decision confidence for the LAST observation
    """
    n = X.shape[0]
    n_states = int(max(2, min(n_states, max(2, n // 30))))

    # ---- Primary: Gaussian HMM ----
    if _HAS_HMM and n >= _MIN_ROWS:
        try:
            from hmmlearn.hmm import GaussianHMM
            model = GaussianHMM(n_components=n_states, covariance_type="diag",
                                n_iter=200, random_state=_RANDOM_STATE)
            model.fit(X)
            states = model.predict(X)
            trans = np.asarray(model.transmat_, dtype=float)
            # Posterior confidence for the most recent observation.
            post = model.predict_proba(X)
            confidence = float(post[-1].max())
            return {"states": states, "trans": _normalise_rows(trans),
                    "engine": "Hidden Markov Model", "confidence": confidence}
        except Exception:  # noqa: BLE001 — fall through to KMeans
            pass

    # ---- Fallback: KMeans + empirical transition matrix ----
    if _HAS_SKLEARN:
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=n_states, n_init=10, random_state=_RANDOM_STATE)
        states = km.fit_predict(X)
        trans = _empirical_transition(states, n_states)
        # Confidence ~ inverse relative distance to nearest vs 2nd-nearest centre.
        d = km.transform(X[-1:].reshape(1, -1))[0]
        order = np.sort(d)
        confidence = float(1.0 - order[0] / (order[1] + 1e-9)) if len(order) > 1 else 0.5
        confidence = float(np.clip(confidence, 0.34, 0.99))
        return {"states": states, "trans": trans,
                "engine": "KMeans (fallback)", "confidence": confidence}

    # ---- Last resort: single regime ----
    states = np.zeros(n, dtype=int)
    return {"states": states, "trans": np.array([[1.0]]),
            "engine": "Single-regime (no model libs)", "confidence": 1.0}


def _normalise_rows(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=float)
    rs = m.sum(axis=1, keepdims=True)
    rs[rs < 1e-12] = 1.0
    return m / rs


def _empirical_transition(states: np.ndarray, n_states: int) -> np.ndarray:
    """Count state→state transitions and normalise to probabilities."""
    trans = np.zeros((n_states, n_states), dtype=float)
    for a, b in zip(states[:-1], states[1:]):
        trans[int(a), int(b)] += 1.0
    return _normalise_rows(trans)


def _current_run_length(states: np.ndarray) -> int:
    """How many consecutive most-recent observations share the latest regime."""
    if len(states) == 0:
        return 0
    last = states[-1]
    run = 0
    for s in states[::-1]:
        if s == last:
            run += 1
        else:
            break
    return int(run)


# --------------------------------------------------------------------------- #
# Regime labelling                                                            #
# --------------------------------------------------------------------------- #
def _label_regimes(stats: pd.DataFrame) -> Dict[int, Dict[str, str]]:
    """Map raw state ids to human labels from their return/volatility signature.

    The highest-volatility state with non-positive return → Volatile-Bearish;
    the most positive-return state → Calm Bull; the remainder → Sideways.
    Badges reuse the app's buy/hold/sell colour classes.
    """
    labels: Dict[int, Dict[str, str]] = {}
    order_vol = stats.sort_values("avg_vol", ascending=False).index.tolist()
    order_ret = stats.sort_values("avg_ret", ascending=False).index.tolist()

    bearish = order_vol[0]
    bullish = next((sid for sid in order_ret if sid != bearish), order_ret[0])

    for sid in stats.index:
        if sid == bullish:
            labels[sid] = {"name": "Calm Bull", "badge": "badge-buy", "emoji": "🟢"}
        elif sid == bearish:
            labels[sid] = {"name": "Volatile-Bearish", "badge": "badge-sell", "emoji": "🔴"}
        else:
            labels[sid] = {"name": "Sideways", "badge": "badge-hold", "emoji": "🟡"}
    return labels


def _regime_stats(states: np.ndarray, ret: pd.Series, vol: pd.Series) -> pd.DataFrame:
    """Per-regime average return, volatility, frequency and mean run-length."""
    df = pd.DataFrame({"state": states, "ret": ret.values, "vol": vol.values})
    rows = []
    # Mean run length (avg consecutive duration) per state.
    runs: Dict[int, List[int]] = {s: [] for s in np.unique(states)}
    cur, length = states[0], 1
    for s in states[1:]:
        if s == cur:
            length += 1
        else:
            runs[cur].append(length)
            cur, length = s, 1
    runs[cur].append(length)

    n_total = len(states)
    for sid, g in df.groupby("state"):
        durations = runs.get(sid, [])
        rows.append({
            "state": int(sid),
            "avg_ret": float(g["ret"].mean()),
            "avg_vol": float(g["vol"].mean()),
            "frequency": float(len(g) / n_total),
            "avg_duration": float(np.mean(durations)) if durations else 0.0,
            "n_days": int(len(g)),
        })
    return pd.DataFrame(rows).set_index("state").sort_index()


# --------------------------------------------------------------------------- #
# Entry point                                                                 #
# --------------------------------------------------------------------------- #
def run(df: Optional[pd.DataFrame] = None, df_full: Optional[pd.DataFrame] = None) -> None:
    page_header("Market Regime Radar",
                "Detect hidden market regimes (Calm Bull / Volatile-Bearish / Sideways) "
                "with a Hidden Markov Model, and see what each has historically implied."
                )

    source = df_full if df_full is not None and not df_full.empty else df
    if source is None or source.empty:
        source = get_engineered_data()
    if source is None or source.empty:
        st.warning("No data available for regime detection.")
        return

    if not _HAS_HMM:
        st.info("`hmmlearn` is not installed, so the **KMeans fallback** engine is "
                "used (same two features, empirical transitions). For the true HMM, "
                "install it with `pip install hmmlearn` and restart.")

    symbols = sorted(source[COLS.symbol].unique().tolist())
    c1, c2 = st.columns([2, 1])
    with c1:
        symbol = st.selectbox("Select stock", symbols, key="regime_symbol")
    with c2:
        n_req = st.slider("Number of regimes", 2, 4, N_REGIMES, key="regime_n")

    s = source[source[COLS.symbol] == symbol].sort_values(COLS.date).reset_index(drop=True)
    close = pd.to_numeric(s[COLS.stock_close], errors="coerce")
    if len(s) < 60 or close.dropna().empty:
        st.warning("Not enough history for this stock to detect regimes reliably.")
        return

    company = str(s[COLS.company].iloc[-1]) if COLS.company in s else symbol

    # ---- Features + model ----
    X, valid_idx = _prepare_features(s, close)
    if X.shape[0] < 40:
        st.warning("Not enough valid observations after cleaning to fit a model.")
        return

    result = _fit_regimes(symbol, X, n_req)
    states = np.asarray(result["states"], dtype=int)
    trans = np.asarray(result["trans"], dtype=float)
    engine = str(result["engine"])
    confidence = float(result["confidence"])

    # Align the per-observation arrays back to the valid rows.
    s_valid = s.loc[valid_idx.values].copy()
    s_valid["__state"] = states
    ret_v = pd.to_numeric(s_valid.get("Daily_Return", close.pct_change().loc[valid_idx.values]),
                          errors="coerce").fillna(0.0)
    vol_v = pd.to_numeric(s_valid.get("Volatility",
                          close.pct_change().rolling(20, min_periods=5).std().loc[valid_idx.values]),
                          errors="coerce").fillna(0.0)

    stats = _regime_stats(states, ret_v, vol_v)
    labels = _label_regimes(stats)
    current_state = int(states[-1])
    current = labels[current_state]

    # ===================================================== SECTION 1 ======
    # Current regime card.
    last_date = pd.to_datetime(s_valid[COLS.date].iloc[-1]).date()
    st.markdown(
        f"<div class='glass'>"
        f"<h3 style='margin:0.1rem 0'>{company} "
        f"<span style='font-size:1rem;opacity:0.7'>({symbol})</span></h3>"
        f"<p style='margin:0.4rem 0;font-size:1.1rem'>Current Market Regime: "
        f"<span class='badge {current['badge']}'>{current['emoji']} {current['name']}</span>"
        f" &nbsp;·&nbsp; Model confidence <b>{confidence*100:.0f}%</b></p>"
        f"<p style='margin:0.2rem 0;opacity:0.75;font-size:0.9rem'>"
        f"Engine: {engine} · {len(stats)} regimes · as of {last_date}</p>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # KPI strip for the current regime.
    k1, k2, k3, k4 = st.columns(4)
    cur_stats = stats.loc[current_state]
    k1.metric("Avg Daily Return", f"{cur_stats['avg_ret']*100:+.2f}%")
    k2.metric("Avg Volatility", f"{cur_stats['avg_vol']*100:.2f}%")
    k3.metric("Time in Regime", f"{cur_stats['frequency']*100:.0f}%")
    k4.metric("Typical Duration", f"{cur_stats['avg_duration']:.0f} days")

    # ===================================================== SECTION 1b =====
    # "Why this regime?" — transparent, evidence-based reasoning derived from
    # the SAME computed statistics (no model logic changed). It contrasts the
    # current regime against the others and against recent observed behaviour.
    st.markdown("###  Why is the stock in this regime?")

    # Recent observed signal (last 20 valid days) vs the regime's profile.
    recent_ret = float(ret_v.tail(20).mean())
    recent_vol = float(vol_v.tail(20).mean())
    cur_run = _current_run_length(states)
    avg_dur = float(cur_stats["avg_duration"])
    stay_p = float(trans[current_state, current_state])

    # Rank the current regime among all regimes for context.
    ret_rank = int((stats["avg_ret"] > cur_stats["avg_ret"]).sum()) + 1
    vol_rank = int((stats["avg_vol"] > cur_stats["avg_vol"]).sum()) + 1
    n_reg = len(stats)
    ret_pos = ("highest" if ret_rank == 1 else "lowest" if ret_rank == n_reg else "mid-range")
    vol_pos = ("highest" if vol_rank == 1 else "lowest" if vol_rank == n_reg else "mid-range")

    reasons: List[str] = []
    reasons.append(
        f"Recent daily returns average <b>{recent_ret*100:+.2f}%</b> and volatility "
        f"<b>{recent_vol*100:.2f}%</b>, which best match the "
        f"<b>{current['name']}</b> state's signature "
        f"({cur_stats['avg_ret']*100:+.2f}% return, {cur_stats['avg_vol']*100:.2f}% volatility).")
    reasons.append(
        f"Within this stock's {n_reg} regimes, {current['name']} has the "
        f"<b>{ret_pos}</b> average return and <b>{vol_pos}</b> volatility — "
        + ("a calm, upward-leaning state."
           if current['badge'] == 'badge-buy' else
           "a turbulent, downside-prone state."
           if current['badge'] == 'badge-sell' else
           "a flat, low-conviction state."))
    reasons.append(
        f"The stock has already spent <b>{cur_run} day(s)</b> in this regime; "
        f"its typical run lasts <b>{avg_dur:.0f} day(s)</b>, and the model puts "
        f"<b>{stay_p*100:.0f}%</b> probability on staying next period.")
    # Where it is most likely to go next (excluding itself).
    nxt = trans[current_state].copy()
    nxt[current_state] = -1.0
    nxt_state = int(np.argmax(nxt))
    if nxt[nxt_state] > 0:
        reasons.append(
            f"If it does shift, the most likely move is to "
            f"<b>{labels[nxt_state]['name']}</b> "
            f"(~{trans[current_state, nxt_state]*100:.0f}%).")
    reasons.append(
        f"Confidence in today's classification is <b>{confidence*100:.0f}%</b> "
        f"(from the {engine}).")

    reasons_html = "".join(f"<li style='margin:3px 0'>{r}</li>" for r in reasons)
    st.markdown(
        f"<div class='glass'>"
        f"<p style='margin:0 0 0.4rem 0'>The "
        f"<span class='badge {current['badge']}'>{current['emoji']} {current['name']}</span> "
        f"label comes from this evidence:</p>"
        f"<ul style='margin:0.2rem 0 0 0;padding-left:1.1rem'>{reasons_html}</ul>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # ===================================================== SECTION 2 ======
    # Regime-coloured price timeline.
    st.markdown("###  Regime Timeline")
    timeline = go.Figure()
    palette_map = {"badge-buy": PALETTE.up, "badge-sell": PALETTE.down, "badge-hold": PALETTE.warn}
    s_valid = s_valid.reset_index(drop=True)
    s_valid["__close"] = pd.to_numeric(s_valid[COLS.stock_close], errors="coerce").values
    dates = pd.to_datetime(s_valid[COLS.date])
    for sid in stats.index:
        mask = s_valid["__state"] == sid
        lbl = labels[sid]
        timeline.add_trace(go.Scatter(
            x=dates[mask], y=s_valid["__close"][mask],
            mode="markers", name=f"{lbl['emoji']} {lbl['name']}",
            marker=dict(size=4, color=palette_map[lbl["badge"]]),
            hovertemplate="%{x|%Y-%m-%d}<br>Close ₹%{y:.2f}<extra>" + lbl["name"] + "</extra>",
        ))
    # Faint continuous price line for context.
    timeline.add_trace(go.Scatter(
        x=dates, y=s_valid["__close"], mode="lines", name="Price",
        line=dict(color=PALETTE.muted, width=1), opacity=0.35, hoverinfo="skip",
        showlegend=False))
    timeline.update_layout(
        template="plotly_dark", height=420,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        title=f"{symbol} — Price coloured by detected regime",
        xaxis_title="Date", yaxis_title="Close (₹)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(timeline, use_container_width=True)

    # ===================================================== SECTION 3 ======
    # Transition-probability matrix + per-regime stats.
    col_a, col_b = st.columns([1, 1])

    with col_a:
        st.markdown("###  Transition Probabilities")
        names = [labels[sid]["name"] for sid in stats.index]
        heat = go.Figure(go.Heatmap(
            z=trans, x=names, y=names,
            colorscale="Tealrose", zmin=0, zmax=1,
            text=[[f"{v*100:.0f}%" for v in row] for row in trans],
            texttemplate="%{text}", textfont=dict(size=12),
            hovertemplate="From %{y} → %{x}: %{z:.2f}<extra></extra>",
            colorbar=dict(title="P"),
        ))
        heat.update_layout(
            template="plotly_dark", height=380,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            title="P(next regime | current regime)",
            xaxis_title="To", yaxis_title="From",
        )
        st.plotly_chart(heat, use_container_width=True)
        stay = trans[current_state, current_state]
        st.caption(f"From **{current['name']}**, the model estimates a "
                   f"**{stay*100:.0f}%** chance of staying next period.")

    with col_b:
        st.markdown("###  Regime Profiles")
        prof = stats.copy()
        prof.insert(0, "Regime", [labels[sid]["name"] for sid in prof.index])
        prof_disp = pd.DataFrame({
            "Regime": prof["Regime"].values,
            "Avg Return": [f"{v*100:+.2f}%" for v in prof["avg_ret"]],
            "Avg Volatility": [f"{v*100:.2f}%" for v in prof["avg_vol"]],
            "Frequency": [f"{v*100:.0f}%" for v in prof["frequency"]],
            "Avg Duration": [f"{v:.0f}d" for v in prof["avg_duration"]],
            "Days": [int(v) for v in prof["n_days"]],
        })
        st.dataframe(prof_disp, use_container_width=True, hide_index=True)
        st.caption("Regimes are labelled from their return/volatility signature; "
                   "they are statistical states, not guaranteed future behaviour.")

    # ---- Export ----
    export = stats.copy()
    export.insert(0, "regime", [labels[sid]["name"] for sid in export.index])
    download_buttons(export.reset_index(), key="regime",
                     label=f"{symbol}_regime_profiles")

    # ===================================================== SECTION 4 ======
    # Inline plain-English explanation of the result.
    if _HAS_LLM:
        results = {
            "symbol": symbol,
            "company": company,
            "engine": engine,
            "current_regime": current["name"],
            "confidence_pct": round(confidence * 100, 1),
            "stay_probability_pct": round(float(trans[current_state, current_state]) * 100, 1),
            "current_regime_avg_daily_return_pct": round(float(cur_stats["avg_ret"]) * 100, 3),
            "current_regime_avg_volatility_pct": round(float(cur_stats["avg_vol"]) * 100, 3),
            "time_in_current_regime_pct": round(float(cur_stats["frequency"]) * 100, 1),
            "recent_20d_avg_return_pct": round(recent_ret * 100, 3),
            "recent_20d_avg_volatility_pct": round(recent_vol * 100, 3),
            "days_in_current_regime": int(cur_run),
            "typical_regime_duration_days": round(avg_dur, 1),
            "most_likely_next_regime": labels[nxt_state]["name"] if nxt[nxt_state] > 0 else current["name"],
            "most_likely_next_regime_pct": round(float(trans[current_state, nxt_state]) * 100, 1) if nxt[nxt_state] > 0 else None,
            "regimes": {labels[sid]["name"]: {
                "avg_return_pct": round(float(stats.loc[sid, "avg_ret"]) * 100, 3),
                "avg_volatility_pct": round(float(stats.loc[sid, "avg_vol"]) * 100, 3),
                "frequency_pct": round(float(stats.loc[sid, "frequency"]) * 100, 1),
            } for sid in stats.index},
        }
        render_llm_explanation(results, title="Market Regime Detection", key="regime")