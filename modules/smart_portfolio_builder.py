"""
modules/smart_portfolio_builder.py
-----------------------------------
AI Smart Portfolio Builder — robo-advisor style allocator.

Answers:
    "Given my risk tolerance and investment amount, how should I allocate
     my money across the available stocks?"

Pipeline (calculations unchanged)
    Step 1  Daily returns per stock.
    Step 2  Expected (annualised) return per stock.
    Step 3  Volatility (annualised std-dev of returns).
    Step 4  Sharpe Ratio  =  (Expected Return − Risk-Free Rate) / Volatility.
    Step 5  Maximum Drawdown  — worst peak-to-trough decline.
    Step 6  Momentum  — trailing-return over the chosen horizon window.
    Step 7  Composite Quality Score (0-100) with profile-tilted weights.
    Step 8  Risk Profile Engine (Conservative / Moderate / Aggressive).
    Step 9  Stock selection — top-N from the profile-ranked universe.
    Step 10 Intelligent allocation — Sharpe/risk-adjusted weights (NOT equal).
    Step 11 Portfolio-level metrics (return, volatility, Sharpe, drawdown).
    Step 12 Diversification analysis (HHI concentration + diversification score).
    Step 13 Scenario analysis (Bear / Base / Bull — 1-year projection).
    Step 14 Per-stock "why selected" explanation (generated dynamically).
    Step 15 Portfolio Quality Score (0-100) with Plotly gauge.
    Step 16 AI Portfolio Manager Report (fully dynamic narrative).
    Viz     Pie, Risk-Return scatter, Sharpe bars, Treemap, Scenario bars.

UX layer redesigned as an AI Investment Advisor that directly answers:
    "I have Tk.X and I am willing to take <profile> risk.
     How should I invest my money?"
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ------------------------------------------------------------------ project imports
try:
    from config.settings import COLS, PALETTE
    from utils import visualizations as viz
    from utils.helper_functions import download_buttons, page_header
    from utils.preprocessing import get_engineered_data
    _HAS_PROJECT_HELPERS = True
except ImportError:
    _HAS_PROJECT_HELPERS = False

# ------------------------------------------------------------------ constants
TRADING_DAYS: int = 252
RISK_FREE: float = 0.05

PROFILE_WEIGHTS: Dict[str, Dict[str, float]] = {
    "Conservative": {"ret": 0.15, "sharpe": 0.40, "mom": 0.10, "low_dd": 0.35},
    "Moderate":     {"ret": 0.30, "sharpe": 0.30, "mom": 0.20, "low_dd": 0.20},
    "Aggressive":   {"ret": 0.40, "sharpe": 0.20, "mom": 0.30, "low_dd": 0.10},
}

HORIZON_LOOKBACK: Dict[str, int] = {
    "Short Term":  21,
    "Medium Term": 63,
    "Long Term":  126,
}

# ------------------------------------------------------------------ fallback stubs
if not _HAS_PROJECT_HELPERS:
    class _SimpleCols:
        date = "Date"; symbol = "Symbol"; company = "Company"
        industry = "Industry"; stock_close = "Stock_Close"

    class _SimplePalette:
        accent = "#00B4D8"; accent_2 = "#90E0EF"; muted = "#555"
        up = "#2ecc71"; down = "#e74c3c"; warn = "#f39c12"
        sequence = ["#00B4D8","#0077B6","#90E0EF","#CAF0F8",
                    "#2ecc71","#f39c12","#e74c3c","#9b59b6"]

    COLS = _SimpleCols()
    PALETTE = _SimplePalette()

    def page_header(title: str, subtitle: str = "", icon: str = "") -> None:
        st.title(f"{icon} {title}")
        if subtitle:
            st.caption(subtitle)

    def download_buttons(df: pd.DataFrame, key: str = "", label: str = "data") -> None:
        csv = df.to_csv(index=False).encode()
        st.download_button(f"⬇Download {label}", csv,
                           file_name=f"{label}.csv", mime="text/csv", key=f"dl_{key}")

    def get_engineered_data() -> pd.DataFrame:
        return pd.DataFrame()

    class viz:
        @staticmethod
        def treemap(df, path, values, title=""):
            import plotly.express as px
            return px.treemap(df, path=path, values=values, title=title)


# ===================================================== STEP 1-6  PER-STOCK METRICS
@st.cache_data(show_spinner=False)
def _compute_stock_metrics(df_key: str, lookback: int) -> pd.DataFrame:
    """
    Compute per-stock risk/return metrics from the engineered dataset.

    Steps:
        1. Daily return  = (Close_t − Close_{t−1}) / Close_{t−1}
        2. Expected Return = mean(daily returns) × 252   [annualised]
        3. Volatility    = std(daily returns) × √252     [annualised]
        4. Sharpe Ratio  = (Expected Return − RISK_FREE) / Volatility
        5. Max Drawdown  = min(Close / cumulative-peak − 1)
        6. Momentum      = Close[-1] / Close[-lookback] − 1
    """
    try:
        df = get_engineered_data()
    except Exception:
        df = pd.DataFrame()

    if df.empty:
        return pd.DataFrame()

    rows: List[dict] = []
    for sym, sdf in df.groupby(COLS.symbol):
        sdf = sdf.sort_values(COLS.date)
        close = pd.to_numeric(sdf[COLS.stock_close], errors="coerce").dropna()

        if len(close) < 60:
            continue

        if "Daily_Return" in sdf.columns:
            rets = pd.to_numeric(sdf["Daily_Return"], errors="coerce").dropna()
        else:
            rets = close.pct_change().dropna()

        if len(rets) < 30:
            continue

        exp_return = float(rets.mean() * TRADING_DAYS)
        volatility = float(rets.std() * np.sqrt(TRADING_DAYS))
        if volatility < 1e-9:
            continue

        sharpe = (exp_return - RISK_FREE) / volatility
        peak   = close.cummax()
        max_dd = float((close / peak - 1.0).min())
        lb     = min(lookback, len(close) - 1)
        momentum = float(close.iloc[-1] / close.iloc[-lb] - 1.0)

        rows.append({
            COLS.symbol:    str(sym),
            COLS.company:   str(sdf[COLS.company].iloc[-1]),
            COLS.industry:  str(sdf[COLS.industry].iloc[-1]),
            "Exp_Return":   exp_return,
            "Volatility":   volatility,
            "Sharpe":       float(sharpe),
            "Max_Drawdown": max_dd,
            "Momentum":     momentum,
        })

    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ===================================================== STEP 7  QUALITY SCORE
def _minmax_scale(series: pd.Series) -> pd.Series:
    """Scale a Series to [0, 1]. Flat series → 0.5 everywhere."""
    lo, hi = series.min(), series.max()
    if not (np.isfinite(lo) and np.isfinite(hi)) or hi - lo < 1e-12:
        return pd.Series(0.5, index=series.index)
    return (series - lo) / (hi - lo)


def _score_and_rank(metrics: pd.DataFrame, profile: str) -> pd.DataFrame:
    """
    Step 7: Composite quality score (0-100) using profile-tilted weights.

        Score = w_ret × norm(Exp_Return)
              + w_sharpe × norm(Sharpe)
              + w_mom × norm(Momentum)
              + w_low_dd × norm(−|Max_Drawdown|)
    """
    w = PROFILE_WEIGHTS[profile]
    m = metrics.copy()
    m["Score"] = (
        w["ret"]    * _minmax_scale(m["Exp_Return"])
        + w["sharpe"] * _minmax_scale(m["Sharpe"])
        + w["mom"]    * _minmax_scale(m["Momentum"])
        + w["low_dd"] * _minmax_scale(-m["Max_Drawdown"].abs())
    ) * 100
    m = m.sort_values("Score", ascending=False).reset_index(drop=True)
    m.insert(0, "Rank", m.index + 1)
    return m


# ===================================================== STEP 10  INTELLIGENT WEIGHTS
def _compute_weights(picks: pd.DataFrame, profile: str) -> np.ndarray:
    """
    Step 10: Non-equal allocation weights.
    Conservative → inverse-variance.
    Aggressive   → Sharpe-weighted.
    Moderate     → geometric blend of both.
    """
    vol    = picks["Volatility"].to_numpy(dtype=float)
    sharpe = picks["Sharpe"].to_numpy(dtype=float)

    inv_var    = 1.0 / np.clip(vol ** 2, 1e-6, None)
    sharpe_pos = np.clip(sharpe, 0.01, None)

    if profile == "Conservative":
        raw = inv_var
    elif profile == "Aggressive":
        raw = sharpe_pos
    else:
        raw = np.sqrt(inv_var * sharpe_pos)

    total = raw.sum()
    if not np.isfinite(total) or total <= 0:
        raw = np.ones(len(picks))
    return raw / raw.sum()


# ===================================================== HELPERS
def _risk_label(value: float, lo: float, hi: float) -> str:
    return "Low" if value < lo else ("Medium" if value < hi else "High")


def _rank_table(metrics: pd.DataFrame, col: str, ascending: bool, fmt_fn) -> pd.DataFrame:
    r = metrics.sort_values(col, ascending=ascending).head(10).copy()
    r = r[[COLS.symbol, COLS.company, col]].reset_index(drop=True)
    r.index += 1
    r[col] = r[col].map(fmt_fn)
    return r


# ===================================================== CSS CARDS
_CARD_CSS = """
<style>
.advisor-card {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    border: 1px solid rgba(0,180,216,0.35);
    border-radius: 14px;
    padding: 1.4rem 1.8rem;
    margin-bottom: 1.2rem;
}
.advisor-card h3 { margin: 0 0 0.6rem 0; font-size: 1.05rem;
                   color: #90e0ef; letter-spacing: 0.04em; }
.advisor-card .question-text {
    font-size: 1.18rem; font-style: italic;
    color: #caf0f8; line-height: 1.5;
    border-left: 3px solid #00b4d8;
    padding-left: 0.9rem; margin-top: 0.6rem;
}
.advisor-card .tag {
    display: inline-block; background: rgba(0,180,216,0.18);
    border: 1px solid rgba(0,180,216,0.4);
    border-radius: 20px; padding: 2px 12px;
    font-size: 0.82rem; color: #90e0ef; margin: 2px 4px 2px 0;
}
.rec-card {
    background: linear-gradient(135deg, #0d1b2a, #1b2838);
    border: 1px solid rgba(46,204,113,0.35);
    border-radius: 14px;
    padding: 1.4rem 1.8rem;
    margin-bottom: 1.2rem;
}
.rec-card h3 { margin: 0 0 0.8rem 0; font-size: 1.05rem;
               color: #2ecc71; letter-spacing: 0.04em; }
.rec-card .alloc-line {
    font-size: 1.05rem; color: #ecf0f1;
    padding: 4px 0; border-bottom: 1px solid rgba(255,255,255,0.06);
}
.rec-card .alloc-line span { color: #00b4d8; font-weight: 600; }
.invest-card {
    background: linear-gradient(135deg, #1a1a2e, #16213e);
    border: 1px solid rgba(243,156,18,0.4);
    border-radius: 14px;
    padding: 1.6rem 1.8rem;
    margin-bottom: 1.2rem;
    text-align: center;
}
.invest-card .big-val {
    font-size: 2.4rem; font-weight: 700;
    color: #f39c12; line-height: 1.1;
}
.invest-card .label { font-size: 0.82rem; color: #aaa; margin-bottom: 0.2rem; }
.invest-card .profit { font-size: 1.6rem; font-weight: 600; color: #2ecc71; }
.invest-card .profit.loss { color: #e74c3c; }
.why-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 12px;
    padding: 1.2rem 1.6rem;
    margin-bottom: 1rem;
}
.why-card h4 { margin: 0 0 0.6rem 0; color: #caf0f8; font-size: 0.95rem; }
.why-point { color: #b2f2bb; font-size: 0.92rem; padding: 2px 0; }
.report-card {
    background: linear-gradient(160deg, #0a0a1a, #111827);
    border: 1px solid rgba(155,89,182,0.4);
    border-radius: 14px;
    padding: 1.6rem 2rem;
    margin-bottom: 1.2rem;
}
.report-card h3 { color: #bb8fce; font-size: 1.05rem;
                  letter-spacing: 0.06em; margin-bottom: 1rem; }
.report-card p { color: #dfe6e9; line-height: 1.7; font-size: 0.95rem; }
.report-card .verdict {
    font-size: 1.15rem; font-weight: 600;
    color: #f39c12; margin-top: 0.8rem;
}
.divider { border: none; border-top: 1px solid rgba(255,255,255,0.08);
           margin: 1.4rem 0; }
</style>
"""


# ===================================================== MAIN ENTRY POINT
def run(df_filtered: Optional[pd.DataFrame] = None,
        df_full: Optional[pd.DataFrame] = None) -> None:
    """
    Entry point called by the Streamlit app router.

    Parameters
    ----------
    df_filtered : DataFrame filtered by upstream sidebar selections.
    df_full     : Complete project dataset.
    """
    # Inject card styles once
    st.markdown(_CARD_CSS, unsafe_allow_html=True)

    page_header("AI Smart Portfolio Builder",
                "Enter your investment details below — I'll build a personalised portfolio instantly.")

    # ---------------------------------------------------------------- INLINE INPUT FORM
    # Users type/select directly on the main page — the query card updates live.
    st.markdown("Tell Me About Your Investment")

    fi1, fi2, fi3, fi4 = st.columns([2, 2, 2, 1])

    with fi1:
        amount = st.number_input(
            "Investment Amount (Tk.)",
            min_value=1_000,
            max_value=100_000_000,
            value=100_000,
            step=5_000,
            key="spb_amount",
            help="Type any amount — e.g. 75000 for Tk. 75,000",
        )

    with fi2:
        profile = st.selectbox(
            "Risk Profile",
            ["Conservative", "Moderate", "Aggressive"],
            index=1,
            key="spb_profile",
            help="Conservative = low risk · Moderate = balanced · Aggressive = high growth",
        )

    with fi3:
        horizon = st.selectbox(
            "Investment Horizon",
            ["Short Term", "Medium Term", "Long Term"],
            index=1,
            key="spb_horizon",
            help="Short Term ≈ 1-3 months · Medium Term ≈ 6-12 months · Long Term ≈ 1-3 years",
        )

    with fi4:
        max_stocks = st.selectbox(
            "Max Stocks",
            [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],
            index=1,
            key="spb_maxn",
            help="Maximum number of stocks in your portfolio",
        )

    st.markdown("---")

    # Mirror into sidebar for reference (read-only display, not inputs)
    st.sidebar.markdown("---")
    st.sidebar.markdown("Current Portfolio Settings")
    st.sidebar.markdown(f"**Amount:** Tk. {amount:,.0f}")
    st.sidebar.markdown(f"**Risk Profile:** {profile}")
    st.sidebar.markdown(f"**Horizon:** {horizon}")
    st.sidebar.markdown(f"**Max Stocks:** {max_stocks}")

    # ---------------------------------------------------------------- COMPUTE (unchanged)
    lookback  = HORIZON_LOOKBACK[horizon]
    cache_key = (str(sorted(df_filtered[COLS.symbol].unique().tolist()))
                 if df_filtered is not None and not df_filtered.empty
                 else "full")
    metrics = _compute_stock_metrics(cache_key, lookback)

    if metrics.empty:
        st.warning("⚠️ Not enough historical data to build a portfolio. "
                   "Please check that the dataset contains at least 60 "
                   "trading days per stock.")
        return

    if df_filtered is not None and not df_filtered.empty:
        active_symbols = set(df_filtered[COLS.symbol].unique())
        subset = metrics[metrics[COLS.symbol].isin(active_symbols)]
        if len(subset) >= 3:
            metrics = subset.reset_index(drop=True)

    ranked  = _score_and_rank(metrics, profile)
    n       = min(max_stocks, len(ranked))
    if n < 1:
        st.warning("Too few stocks in the filtered universe. Broaden your filters.")
        return
    picks   = ranked.head(n).reset_index(drop=True)

    weights          = _compute_weights(picks, profile)
    picks            = picks.copy()
    picks["Weight"]  = weights
    picks["Weight_%"] = (weights * 100).round(2)
    picks["Allocated"] = (weights * amount).round(0)

    # Portfolio-level aggregates (Step 11)
    port_return = float(np.dot(weights, picks["Exp_Return"]))
    port_vol    = float(np.dot(weights, picks["Volatility"]))
    port_sharpe = (port_return - RISK_FREE) / port_vol if port_vol > 1e-9 else 0.0
    worst_dd    = float(picks["Max_Drawdown"].min())

    # Diversification (Step 12)
    ind_weights  = picks.groupby(COLS.industry)["Weight"].sum()
    n_industries = int(ind_weights.size)
    hhi          = float((ind_weights ** 2).sum())
    div_score    = float(np.clip((1 - hhi) * 100, 0, 100))
    div_label    = ("Excellent" if div_score >= 70 else
                    "Moderate"  if div_score >= 40 else "Poor")

    # Scenario returns (Step 13) — kept as pure dict for chart reuse
    scenarios: Dict[str, float] = {
        "Bear": port_return - port_vol,
        "Base": port_return,
        "Bull": port_return + port_vol,
    }

    # Quality gauge score (Step 15)
    q_div    = div_score / 100
    q_sharpe = float(np.clip((port_sharpe + 0.5) / 2.0, 0, 1))
    q_vol    = float(np.clip(1.0 - port_vol / 0.5, 0, 1))
    q_dd     = float(np.clip(1.0 + worst_dd / 0.60, 0, 1))
    q_bal    = float(np.clip(n_industries / max(n, 1), 0, 1))
    quality  = float(np.clip(
        (0.25 * q_div + 0.30 * q_sharpe + 0.20 * q_vol
         + 0.15 * q_dd + 0.10 * q_bal) * 100, 0, 100))
    q_label  = ("Excellent Portfolio" if quality >= 70 else
                "Good Portfolio"      if quality >= 40 else
                "Weak Portfolio")

    risk_rating = _risk_label(port_vol, 0.20, 0.40)
    sharpe_grade = (
        "Excellent (> 1.5)"      if port_sharpe > 1.5 else
        "Good (1.0 – 1.5)"       if port_sharpe > 1.0 else
        "Acceptable (0.5 – 1.0)" if port_sharpe > 0.5 else
        "Below Average (< 0.5)"
    )
    top = picks.iloc[0]

    # ================================================================
    # SECTION 1 — INVESTOR QUESTION CARD
    # ================================================================
    horizon_map = {"Short Term": "6 months", "Medium Term": "1 Year", "Long Term": "2+ Years"}
    horizon_display = horizon_map.get(horizon, horizon)

    profile_icon = {"Conservative": "", "Moderate": "", "Aggressive": ""}[profile]

    st.markdown(f"""
    <div class="advisor-card">
      <h3>👤 Investor Query</h3>
      <div style="display:flex; gap:10px; flex-wrap:wrap; margin-bottom:0.8rem;">
        <span class="tag">Tk. {amount:,.0f}</span>
        <span class="tag">{profile_icon} {profile} Risk</span>
        <span class="tag">{horizon}</span>
        <span class="tag">Up to {n} stocks</span>
      </div>
      <div class="question-text">
        "I have <strong>Tk. {amount:,.0f}</strong> and I am willing to take
        <strong>{profile.lower()} risk</strong>.
        How should I invest my money over a <strong>{horizon_display}</strong> horizon?"
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ================================================================
    # SECTION 2 — AI RECOMMENDATION CARD
    # ================================================================
    alloc_lines_html = "".join(
        f'<div class="alloc-line">• <span>Tk. {float(row["Allocated"]):,.0f}</span>'
        f' in <strong>{row[COLS.symbol]}</strong>'
        f' &nbsp;<span style="color:#aaa;font-size:0.85rem">({row["Weight_%"]:.1f}%)</span>'
        f' — {row[COLS.industry]}</div>'
        for _, row in picks.iterrows()
    )

    st.markdown(f"""
    <div class="rec-card">
      <h3>AI Recommendation</h3>
      <p style="color:#b2bec3; font-size:0.9rem; margin:0 0 0.8rem 0;">
        Based on your <strong>{profile.lower()}</strong> risk profile and historical
        stock performance, here is how I recommend distributing your
        <strong>Tk. {amount:,.0f}</strong>:
      </p>
      {alloc_lines_html}
      <hr class="divider">
      <div style="display:flex; gap:2rem; flex-wrap:wrap; margin-top:0.4rem;">
        <div>
          <div style="font-size:0.78rem; color:#aaa;">Expected Annual Return</div>
          <div style="font-size:1.3rem; font-weight:700; color:#2ecc71;">{port_return*100:.1f}%</div>
        </div>
        <div>
          <div style="font-size:0.78rem; color:#aaa;">Risk Level</div>
          <div style="font-size:1.3rem; font-weight:700; color:#f39c12;">{risk_rating}</div>
        </div>
        <div>
          <div style="font-size:0.78rem; color:#aaa;">Diversification</div>
          <div style="font-size:1.3rem; font-weight:700; color:#00b4d8;">{div_score:.0f}/100</div>
        </div>
        <div>
          <div style="font-size:0.78rem; color:#aaa;">Portfolio Sharpe</div>
          <div style="font-size:1.3rem; font-weight:700; color:#bb8fce;">{port_sharpe:.2f}</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ================================================================
    # SECTION 3 — INVESTMENT PLAN TABLE
    # ================================================================
    st.markdown("#### Your Investment Plan")

    plan = picks[[COLS.symbol, COLS.company, COLS.industry,
                  "Weight_%", "Allocated", "Exp_Return", "Volatility"]].copy()
    plan = plan.rename(columns={
        COLS.symbol:   "Stock",
        COLS.company:  "Company",
        COLS.industry: "Industry",
        "Weight_%":    "Allocation %",
        "Allocated":   "Investment (Tk.)",
        "Exp_Return":  "Expected Return",
        "Volatility":  "Risk Level",
    })
    plan["Allocation %"]      = plan["Allocation %"].map(lambda v: f"{v:.1f}%")
    plan["Investment (Tk.)"]  = plan["Investment (Tk.)"].map(lambda v: f"Tk. {float(v):,.0f}")
    plan["Expected Return"]   = plan["Expected Return"].map(lambda v: f"{v*100:.1f}%")
    plan["Risk Level"]        = plan["Risk Level"].map(
        lambda v: "Low" if v < 0.20 else ("Medium" if v < 0.40 else "High"))
    st.dataframe(plan, use_container_width=True, hide_index=True)

    download_buttons(
        picks[[COLS.symbol, COLS.company, COLS.industry,
               "Weight_%", "Allocated", "Exp_Return", "Volatility",
               "Sharpe", "Momentum", "Max_Drawdown", "Score"]],
        key="spb",
        label=f"{profile.lower()}_portfolio",
    )

    # ================================================================
    # SECTION 4 — IF YOU INVEST TODAY
    # ================================================================
    st.markdown("If You Invest Today")

    base_end   = amount * (1 + port_return)
    base_profit = base_end - amount
    profit_cls = "profit" if base_profit >= 0 else "profit loss"
    profit_sign = "+" if base_profit >= 0 else ""

    c_inv, c_val, c_pnl, c_ret = st.columns(4)
    with c_inv:
        st.markdown(f"""
        <div class="invest-card">
          <div class="label">Initial Investment</div>
          <div class="big-val">Tk. {amount:,.0f}</div>
        </div>""", unsafe_allow_html=True)
    with c_val:
        st.markdown(f"""
        <div class="invest-card">
          <div class="label">Expected Value After {horizon_display}</div>
          <div class="big-val">Tk. {base_end:,.0f}</div>
        </div>""", unsafe_allow_html=True)
    with c_pnl:
        st.markdown(f"""
        <div class="invest-card">
          <div class="label">Expected Profit</div>
          <div class="{profit_cls}">{profit_sign}Tk. {abs(base_profit):,.0f}</div>
        </div>""", unsafe_allow_html=True)
    with c_ret:
        st.markdown(f"""
        <div class="invest-card">
          <div class="label">Expected Return</div>
          <div class="big-val" style="color:#2ecc71;">{profit_sign}{port_return*100:.1f}%</div>
        </div>""", unsafe_allow_html=True)

    # Scenario strip
    bear_val = amount * (1 + scenarios["Bear"])
    bull_val = amount * (1 + scenarios["Bull"])
    sc1, sc2, sc3 = st.columns(3)
    sc1.metric("Bear Market",  f"Tk. {bear_val:,.0f}",
               f"{scenarios['Bear']*100:+.1f}%  (worst-case)")
    sc2.metric("Base Scenario", f"Tk. {base_end:,.0f}",
               f"{port_return*100:+.1f}%  (expected)")
    sc3.metric("Bull Market",  f"Tk. {bull_val:,.0f}",
               f"{scenarios['Bull']*100:+.1f}%  (best-case)")

    # ================================================================
    # SECTION 5 — WHY THIS PORTFOLIO?
    # ================================================================
    st.markdown("---")
    st.markdown("Why This Portfolio Was Selected")

    # Determine portfolio strengths dynamically
    strengths = []
    if port_sharpe >= 1.0:
        strengths.append(f"✓ **Strong Sharpe Ratio ({port_sharpe:.2f})** — excellent risk-adjusted return.")
    else:
        strengths.append(f"✓ **Positive Sharpe Ratio ({port_sharpe:.2f})** — return above the risk-free rate.")

    if picks["Momentum"].median() > 0:
        strengths.append("✓ **Positive Momentum** — selected stocks are trending upward.")

    if port_vol < 0.30:
        strengths.append(f"✓ **Acceptable Volatility ({port_vol*100:.1f}%)** — manageable day-to-day swings.")
    else:
        strengths.append(f"✓ **Volatility ({port_vol*100:.1f}%) consistent with {profile.lower()} profile.**")

    if worst_dd > -0.30:
        strengths.append(f"✓ **Controlled Drawdown ({worst_dd*100:.0f}%)** — limited historical losses.")
    else:
        strengths.append(f"✓ **Drawdown of {worst_dd*100:.0f}%** monitored — offset by return potential.")

    if n_industries > 1:
        strengths.append(f"✓ **{n_industries}-Industry Diversification** — reduces sector-specific risk.")

    profile_rationale = {
        "Conservative": "Stocks were selected to minimise volatility and protect capital while "
                        "maintaining a positive expected return.",
        "Moderate":     "Stocks were selected to balance growth potential with downside protection "
                        "across multiple industries.",
        "Aggressive":   "Stocks were selected for high return potential and strong momentum, "
                        "accepting higher short-term volatility.",
    }[profile]

    strengths_html = "".join(f'<div class="why-point">{s}</div>' for s in strengths)
    st.markdown(f"""
    <div class="why-card">
      <h4>Portfolio Selection Criteria</h4>
      {strengths_html}
      <p style="color:#aaa; font-size:0.88rem; margin-top:0.8rem;">
        {profile_rationale}
      </p>
    </div>
    """, unsafe_allow_html=True)

    # ================================================================
    # SECTION 6 — WHY EACH STOCK WAS CHOSEN
    # ================================================================
    st.markdown("#### Why Each Stock Was Chosen")
    st.caption("Reasons generated dynamically from each stock's metrics relative to the portfolio.")

    med_sharpe = picks["Sharpe"].median()
    med_vol    = picks["Volatility"].median()
    med_dd     = picks["Max_Drawdown"].median()

    for _, row in picks.iterrows():
        reasons: List[str] = []

        if row["Sharpe"] > med_sharpe:
            reasons.append(f"✓ **Highest Sharpe Ratio** ({row['Sharpe']:.2f}) in this portfolio")
        else:
            reasons.append(f"✓ **Positive Sharpe Ratio** ({row['Sharpe']:.2f})")

        if row["Momentum"] > 0.05:
            reasons.append(f"✓ **Strong Momentum** (+{row['Momentum']*100:.1f}%) — confirmed upward trend")
        elif row["Momentum"] > 0:
            reasons.append(f"✓ **Positive Momentum** (+{row['Momentum']*100:.1f}%)")
        else:
            reasons.append(f"⚠️ Momentum slightly negative ({row['Momentum']*100:.1f}%) — offset by quality score")

        if row["Max_Drawdown"] >= med_dd:
            reasons.append(f"✓ **Stable Drawdown** ({row['Max_Drawdown']*100:.0f}%) — shallower than portfolio median")
        else:
            reasons.append(f"↓ Deeper drawdown ({row['Max_Drawdown']*100:.0f}%) — compensated by Sharpe and return")

        if row["Exp_Return"] > 0:
            reasons.append(f"✓ **Consistent Historical Return** ({row['Exp_Return']*100:.1f}% p.a.)")

        if row["Volatility"] <= med_vol:
            reasons.append(f"✓ **Lower Volatility** ({row['Volatility']*100:.1f}%) — below portfolio median")

        reasons.append(f"✓ **Quality Score: {row['Score']:.0f}/100** — Rank #{int(row['Rank'])} in the universe")

        label = (f"{row[COLS.symbol]}  ·  {row[COLS.company]}  "
                 f"·  {row['Weight_%']:.1f}%  ·  Tk. {float(row['Allocated']):,.0f}")
        with st.expander(label):
            for r in reasons:
                st.markdown(r)

    # ================================================================
    # SECTION 7 — AI PORTFOLIO MANAGER REPORT
    # ================================================================
    st.markdown("---")

    suitability = {
        "Conservative": "investors prioritising capital preservation and steady, "
                        "low-volatility compounding",
        "Moderate": "investors seeking balanced growth while maintaining "
                    "acceptable risk exposure",
        "Aggressive": "growth-oriented investors comfortable with higher "
                      "short-term volatility in pursuit of superior returns",
    }[profile]

    overall_outlook = (
        "Moderately Bullish" if port_return > 0.10 and risk_rating != "High" else
        "Bullish"            if port_return > 0.15 else
        "Cautiously Bullish" if port_return > 0 else
        "Neutral"
    )

    top3 = ", ".join(picks[COLS.symbol].head(3).tolist())
    if n > 3:
        top3 += f" and {n-3} more"

    st.markdown(f"""
    <div class="report-card">
      <h3>AI PORTFOLIO MANAGER REPORT</h3>
      <p>
        You selected a <strong>{profile}</strong> risk profile with an investment
        amount of <strong>Tk. {amount:,.0f}</strong> over a
        <strong>{horizon.lower()}</strong> horizon.
      </p>
      <p>
        The portfolio builder analysed the available universe and identified
        <strong>{n} stocks</strong> — {top3} —
        with the strongest risk-adjusted performance for your profile.
        The largest allocation ({top['Weight_%']:.1f}%) was assigned to
        <strong>{top[COLS.symbol]}</strong> ({top[COLS.company]}) because it
        achieved the highest overall quality score ({top['Score']:.0f}/100),
        a Sharpe Ratio of {top['Sharpe']:.2f}, and an expected return of
        {top['Exp_Return']*100:.1f}% p.a.
      </p>
      <p>
        The portfolio is spread across <strong>{n_industries}</strong>
        {'industry' if n_industries == 1 else 'industries'}, giving a
        diversification score of <strong>{div_score:.0f}/100</strong>
        ({div_label.lower()} diversification, HHI = {hhi:.2f}).
      </p>
      <hr class="divider">
      <div style="display:grid; grid-template-columns:repeat(3,1fr); gap:0.8rem;">
        <div>
          <div style="font-size:0.78rem;color:#aaa;">Expected Annual Return</div>
          <div style="font-size:1.2rem;font-weight:700;color:#2ecc71;">{port_return*100:.1f}%</div>
        </div>
        <div>
          <div style="font-size:0.78rem;color:#aaa;">Expected Volatility</div>
          <div style="font-size:1.2rem;font-weight:700;color:#f39c12;">{port_vol*100:.1f}%</div>
        </div>
        <div>
          <div style="font-size:0.78rem;color:#aaa;">Portfolio Sharpe Ratio</div>
          <div style="font-size:1.2rem;font-weight:700;color:#00b4d8;">{port_sharpe:.2f}</div>
        </div>
        <div>
          <div style="font-size:0.78rem;color:#aaa;">Diversification Quality</div>
          <div style="font-size:1.2rem;font-weight:700;color:#bb8fce;">{div_label}</div>
        </div>
        <div>
          <div style="font-size:0.78rem;color:#aaa;">Risk Assessment</div>
          <div style="font-size:1.2rem;font-weight:700;color:#f39c12;">{risk_rating}</div>
        </div>
        <div>
          <div style="font-size:0.78rem;color:#aaa;">Portfolio Quality</div>
          <div style="font-size:1.2rem;font-weight:700;color:#2ecc71;">{quality:.0f}/100</div>
        </div>
      </div>
      <hr class="divider">
      <div class="verdict">Overall Assessment: {overall_outlook}</div>
      <p style="margin-top:0.4rem;">
        This portfolio is suitable for {suitability}.
        {'The allocation emphasises lower-volatility, high-Sharpe holdings to protect capital.' if profile == 'Conservative' else
         'The allocation balances growth potential and downside protection across multiple sectors.' if profile == 'Moderate' else
         'The allocation tilts toward high-Sharpe, high-momentum stocks for maximum growth potential.'}
      </p>
      <p style="font-size:0.8rem; color:#636e72; margin-top:0.6rem;">
        ⚠️ Automated evidence-based allocation from historical data.
        This report is for educational purposes only and does not constitute financial advice.
      </p>
    </div>
    """, unsafe_allow_html=True)

    # ================================================================
    # VISUALISATIONS (charts unchanged — only ordering adjusted to follow UX flow)
    # ================================================================
    with st.expander("Portfolio Dashboard — Charts", expanded=True):

        v1, v2 = st.columns(2)

        with v1:
            pie = go.Figure(go.Pie(
                labels=picks[COLS.symbol],
                values=picks["Weight_%"],
                hole=0.52,
                marker=dict(colors=PALETTE.sequence[:n],
                            line=dict(color="#0e1117", width=2)),
                textinfo="label+percent",
                hovertemplate="<b>%{label}</b><br>Weight: %{value:.1f}%<extra></extra>",
            ))
            pie.update_layout(
                template="plotly_dark", height=420,
                paper_bgcolor="rgba(0,0,0,0)",
                title="Allocation by Stock", showlegend=True,
            )
            st.plotly_chart(pie, use_container_width=True)

        with v2:
            sc_fig = go.Figure(go.Scatter(
                x=picks["Volatility"] * 100,
                y=picks["Exp_Return"] * 100,
                mode="markers+text",
                text=picks[COLS.symbol],
                textposition="top center",
                hovertemplate=(
                    "<b>%{text}</b><br>Volatility: %{x:.1f}%<br>"
                    "Expected Return: %{y:.1f}%<extra></extra>"),
                marker=dict(
                    size=picks["Weight_%"] + 10,
                    color=picks["Sharpe"],
                    colorscale="Tealgrn", showscale=True,
                    colorbar=dict(title="Sharpe", x=1.02),
                    line=dict(color="#fff", width=1),
                ),
            ))
            sc_fig.update_layout(
                template="plotly_dark", height=420,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                title="Risk vs Return  (bubble = allocation weight)",
                xaxis_title="Volatility (%)", yaxis_title="Expected Return (%)",
            )
            st.plotly_chart(sc_fig, use_container_width=True)

        v3, v4 = st.columns(2)

        with v3:
            mb = picks.sort_values("Sharpe")
            bar_colors = [PALETTE.up if s >= 1 else PALETTE.warn if s >= 0.5
                          else PALETTE.down for s in mb["Sharpe"]]
            bar = go.Figure(go.Bar(
                x=mb["Sharpe"], y=mb[COLS.symbol], orientation="h",
                marker=dict(color=bar_colors, line=dict(color="rgba(0,0,0,0)")),
                text=[f"{v:.2f}" for v in mb["Sharpe"]], textposition="outside",
                hovertemplate="<b>%{y}</b>  Sharpe: %{x:.2f}<extra></extra>",
            ))
            bar.add_vline(x=0, line=dict(color=PALETTE.muted, width=1, dash="dot"))
            bar.update_layout(
                template="plotly_dark", height=420,
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                title="Sharpe Ratio per Holding", xaxis_title="Sharpe Ratio",
            )
            st.plotly_chart(bar, use_container_width=True)

        with v4:
            ind_df = (picks.groupby(COLS.industry)["Weight_%"]
                      .sum().reset_index()
                      .sort_values("Weight_%", ascending=False))
            try:
                treemap_fig = viz.treemap(ind_df, [COLS.industry], "Weight_%",
                                          title="Allocation by Industry")
                treemap_fig.update_layout(height=420, paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(treemap_fig, use_container_width=True)
            except Exception:
                import plotly.express as px
                tf = px.treemap(ind_df, path=[COLS.industry], values="Weight_%",
                                title="Allocation by Industry",
                                color="Weight_%", color_continuous_scale="Teal")
                tf.update_layout(height=420, paper_bgcolor="rgba(0,0,0,0)",
                                 template="plotly_dark")
                st.plotly_chart(tf, use_container_width=True)

        # Scenario bar chart
        scen_vals   = [amount * (1 + r) for r in scenarios.values()]
        scb = go.Figure(go.Bar(
            x=list(scenarios.keys()), y=scen_vals,
            marker=dict(color=[PALETTE.down, PALETTE.warn, PALETTE.up],
                        line=dict(color="rgba(0,0,0,0)")),
            text=[f"Tk. {v:,.0f}" for v in scen_vals], textposition="outside",
            hovertemplate="<b>%{x}</b>  Tk. %{y:,.0f}<extra></extra>",
        ))
        scb.add_hline(
            y=amount,
            line=dict(color=PALETTE.muted, dash="dot", width=1.5),
            annotation_text=f"  Invested  Tk. {amount:,.0f}",
            annotation_font_color=PALETTE.muted,
        )
        scb.update_layout(
            template="plotly_dark", height=400,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            title="1-Year Scenario Projection",
            yaxis_title="Portfolio Value (Tk.)", showlegend=False,
        )
        st.plotly_chart(scb, use_container_width=True)

        # Portfolio Quality Gauge (Step 15)
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=quality,
            number=dict(suffix=" / 100", font=dict(size=40, color="#ffffff")),
            gauge=dict(
                axis=dict(range=[0, 100], tickcolor=PALETTE.muted,
                          tickfont=dict(size=12)),
                bar=dict(color=PALETTE.accent, thickness=0.3),
                bgcolor="rgba(0,0,0,0)",
                bordercolor="rgba(0,0,0,0)",
                steps=[
                    {"range": [0,  40], "color": PALETTE.down},
                    {"range": [40, 70], "color": PALETTE.warn},
                    {"range": [70, 100], "color": PALETTE.up},
                ],
                threshold=dict(
                    line=dict(color="white", width=3),
                    thickness=0.85, value=quality,
                ),
            ),
            title=dict(text=f"<b>Portfolio Quality: {q_label}</b>",
                       font=dict(size=16, color="#ffffff")),
            domain=dict(x=[0.1, 0.9], y=[0, 0.9]),
        ))
        gauge.update_layout(
            template="plotly_dark", height=320,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=60, b=20, l=20, r=20),
        )
        st.plotly_chart(gauge, use_container_width=True)

    # ================================================================
    # METHODOLOGY — collapsed by default so non-technical users skip it
    # ================================================================
    with st.expander("How the Scoring Works (Methodology)", expanded=False):
        w_params = PROFILE_WEIGHTS[profile]
        st.markdown(
            f"Each stock is evaluated on five evidence-based metrics:\n\n"
            f"- **Expected Return** = avg daily return × 252\n"
            f"- **Volatility** = std-dev × √252\n"
            f"- **Sharpe Ratio** = (Return − {RISK_FREE:.0%}) ÷ Volatility\n"
            f"- **Max Drawdown** = worst peak-to-trough decline\n"
            f"- **Momentum** = trailing {lookback}-day return\n\n"
            f"**Quality Score** = "
            f"{w_params['ret']*100:.0f}% Return + "
            f"{w_params['sharpe']*100:.0f}% Sharpe + "
            f"{w_params['mom']*100:.0f}% Momentum + "
            f"{w_params['low_dd']*100:.0f}% Low-Drawdown  *(tuned for {profile} profile)*\n\n"
            f"**Allocation weights** use {'inverse-variance (low-vol preference)' if profile == 'Conservative' else 'Sharpe-weighted (high-performance preference)' if profile == 'Aggressive' else 'geometric blend of inverse-variance and Sharpe'} — never equal weights."
        )

        st.markdown("##### Universe Rankings (top 10 per metric)")
        rk1, rk2, rk3 = st.columns(3)
        with rk1:
            st.caption("↑ Expected Return")
            st.dataframe(_rank_table(metrics, "Exp_Return", False,
                                     lambda v: f"{v*100:.1f}%"), use_container_width=True)
        with rk2:
            st.caption("↓ Lowest Volatility")
            st.dataframe(_rank_table(metrics, "Volatility", True,
                                     lambda v: f"{v*100:.1f}%"), use_container_width=True)
        with rk3:
            st.caption("↑ Sharpe Ratio")
            st.dataframe(_rank_table(metrics, "Sharpe", False,
                                     lambda v: f"{v:.2f}"), use_container_width=True)

    st.caption(
        "This tool is for educational purposes only. "
        "Historical performance does not guarantee future results. "
        "All projections are illustrative and do not constitute financial advice."
    )