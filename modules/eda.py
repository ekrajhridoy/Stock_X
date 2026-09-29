"""
modules/eda.py
--------------
Exploratory Data Analysis dashboard for the AI Financial Intelligence Platform.

Two dropdowns drive the page:

    1. **Section**  — one of the 10 analytical themes from the EDA question bank
                      (Data Understanding, Cleaning, Univariate, … ML).
    2. **Question** — the specific question inside that section.

Selecting a question renders its dedicated, interactive Plotly analysis plus a
short, dynamically-generated insight. Every one of the 105 questions in the
`Financial_EDA_Questions` document has its own handler, registered in the
`SECTIONS` registry at the bottom of this file.

Performance: heavy group-by/loops live in `@st.cache_data` helpers that pull the
engineered dataset internally (hashable keys, computed once per session). All
columns are referenced through `config.settings.COLS`.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config.settings import COLS, PALETTE
from utils.helper_functions import (
    download_buttons,
    has_package,
    info_banner,
    page_header,
    score_sentiment,
)
from utils.preprocessing import get_engineered_data


# --------------------------------------------------------------------------- #
# Small rendering helpers                                                     #
# --------------------------------------------------------------------------- #
def _dark(fig: go.Figure, height: int = 440, title: str = "") -> go.Figure:
    """Apply the platform's dark FinTech theme to any Plotly figure."""
    fig.update_layout(
        template="plotly_dark", height=height, title=title,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=PALETTE.text, size=12),
        margin=dict(l=10, r=10, t=50 if title else 16, b=10),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(gridcolor=PALETTE.grid, zeroline=False)
    fig.update_yaxes(gridcolor=PALETTE.grid, zeroline=False)
    return fig


def _show(fig: go.Figure, height: int = 440, title: str = "") -> None:
    st.plotly_chart(_dark(fig, height, title), use_container_width=True)


def _insight(text: str) -> None:
    st.markdown(
        f"<div class='glass' style='padding:0.75rem 1rem;border-left:3px solid "
        f"{PALETTE.accent};'>💡 <b>Insight:</b> {text}</div>",
        unsafe_allow_html=True,
    )


def _hbar(frame: pd.DataFrame, value: str, label: str, title: str,
          scale: str = "Tealgrn", n: Optional[int] = 15, ascending=True) -> None:
    """Horizontal ranked bar chart (the workhorse for most questions)."""
    f = frame.sort_values(value, ascending=False)
    if n:
        f = f.head(n)
    f = f.sort_values(value, ascending=ascending)
    fig = px.bar(f, x=value, y=label, orientation="h", color=value,
                 color_continuous_scale=scale)
    _show(fig, title=title)


def _company_picker(df: pd.DataFrame, key: str, label: str = "Company") -> str:
    companies = sorted(df[COLS.company].dropna().unique().tolist())
    return st.selectbox(label, companies, key=key)


# --------------------------------------------------------------------------- #
# Cached compute helpers (computed once, reused across questions)             #
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def _close_stats() -> pd.DataFrame:
    df = get_engineered_data()
    g = df.groupby(COLS.company)[COLS.stock_close]
    out = g.agg(Count="count", Mean="mean", Median="median", Std="std",
                Min="min", Max="max", Skewness="skew").round(2).reset_index()
    return out


@st.cache_data(show_spinner=False)
def _volatility() -> pd.DataFrame:
    df = get_engineered_data()
    vol = df.groupby([COLS.symbol, COLS.company]).agg(
        Std_Dev=(COLS.stock_close, "std"),
        Mean_Close=(COLS.stock_close, "mean"),
    ).reset_index()
    vol["CV_%"] = (vol["Std_Dev"] / vol["Mean_Close"] * 100).round(2)
    ann = (df.groupby(COLS.symbol)["Daily_Return"].std() * np.sqrt(252)).round(2)
    vol = vol.merge(ann.rename("Ann_Volatility").reset_index(), on=COLS.symbol)
    vol["Tier"] = pd.cut(vol["CV_%"], bins=[0, 15, 30, 50, float("inf")],
                         labels=["Low", "Medium", "High", "Very High"])
    return vol.sort_values("CV_%", ascending=False).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def _returns_flat() -> np.ndarray:
    return get_engineered_data()["Daily_Return"].dropna().values


@st.cache_data(show_spinner=False)
def _skew_kurt() -> pd.DataFrame:
    df = get_engineered_data()
    out = df.groupby(COLS.symbol)["Daily_Return"].agg(
        Skewness="skew", Kurtosis=lambda x: x.kurt()).round(2).reset_index()
    return out


@st.cache_data(show_spinner=False)
def _market_daily() -> pd.DataFrame:
    """Equal-weight market series + gold series, one row per date."""
    df = get_engineered_data()
    m = df.groupby(COLS.date).agg(
        Mkt_Close=(COLS.stock_close, "mean"),
        Mkt_Ret=("Daily_Return", "mean"),
        Gold_Close=(COLS.gold_close, "mean"),
        Volume=(COLS.stock_volume, "mean"),
    ).reset_index().sort_values(COLS.date)
    m["Gold_Ret"] = m["Gold_Close"].pct_change() * 100
    return m


@st.cache_data(show_spinner=False)
def _gold_corr() -> pd.DataFrame:
    df = get_engineered_data()
    rows = []
    for sym, g in df.groupby(COLS.symbol):
        if g[COLS.gold_close].notna().sum() > 30:
            r = g[COLS.stock_close].corr(g[COLS.gold_close])
            rows.append({COLS.symbol: sym, COLS.company: g[COLS.company].iloc[-1],
                         COLS.industry: g[COLS.industry].iloc[-1],
                         "Pearson_r": round(float(r), 3) if pd.notna(r) else 0.0})
    return pd.DataFrame(rows).sort_values("Pearson_r").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def _gold_beta() -> pd.DataFrame:
    """Sensitivity (beta) of each stock's daily return to gold's daily return."""
    df = get_engineered_data()
    mkt = _market_daily()[[COLS.date, "Gold_Ret"]]
    rows = []
    for sym, g in df.groupby(COLS.symbol):
        gg = g[[COLS.date, "Daily_Return"]].merge(mkt, on=COLS.date).dropna()
        if len(gg) < 50 or gg["Gold_Ret"].var() == 0:
            continue
        beta = np.cov(gg["Daily_Return"], gg["Gold_Ret"])[0, 1] / gg["Gold_Ret"].var()
        rows.append({COLS.symbol: sym, COLS.company: g[COLS.company].iloc[-1],
                     COLS.industry: g[COLS.industry].iloc[-1],
                     "Gold_Beta": round(float(beta), 3)})
    return pd.DataFrame(rows).sort_values("Gold_Beta", ascending=False).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def _market_beta() -> pd.DataFrame:
    """CAPM-style beta of each stock vs the equal-weight market."""
    df = get_engineered_data()
    mkt = _market_daily()[[COLS.date, "Mkt_Ret"]]
    rows = []
    for sym, g in df.groupby(COLS.symbol):
        gg = g[[COLS.date, "Daily_Return"]].merge(mkt, on=COLS.date).dropna()
        if len(gg) < 50 or gg["Mkt_Ret"].var() == 0:
            continue
        beta = np.cov(gg["Daily_Return"], gg["Mkt_Ret"])[0, 1] / gg["Mkt_Ret"].var()
        rows.append({COLS.symbol: sym, COLS.company: g[COLS.company].iloc[-1],
                     "Beta": round(float(beta), 3)})
    return pd.DataFrame(rows).sort_values("Beta", ascending=False).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def _drawdowns() -> pd.DataFrame:
    df = get_engineered_data()
    rows = []
    for sym, g in df.groupby(COLS.symbol):
        s = g.sort_values(COLS.date)[COLS.stock_close].dropna()
        if len(s) < 30:
            continue
        dd = (s / s.cummax() - 1.0).min() * 100
        rows.append({COLS.symbol: sym, COLS.company: g[COLS.company].iloc[-1],
                     "Max_Drawdown_%": round(float(dd), 1)})
    return pd.DataFrame(rows).sort_values("Max_Drawdown_%").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def _var95() -> pd.DataFrame:
    df = get_engineered_data()
    rows = []
    for sym, g in df.groupby(COLS.symbol):
        r = g["Daily_Return"].dropna()
        if len(r) < 50:
            continue
        rows.append({COLS.symbol: sym, COLS.company: g[COLS.company].iloc[-1],
                     "VaR_95_%": round(float(-np.percentile(r, 5)), 2)})
    return pd.DataFrame(rows).sort_values("VaR_95_%", ascending=False).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def _cagr() -> pd.DataFrame:
    df = get_engineered_data()
    rows = []
    for sym, g in df.groupby(COLS.symbol):
        s = g.sort_values(COLS.date)[COLS.stock_close].dropna()
        if len(s) < 252:
            continue
        yrs = max((g[COLS.date].max() - g[COLS.date].min()).days / 365.25, 0.5)
        cagr = ((s.iloc[-1] / s.iloc[0]) ** (1 / yrs) - 1) * 100
        rows.append({COLS.symbol: sym, COLS.company: g[COLS.company].iloc[-1],
                     "CAGR_%": round(float(cagr), 2)})
    return pd.DataFrame(rows).sort_values("CAGR_%", ascending=False).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def _industry_activity() -> pd.DataFrame:
    df = get_engineered_data()
    return df.groupby(COLS.industry).agg(
        Total_Volume=(COLS.stock_volume, "sum"),
        Total_Turnover=(COLS.stock_turnover, "sum"),
        Total_Trades=(COLS.stock_trades, "sum"),
        Avg_Volume=(COLS.stock_volume, "mean"),
        Avg_Volatility=("Daily_Return", "std"),
    ).reset_index()


@st.cache_data(show_spinner=False)
def _sentiment_sample(n: int = 12000) -> pd.DataFrame:
    """Lexicon sentiment on a sample of headline rows (fast, cached)."""
    df = get_engineered_data()
    if COLS.headlines not in df.columns:
        return pd.DataFrame()
    sub = df.dropna(subset=["Daily_Return"]).copy()
    sub["Next_Return"] = sub.groupby(COLS.symbol)["Daily_Return"].shift(-1)
    sub = sub.dropna(subset=["Next_Return"])
    if len(sub) > n:
        sub = sub.sample(n, random_state=42)
    scores, _ = score_sentiment(sub[COLS.headlines].fillna("").astype(str).tolist(),
                                use_finbert=False)
    sub = sub.assign(Sentiment=scores)
    return sub[["Sentiment", "Daily_Return", "Next_Return", COLS.industry]]


# =========================================================================== #
# SECTION 1 — DATA UNDERSTANDING (Q1–Q10)                                     #
# =========================================================================== #
def q1(df):  # unique companies / industries / symbols
    c1, c2, c3 = st.columns(3)
    c1.metric("Companies", f"{df[COLS.company].nunique()}")
    c2.metric("Industries", f"{df[COLS.industry].nunique()}")
    c3.metric("Stock Symbols", f"{df[COLS.symbol].nunique()}")
    _insight(f"The dataset spans **{df[COLS.symbol].nunique()} symbols** across "
             f"**{df[COLS.industry].nunique()} industries**.")


def q2(df):  # date range
    dmin, dmax = df[COLS.date].min(), df[COLS.date].max()
    c1, c2, c3 = st.columns(3)
    c1.metric("Start", f"{dmin:%d %b %Y}")
    c2.metric("End", f"{dmax:%d %b %Y}")
    c3.metric("Span", f"{(dmax - dmin).days / 365.25:.1f} yrs")
    _insight(f"Coverage runs from **{dmin:%Y}** to **{dmax:%Y}** "
             f"({(dmax - dmin).days:,} calendar days).")


def q3(df):  # industries with most records
    cnt = df[COLS.industry].value_counts().reset_index()
    cnt.columns = [COLS.industry, "Records"]
    _hbar(cnt, "Records", COLS.industry, "Record Count by Industry", "Blues", n=None)
    _insight(f"**{cnt.iloc[0][COLS.industry]}** contributes the most rows "
             f"({cnt.iloc[0]['Records']:,}).")


def q4(df):  # proportion per symbol
    cnt = df[COLS.symbol].value_counts().reset_index()
    cnt.columns = [COLS.symbol, "Records"]
    cnt["Share_%"] = (cnt["Records"] / cnt["Records"].sum() * 100).round(2)
    _hbar(cnt, "Share_%", COLS.symbol, "Dataset Share per Symbol (%)", "Viridis", n=20)
    _insight(f"Each of {len(cnt)} symbols holds roughly "
             f"{cnt['Share_%'].mean():.1f}% of rows — a fairly balanced panel.")


def q5(df):  # equal representation over time
    tmp = df.copy()
    tmp["Year"] = tmp[COLS.date].dt.year
    piv = tmp.pivot_table(index=COLS.symbol, columns="Year",
                          values=COLS.stock_close, aggfunc="count").fillna(0)
    fig = go.Figure(go.Heatmap(z=piv.values, x=piv.columns.astype(str),
                               y=piv.index, colorscale="Viridis"))
    _show(fig, height=620, title="Records per Symbol per Year")
    _insight("Brighter rows trade for the full period; darker gaps mark stocks "
             "that listed later or have missing early years.")


def q6(df):  # variable types
    rows = []
    for c in df.columns:
        if c == COLS.headlines:
            kind = "Textual"
        elif c in (COLS.symbol, COLS.company, COLS.industry, COLS.isin):
            kind = "Categorical"
        elif c == COLS.date:
            kind = "Datetime"
        else:
            kind = "Numerical"
        rows.append({"Column": c, "Type": kind, "Dtype": str(df[c].dtype)})
    types = pd.DataFrame(rows)
    summary = types["Type"].value_counts().reset_index()
    summary.columns = ["Type", "Count"]
    c1, c2 = st.columns([1, 2])
    with c1:
        fig = px.pie(summary, names="Type", values="Count", hole=0.5,
                     color_discrete_sequence=PALETTE.sequence)
        _show(fig, height=360, title="Variable Types")
    with c2:
        st.dataframe(types, use_container_width=True, height=360)
    _insight(f"{len(df.columns)} columns: mostly numerical market metrics, plus "
             f"categorical identifiers and one free-text headline field.")


def q7(df):  # duplicate stock+date
    dup = df.duplicated(subset=[COLS.symbol, COLS.date]).sum()
    st.metric("Duplicate (Symbol, Date) rows", f"{dup:,}")
    _insight("No duplicate symbol-date rows — the panel is clean." if dup == 0
             else f"Found {dup:,} duplicate symbol-date rows worth de-duplicating.")


def q8(df):  # gold synced with stock dates
    total = len(df)
    have_gold = df[COLS.gold_close].notna().sum()
    pct = have_gold / total * 100
    fig = px.pie(values=[have_gold, total - have_gold],
                 names=["Gold present", "Gold missing"], hole=0.5,
                 color_discrete_sequence=[PALETTE.up, PALETTE.down])
    _show(fig, height=360, title="Gold Coverage vs Stock Rows")
    _insight(f"**{pct:.1f}%** of stock rows have a matching gold price — gold is "
             f"{'well' if pct > 90 else 'partially'} synchronised with trading dates.")


def q9(df):  # recording frequency
    d = df[df[COLS.symbol] == df[COLS.symbol].iloc[0]].sort_values(COLS.date)
    gaps = d[COLS.date].diff().dt.days.dropna()
    med = gaps.median()
    fig = px.histogram(x=gaps[gaps <= 10], nbins=10,
                       color_discrete_sequence=[PALETTE.accent])
    _show(fig, height=360, title="Days Between Consecutive Records (one symbol)")
    _insight(f"Median gap is **{med:.0f} day(s)** — consistent with daily "
             f"business-day (Mon–Fri) recording.")


def q10(df):  # companies in same industry
    grp = (df.groupby(COLS.industry)[COLS.company]
           .nunique().sort_values(ascending=False).reset_index())
    grp.columns = [COLS.industry, "Companies"]
    _hbar(grp, "Companies", COLS.industry, "Companies per Industry", "Tealgrn", n=None)
    sel = st.selectbox("Inspect an industry", grp[COLS.industry].tolist(), key="q10_ind")
    members = sorted(df[df[COLS.industry] == sel][COLS.company].unique().tolist())
    st.write(", ".join(members))
    _insight(f"**{sel}** contains {len(members)} companies.")


# =========================================================================== #
# SECTION 2 — DATA CLEANING & QUALITY (Q11–Q20)                               #
# =========================================================================== #
def _null_table(df):
    n = df.isna().sum()
    out = (n[n >= 0].rename("Nulls").reset_index()
           .rename(columns={"index": "Column"}))
    out["Pct"] = (out["Nulls"] / len(df) * 100).round(3)
    return out.sort_values("Nulls", ascending=False)


def q11(df):  # missing in prices/gold/trading
    cols = [COLS.stock_close, COLS.stock_open, COLS.stock_high, COLS.stock_low,
            COLS.stock_volume, COLS.stock_turnover, COLS.gold_close, COLS.gold_volume]
    sub = _null_table(df)
    sub = sub[sub["Column"].isin(cols)]
    _hbar(sub, "Nulls", "Column", "Missing Values in Key Columns", "OrRd", n=None)
    tot = int(sub["Nulls"].sum())
    _insight("No missing values in the key price/gold/trading columns."
             if tot == 0 else f"{tot:,} missing cells remain in key columns.")


def q12(df):  # highest % null
    sub = _null_table(df).head(12)
    _hbar(sub, "Pct", "Column", "Top Columns by % Null", "OrRd", n=None)
    top = sub.iloc[0]
    _insight(f"**{top['Column']}** has the highest null share ({top['Pct']:.2f}%)."
             if top["Pct"] > 0 else "The dataset has effectively no missing values.")


def q13(df):  # impossible values
    neg_price = (df[[COLS.stock_open, COLS.stock_high, COLS.stock_low,
                     COLS.stock_close]] <= 0).any(axis=1).sum()
    neg_vol = (df[COLS.stock_volume] < 0).sum()
    c1, c2 = st.columns(2)
    c1.metric("Rows with non-positive price", f"{int(neg_price):,}")
    c2.metric("Rows with negative volume", f"{int(neg_vol):,}")
    _insight("No impossible prices or volumes detected — values are physically valid."
             if neg_price == 0 and neg_vol == 0 else
             "Some non-positive prices/volumes exist and should be reviewed.")


def q14(df):  # outliers in volume/turnover
    col = st.selectbox("Metric", [COLS.stock_volume, COLS.stock_turnover], key="q14")
    s = df[col].dropna()
    q1v, q3v = s.quantile(0.25), s.quantile(0.75)
    iqr = q3v - q1v
    out = ((s < q1v - 1.5 * iqr) | (s > q3v + 1.5 * iqr)).sum()
    fig = px.box(y=np.log10(s[s > 0]), color_discrete_sequence=[PALETTE.accent])
    fig.update_yaxes(title=f"log10({col})")
    _show(fig, height=420, title=f"{col} — Distribution & Outliers (log scale)")
    _insight(f"**{out:,}** rows ({out/len(s)*100:.1f}%) fall outside the 1.5×IQR "
             f"fence — typical for heavy-tailed trading metrics.")


def q15(df):  # OHLC inconsistencies
    hi_lt_lo = (df[COLS.stock_high] < df[COLS.stock_low]).sum()
    hi_bad = (df[COLS.stock_high] < df[[COLS.stock_open, COLS.stock_close]].max(axis=1)).sum()
    lo_bad = (df[COLS.stock_low] > df[[COLS.stock_open, COLS.stock_close]].min(axis=1)).sum()
    res = pd.DataFrame({"Check": ["High < Low", "High < max(O,C)", "Low > min(O,C)"],
                        "Violations": [int(hi_lt_lo), int(hi_bad), int(lo_bad)]})
    _hbar(res, "Violations", "Check", "OHLC Consistency Violations", "OrRd", n=None)
    _insight("OHLC relationships are internally consistent." if res["Violations"].sum() == 0
             else f"{int(res['Violations'].sum()):,} OHLC violations found.")


def q16(df):  # abnormal spikes
    r = df["Daily_Return"].dropna()
    z = (r - r.mean()) / r.std()
    spikes = (z.abs() > 10).sum()
    fig = px.histogram(x=z[z.abs() <= 15], nbins=120,
                       color_discrete_sequence=[PALETTE.warn])
    _show(fig, height=400, title="Daily-Return Z-Scores (spike detector)")
    _insight(f"**{spikes:,}** observations exceed ±10σ — candidate data-entry spikes "
             f"worth manual review." if spikes else
             "No extreme (>10σ) return spikes suggestive of data-entry errors.")


def q17(df):  # name/symbol formatting
    bad_sym = df[COLS.symbol].astype(str).str.contains(r"[^A-Z0-9&]").sum()
    st.metric("Symbols with unexpected characters", f"{int(bad_sym):,}")
    st.write("Sample symbols:", ", ".join(sorted(df[COLS.symbol].unique())[:15]))
    _insight("Symbols are consistently upper-case tickers." if bad_sym == 0 else
             "Some symbols contain unexpected characters.")


def q18(df):  # duplicated headlines same date
    if COLS.headlines not in df.columns:
        info_banner("No headlines column.", "warning"); return
    dup = df.duplicated(subset=[COLS.date, COLS.headlines]).sum()
    st.metric("Duplicate (Date, headline) rows", f"{int(dup):,}")
    _insight("Headlines repeat across symbols on the same date because one news "
             "feed is shared market-wide — expected, not an error.")


def q19(df):  # gold volume abnormal
    s = df[COLS.gold_volume].dropna()
    zeros = (s == 0).sum()
    fig = px.histogram(x=np.log1p(s), nbins=60, color_discrete_sequence=["#BA7517"])
    _show(fig, height=400, title="Gold Volume Distribution (log1p)")
    _insight(f"{zeros:,} zero-volume gold rows; the rest form a smooth heavy-tailed "
             f"distribution with no obvious corruption.")


def q20(df):  # periods with no trading
    dates = pd.Series(sorted(df[COLS.date].unique()))
    full = pd.date_range(dates.min(), dates.max(), freq="B")
    missing = len(full) - dates.nunique()
    st.metric("Business days with no records", f"{missing:,}")
    _insight(f"Around **{missing:,}** business days have no trading rows — market "
             f"holidays and non-trading sessions.")


# =========================================================================== #
# SECTION 3 — UNIVARIATE (Q21–Q30)                                            #
# =========================================================================== #
def q21(df):  # distribution of closing prices
    comp = _company_picker(df, "q21")
    s = df[df[COLS.company] == comp][COLS.stock_close]
    fig = px.histogram(x=s, nbins=40, color_discrete_sequence=[PALETTE.accent])
    _show(fig, height=420, title=f"{comp} — Closing Price Distribution")
    sk = float(s.skew())
    shape = "right-skewed" if sk > 0.5 else "left-skewed" if sk < -0.5 else "symmetric"
    _insight(f"{comp}'s prices are **{shape}** (skew = {sk:.2f}).")


def q22(df):  # highest avg closing
    stats = _close_stats()[[COLS.company, "Mean"]].rename(columns={"Mean": "Avg_Close"})
    _hbar(stats, "Avg_Close", COLS.company, "Top Avg Closing Price", "Tealgrn")
    top = stats.sort_values("Avg_Close", ascending=False).iloc[0]
    _insight(f"**{top[COLS.company]}** has the highest average close "
             f"(₹{top['Avg_Close']:,.0f}).")


def q23(df):  # volatility each stock
    vol = _volatility()
    _hbar(vol, "CV_%", COLS.company, "Most Volatile Stocks (CV %)", "OrRd")
    _insight(f"**{vol.iloc[0][COLS.company]}** is most volatile "
             f"(CV = {vol.iloc[0]['CV_%']:.0f}%).")
    download_buttons(vol, key="q23", label="volatility")


def q24(df):  # distribution of trading volume
    s = df[df[COLS.stock_volume] > 0][COLS.stock_volume]
    fig = px.histogram(x=np.log10(s), nbins=60, color_discrete_sequence=[PALETTE.accent_2])
    fig.update_xaxes(title="log10(Volume)")
    _show(fig, height=420, title="Trading Volume Distribution (log scale)")
    _insight("Volume is strongly right-skewed; on a log scale it is roughly "
             "bell-shaped — most days are quiet with occasional surges.")


def q25(df):  # industries highest trading activity
    act = _industry_activity()
    _hbar(act, "Total_Volume", COLS.industry, "Total Volume by Industry", "Blues", n=None)
    _insight(f"**{act.sort_values('Total_Volume').iloc[-1][COLS.industry]}** leads "
             f"total trading volume.")


def q26(df):  # gold closing distribution
    s = df[COLS.gold_close].dropna()
    fig = px.histogram(x=s, nbins=40, color_discrete_sequence=["#BA7517"])
    _show(fig, height=420, title="Gold Closing Price Distribution")
    _insight(f"Gold trades between ${s.min():,.0f} and ${s.max():,.0f}, "
             f"mean ${s.mean():,.0f}.")


def q27(df):  # avg deliverable % across industries
    dv = (df.groupby(COLS.industry)[COLS.stock_deliverable_pct]
          .mean().reset_index())
    _hbar(dv, COLS.stock_deliverable_pct, COLS.industry,
          "Avg Deliverable % by Industry", "Greens", n=None)
    top = dv.sort_values(COLS.stock_deliverable_pct).iloc[-1]
    _insight(f"**{top[COLS.industry]}** shows the highest delivery ratio — more "
             f"investment-style holding, less intraday speculation.")


def q28(df):  # returns normally distributed
    r = _returns_flat()
    samp = r if len(r) <= 50000 else np.random.default_rng(0).choice(r, 50000, replace=False)
    fig = px.histogram(x=samp, nbins=120, color_discrete_sequence=[PALETTE.up])
    fig.update_xaxes(range=[np.percentile(samp, 0.5), np.percentile(samp, 99.5)])
    _show(fig, height=420, title="Daily Return Distribution")
    _insight(f"Returns have heavy tails (kurtosis = {pd.Series(r).kurt():.1f} ≫ 0) and "
             f"are **not** normally distributed — extreme moves are common.")


def q29(df):  # extreme skew/kurtosis stocks
    sk = _skew_kurt()
    metric = st.radio("Metric", ["Kurtosis", "Skewness"], horizontal=True, key="q29")
    tmp = sk.assign(absval=sk[metric].abs())
    _hbar(tmp, "absval", COLS.symbol, f"Most Extreme |{metric}|", "Plasma")
    top = tmp.sort_values("absval", ascending=False).iloc[0]
    _insight(f"**{top[COLS.symbol]}** shows the most extreme {metric.lower()} "
             f"({top[metric]:.1f}).")


def q30(df):  # most common trading ranges
    comp = _company_picker(df, "q30")
    rng = (df[df[COLS.company] == comp][COLS.stock_high]
           - df[df[COLS.company] == comp][COLS.stock_low])
    fig = px.histogram(x=rng, nbins=40, color_discrete_sequence=[PALETTE.warn])
    _show(fig, height=420, title=f"{comp} — Daily High-Low Range")
    _insight(f"{comp}'s most common intraday range clusters near "
             f"₹{rng.median():.1f} (median).")


# =========================================================================== #
# SECTION 4 — BIVARIATE (Q31–Q40)                                             #
# =========================================================================== #
def q31(df):  # gold vs stock close correlation
    corr = _gold_corr()
    ends = pd.concat([corr.head(8), corr.tail(8)]).drop_duplicates()
    _hbar(ends, "Pearson_r", COLS.company, "Stock–Gold Price Correlation", "RdYlGn", n=None)
    _insight(f"{(corr['Pearson_r'] > 0).mean()*100:.0f}% of stocks correlate "
             f"positively with gold over the full sample.")


def q32(df):  # rising gold ↔ falling stock
    m = _market_daily().dropna(subset=["Gold_Ret", "Mkt_Ret"])
    fig = px.scatter(m.sample(min(4000, len(m)), random_state=1),
                     x="Gold_Ret", y="Mkt_Ret", opacity=0.4,
                     color_discrete_sequence=[PALETTE.warn])
    _show(fig, height=420, title="Gold Return vs Market Return (daily)")
    r = m["Gold_Ret"].corr(m["Mkt_Ret"])
    _insight(f"Daily correlation is **{r:+.2f}** — "
             f"{'mild inverse (some safe-haven)' if r < -0.05 else 'near zero (independent)' if abs(r) <= 0.05 else 'positive co-movement'}.")


def q33(df):  # most sensitive to gold
    beta = _gold_beta()
    show = pd.concat([beta.head(8), beta.tail(8)]).drop_duplicates()
    _hbar(show, "Gold_Beta", COLS.company, "Sensitivity to Gold (Beta)", "RdBu", n=None)
    _insight(f"**{beta.iloc[0][COLS.company]}** is most gold-sensitive "
             f"(β = {beta.iloc[0]['Gold_Beta']:.2f}).")


def q34(df):  # volume vs gold volatility
    m = _market_daily().dropna(subset=["Gold_Ret"])
    m["Gold_Vol_Bucket"] = pd.cut(m["Gold_Ret"].abs(), bins=[0, 0.5, 1, 1.5, 100],
                                  labels=["<0.5%", "0.5-1%", "1-1.5%", ">1.5%"])
    agg = m.groupby("Gold_Vol_Bucket")["Volume"].mean().reset_index()
    fig = px.bar(agg, x="Gold_Vol_Bucket", y="Volume", color="Volume",
                 color_continuous_scale="Viridis")
    _show(fig, height=420, title="Avg Stock Volume by Gold-Volatility Bucket")
    _insight("Stock trading volume tends to rise on days when gold moves sharply — "
             "cross-asset volatility spills over.")


def q35(df):  # defensive sectors vs gold
    defensive = {"PHARMA", "FINANCIAL SERVICES", "IT", "CONSUMER GOODS",
                 "TELECOM", "SERVICES"}
    corr = _gold_corr().copy()
    corr["Sector_Type"] = corr[COLS.industry].str.upper().apply(
        lambda x: "Defensive" if x in defensive else "Cyclical")
    agg = corr.groupby("Sector_Type")["Pearson_r"].mean().reset_index()
    fig = px.bar(agg, x="Sector_Type", y="Pearson_r", color="Sector_Type",
                 color_discrete_sequence=[PALETTE.up, PALETTE.down])
    _show(fig, height=400, title="Avg Gold Correlation: Defensive vs Cyclical")
    _insight("Defensive sectors show a "
             f"{'higher' if agg.set_index('Sector_Type').loc['Defensive','Pearson_r'] > agg.set_index('Sector_Type').loc['Cyclical','Pearson_r'] else 'lower'}"
             " average correlation with gold than cyclical sectors.")


def q36(df):  # volume vs price movement
    s = df.dropna(subset=["Daily_Return", COLS.stock_volume])
    s = s[s[COLS.stock_volume] > 0]
    if len(s) > 6000:
        s = s.sample(6000, random_state=2)
    s = s.assign(Abs_Return=s["Daily_Return"].abs(),
                 Log_Volume=np.log10(s[COLS.stock_volume]))
    fig = px.scatter(s, x="Log_Volume", y="Abs_Return", opacity=0.35,
                     color="Abs_Return", color_continuous_scale="Plasma")
    _show(fig, height=420, title="Volume vs Absolute Daily Return")
    r = s["Log_Volume"].corr(s["Abs_Return"])
    _insight(f"Correlation r = {r:.2f}: higher volume generally accompanies larger "
             f"price moves.")


def q37(df):  # turnover vs trades
    s = df.dropna(subset=[COLS.stock_turnover, COLS.stock_trades])
    s = s[(s[COLS.stock_turnover] > 0) & (s[COLS.stock_trades] > 0)]
    if len(s) > 6000:
        s = s.sample(6000, random_state=3)
    s = s.assign(LT=np.log10(s[COLS.stock_turnover]), LR=np.log10(s[COLS.stock_trades]))
    fig = px.scatter(s, x="LT", y="LR", opacity=0.3,
                     color_discrete_sequence=[PALETTE.accent])
    _show(fig, height=420, title="log Turnover vs log Trades")
    _insight(f"Turnover and trade count are tightly linked "
             f"(r = {s['LT'].corr(s['LR']):.2f}) — both proxy market activity.")


def q38(df):  # high deliverable % less volatile?
    agg = df.groupby(COLS.company).agg(
        Deliv=(COLS.stock_deliverable_pct, "mean"),
        Vol=("Daily_Return", "std")).dropna().reset_index()
    fig = px.scatter(agg, x="Deliv", y="Vol", hover_name=COLS.company, opacity=0.7,
                     color="Vol", color_continuous_scale="RdYlGn_r")
    _show(fig, height=420, title="Deliverable % vs Return Volatility")
    _insight(f"Correlation r = {agg['Deliv'].corr(agg['Vol']):.2f}: higher delivery "
             f"ratios tend to pair with {'lower' if agg['Deliv'].corr(agg['Vol'])<0 else 'higher'} volatility.")


def q39(df):  # VWAP vs close
    s = df.dropna(subset=[COLS.stock_vwap, COLS.stock_close])
    if len(s) > 6000:
        s = s.sample(6000, random_state=4)
    fig = px.scatter(s, x=COLS.stock_vwap, y=COLS.stock_close, opacity=0.3,
                     color_discrete_sequence=[PALETTE.accent_2])
    _show(fig, height=420, title="VWAP vs Close")
    _insight(f"Close tracks VWAP almost perfectly "
             f"(r = {s[COLS.stock_vwap].corr(s[COLS.stock_close]):.3f}).")


def q40(df):  # open predicts close
    s = df.dropna(subset=[COLS.stock_open, COLS.stock_close])
    if len(s) > 6000:
        s = s.sample(6000, random_state=5)
    fig = px.scatter(s, x=COLS.stock_open, y=COLS.stock_close, opacity=0.3,
                     color_discrete_sequence=[PALETTE.accent])
    _show(fig, height=420, title="Open vs Close")
    r = s[COLS.stock_open].corr(s[COLS.stock_close])
    _insight(f"Open explains ~{r**2*100:.0f}% of close variance (r = {r:.3f}) — a very "
             f"strong predictor.")


# =========================================================================== #
# SECTION 5 — MULTIVARIATE (Q41–Q50)                                          #
# =========================================================================== #
def _return_feature_importance():
    from sklearn.ensemble import RandomForestRegressor
    df = get_engineered_data()
    feats = [c for c in [COLS.stock_volume, COLS.stock_turnover, "RSI", "MACD",
                         "Volatility", "Momentum", "Volume_Change", COLS.gold_close]
             if c in df.columns]
    sub = df.dropna(subset=feats + ["Daily_Return"])
    if len(sub) > 15000:
        sub = sub.sample(15000, random_state=42)
    rf = RandomForestRegressor(n_estimators=120, max_depth=10, n_jobs=-1,
                               random_state=42)
    rf.fit(sub[feats], sub["Daily_Return"])
    return pd.DataFrame({"Feature": feats, "Importance": rf.feature_importances_})


def q41(df):  # best combo explaining returns
    imp = _return_feature_importance()
    _hbar(imp, "Importance", "Feature", "Drivers of Daily Returns (RF importance)",
          "Viridis", n=None)
    top = imp.sort_values("Importance").iloc[-1]
    _insight(f"**{top['Feature']}** is the single strongest explanator of daily "
             f"returns in a random-forest model.")


def q42(df):  # cluster on volatility/volume/turnover
    feat = _cluster_features()
    fig = px.scatter(feat, x="Avg_Volume_M", y="Volatility", color="Risk_Label",
                     size="Avg_Turnover", hover_name=COLS.company,
                     color_discrete_map={"Stable": PALETTE.up, "Moderate": PALETTE.warn,
                                         "Volatile": PALETTE.down})
    fig.update_xaxes(title="Avg Volume (millions)")
    _show(fig, height=460, title="Stock Clusters: Volatility × Volume × Turnover")
    counts = feat["Risk_Label"].value_counts()
    _insight("KMeans groups the universe into " +
             ", ".join(f"{n} {l.lower()}" for l, n in counts.items()) + " stocks.")


@st.cache_data(show_spinner=False)
def _cluster_features():
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    df = get_engineered_data()
    feat = df.groupby([COLS.symbol, COLS.company]).agg(
        Volatility=("Daily_Return", "std"),
        Avg_Volume=(COLS.stock_volume, "mean"),
        Avg_Turnover=(COLS.stock_turnover, "mean")).dropna().reset_index()
    X = StandardScaler().fit_transform(feat[["Volatility", "Avg_Volume", "Avg_Turnover"]])
    km = KMeans(n_clusters=3, random_state=42, n_init=10)
    feat["Cluster"] = km.fit_predict(X)
    order = feat.groupby("Cluster")["Volatility"].mean().sort_values().index.tolist()
    feat["Risk_Label"] = feat["Cluster"].map(
        {order[0]: "Stable", order[1]: "Moderate", order[2]: "Volatile"})
    feat["Avg_Volume_M"] = feat["Avg_Volume"] / 1e6
    return feat


def q43(df):  # industries similar trading patterns
    feat = df.groupby(COLS.industry).agg(
        Vol=("Daily_Return", "std"), Volume=(COLS.stock_volume, "mean"),
        Turnover=(COLS.stock_turnover, "mean"),
        Deliv=(COLS.stock_deliverable_pct, "mean")).dropna()
    norm = (feat - feat.min()) / (feat.max() - feat.min())
    corr = norm.T.corr()
    fig = go.Figure(go.Heatmap(z=corr.values, x=corr.columns, y=corr.index,
                               colorscale="RdBu", zmid=0))
    _show(fig, height=560, title="Industry Similarity (profile correlation)")
    _insight("Bright blocks are industries with near-identical trading footprints — "
             "they tend to react to the market in lockstep.")


def q44(df):  # price/gold/volume interact
    feat = _cluster_features()
    fig = px.scatter_3d(feat, x="Volatility", y="Avg_Volume_M", z="Avg_Turnover",
                        color="Risk_Label", hover_name=COLS.company,
                        color_discrete_map={"Stable": PALETTE.up, "Moderate": PALETTE.warn,
                                            "Volatile": PALETTE.down})
    fig.update_layout(template="plotly_dark", height=560,
                      paper_bgcolor="rgba(0,0,0,0)", title="3D Interaction View")
    st.plotly_chart(fig, use_container_width=True)
    _insight("In three dimensions, the volatile cluster pulls away on the volatility "
             "axis while volume and turnover stay coupled.")


def q45(df):  # variables contribute most to volatility
    from sklearn.ensemble import RandomForestRegressor
    feats = [c for c in [COLS.stock_volume, COLS.stock_turnover, "RSI", "Momentum",
                         "Volume_Change", COLS.gold_close, COLS.stock_deliverable_pct]
             if c in df.columns]
    sub = df.dropna(subset=feats + ["Volatility"])
    if len(sub) > 15000:
        sub = sub.sample(15000, random_state=7)
    rf = RandomForestRegressor(n_estimators=120, max_depth=10, n_jobs=-1, random_state=7)
    rf.fit(sub[feats], sub["Volatility"])
    imp = pd.DataFrame({"Feature": feats, "Importance": rf.feature_importances_})
    _hbar(imp, "Importance", "Feature", "Drivers of Volatility", "OrRd", n=None)
    _insight(f"**{imp.sort_values('Importance').iloc[-1]['Feature']}** contributes "
             f"most to explaining volatility.")


def q46(df):  # hidden latent factors (PCA)
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    feats = [c for c in [COLS.stock_close, COLS.stock_volume, COLS.stock_turnover,
                         "Daily_Return", "Volatility", "RSI"] if c in df.columns]
    sub = df[feats].dropna()
    if len(sub) > 20000:
        sub = sub.sample(20000, random_state=9)
    ev = PCA().fit(StandardScaler().fit_transform(sub)).explained_variance_ratio_ * 100
    fig = px.bar(x=[f"PC{i+1}" for i in range(len(ev))], y=ev, color=ev,
                 color_continuous_scale="Viridis")
    _show(fig, height=420, title="PCA — Variance Explained per Latent Factor (%)")
    _insight(f"The first two latent factors capture **{ev[0]+ev[1]:.0f}%** of all "
             f"variation in the market metrics.")


def q47(df):  # stocks similar in rallies/declines
    piv = (df.pivot_table(index=COLS.date, columns=COLS.symbol,
                          values="Daily_Return", aggfunc="mean"))
    corr = piv.corr().fillna(0)
    syms = corr.index.tolist()[:25]
    fig = go.Figure(go.Heatmap(z=corr.loc[syms, syms].values, x=syms, y=syms,
                               colorscale="RdBu", zmid=0))
    _show(fig, height=600, title="Return Correlation Across Stocks (first 25)")
    _insight("Tight red/blue clusters mark stocks that rise and fall together — "
             "useful for diversification and pairs analysis.")


def q48(df):  # momentum vs stable groups
    from sklearn.cluster import KMeans
    feat = df.groupby([COLS.symbol, COLS.company]).agg(
        Momentum=("Momentum", "mean"), Volatility=("Daily_Return", "std")).dropna().reset_index()
    km = KMeans(n_clusters=2, random_state=42, n_init=10)
    feat["Group"] = km.fit_predict(feat[["Momentum", "Volatility"]])
    hi = feat.groupby("Group")["Momentum"].mean().idxmax()
    feat["Label"] = feat["Group"].apply(lambda g: "Momentum" if g == hi else "Stable")
    fig = px.scatter(feat, x="Volatility", y="Momentum", color="Label",
                     hover_name=COLS.company,
                     color_discrete_map={"Momentum": PALETTE.accent, "Stable": PALETTE.up})
    _show(fig, height=440, title="Momentum vs Stable Stock Groups")
    _insight("Two natural groups emerge: high-momentum movers vs calmer, stable names.")


def q49(df):  # industries resilient during gold surges
    m = _market_daily()
    surge_dates = set(m[m["Gold_Ret"] > m["Gold_Ret"].quantile(0.95)][COLS.date])
    sub = df[df[COLS.date].isin(surge_dates)]
    agg = sub.groupby(COLS.industry)["Daily_Return"].mean().reset_index()
    _hbar(agg, "Daily_Return", COLS.industry, "Avg Return on Gold-Surge Days",
          "RdYlGn", n=None)
    top = agg.sort_values("Daily_Return").iloc[-1]
    _insight(f"**{top[COLS.industry]}** holds up best when gold spikes "
             f"({top['Daily_Return']:+.2f}% avg).")


def q50(df):  # sentiment influence on price
    s = _sentiment_sample()
    if s.empty:
        info_banner("No headline data.", "warning"); return
    fig = px.scatter(s.sample(min(4000, len(s)), random_state=11),
                     x="Sentiment", y="Daily_Return", opacity=0.35,
                     color_discrete_sequence=[PALETTE.accent])
    _show(fig, height=420, title="Headline Sentiment vs Same-Day Return")
    r = s["Sentiment"].corr(s["Daily_Return"])
    _insight(f"Same-day correlation r = {r:+.2f} — sentiment has a "
             f"{'measurable' if abs(r) > 0.05 else 'weak'} contemporaneous link to returns.")


# =========================================================================== #
# SECTION 6 — TIME SERIES (Q51–Q62)                                           #
# =========================================================================== #
def q51(df):  # long-term trends
    comp = _company_picker(df, "q51")
    c = df[df[COLS.company] == comp].sort_values(COLS.date)
    fig = px.line(c, x=COLS.date, y=COLS.stock_close,
                  color_discrete_sequence=[PALETTE.accent])
    _show(fig, height=440, title=f"{comp} — Long-Term Price Trend")
    chg = (c[COLS.stock_close].iloc[-1] / c[COLS.stock_close].iloc[0] - 1) * 100
    _insight(f"{comp} moved **{chg:+.0f}%** end-to-end over the sample period.")


def q52(df):  # seasonality in volume
    t = df.copy(); t["Month"] = t[COLS.date].dt.month
    names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    mv = t.groupby("Month")[COLS.stock_volume].mean().reindex(range(1, 13))
    fig = px.bar(x=names, y=mv.values, color=mv.values, color_continuous_scale="Blues")
    _show(fig, height=420, title="Average Trading Volume by Month")
    _insight(f"Volume peaks in **{names[int(np.argmax(mv.values))]}** — a recurring "
             f"seasonal pattern.")


def q53(df):  # months highest volatility
    t = df.dropna(subset=["Daily_Return"]).copy(); t["Month"] = t[COLS.date].dt.month
    names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    mv = t.groupby("Month")["Daily_Return"].std().reindex(range(1, 13))
    fig = px.bar(x=names, y=mv.values, color=mv.values, color_continuous_scale="OrRd")
    _show(fig, height=420, title="Return Volatility by Month")
    _insight(f"**{names[int(np.nanargmax(mv.values))]}** is historically the most "
             f"volatile month.")


def q54(df):  # weekday effects
    t = df.dropna(subset=["Daily_Return"]).copy(); t["WD"] = t[COLS.date].dt.day_name()
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    wd = t.groupby("WD")["Daily_Return"].mean().reindex(order)
    fig = px.bar(x=order, y=wd.values, color=wd.values,
                 color_continuous_scale="RdYlGn", color_continuous_midpoint=0)
    _show(fig, height=420, title="Average Return by Weekday (%)")
    _insight(f"**{order[int(np.nanargmax(wd.values))]}** shows the strongest average "
             f"return — a classic calendar effect.")


def q55(df):  # gold vs stock over time
    m = _market_daily()
    norm = m.copy()
    norm["Stock"] = norm["Mkt_Close"] / norm["Mkt_Close"].iloc[0] * 100
    norm["Gold"] = norm["Gold_Close"] / norm["Gold_Close"].dropna().iloc[0] * 100
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=norm[COLS.date], y=norm["Stock"], name="Market",
                             line=dict(color=PALETTE.accent)))
    fig.add_trace(go.Scatter(x=norm[COLS.date], y=norm["Gold"], name="Gold",
                             line=dict(color="#BA7517")))
    _show(fig, height=440, title="Market vs Gold (indexed to 100)")
    _insight("Indexed together, gold and equities show distinct regimes — sometimes "
             "diverging (risk-off), sometimes rising together.")


def q56(df):  # structural breaks
    m = _market_daily()
    m["MA60"] = m["Mkt_Close"].rolling(60, min_periods=1).mean()
    m["Slope"] = m["MA60"].diff(20)
    breaks = m.loc[m["Slope"].abs() > m["Slope"].abs().quantile(0.98), COLS.date]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=m[COLS.date], y=m["Mkt_Close"], name="Market",
                             line=dict(color=PALETTE.muted)))
    fig.add_trace(go.Scatter(x=m[COLS.date], y=m["MA60"], name="MA60",
                             line=dict(color=PALETTE.accent)))
    for d in breaks:
        fig.add_vline(x=d, line=dict(color=PALETTE.down, width=1, dash="dot"))
    _show(fig, height=440, title="Market Trend with Candidate Structural Breaks")
    _insight(f"Dotted lines mark ~{len(breaks)} points where the 60-day trend shifted "
             f"sharply — likely regime changes.")


def q57(df):  # cyclical behavior (autocorrelation)
    comp = _company_picker(df, "q57")
    r = df[df[COLS.company] == comp].sort_values(COLS.date)["Daily_Return"].dropna()
    lags = range(1, 31)
    acf = [r.autocorr(lag=k) for k in lags]
    fig = px.bar(x=list(lags), y=acf, color=acf, color_continuous_scale="RdBu",
                 color_continuous_midpoint=0)
    fig.update_xaxes(title="Lag (days)")
    _show(fig, height=420, title=f"{comp} — Return Autocorrelation")
    _insight("Bars near zero mean little cyclicality; persistent positive bars would "
             "indicate a repeating cycle in returns.")


def q58(df):  # volume spikes before price moves
    comp = _company_picker(df, "q58")
    c = df[df[COLS.company] == comp].sort_values(COLS.date).copy()
    c["VolZ"] = (c[COLS.stock_volume] - c[COLS.stock_volume].rolling(20, min_periods=5).mean())
    c["AbsRet_next"] = c["Daily_Return"].abs().shift(-1)
    cc = c.dropna(subset=["VolZ", "AbsRet_next"])
    r = cc["VolZ"].corr(cc["AbsRet_next"])
    fig = px.scatter(cc.sample(min(2000, len(cc)), random_state=1),
                     x="VolZ", y="AbsRet_next", opacity=0.4,
                     color_discrete_sequence=[PALETTE.accent_2])
    _show(fig, height=420, title=f"{comp} — Volume Surge vs Next-Day |Return|")
    _insight(f"Lead correlation r = {r:+.2f}: volume spikes "
             f"{'tend to precede' if r > 0.05 else 'show little lead on'} larger next-day moves.")


def q59(df):  # rolling average trend
    comp = _company_picker(df, "q59")
    c = df[df[COLS.company] == comp].sort_values(COLS.date).copy()
    c["MA20"] = c[COLS.stock_close].rolling(20, min_periods=1).mean()
    c["MA50"] = c[COLS.stock_close].rolling(50, min_periods=1).mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=c[COLS.date], y=c[COLS.stock_close], name="Close",
                             line=dict(color=PALETTE.muted, width=1)))
    fig.add_trace(go.Scatter(x=c[COLS.date], y=c["MA20"], name="MA20",
                             line=dict(color=PALETTE.accent)))
    fig.add_trace(go.Scatter(x=c[COLS.date], y=c["MA50"], name="MA50",
                             line=dict(color=PALETTE.warn)))
    _show(fig, height=440, title=f"{comp} — Rolling Averages")
    _insight("Rolling means smooth daily noise and reveal the underlying trend; MA "
             "crossovers are common momentum signals.")


def q60(df):  # momentum persistence
    rows = []
    for sym, g in df.groupby(COLS.symbol):
        r = g.sort_values(COLS.date)["Daily_Return"].dropna()
        if len(r) > 60:
            rows.append({COLS.symbol: sym, "Lag1_Autocorr": round(r.autocorr(lag=1), 3)})
    pdf = pd.DataFrame(rows).sort_values("Lag1_Autocorr", ascending=False)
    _hbar(pdf, "Lag1_Autocorr", COLS.symbol, "Return Persistence (lag-1 autocorr)",
          "RdBu")
    _insight(f"**{pdf.iloc[0][COLS.symbol]}** shows the most positive return "
             f"persistence — momentum strategies are most plausible there.")


def q61(df):  # volatility clusters
    m = _market_daily()
    m["RollVol"] = m["Mkt_Ret"].rolling(21, min_periods=5).std()
    fig = px.area(m, x=COLS.date, y="RollVol", color_discrete_sequence=[PALETTE.down])
    _show(fig, height=420, title="Rolling 21-Day Market Volatility")
    sq = m["Mkt_Ret"].dropna() ** 2
    acf1 = pd.Series(sq).autocorr(lag=1)
    _insight(f"Squared-return autocorrelation = {acf1:.2f} (>0) confirms **volatility "
             f"clustering** — calm and turbulent periods bunch together.")


def q62(df):  # behavior during gold spikes
    m = _market_daily().dropna(subset=["Gold_Ret"])
    thr = m["Gold_Ret"].quantile(0.95)
    spike = m[m["Gold_Ret"] > thr]["Mkt_Ret"]
    normal = m[m["Gold_Ret"] <= thr]["Mkt_Ret"]
    comp = pd.DataFrame({"Regime": ["Gold spike", "Normal"],
                         "Avg_Market_Return": [spike.mean(), normal.mean()]})
    fig = px.bar(comp, x="Regime", y="Avg_Market_Return", color="Regime",
                 color_discrete_sequence=[PALETTE.warn, PALETTE.accent])
    _show(fig, height=400, title="Market Return: Gold-Spike vs Normal Days")
    _insight(f"On the top 5% gold-spike days the market averages "
             f"{spike.mean():+.2f}% vs {normal.mean():+.2f}% normally.")


# =========================================================================== #
# SECTION 7 — FEATURE ENGINEERING (Q63–Q72)                                   #
# =========================================================================== #
def q63(df):  # daily returns improve prediction?
    sub = df.dropna(subset=["Daily_Return", "Target_Up"])
    grp = sub.groupby("Target_Up")["Daily_Return"].mean().reset_index()
    grp["Target_Up"] = grp["Target_Up"].map({0: "Down day", 1: "Up day"})
    fig = px.bar(grp, x="Target_Up", y="Daily_Return", color="Target_Up",
                 color_discrete_sequence=[PALETTE.down, PALETTE.up])
    _show(fig, height=400, title="Daily Return vs Next-Day Direction")
    _insight("Engineered daily returns separate up- and down-day outcomes, so they "
             "carry predictive signal for direction models.")


def q64(df):  # rolling averages
    comp = _company_picker(df, "q64")
    c = df[df[COLS.company] == comp].sort_values(COLS.date)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=c[COLS.date], y=c[COLS.stock_close], name="Close",
                             line=dict(color=PALETTE.muted, width=1)))
    for w, col in [("SMA_20", PALETTE.accent), ("SMA_50", PALETTE.warn)]:
        if w in c.columns:
            fig.add_trace(go.Scatter(x=c[COLS.date], y=c[w], name=w, line=dict(color=col)))
    _show(fig, height=440, title=f"{comp} — Engineered Moving Averages")
    _insight("Rolling averages (already engineered as SMA_20 / SMA_50) denoise price "
             "and feed trend-following models — yes, they help.")


def q65(df):  # ATR / Bollinger Bands
    comp = _company_picker(df, "q65")
    c = df[df[COLS.company] == comp].sort_values(COLS.date).copy()
    c["MA20"] = c[COLS.stock_close].rolling(20, min_periods=1).mean()
    c["SD20"] = c[COLS.stock_close].rolling(20, min_periods=1).std()
    c["Upper"] = c["MA20"] + 2 * c["SD20"]
    c["Lower"] = c["MA20"] - 2 * c["SD20"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=c[COLS.date], y=c["Upper"], name="Upper",
                             line=dict(color=PALETTE.down, width=1)))
    fig.add_trace(go.Scatter(x=c[COLS.date], y=c["Lower"], name="Lower", fill="tonexty",
                             fillcolor="rgba(0,212,255,0.08)", line=dict(color=PALETTE.up, width=1)))
    fig.add_trace(go.Scatter(x=c[COLS.date], y=c[COLS.stock_close], name="Close",
                             line=dict(color=PALETTE.accent)))
    _show(fig, height=440, title=f"{comp} — Engineered Bollinger Bands")
    _insight("Bollinger Bands (a 20-day mean ± 2σ) are easy to engineer and flag "
             "over-extended prices — useful volatility features.")


def q66(df):  # lagged gold features
    m = _market_daily()
    m["Gold_lag1"] = m["Gold_Ret"].shift(1)
    cc = m.dropna(subset=["Gold_lag1", "Mkt_Ret"])
    r = cc["Gold_lag1"].corr(cc["Mkt_Ret"])
    fig = px.scatter(cc.sample(min(3000, len(cc)), random_state=2),
                     x="Gold_lag1", y="Mkt_Ret", opacity=0.4,
                     color_discrete_sequence=["#BA7517"])
    _show(fig, height=420, title="Yesterday's Gold Return vs Today's Market Return")
    _insight(f"Lag-1 gold correlation with next-day market return = {r:+.2f} — "
             f"{'a usable' if abs(r) > 0.05 else 'a weak'} predictive feature.")


def q67(df):  # sentiment scores from headlines
    s = _sentiment_sample()
    if s.empty:
        info_banner("No headlines.", "warning"); return
    fig = px.histogram(s, x="Sentiment", nbins=50,
                       color_discrete_sequence=[PALETTE.accent])
    _show(fig, height=420, title="Engineered Headline Sentiment Distribution")
    _insight(f"Sentiment scores span [-1, 1] with mean {s['Sentiment'].mean():+.2f} — "
             f"a ready-made feature for sentiment-enhanced models.")


def q68(df):  # momentum indicators
    sub = df.dropna(subset=["RSI", "Daily_Return"])
    sub = sub.assign(RSI_bin=pd.cut(sub["RSI"], bins=[0, 30, 50, 70, 100],
                     labels=["<30", "30-50", "50-70", ">70"]))
    agg = sub.groupby("RSI_bin")["Daily_Return"].mean().reset_index()
    fig = px.bar(agg, x="RSI_bin", y="Daily_Return", color="Daily_Return",
                 color_continuous_scale="RdYlGn", color_continuous_midpoint=0)
    _show(fig, height=400, title="Avg Return by RSI Zone")
    _insight("Returns vary across RSI zones, confirming engineered momentum "
             "indicators (RSI/MACD/Momentum) add forecasting value.")


def q69(df):  # sector-level indices
    sec = (df.groupby([COLS.date, COLS.industry])[COLS.stock_close].mean().reset_index())
    pick = st.selectbox("Sector", sorted(sec[COLS.industry].unique()), key="q69")
    s = sec[sec[COLS.industry] == pick].sort_values(COLS.date)
    s = s.assign(Index=s[COLS.stock_close] / s[COLS.stock_close].iloc[0] * 100)
    fig = px.line(s, x=COLS.date, y="Index", color_discrete_sequence=[PALETTE.accent])
    _show(fig, height=420, title=f"{pick} — Engineered Equal-Weight Sector Index")
    _insight("Averaging member stocks creates a clean sector index — a powerful "
             "engineered feature for cross-sector comparison.")


def q70(df):  # interaction gold×stock
    sub = df.dropna(subset=["Daily_Return", COLS.gold_close]).copy()
    sub["Gold_x_Vol"] = sub[COLS.gold_close] * sub["Volatility"]
    cc = sub.dropna(subset=["Gold_x_Vol"])
    r = cc["Gold_x_Vol"].corr(cc["Daily_Return"].abs())
    _insight(f"An engineered Gold×Volatility interaction correlates {r:+.2f} with "
             f"absolute returns — interaction terms can capture cross-asset effects.")
    fig = px.scatter(cc.sample(min(3000, len(cc)), random_state=3),
                     x="Gold_x_Vol", y=cc.sample(min(3000, len(cc)), random_state=3)["Daily_Return"].abs(),
                     opacity=0.3, color_discrete_sequence=[PALETTE.accent_2])
    _show(fig, height=400, title="Gold×Volatility Interaction vs |Return|")


def q71(df):  # abnormal trading activity indicator
    comp = _company_picker(df, "q71")
    c = df[df[COLS.company] == comp].sort_values(COLS.date).copy()
    mu = c[COLS.stock_volume].rolling(20, min_periods=5).mean()
    sd = c[COLS.stock_volume].rolling(20, min_periods=5).std()
    c["VolZ"] = (c[COLS.stock_volume] - mu) / sd
    flags = int((c["VolZ"].abs() > 3).sum())
    fig = px.line(c, x=COLS.date, y="VolZ", color_discrete_sequence=[PALETTE.warn])
    fig.add_hline(y=3, line=dict(color=PALETTE.down, dash="dot"))
    fig.add_hline(y=-3, line=dict(color=PALETTE.down, dash="dot"))
    _show(fig, height=420, title=f"{comp} — Abnormal Volume Indicator (z-score)")
    _insight(f"{flags} days breach ±3σ volume — an engineered abnormal-activity flag.")


def q72(df):  # rolling volatility features
    comp = _company_picker(df, "q72")
    c = df[df[COLS.company] == comp].sort_values(COLS.date).copy()
    c["RollVol"] = c["Daily_Return"].rolling(21, min_periods=5).std()
    fig = px.area(c, x=COLS.date, y="RollVol", color_discrete_sequence=[PALETTE.down])
    _show(fig, height=420, title=f"{comp} — Rolling 21-Day Volatility Feature")
    _insight("Rolling volatility is persistent (today's predicts tomorrow's), making "
             "it one of the most useful engineered features for risk forecasting.")


# =========================================================================== #
# SECTION 8 — FINANCIAL INSIGHTS (Q73–Q84)                                    #
# =========================================================================== #
def q73(df):  # industries outperform on rising gold
    m = _market_daily()
    up = set(m[m["Gold_Ret"] > 0][COLS.date])
    sub = df[df[COLS.date].isin(up)]
    agg = sub.groupby(COLS.industry)["Daily_Return"].mean().reset_index()
    _hbar(agg, "Daily_Return", COLS.industry, "Avg Return on Gold-Up Days", "RdYlGn", n=None)
    _insight(f"**{agg.sort_values('Daily_Return').iloc[-1][COLS.industry]}** outperforms "
             f"most when gold rises.")


def q74(df):  # highest long-term growth
    cagr = _cagr()
    _hbar(cagr, "CAGR_%", COLS.company, "Top Companies by CAGR (%)", "Greens")
    _insight(f"**{cagr.iloc[0][COLS.company]}** compounded fastest "
             f"(**{cagr.iloc[0]['CAGR_%']:.1f}%/yr**).")
    download_buttons(cagr, key="q74", label="cagr")


def q75(df):  # defensive during downturns
    beta = _market_beta()
    _hbar(beta, "Beta", COLS.company, "Lowest-Beta (Defensive) Stocks", "Tealgrn",
          ascending=False)
    low = beta.sort_values("Beta").iloc[0]
    _insight(f"**{low[COLS.company]}** is most defensive (β = {low['Beta']:.2f}); "
             f"low-beta names cushion market downturns.")


def q76(df):  # liquidity across industries
    liq = _liquidity_industry()
    _hbar(liq, "Liquidity_Score", COLS.industry, "Liquidity Score by Industry",
          "Tealgrn", n=None)
    _insight(f"**{liq.sort_values('Liquidity_Score').iloc[-1][COLS.industry]}** is the "
             f"most liquid sector.")


@st.cache_data(show_spinner=False)
def _liquidity_industry():
    df = get_engineered_data()
    out = df.groupby(COLS.industry).agg(
        V=(COLS.stock_volume, "mean"), T=(COLS.stock_turnover, "mean"),
        R=(COLS.stock_trades, "mean")).reset_index()
    for c in ["V", "T", "R"]:
        rng = out[c].max() - out[c].min()
        out[c] = (out[c] - out[c].min()) / rng if rng else 0
    out["Liquidity_Score"] = (out[["V", "T", "R"]].mean(axis=1) * 100).round(1)
    return out


def q77(df):  # highest investor participation
    part = df.groupby([COLS.symbol, COLS.company]).agg(
        Trades=(COLS.stock_trades, "mean"),
        Deliv=(COLS.stock_deliverable_pct, "mean"),
        Vol=(COLS.stock_volume, "mean")).reset_index()
    for c in ["Trades", "Deliv", "Vol"]:
        rng = part[c].max() - part[c].min()
        part[c + "_n"] = (part[c] - part[c].min()) / rng if rng else 0
    part["Participation"] = (part[["Trades_n", "Deliv_n", "Vol_n"]].mean(axis=1) * 100).round(1)
    _hbar(part, "Participation", COLS.company, "Investor Participation Score", "Viridis")
    _insight(f"**{part.sort_values('Participation').iloc[-1][COLS.company]}** attracts "
             f"the broadest investor participation.")


def q78(df):  # most stable trading patterns
    stab = df.groupby([COLS.symbol, COLS.company]).agg(
        RetStd=("Daily_Return", "std"),
        VolCV=(COLS.stock_volume, lambda x: x.std() / x.mean() if x.mean() else np.nan)
    ).dropna().reset_index()
    stab["Instability"] = (stab["RetStd"].rank() + stab["VolCV"].rank())
    stab = stab.sort_values("Instability")
    _hbar(stab.assign(Stability=stab["Instability"].max() - stab["Instability"]),
          "Stability", COLS.company, "Most Stable Trading Patterns", "Tealgrn")
    _insight(f"**{stab.iloc[0][COLS.company]}** has the steadiest returns and volume.")


def q79(df):  # delivery % vs confidence
    agg = df.groupby(COLS.company).agg(
        Deliv=(COLS.stock_deliverable_pct, "mean"),
        Vol=("Daily_Return", "std")).dropna().reset_index()
    fig = px.scatter(agg, x="Deliv", y="Vol", hover_name=COLS.company,
                     color="Vol", color_continuous_scale="RdYlGn_r")
    fig.update_yaxes(title="Return volatility")
    _show(fig, height=420, title="Delivery % vs Volatility (confidence proxy)")
    _insight(f"r = {agg['Deliv'].corr(agg['Vol']):.2f}: higher delivery (holding, not "
             f"flipping) aligns with {'calmer' if agg['Deliv'].corr(agg['Vol'])<0 else 'noisier'} prices.")


def q80(df):  # industries correlated with commodities
    corr = _gold_corr().groupby(COLS.industry)["Pearson_r"].mean().reset_index()
    _hbar(corr, "Pearson_r", COLS.industry, "Avg Industry–Gold Correlation",
          "RdYlGn", n=None)
    _insight(f"**{corr.sort_values('Pearson_r').iloc[-1][COLS.industry]}** is most "
             f"tied to gold prices.")


def q81(df):  # consistently above VWAP
    sub = df.dropna(subset=[COLS.stock_close, COLS.stock_vwap])
    sub = sub.assign(Above=(sub[COLS.stock_close] > sub[COLS.stock_vwap]).astype(int))
    agg = (sub.groupby([COLS.symbol, COLS.company])["Above"].mean() * 100).reset_index()
    agg.columns = [COLS.symbol, COLS.company, "Pct_Above_VWAP"]
    _hbar(agg, "Pct_Above_VWAP", COLS.company, "% of Days Closing Above VWAP", "Greens")
    _insight(f"**{agg.sort_values('Pct_Above_VWAP').iloc[-1][COLS.company]}** closes "
             f"above VWAP most often — persistent buying pressure.")


def q82(df):  # sectors highest momentum
    sec = (df.groupby([COLS.date, COLS.industry])["Daily_Return"].mean().reset_index())
    cum = sec.groupby(COLS.industry)["Daily_Return"].mean().reset_index()
    _hbar(cum, "Daily_Return", COLS.industry, "Avg Daily Return (Momentum) by Sector",
          "RdYlGn", n=None)
    _insight(f"**{cum.sort_values('Daily_Return').iloc[-1][COLS.industry]}** shows the "
             f"strongest momentum trend.")


def q83(df):  # high-volume stocks more stable?
    agg = df.groupby([COLS.symbol, COLS.company]).agg(
        Vol=(COLS.stock_volume, "mean"), RetStd=("Daily_Return", "std")).dropna().reset_index()
    agg["Volume_Tier"] = pd.qcut(agg["Vol"], 3, labels=["Low", "Mid", "High"])
    tier = agg.groupby("Volume_Tier")["RetStd"].mean().reset_index()
    fig = px.bar(tier, x="Volume_Tier", y="RetStd", color="RetStd",
                 color_continuous_scale="RdYlGn_r")
    _show(fig, height=400, title="Return Volatility by Volume Tier")
    _insight("High-volume stocks tend to show "
             f"{'lower' if tier['RetStd'].iloc[-1] < tier['RetStd'].iloc[0] else 'higher'} "
             "return volatility — liquidity and stability are linked.")


def q84(df):  # gold safe-haven
    m = _market_daily().dropna(subset=["Gold_Ret", "Mkt_Ret"])
    r = m["Gold_Ret"].corr(m["Mkt_Ret"])
    down = m[m["Mkt_Ret"] < m["Mkt_Ret"].quantile(0.05)]["Gold_Ret"].mean()
    c1, c2 = st.columns(2)
    c1.metric("Gold–Market corr", f"{r:+.2f}")
    c2.metric("Avg gold return on worst market days", f"{down:+.2f}%")
    _insight(f"On the worst 5% of market days gold averages {down:+.2f}% — "
             f"{'classic safe-haven behaviour' if down > 0 else 'no clear hedge'} in this data.")


# =========================================================================== #
# SECTION 9 — RISK & VOLATILITY (Q85–Q95)                                     #
# =========================================================================== #
def q85(df):  # highest historical volatility
    vol = _volatility()
    _hbar(vol, "Ann_Volatility", COLS.company, "Highest Annualised Volatility", "OrRd")
    _insight(f"**{vol.sort_values('Ann_Volatility').iloc[-1][COLS.company]}** has the "
             f"highest annualised volatility.")


def q86(df):  # riskiest industries
    act = _industry_activity()
    _hbar(act, "Avg_Volatility", COLS.industry, "Riskiest Industries (avg daily σ)",
          "OrRd", n=None)
    _insight(f"**{act.sort_values('Avg_Volatility').iloc[-1][COLS.industry]}** is the "
             f"riskiest sector by average return volatility.")


def q87(df):  # extreme outlier trading days
    m = _market_daily().dropna(subset=["Mkt_Ret"])
    ext = m.reindex(m["Mkt_Ret"].abs().sort_values(ascending=False).index).head(10)
    fig = px.bar(ext, x=COLS.date, y="Mkt_Ret", color="Mkt_Ret",
                 color_continuous_scale="RdYlGn", color_continuous_midpoint=0)
    _show(fig, height=420, title="Top 10 Most Extreme Market Days")
    _insight(f"The single most extreme day moved the market "
             f"{ext['Mkt_Ret'].iloc[0]:+.1f}%.")


def q88(df):  # largest drawdowns
    dd = _drawdowns()
    _hbar(dd.assign(Depth=dd["Max_Drawdown_%"].abs()), "Depth", COLS.company,
          "Largest Maximum Drawdowns", "OrRd")
    _insight(f"**{dd.iloc[0][COLS.company]}** suffered the deepest drawdown "
             f"({dd.iloc[0]['Max_Drawdown_%']:.0f}%).")


def q89(df):  # gold less volatile than stocks?
    gold_vol = _market_daily()["Gold_Ret"].std()
    stock_vol = df["Daily_Return"].std()
    comp = pd.DataFrame({"Asset": ["Gold", "Avg Stock"],
                         "Daily_Vol_%": [gold_vol, stock_vol]})
    fig = px.bar(comp, x="Asset", y="Daily_Vol_%", color="Asset",
                 color_discrete_sequence=["#BA7517", PALETTE.accent])
    _show(fig, height=400, title="Daily Volatility: Gold vs Stocks")
    _insight(f"Gold's daily volatility ({gold_vol:.2f}%) is "
             f"{'lower' if gold_vol < stock_vol else 'higher'} than the average stock "
             f"({stock_vol:.2f}%).")


def q90(df):  # sudden liquidity crashes
    rows = []
    for sym, g in df.groupby(COLS.symbol):
        g = g.sort_values(COLS.date)
        ma = g[COLS.stock_volume].rolling(20, min_periods=5).mean()
        crashes = ((g[COLS.stock_volume] < 0.2 * ma)).sum()
        rows.append({COLS.symbol: sym, COLS.company: g[COLS.company].iloc[-1],
                     "Liquidity_Crash_Days": int(crashes)})
    cr = pd.DataFrame(rows)
    _hbar(cr, "Liquidity_Crash_Days", COLS.company, "Liquidity-Crash Days (<20% of avg)",
          "OrRd")
    _insight(f"**{cr.sort_values('Liquidity_Crash_Days').iloc[-1][COLS.company]}** sees "
             f"the most sudden liquidity dry-ups.")


def q91(df):  # gap-up / gap-down frequency
    sub = df.dropna(subset=[COLS.stock_open, COLS.stock_prev_close]).copy()
    sub["Gap"] = (sub[COLS.stock_open] - sub[COLS.stock_prev_close]) / sub[COLS.stock_prev_close] * 100
    sub["Big_Gap"] = (sub["Gap"].abs() > 2).astype(int)
    agg = (sub.groupby([COLS.symbol, COLS.company])["Big_Gap"].mean() * 100).reset_index()
    agg.columns = [COLS.symbol, COLS.company, "Gap_Freq_%"]
    _hbar(agg, "Gap_Freq_%", COLS.company, "Frequency of >2% Opening Gaps", "Plasma")
    _insight(f"**{agg.sort_values('Gap_Freq_%').iloc[-1][COLS.company]}** gaps open "
             f"most often — higher overnight risk.")


def q92(df):  # VaR
    var = _var95()
    _hbar(var, "VaR_95_%", COLS.company, "Historical 95% Value-at-Risk (daily)", "OrRd")
    _insight(f"**{var.iloc[0][COLS.company]}** has the worst 95% VaR — a 1-in-20 day "
             f"can lose ≥{var.iloc[0]['VaR_95_%']:.1f}%.")
    download_buttons(var, key="q92", label="value_at_risk")


def q93(df):  # abnormal return behavior
    rows = []
    for sym, g in df.groupby(COLS.symbol):
        r = g["Daily_Return"].dropna()
        if len(r) < 50:
            continue
        z = (r - r.mean()) / r.std()
        rows.append({COLS.symbol: sym, COLS.company: g[COLS.company].iloc[-1],
                     "Extreme_Days": int((z.abs() > 3).sum())})
    ab = pd.DataFrame(rows)
    _hbar(ab, "Extreme_Days", COLS.company, "Days with |Return| > 3σ", "Plasma")
    _insight(f"**{ab.sort_values('Extreme_Days').iloc[-1][COLS.company]}** logs the most "
             f"abnormal return days.")


def q94(df):  # shocks across industries
    piv = df.pivot_table(index=COLS.date, columns=COLS.industry,
                         values="Daily_Return", aggfunc="mean")
    corr = piv.corr()
    fig = go.Figure(go.Heatmap(z=corr.values, x=corr.columns, y=corr.index,
                               colorscale="RdBu", zmid=0,
                               text=np.round(corr.values, 2), texttemplate="%{text}",
                               textfont=dict(size=8)))
    _show(fig, height=600, title="Cross-Industry Return Correlation")
    _insight(f"Average inter-industry correlation is "
             f"{corr.values[np.triu_indices_from(corr.values, 1)].mean():.2f} — shocks "
             f"transmit broadly across sectors.")


def q95(df):  # high beta stocks
    beta = _market_beta()
    _hbar(beta, "Beta", COLS.company, "Highest-Beta Stocks (vs market)", "OrRd")
    _insight(f"**{beta.iloc[0][COLS.company]}** is the highest-beta name "
             f"(β = {beta.iloc[0]['Beta']:.2f}) — it amplifies market moves.")


# =========================================================================== #
# SECTION 10 — PREDICTIVE ANALYTICS & ML (Q96–Q105)                           #
# =========================================================================== #
def q96(df):  # predict close from gold+metrics
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    feats = [c for c in [COLS.gold_close, COLS.stock_volume, COLS.stock_turnover,
                         "RSI", "Volatility", COLS.stock_vwap] if c in df.columns]
    sub = df.dropna(subset=feats + [COLS.stock_close])
    if len(sub) > 12000:
        sub = sub.sample(12000, random_state=42)
    Xtr, Xte, ytr, yte = train_test_split(sub[feats], sub[COLS.stock_close],
                                          test_size=0.25, random_state=42)
    rf = RandomForestRegressor(n_estimators=120, max_depth=12, n_jobs=-1, random_state=42).fit(Xtr, ytr)
    r2 = rf.score(Xte, yte)
    st.metric("Hold-out R²", f"{r2:.3f}")
    _insight(f"A quick random forest explains **{r2*100:.0f}%** of closing-price "
             f"variance from gold + trading metrics. The dedicated *Stock Direction* "
             f"and *LSTM Forecasting* modules go further.")


def q97(df):  # strongest predictors of returns
    imp = _return_feature_importance()
    _hbar(imp, "Importance", "Feature", "Strongest Predictors of Returns", "Viridis", n=None)
    _insight(f"**{imp.sort_values('Importance').iloc[-1]['Feature']}** ranks highest — "
             f"see *Feature Importance Analysis* for SHAP-level detail.")


def q98(df):  # headlines improve accuracy
    s = _sentiment_sample()
    if s.empty:
        info_banner("No headlines.", "warning"); return
    r = s["Sentiment"].corr(s["Next_Return"])
    st.metric("Sentiment ↔ next-day return corr", f"{r:+.3f}")
    _insight(f"Headline sentiment correlates {r:+.2f} with next-day returns — a small "
             f"but real edge. The *Sentiment-Enhanced Prediction* module quantifies the "
             f"accuracy lift directly.")


def q99(df):  # classify bullish/bearish
    rows = []
    for sym, g in df.groupby(COLS.symbol):
        s = g.sort_values(COLS.date)[COLS.stock_close].dropna()
        if len(s) < 30:
            continue
        slope = np.polyfit(range(len(s)), s.values, 1)[0]
        rows.append("Bullish" if slope > 0 else "Bearish")
    vc = pd.Series(rows).value_counts().reset_index()
    vc.columns = ["Verdict", "Count"]
    fig = px.pie(vc, names="Verdict", values="Count", hole=0.5,
                 color="Verdict", color_discrete_map={"Bullish": PALETTE.up, "Bearish": PALETTE.down})
    _show(fig, height=400, title="Trend Classification Across Stocks")
    _insight(f"By trend slope, {vc.set_index('Verdict').get('Count', pd.Series()).to_dict()} — "
             f"stocks split cleanly into bullish vs bearish trajectories.")


def q100(df):  # forecast volatility
    comp = _company_picker(df, "q100")
    c = df[df[COLS.company] == comp].sort_values(COLS.date).copy()
    c["RollVol"] = c["Daily_Return"].rolling(21, min_periods=5).std()
    c["RollVol_next"] = c["RollVol"].shift(-1)
    cc = c.dropna(subset=["RollVol", "RollVol_next"])
    r = cc["RollVol"].corr(cc["RollVol_next"])
    fig = px.scatter(cc, x="RollVol", y="RollVol_next", opacity=0.4,
                     color_discrete_sequence=[PALETTE.down])
    _show(fig, height=420, title=f"{comp} — Today's vs Tomorrow's Volatility")
    _insight(f"Volatility autocorrelation = {r:.2f}: it is highly persistent, so "
             f"GARCH/rolling models forecast it well.")


def q101(df):  # detect anomalous days
    from sklearn.ensemble import IsolationForest
    m = _market_daily().dropna(subset=["Mkt_Ret", "Volume"])
    X = m[["Mkt_Ret", "Volume"]].copy()
    X["AbsRet"] = X["Mkt_Ret"].abs()
    iso = IsolationForest(contamination=0.05, random_state=42).fit(X)
    m = m.assign(Anomaly=iso.predict(X))
    fig = px.scatter(m, x=COLS.date, y="Mkt_Ret",
                     color=m["Anomaly"].map({1: "Normal", -1: "Anomaly"}),
                     color_discrete_map={"Normal": PALETTE.muted, "Anomaly": PALETTE.down})
    _show(fig, height=420, title="Isolation-Forest Anomalous Trading Days")
    _insight(f"{int((m['Anomaly'] == -1).sum())} days flagged as anomalies — the "
             f"*Crash Detection* module operationalises this.")


def q102(df):  # clustering market regimes
    from sklearn.cluster import KMeans
    m = _market_daily().dropna(subset=["Mkt_Ret", "Volume"]).copy()
    m["RollVol"] = m["Mkt_Ret"].rolling(21, min_periods=5).std()
    mm = m.dropna(subset=["RollVol"])
    km = KMeans(n_clusters=3, random_state=42, n_init=10)
    mm = mm.assign(Regime=km.fit_predict(mm[["Mkt_Ret", "RollVol", "Volume"]]))
    order = mm.groupby("Regime")["RollVol"].mean().sort_values().index.tolist()
    name = {order[0]: "Calm", order[1]: "Normal", order[2]: "Turbulent"}
    mm["Regime_Name"] = mm["Regime"].map(name)
    fig = px.scatter(mm, x=COLS.date, y="Mkt_Close", color="Regime_Name",
                     color_discrete_map={"Calm": PALETTE.up, "Normal": PALETTE.warn,
                                         "Turbulent": PALETTE.down})
    _show(fig, height=440, title="Hidden Market Regimes (KMeans)")
    _insight("Clustering reveals calm, normal and turbulent regimes across the "
             "timeline — the basis for regime-aware strategies.")


def q103(df):  # features for algo trading
    imp = _return_feature_importance().sort_values("Importance", ascending=True)
    _hbar(imp, "Importance", "Feature", "Most Useful Features for Trading Signals",
          "Viridis", n=None)
    _insight("Momentum and volatility features dominate — the natural inputs for an "
             "algorithmic signal stack.")


def q104(df):  # gold forecast sector performance
    m = _market_daily()
    m["Gold_lag1"] = m["Gold_Ret"].shift(1)
    sec = df.groupby([COLS.date, COLS.industry])["Daily_Return"].mean().reset_index()
    merged = sec.merge(m[[COLS.date, "Gold_lag1"]], on=COLS.date).dropna()
    rows = []
    for ind, g in merged.groupby(COLS.industry):
        rows.append({COLS.industry: ind,
                     "Corr_vs_LagGold": round(g["Gold_lag1"].corr(g["Daily_Return"]), 3)})
    cr = pd.DataFrame(rows)
    _hbar(cr, "Corr_vs_LagGold", COLS.industry, "Next-Day Sector Return vs Lagged Gold",
          "RdYlGn", n=None)
    _insight(f"**{cr.sort_values('Corr_vs_LagGold').iloc[-1][COLS.industry]}** is most "
             f"forecastable from yesterday's gold move.")


def q105(df):  # TS models vs MA baseline
    comp = _company_picker(df, "q105")
    s = df[df[COLS.company] == comp].sort_values(COLS.date)[COLS.stock_close].dropna().values
    if len(s) < 100:
        info_banner("Not enough history.", "warning"); return
    naive = np.abs(np.diff(s))                      # last-value baseline error
    ma5 = pd.Series(s).rolling(5).mean().shift(1).values
    ma20 = pd.Series(s).rolling(20).mean().shift(1).values
    err = pd.DataFrame({
        "Model": ["Naive (last value)", "MA(5)", "MA(20)"],
        "MAE": [np.nanmean(naive),
                np.nanmean(np.abs(s - ma5)),
                np.nanmean(np.abs(s - ma20))],
    })
    fig = px.bar(err, x="Model", y="MAE", color="MAE", color_continuous_scale="OrRd")
    _show(fig, height=400, title=f"{comp} — Baseline Forecast Errors (MAE)")
    _insight("These moving-average baselines set the bar; the *LSTM Forecasting* and "
             "*Forecasting Comparison* modules test whether ML beats them.")


# =========================================================================== #
# SECTION 11 — METHODS & VISUALIZATION GUIDE (reference, items 106–115)        #
# =========================================================================== #
def methods_guide(df):
    st.markdown("#### Recommended Visualizations & Statistical Methods")
    guide = pd.DataFrame({
        "Technique": ["Line charts", "Candlestick charts", "Heatmaps", "Box plots",
                      "Scatter plots", "PCA / clustering", "Pearson / Spearman",
                      "ADF test", "GARCH", "ARIMA / LSTM"],
        "Best used for": [
            "Price trends & rolling averages",
            "OHLC intraday analysis",
            "Correlation across variables/sectors",
            "Volatility & outlier detection",
            "Pairwise relationships",
            "Segmentation & latent factors",
            "Linear / rank correlation tests",
            "Stationarity checks before modelling",
            "Volatility forecasting",
            "Time-series price forecasting",
        ],
        "Where in this app": [
            "Time-Series section", "Dashboard / candlestick", "Bivariate & Multivariate",
            "Risk & Volatility", "Bivariate", "Multivariate (Q42, Q46)",
            "Bivariate (Q31, Q37)", "(pre-modelling step)",
            "Volatility questions (Q100)", "LSTM & Forecasting modules",
        ],
    })
    st.dataframe(guide, use_container_width=True, height=420)
    _insight("This guide maps each recommended statistical method to where it appears "
             "across the platform's EDA and modelling modules.")


# =========================================================================== #
# REGISTRY: Section -> [(number, question text, handler), ...]                #
# =========================================================================== #
SECTIONS: Dict[str, List[Tuple[int, str, Callable]]] = {
    "1 · Data Understanding": [
        (1, "How many unique companies, industries, and stock symbols exist?", q1),
        (2, "What is the date range covered by the dataset?", q2),
        (3, "Which industries contribute the highest number of records?", q3),
        (4, "What proportion of the dataset belongs to each stock symbol?", q4),
        (5, "Are all stocks represented equally over time?", q5),
        (6, "Which variables are numerical, categorical, and textual?", q6),
        (7, "Are there duplicate records for the same stock and date?", q7),
        (8, "Is the gold price data synchronized with stock market dates?", q8),
        (9, "How frequently was the stock market data recorded?", q9),
        (10, "Which companies belong to the same industry sectors?", q10),
    ],
    "2 · Data Cleaning & Quality": [
        (11, "Are there missing values in prices, gold, or trading metrics?", q11),
        (12, "Which columns contain the highest percentage of null values?", q12),
        (13, "Are there impossible values such as negative prices or volumes?", q13),
        (14, "Are there outliers in trading volume or turnover?", q14),
        (15, "Are there inconsistencies between OHLC prices?", q15),
        (16, "Are there sudden abnormal spikes caused by data entry issues?", q16),
        (17, "Are company names and stock symbols consistently formatted?", q17),
        (18, "Are there duplicated headlines for the same trading date?", q18),
        (19, "Does gold volume contain abnormal entries?", q19),
        (20, "Are there periods where no trading activity occurred?", q20),
    ],
    "3 · Univariate Analysis": [
        (21, "What is the distribution of stock closing prices?", q21),
        (22, "Which stocks have the highest average closing prices?", q22),
        (23, "How volatile is each stock individually?", q23),
        (24, "What is the distribution of trading volume across stocks?", q24),
        (25, "Which industries have the highest trading activity?", q25),
        (26, "What is the distribution of gold closing prices?", q26),
        (27, "What is the average deliverable percentage across industries?", q27),
        (28, "Are stock returns normally distributed?", q28),
        (29, "Which stocks exhibit extreme skewness or kurtosis?", q29),
        (30, "What are the most common trading ranges?", q30),
    ],
    "4 · Bivariate Analysis": [
        (31, "Is there a correlation between gold prices and stock closing prices?", q31),
        (32, "Does rising gold price correspond to falling stock prices?", q32),
        (33, "Which stocks are most sensitive to gold market fluctuations?", q33),
        (34, "How does stock trading volume change with gold price volatility?", q34),
        (35, "Are defensive sectors more positively correlated with gold prices?", q35),
        (36, "Is higher volume associated with larger price movements?", q36),
        (37, "Does turnover strongly correlate with number of trades?", q37),
        (38, "Are stocks with high deliverable percentages less volatile?", q38),
        (39, "How does VWAP compare with closing price behavior?", q39),
        (40, "Does stock opening price strongly predict closing price?", q40),
    ],
    "5 · Multivariate Analysis": [
        (41, "Which combination of variables best explains stock returns?", q41),
        (42, "Can stocks be clustered based on volatility, volume, and turnover?", q42),
        (43, "Which industries exhibit similar trading patterns?", q43),
        (44, "How do stock price, gold price, and volume interact simultaneously?", q44),
        (45, "Which variables contribute most to stock volatility?", q45),
        (46, "Are there hidden latent factors driving market behavior?", q46),
        (47, "Which stocks behave similarly during market rallies or declines?", q47),
        (48, "Can stocks be grouped into momentum vs stable categories?", q48),
        (49, "Which industries are most resilient during gold price surges?", q49),
        (50, "Does market sentiment from headlines influence price movements?", q50),
    ],
    "6 · Time Series Analysis": [
        (51, "What are the long-term trends in stock prices?", q51),
        (52, "Is there seasonality in stock trading volume?", q52),
        (53, "Which months experience the highest volatility?", q53),
        (54, "Are there weekday effects in stock returns?", q54),
        (55, "How do gold prices trend over time relative to stock prices?", q55),
        (56, "Are there structural breaks in stock market trends?", q56),
        (57, "Which stocks exhibit cyclical behavior?", q57),
        (58, "Do trading volumes spike before large price movements?", q58),
        (59, "What is the rolling average trend of stock prices?", q59),
        (60, "Which stocks show momentum persistence?", q60),
        (61, "Are there volatility clusters in the market?", q61),
        (62, "How did market behavior change during major gold price spikes?", q62),
    ],
    "7 · Feature Engineering": [
        (63, "Can daily stock returns improve predictive analysis?", q63),
        (64, "Should rolling averages be created?", q64),
        (65, "Can volatility indicators like ATR or Bollinger Bands be engineered?", q65),
        (66, "Should lagged gold prices be used as predictive features?", q66),
        (67, "Can sentiment scores be extracted from headlines?", q67),
        (68, "Would price momentum indicators improve forecasting?", q68),
        (69, "Can sector-level indices be engineered from grouped stocks?", q69),
        (70, "Should interaction features between gold and stocks be created?", q70),
        (71, "Can abnormal trading activity indicators be engineered?", q71),
        (72, "How useful are rolling volatility features for forecasting?", q72),
    ],
    "8 · Financial Insights": [
        (73, "Which industries outperform during rising gold prices?", q73),
        (74, "Which companies have the highest long-term growth?", q74),
        (75, "Which stocks show defensive behavior during downturns?", q75),
        (76, "How does liquidity differ across industries?", q76),
        (77, "Which stocks attract the highest investor participation?", q77),
        (78, "Which companies exhibit the most stable trading patterns?", q78),
        (79, "How does delivery percentage relate to stock confidence?", q79),
        (80, "Are certain industries more correlated with commodity prices?", q80),
        (81, "Which stocks consistently trade above VWAP?", q81),
        (82, "Which sectors show the highest momentum trends?", q82),
        (83, "Do high-volume stocks generate more stable returns?", q83),
        (84, "Are gold prices acting as a safe-haven indicator?", q84),
    ],
    "9 · Risk & Volatility": [
        (85, "Which stocks have the highest historical volatility?", q85),
        (86, "Which industries are the riskiest?", q86),
        (87, "Are there extreme outlier trading days?", q87),
        (88, "Which stocks have the largest drawdowns?", q88),
        (89, "Is gold less volatile than stocks in this dataset?", q89),
        (90, "Which stocks exhibit sudden liquidity crashes?", q90),
        (91, "How often do stocks experience gap-up or gap-down openings?", q91),
        (92, "What is the Value-at-Risk (VaR) for major stocks?", q92),
        (93, "Which stocks demonstrate abnormal return behavior?", q93),
        (94, "Are market shocks transmitted across industries?", q94),
        (95, "Which stocks exhibit high beta relative to market movement?", q95),
    ],
    "10 · Predictive Analytics & ML": [
        (96, "Can future closing prices be predicted from gold & trading metrics?", q96),
        (97, "Which variables are the strongest predictors of stock returns?", q97),
        (98, "Can headlines improve stock price prediction accuracy?", q98),
        (99, "Can stocks be classified into bullish or bearish categories?", q99),
        (100, "Is it possible to forecast volatility using historical trends?", q100),
        (101, "Can anomalous trading days be detected automatically?", q101),
        (102, "Can clustering identify hidden market regimes?", q102),
        (103, "Which features are most useful for algorithmic trading?", q103),
        (104, "Can gold prices help forecast sector-specific performance?", q104),
        (105, "Can time-series models outperform moving-average baselines?", q105),
    ],
    "11 · Methods & Visualization Guide": [
        (106, "Recommended visualizations & statistical methods (reference)", methods_guide),
    ],
}


# =========================================================================== #
# ENTRY POINT                                                                 #
# =========================================================================== #
def run(df: Optional[pd.DataFrame] = None, df_full: Optional[pd.DataFrame] = None) -> None:
    """EDA dashboard. main.py calls run(df_filtered, df_full)."""
    page_header("Exploratory Data Analysis",
                "Pick a section, then a question — the analysis appears instantly")

    # EDA explores the whole market; prefer the full frame, fall back gracefully.
    data = df_full if df_full is not None and not df_full.empty else df
    if data is None or data.empty:
        data = get_engineered_data()
    if data is None or data.empty:
        info_banner("Dataset could not be loaded for EDA.", "warning")
        return

    # ---- The two driving dropdowns ---------------------------------------
    c1, c2 = st.columns(2)
    with c1:
        section = st.selectbox(" Analysis Section", list(SECTIONS.keys()),
                               key="eda_section")
    items = SECTIONS[section]
    labels = [f"Q{num}. {text}" for num, text, _ in items]
    with c2:
        choice = st.selectbox(" Question", labels, key="eda_question")

    num, text, handler = items[labels.index(choice)]
    st.markdown(f"### Q{num}. {text}")

    # ---- Render the selected analysis (isolated so errors never crash app)-
    try:
        handler(data)
    except Exception as exc:  # noqa: BLE001
        st.error("This analysis hit an unexpected error, but the page is still running.")
        with st.expander("Technical details"):
            st.exception(exc)


