"""
modules/dashboard.py
--------------------
Professional landing dashboard: animated KPI cards + market-overview charts.
Reacts to the global sidebar filters (receives the already-filtered frame).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

from config.settings import COLS, PALETTE
from utils import visualizations as viz
from utils.helper_functions import download_buttons, page_header
from utils.preprocessing import get_engineered_data

try:
    from st_aggrid import AgGrid, GridOptionsBuilder
    _HAS_AGGRID = True
except Exception:  # noqa: BLE001
    _HAS_AGGRID = False


def _fmt(n: float, prefix: str = "", suffix: str = "") -> str:
    if n is None or (isinstance(n, float) and not np.isfinite(n)):
        return "—"
    a = abs(n)
    if a >= 1e9:
        return f"{prefix}{n/1e9:.2f}B{suffix}"
    if a >= 1e6:
        return f"{prefix}{n/1e6:.2f}M{suffix}"
    if a >= 1e3:
        return f"{prefix}{n/1e3:.2f}K{suffix}"
    return f"{prefix}{n:,.2f}{suffix}"


def display_grid(df: pd.DataFrame, key: str) -> None:
    """Interactive table with search/sort/filter/pagination, AgGrid if present."""
    if _HAS_AGGRID and not df.empty:
        gob = GridOptionsBuilder.from_dataframe(df)
        gob.configure_default_column(filterable=True, sortable=True, resizable=True)
        gob.configure_pagination(paginationAutoPageSize=False, paginationPageSize=12)
        gob.configure_grid_options(domLayout="normal")
        AgGrid(
            df, gridOptions=gob.build(), height=380, theme="streamlit",
            fit_columns_on_grid_load=False, key=f"grid_{key}",
        )
    else:
        st.dataframe(df, use_container_width=True, height=380)


def run(df: Optional[pd.DataFrame] = None, df_full: Optional[pd.DataFrame] = None) -> None:
    if df is None:
        df = get_engineered_data()
    if df.empty:
        st.warning("No data available for the current filters.")
        return

    page_header(
        "Stock_X Dashboard",
        "Real-time market intelligence across stocks, gold and financial news"
    )

    # ---- KPI computations -------------------------------------------------
    total_records = len(df)
    avg_price = df[COLS.stock_close].mean()
    avg_gold = df[COLS.gold_close].mean()
    avg_volume = df[COLS.stock_volume].mean()
    avg_return = df["Daily_Return"].mean() * 100 if "Daily_Return" in df else 0.0
    volatility = df["Volatility"].mean() if "Volatility" in df else 0.0

    # Market trend from recent average returns.
    recent = df.sort_values(COLS.date).tail(max(50, len(df) // 20))
    trend_val = recent["Daily_Return"].mean() if "Daily_Return" in recent else 0.0
    trend = "Bullish ▲" if trend_val > 0 else ("Bearish ▼" if trend_val < 0 else "Flat ◆")
    trend_accent = PALETTE.up if trend_val > 0 else (PALETTE.down if trend_val < 0 else PALETTE.warn)

    # Simple RSI-based signal.
    last_rsi = df.sort_values(COLS.date)["RSI"].iloc[-1] if "RSI" in df else 50.0
    if last_rsi < 35:
        signal, sig_cls = "BUY", PALETTE.up
    elif last_rsi > 65:
        signal, sig_cls = "SELL", PALETTE.down
    else:
        signal, sig_cls = "HOLD", PALETTE.warn

    sentiment = _quick_sentiment(df)

    cards = [
        viz.kpi_card("Total Records", _fmt(total_records), icon=""),
        viz.kpi_card("Avg Stock Price", _fmt(avg_price, "₹"), icon="", accent=PALETTE.accent_2),
        viz.kpi_card("Avg Gold Price", _fmt(avg_gold, "$"), icon="", accent=PALETTE.warn),
        viz.kpi_card("Avg Volume", _fmt(avg_volume), icon="", accent=PALETTE.accent),
    ]
    viz.render_kpi_row(cards)

    cards2 = [
        viz.kpi_card("Avg Daily Return", f"{avg_return:.3f}%", icon="",
                     accent=PALETTE.up if avg_return >= 0 else PALETTE.down),
        viz.kpi_card("Volatility Index", f"{volatility:.3f}", icon="", accent=PALETTE.warn),
        viz.kpi_card("Market Trend", trend, icon="", accent=trend_accent),
        viz.kpi_card("Current Signal", signal, f"RSI {last_rsi:.0f}", icon="", accent=sig_cls),
    ]
    viz.render_kpi_row(cards2)

    cards3 = [
        viz.kpi_card("Sentiment Score", f"{sentiment:+.2f}", icon="",
                     accent=PALETTE.up if sentiment >= 0 else PALETTE.down),
        viz.kpi_card("Companies", _fmt(df[COLS.company].nunique()), icon="", accent=PALETTE.accent),
        viz.kpi_card("Industries", _fmt(df[COLS.industry].nunique()), icon="", accent=PALETTE.accent_2),
        viz.kpi_card("Date Span", f"{df[COLS.date].dt.year.min()}–{df[COLS.date].dt.year.max()}",
                     icon="", accent=PALETTE.warn),
    ]
    viz.render_kpi_row(cards3)

    st.markdown("###")

    # ---- Overview charts --------------------------------------------------
    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown("### Top Trading Shares")
        st.caption("Most actively traded shares within the selected date range")

        if df.empty:
            st.warning("No data available for the current filters.")
        else:
            min_date = df[COLS.date].min().date()
            max_date = df[COLS.date].max().date()

            date_range = st.date_input(
                "Select date range",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date,
                key="top_shares_date_range",
            )

            if isinstance(date_range, tuple) and len(date_range) == 2:
                start_date, end_date = date_range
            else:
                start_date, end_date = min_date, max_date

            mask = (df[COLS.date].dt.date >= start_date) & (df[COLS.date].dt.date <= end_date)
            lb_df = df.loc[mask]

            if lb_df.empty:
                st.warning("No data available for the selected date range.")
            else:
                leaderboard = (
                    lb_df.groupby([COLS.symbol, COLS.company], as_index=False)
                    .agg({COLS.stock_volume: "sum"})
                )

                leaderboard = leaderboard.sort_values(
                    COLS.stock_volume, ascending=False
                ).head(5).reset_index(drop=True)

                leaderboard.insert(0, "Rank", [f"#{i+1}" for i in range(len(leaderboard))])
                leaderboard["Total Volume"] = leaderboard[COLS.stock_volume].apply(_fmt)

                display_df = leaderboard[["Rank", COLS.symbol, COLS.company, "Total Volume"]].rename(columns={
                    COLS.symbol: "Symbol",
                    COLS.company: "Company",
                })

                theme = st.session_state.get("theme", "Dark")
                if theme == "Dark":
                    panel, panel2, text, muted = "#141a2a", "#1c2233", "#e6edf3", "#8b97a7"
                else:
                    panel, panel2, text, muted = "#ffffff", "#eef1f7", "#10141f", "#5b6677"

                rank_styles = {
                    "#1": ("linear-gradient(135deg, rgba(255,215,0,0.22), rgba(255,215,0,0.06))", "#ffd700"),
                    "#2": ("linear-gradient(135deg, rgba(192,192,192,0.20), rgba(192,192,192,0.05))", "#c0c0c0"),
                    "#3": ("linear-gradient(135deg, rgba(205,127,50,0.20), rgba(205,127,50,0.05))", "#cd7f32"),
                }

                rows_html = ""
                for _, row in display_df.iterrows():
                    bg, accent = rank_styles.get(
                        row["Rank"],
                        (f"linear-gradient(160deg, {panel2} 0%, {panel} 100%)", "#00d4ff"),
                    )
                    rows_html += f"""
                    <div style="
                        display:flex; align-items:center; gap:14px;
                        background:{bg};
                        border:1px solid rgba(255,255,255,0.08);
                        border-left:4px solid {accent};
                        border-radius:14px; padding:16px 18px; margin-bottom:8px;
                        box-shadow:0 6px 18px rgba(0,0,0,0.18);
                        flex:1;
                    ">
                        <div style="font-size:18px; font-weight:800; color:{accent}; min-width:42px;">{row['Rank']}</div>
                        <div style="flex:1;">
                            <div style="font-size:14px; font-weight:700; color:{text};">{row['Symbol']}</div>
                            <div style="font-size:12px; color:{muted};">{row['Company']}</div>
                        </div>
                        <div style="text-align:right;">
                            <div style="font-size:11px; color:{muted}; text-transform:uppercase; letter-spacing:.5px;">Volume</div>
                            <div style="font-size:15px; font-weight:800; color:{text};">{row['Total Volume']}</div>
                        </div>
                    </div>
                    """

                st.markdown(
                    f"""
                    <div style="
                        background:{panel};
                        border:1px solid rgba(255,255,255,0.08);
                        border-radius:18px; padding:18px;
                        box-shadow:0 8px 24px rgba(0,0,0,0.20);
                        height:450px;
                        display:flex; flex-direction:column;
                        justify-content:space-between;
                    ">
                        {rows_html}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
    with c2:
        gauge = viz.gauge_chart(float(last_rsi), "RSI Momentum Gauge", 0, 100, "")
        st.plotly_chart(gauge, use_container_width=True)

        if last_rsi < 35:
            rsi_note = "🟢 RSI below 30–35 suggests the market may be **oversold** — a potential buying opportunity."
        elif last_rsi > 65:
            rsi_note = "🔴 RSI above 65–70 suggests the market may be **overbought** — caution advised."
        else:
            rsi_note = "🟡 RSI is in a **neutral zone**, indicating balanced buying and selling pressure."

        st.markdown(
            f"""
            <div style="
                margin-top:10px; padding:14px 16px; border-radius:14px;
                background:{panel}; border:1px solid rgba(255,255,255,0.08);
                font-size:13px; line-height:1.6; color:{text};
            ">
                {rsi_note}<br>
                RSI ranges from 0–100: values near <b>0–30</b> = oversold, <b>70–100</b> = overbought.
            </div>
            """,
            unsafe_allow_html=True,
        )

    c3, c4 = st.columns(2)
    with c3:
        vol_by_ind = (
            df.groupby(COLS.industry)[COLS.stock_volume].sum().reset_index()
            .sort_values(COLS.stock_volume, ascending=True)   # ascending → biggest bar on top
        )
        st.plotly_chart(
            viz.bar_chart(vol_by_ind, x=COLS.stock_volume, y=COLS.industry,
                          color=COLS.stock_volume, orientation="h",
                          title="Total Volume by Industry"),
            use_container_width=True,
        )
    with c4:
        tm = (
            df.groupby([COLS.industry, COLS.company])[COLS.stock_turnover]
            .sum().reset_index()
        )
        tm = tm[tm[COLS.stock_turnover] > 0]
        st.plotly_chart(
            viz.treemap(tm, [COLS.industry, COLS.company], COLS.stock_turnover,
                        title="Market Turnover Treemap"),
            use_container_width=True,
        )

    # Stock vs Gold dual view.
    daily = (
        df.groupby(COLS.date)[[COLS.stock_close, COLS.gold_close]].mean().reset_index()
    )
    st.plotly_chart(
        viz.line_chart(daily, COLS.date, [COLS.stock_close, COLS.gold_close],
                       "Average Stock Close vs Gold Close Over Time"),
        use_container_width=True,
    )

    # Correlation matrix.
    corr_cols = [COLS.stock_close, COLS.stock_volume, COLS.gold_close,
                 "Daily_Return", "Volatility", "RSI", "MACD"]
    st.plotly_chart(
        viz.correlation_heatmap(df, corr_cols), use_container_width=True
    )

    # ---- Interactive data table ------------------------------------------
    st.markdown("###Market Data Explorer")
    show_cols = [COLS.date, COLS.symbol, COLS.company, COLS.industry,
                 COLS.stock_open, COLS.stock_close, COLS.stock_volume,
                 COLS.gold_close, "Daily_Return", "RSI"]
    show_cols = [c for c in show_cols if c in df.columns]
    table = df[show_cols].sort_values(COLS.date, ascending=False).head(500)
    display_grid(table, key="dashboard")
    download_buttons(df[show_cols], key="dashboard", label="market_data")


def _quick_sentiment(df: pd.DataFrame) -> float:
    """Very cheap lexicon sentiment over a sample of headlines for the KPI card."""
    if COLS.headlines not in df.columns:
        return 0.0
    pos = {"gain", "rise", "rises", "up", "surge", "growth", "profit", "boost",
           "rally", "high", "beat", "strong", "optimistic", "recovery", "approve"}
    neg = {"fall", "falls", "drop", "down", "loss", "slump", "weak", "fear",
           "crisis", "cut", "decline", "warn", "risk", "slip", "fails", "blast"}
    sample = df[COLS.headlines].dropna().astype(str).head(2000)
    score, n = 0, 0
    for text in sample:
        words = text.lower().split()
        p = sum(w.strip(".,/") in pos for w in words)
        q = sum(w.strip(".,/") in neg for w in words)
        if p + q:
            score += (p - q) / (p + q)
            n += 1
    return score / n if n else 0.0
