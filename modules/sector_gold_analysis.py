"""
modules/sector_gold_analysis.py
--------------------------------
Model 7 — How sensitive is each sector to gold prices?

Outputs: sector ranking, gold-sensitivity analysis, sector dashboard.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

from config.settings import APP, COLS, PALETTE
from utils import visualizations as viz
from utils.helper_functions import download_buttons, page_header
from utils.preprocessing import get_engineered_data


@st.cache_data(show_spinner=False)
def _sector_gold_table(_k: int) -> pd.DataFrame:
    df = get_engineered_data()
    df = df.copy()

    df["Gold_Return"] = (
        df.groupby(COLS.symbol)[COLS.gold_close]
        .pct_change()
    )

    rows = []

    for ind, grp in df.groupby(COLS.industry):

        daily = (
            grp.groupby(COLS.date)
            .agg(
                Stock_Return=("Daily_Return", "mean"),
                Gold_Return=("Gold_Return", "mean"),
            )
            .dropna()
        )

        if len(daily) < 30:
            continue

        beta = (
            np.polyfit(
                daily["Gold_Return"],
                daily["Stock_Return"],
                1,
            )[0]
            if daily["Gold_Return"].std() > 0
            else 0.0
        )

        corr = daily["Stock_Return"].corr(
            daily["Gold_Return"]
        )

        rows.append(
            {
                COLS.industry: ind,
                "Gold_Beta": beta,
                "Gold_Correlation": corr,
                "Avg_Return": grp["Daily_Return"].mean(),
                "Avg_Volatility": grp["Volatility"].mean(),
                "Stocks": grp[COLS.symbol].nunique(),
            }
        )

    return pd.DataFrame(rows).dropna()


def run(
    df: Optional[pd.DataFrame] = None,
    df_full: Optional[pd.DataFrame] = None,
) -> None:

    page_header(
        "Gold Sensitivity Analyzer",
        "Discover which sectors rise or fall with changes in gold prices."
    )

    table = _sector_gold_table(0)

    if table.empty:
        st.warning(
            "Not enough data to compute sector sensitivities."
        )
        return

    table["Abs_Sensitivity"] = table["Gold_Beta"].abs()

    ranked = (
        table.sort_values(
            "Abs_Sensitivity",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    most = ranked.iloc[0]
    least = ranked.iloc[-1]

    k1, k2, k3 = st.columns(3)

    k1.metric(
        "Most Gold-Sensitive",
        most[COLS.industry],
        f"β={most['Gold_Beta']:.3f}",
    )

    k2.metric(
        "Least Sensitive",
        least[COLS.industry],
        f"β={least['Gold_Beta']:.3f}",
    )

    k3.metric(
        "Sectors Analysed",
        f"{len(ranked)}",
    )

    # ---- Sensitivity ranking ---------------------------------------------
    st.plotly_chart(
        viz.bar_chart(
            ranked.sort_values("Gold_Beta"),
            "Gold_Beta",
            COLS.industry,
            color="Gold_Beta",
            orientation="h",
            title="Gold Sensitivity (β) by Sector",
        ),
        use_container_width=True,
    )

    # ---- Correlation analysis --------------------------------------------
    g1, g2 = st.columns(2)

    with g1:
        st.plotly_chart(
            viz.scatter_chart(
                ranked,
                "Gold_Correlation",
                "Avg_Return",
                color=COLS.industry,
                title="Gold Correlation vs Average Return",
            ),
            use_container_width=True,
        )

    with g2:
        st.plotly_chart(
            viz.bar_chart(
                ranked,
                COLS.industry,
                "Gold_Correlation",
                color="Gold_Correlation",
                title="Gold Correlation by Sector",
            ),
            use_container_width=True,
        )

    # ---- Sector dashboard table ------------------------------------------
    st.markdown("Sector Dashboard")

    show = ranked[
        [
            COLS.industry,
            "Gold_Beta",
            "Gold_Correlation",
            "Avg_Return",
            "Avg_Volatility",
            "Stocks",
        ]
    ].round(4)

    st.dataframe(
        show,
        use_container_width=True,
        height=380,
    )

    download_buttons(
        show,
        key="sector_gold",
        label="sector_gold_analysis",
    )