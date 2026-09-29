"""
modules/volatility_clustering.py
--------------------------------
Model 5 — Group stocks into risk tiers by volatility behaviour.

Models : KMeans (clustering) + Random Forest (risk classifier).
Outputs: stable / medium / high-risk buckets, cluster plot, risk heatmap.
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
def _symbol_profiles(_df_key: int) -> pd.DataFrame:
    df = get_engineered_data()
    agg = df.groupby([COLS.symbol, COLS.company, COLS.industry]).agg(
        Volatility=("Volatility", "mean"),
        Avg_Return=("Daily_Return", "mean"),
        Return_Std=("Daily_Return", "std"),
        Avg_Volume=(COLS.stock_volume, "mean"),
        Avg_Price=(COLS.stock_close, "mean"),
    ).reset_index()
    return agg.dropna()


def run(df: Optional[pd.DataFrame] = None, df_full: Optional[pd.DataFrame] = None) -> None:
    page_header("Volatility Clustering",
                "Segment the universe into risk tiers with unsupervised learning", "🌀")

    from sklearn.cluster import KMeans
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler

    profiles = _symbol_profiles(0)
    if len(profiles) < 3:
        st.warning("Not enough symbols to cluster.")
        return

    k = st.slider("Number of risk clusters", 3, 5, 3)
    feat_cols = ["Volatility", "Return_Std", "Avg_Volume", "Avg_Return"]
    X = StandardScaler().fit_transform(profiles[feat_cols].values)

    with st.spinner("Clustering…"):
        km = KMeans(n_clusters=k, random_state=APP.random_state, n_init=10)
        labels = km.fit_predict(X)
        profiles["Cluster"] = labels

    # Order clusters by mean volatility → human-readable risk labels.
    order = (profiles.groupby("Cluster")["Volatility"].mean()
             .sort_values().index.tolist())
    names = (["Stable", "Medium Risk", "High Risk"] if k == 3
             else ["Very Low", "Low", "Medium", "High", "Very High"][:k])
    risk_map = {c: names[i] for i, c in enumerate(order)}
    profiles["Risk_Tier"] = profiles["Cluster"].map(risk_map)

    # Train RF to confirm separability / get feature drivers.
    rf = RandomForestClassifier(n_estimators=200, random_state=APP.random_state)
    rf.fit(X, labels)

    # ---- KPIs -------------------------------------------------------------
    counts = profiles["Risk_Tier"].value_counts()
    cols = st.columns(len(counts))
    for col, (tier, cnt) in zip(cols, counts.items()):
        col.metric(tier, f"{cnt} stocks")

    # ---- Cluster scatter --------------------------------------------------
    st.plotly_chart(
        viz.scatter_chart(
            profiles, "Volatility", "Avg_Return", color="Risk_Tier",
            title="Risk Map — Volatility vs Average Return",
        ),
        use_container_width=True,
    )

    # ---- Risk heatmap by industry ----------------------------------------
    pivot = (profiles.pivot_table(index=COLS.industry, columns="Risk_Tier",
                                  values=COLS.symbol, aggfunc="count")
             .fillna(0))
    st.plotly_chart(
        viz.heatmap(pivot.values, list(pivot.columns), list(pivot.index),
                    title="Risk Tier Distribution by Industry",
                    colorscale="Inferno"),
        use_container_width=True,
    )

    # ---- Tables -----------------------------------------------------------
    st.markdown("### 📋 Stocks by Risk Tier")
    tabs = st.tabs(names)
    for tab, tier in zip(tabs, names):
        with tab:
            sub = (profiles[profiles["Risk_Tier"] == tier]
                   [[COLS.symbol, COLS.company, COLS.industry,
                     "Volatility", "Avg_Return", "Avg_Volume"]]
                   .sort_values("Volatility", ascending=False))
            st.dataframe(sub, use_container_width=True, height=300)

    download_buttons(
        profiles[[COLS.symbol, COLS.company, COLS.industry, "Risk_Tier",
                  "Volatility", "Avg_Return"]],
        key="volclust", label="volatility_clusters",
    )
