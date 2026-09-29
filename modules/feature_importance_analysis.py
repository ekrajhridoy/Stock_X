"""
modules/feature_importance_analysis.py
---------------------------------------
Model 2 — Which features drive the next-day move?

Models : Random Forest / XGBoost + optional SHAP.
Outputs: SHAP summary plot, SHAP dependence plot, top features,
         feature-importance ranking.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

from config.settings import APP, COLS
from utils import visualizations as viz
from utils.feature_engineering import available_features
from utils.helper_functions import download_buttons, has_package, page_header
from utils.preprocessing import get_engineered_data


def _build(name: str):
    rs = APP.random_state
    if name == "XGBoost" and has_package("xgboost"):
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=250, max_depth=5, learning_rate=0.05,
                             eval_metric="logloss", random_state=rs, n_jobs=-1), name
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(n_estimators=250, max_depth=8, random_state=rs,
                                  n_jobs=-1), "Random Forest"


def run(df: Optional[pd.DataFrame] = None, df_full: Optional[pd.DataFrame] = None) -> None:
    page_header("Feature Importance Analysis",
                "Understand what drives predictions with permutation & SHAP", "🧩")

    if df is None:
        df = get_engineered_data()
    if df.empty:
        st.warning("No data available.")
        return

    symbols = ["(All stocks — sampled)"] + sorted(df[COLS.symbol].unique().tolist())
    c1, c2 = st.columns([2, 1])
    with c1:
        symbol = st.selectbox("Scope", symbols, key="fi_symbol")
    with c2:
        algo = st.selectbox("Model", ["Random Forest", "XGBoost"], key="fi_algo")

    work = df if symbol.startswith("(All") else df[df[COLS.symbol] == symbol]
    work = work.dropna(subset=["Tomorrow_Close"])
    feats = available_features(work)
    if len(work) < 100 or not feats:
        st.warning("Not enough data to compute importance.")
        return

    work = work.sample(min(len(work), 8000), random_state=APP.random_state)
    X = work[feats].values
    y = work["Target_Up"].values

    with st.spinner("Fitting model & computing importances…"):
        model, used = _build(algo)
        model.fit(X, y)

    # ---- Built-in / permutation importance --------------------------------
    imp = getattr(model, "feature_importances_", None)
    if imp is None:
        from sklearn.inspection import permutation_importance
        imp = permutation_importance(model, X, y, n_repeats=5,
                                     random_state=APP.random_state).importances_mean
    rank = (pd.DataFrame({"Feature": feats, "Importance": imp})
            .sort_values("Importance", ascending=False).reset_index(drop=True))

    st.markdown(f"**Model used:** {used}")
    cL, cR = st.columns([3, 2])
    with cL:
        st.plotly_chart(
            viz.bar_chart(rank.sort_values("Importance").tail(15),
                          "Importance", "Feature", orientation="h",
                          title="Feature Importance Ranking"),
            use_container_width=True,
        )
    with cR:
        st.markdown("#### 🏆 Top Features")
        for i, row in rank.head(8).iterrows():
            st.markdown(f"**{i+1}. {row['Feature']}** — `{row['Importance']:.4f}`")

    # ---- SHAP -------------------------------------------------------------
    st.markdown("### 🔍 SHAP Analysis")
    if has_package("shap"):
        try:
            _render_shap(model, work, feats)
        except Exception as exc:  # noqa: BLE001
            st.info(f"SHAP could not run for this model configuration ({exc}). "
                    "Showing model-based importance above instead.")
    else:
        st.info("⚙️ `shap` not installed. Install with `pip install shap` to see "
                "SHAP summary & dependence plots. Model importance shown above.")

    download_buttons(rank, key="feat_imp", label="feature_importance_ranking")


def _render_shap(model, work: pd.DataFrame, feats) -> None:
    import shap

    sample = work[feats].sample(min(len(work), 400), random_state=APP.random_state)
    try:
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(sample)
    except Exception:  # noqa: BLE001
        explainer = shap.Explainer(model.predict, sample)
        sv = explainer(sample).values

    if isinstance(sv, list):  # binary classifier returns list
        sv = sv[1] if len(sv) > 1 else sv[0]
    sv = np.asarray(sv)
    if sv.ndim == 3:
        sv = sv[:, :, 1]

    mean_abs = np.abs(sv).mean(axis=0)
    shap_rank = (pd.DataFrame({"Feature": feats, "Mean|SHAP|": mean_abs})
                 .sort_values("Mean|SHAP|"))
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(
            viz.bar_chart(shap_rank.tail(15), "Mean|SHAP|", "Feature",
                          orientation="h", title="SHAP Summary (mean |impact|)"),
            use_container_width=True,
        )
    with c2:
        top_feat = shap_rank["Feature"].iloc[-1]
        idx = feats.index(top_feat)
        dep = pd.DataFrame({top_feat: sample[top_feat].values, "SHAP": sv[:, idx]})
        st.plotly_chart(
            viz.scatter_chart(dep, top_feat, "SHAP",
                              title=f"SHAP Dependence — {top_feat}"),
            use_container_width=True,
        )
