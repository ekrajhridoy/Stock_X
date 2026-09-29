"""
modules/explainable_ai.py
-------------------------
Model 13 — Explainable AI.

Base model: XGBoost (Random Forest fallback).
Explainers : SHAP + LIME (both optional, each with a permutation-importance
             fallback so the dashboards always render).
Outputs    : SHAP dashboard, LIME dashboard, waterfall plot, force-style plot,
             feature explanation.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from config.settings import APP, COLS, PALETTE
from utils import visualizations as viz
from utils.feature_engineering import available_features
from utils.helper_functions import download_buttons, has_package, page_header
from utils.preprocessing import get_engineered_data


def _build_model():
    rs = APP.random_state
    if has_package("xgboost"):
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators=250, max_depth=5, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, eval_metric="logloss",
            random_state=rs, n_jobs=-1,
        ), "XGBoost"
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(
        n_estimators=250, max_depth=12, random_state=rs, n_jobs=-1,
    ), "Random Forest (fallback)"


@st.cache_data(show_spinner=False)
def _prepare(symbol: str) -> Tuple[pd.DataFrame, List[str]]:
    df = get_engineered_data()
    sdf = df[df[COLS.symbol] == symbol].sort_values(COLS.date).copy()
    sdf = sdf.dropna(subset=["Tomorrow_Close"])
    feats = available_features(sdf)
    return sdf, feats


def _shap_values(model, X: np.ndarray, feats: List[str]):
    """Return mean |shap| per feature, or None if SHAP unavailable."""
    if not has_package("shap"):
        return None, None
    try:
        import shap
        try:
            explainer = shap.TreeExplainer(model)
            sv = explainer.shap_values(X)
        except Exception:  # noqa: BLE001
            explainer = shap.Explainer(model, X)
            sv = explainer(X).values
        if isinstance(sv, list):           # binary -> take positive class
            sv = sv[1] if len(sv) > 1 else sv[0]
        sv = np.asarray(sv)
        if sv.ndim == 3:                   # (n, feats, classes)
            sv = sv[:, :, -1]
        mean_abs = np.abs(sv).mean(axis=0)
        return mean_abs, sv
    except Exception:  # noqa: BLE001
        return None, None


def run(df: Optional[pd.DataFrame] = None, df_full: Optional[pd.DataFrame] = None) -> None:
    page_header("Explainable AI",
                "SHAP & LIME explanations for the prediction engine", "🔍")

    if df is None:
        df = get_engineered_data()
    if df.empty:
        st.warning("No data available.")
        return

    symbols = sorted(df[COLS.symbol].unique().tolist())
    symbol = st.selectbox("Select stock", symbols, key="xai_symbol")
    sdf, feats = _prepare(symbol)
    if len(sdf) < 100 or not feats:
        st.warning("Not enough history for this stock.")
        return

    X = sdf[feats].values
    y = sdf["Target_Up"].values
    cut = int(len(sdf) * 0.8)
    with st.spinner("Training base model…"):
        model, used = _build_model()
        model.fit(X[:cut], y[:cut])

    if used.endswith("(fallback)"):
        st.info("ℹ️ XGBoost not installed — explaining a Random Forest base model "
                "instead. Install `xgboost` for the specified engine.")

    X_explain = X[cut:][:300]  # keep explanation light

    # ===== SHAP dashboard =================================================
    st.markdown("### SHAP Dashboard")
    mean_abs, sv = _shap_values(model, X_explain, feats)
    if mean_abs is None:
        st.info("ℹ️ SHAP not installed — showing model feature importance as a "
                "proxy. Install `shap` for full Shapley explanations.")
        imp = getattr(model, "feature_importances_", np.ones(len(feats)))
        mean_abs = np.asarray(imp, dtype=float)

    shap_df = (pd.DataFrame({"Feature": feats, "Impact": mean_abs})
               .sort_values("Impact", ascending=True))
    st.plotly_chart(viz.bar_chart(shap_df.tail(15), "Impact", "Feature",
                    orientation="h", title="Mean |SHAP| — Global Importance"),
                    use_container_width=True)

    # ---- Waterfall (single most recent prediction) -----------------------
    st.markdown("#### Waterfall — Latest Prediction")
    if sv is not None and len(sv) > 0:
        contrib = sv[-1]
    else:
        # proxy contributions: importance * standardized feature value
        last = X_explain[-1] if len(X_explain) else X[-1]
        mu = X[:cut].mean(axis=0)
        sd = X[:cut].std(axis=0) + 1e-9
        contrib = mean_abs * ((last - mu) / sd)
    wf = (pd.DataFrame({"Feature": feats, "Contribution": contrib})
          .reindex(np.argsort(np.abs(contrib))).tail(12))
    colors = [PALETTE.up if c >= 0 else PALETTE.down for c in wf["Contribution"]]
    import plotly.graph_objects as go
    fig = go.Figure(go.Bar(
        x=wf["Contribution"], y=wf["Feature"], orientation="h",
        marker_color=colors,
    ))
    fig.update_layout(template="plotly_dark", height=420,
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      title="Feature Contributions (push toward Up / Down)")
    st.plotly_chart(fig, use_container_width=True)

    # ===== LIME dashboard =================================================
    st.markdown("### LIME Dashboard")
    lime_df = _lime_explain(model, X[:cut], X_explain, feats)
    if lime_df is None:
        st.info("ℹ️ LIME not installed — using a local linear surrogate as a "
                "fallback. Install `lime` for the canonical implementation.")
        lime_df = _local_surrogate(model, X[:cut], X_explain[-1], feats)

    st.plotly_chart(viz.bar_chart(lime_df, "Weight", "Feature", orientation="h",
                    title="LIME — Local Explanation (latest sample)"),
                    use_container_width=True)

    # ===== Feature explanation table ======================================
    st.markdown("#### Feature Explanation")
    expl = shap_df.sort_values("Impact", ascending=False).copy()
    expl["Rank"] = range(1, len(expl) + 1)
    expl["Impact"] = expl["Impact"].round(4)
    st.dataframe(expl[["Rank", "Feature", "Impact"]], use_container_width=True,
                 height=340)
    download_buttons(expl[["Rank", "Feature", "Impact"]], key="xai",
                     label=f"{symbol}_explainability")


def _lime_explain(model, X_train: np.ndarray, X_explain: np.ndarray,
                  feats: List[str]) -> Optional[pd.DataFrame]:
    if not has_package("lime") or len(X_explain) == 0:
        return None
    try:
        from lime.lime_tabular import LimeTabularExplainer
        expl = LimeTabularExplainer(
            X_train, feature_names=feats, class_names=["Down", "Up"],
            discretize_continuous=True, mode="classification",
        )
        exp = expl.explain_instance(
            X_explain[-1], model.predict_proba, num_features=min(12, len(feats)),
        )
        items = exp.as_list()
        return (pd.DataFrame(items, columns=["Feature", "Weight"])
                .sort_values("Weight"))
    except Exception:  # noqa: BLE001
        return None


def _local_surrogate(model, X_train: np.ndarray, x0: np.ndarray,
                     feats: List[str]) -> pd.DataFrame:
    """Fit a tiny weighted linear model around x0 as a LIME-style fallback."""
    from sklearn.linear_model import Ridge
    rng = np.random.default_rng(APP.random_state)
    sd = X_train.std(axis=0) + 1e-9
    samples = x0 + rng.normal(0, 0.5, size=(400, len(x0))) * sd
    try:
        proba = model.predict_proba(samples)[:, 1]
    except Exception:  # noqa: BLE001
        proba = model.predict(samples).astype(float)
    dist = np.linalg.norm((samples - x0) / sd, axis=1)
    weights = np.exp(-(dist ** 2) / (2 * (np.median(dist) + 1e-9) ** 2))
    lr = Ridge(alpha=1.0)
    lr.fit((samples - x0) / sd, proba, sample_weight=weights)
    return (pd.DataFrame({"Feature": feats, "Weight": lr.coef_})
            .sort_values("Weight"))
