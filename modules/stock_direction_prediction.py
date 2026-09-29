"""
modules/stock_direction_prediction.py
--------------------------------------
Model 1 — Can AI predict whether a stock rises or falls tomorrow?

Models : XGBoost / LightGBM (with a GradientBoosting fallback that guarantees the
         page always works even without those libraries installed).
Target : Tomorrow_Close > Today_Close
Outputs: BUY/SELL prediction, probability, accuracy/precision/recall/F1,
         confusion matrix, ROC curve, feature importance.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config.settings import APP, COLS, PALETTE
from utils import visualizations as viz
from utils.feature_engineering import available_features
from utils.helper_functions import (
    classification_metrics,
    download_buttons,
    has_package,
    page_header,
    score_sentiment,
)
from utils.preprocessing import get_engineered_data

# Sentiment feature columns appended to the model's feature set when headlines
# are available. They extend — never replace — the existing technical features.
SENTIMENT_FEATURES = [
    "Sentiment_Score", "Positive_Score", "Negative_Score", "Neutral_Score",
    "Sentiment_3D_Mean", "Sentiment_7D_Mean", "Sentiment_14D_Mean",
    "Sentiment_Momentum",
]


def _build_model(name: str):
    rs = APP.random_state
    if name == "XGBoost" and has_package("xgboost"):
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, eval_metric="logloss",
            random_state=rs, n_jobs=-1,
        ), "XGBoost"
    if name == "LightGBM" and has_package("lightgbm"):
        from lightgbm import LGBMClassifier
        return LGBMClassifier(
            n_estimators=300, max_depth=-1, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, random_state=rs,
            n_jobs=-1, verbose=-1,
        ), "LightGBM"
    # Reliable fallback.
    from sklearn.ensemble import GradientBoostingClassifier
    return GradientBoostingClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05, random_state=rs
    ), "Gradient Boosting (fallback)"


@st.cache_resource(show_spinner=False)
def _get_vader():
    """Return a VADER analyzer if the package is available, else None (Level 2)."""
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        return SentimentIntensityAnalyzer()
    except Exception:  # noqa: BLE001
        try:
            import nltk
            from nltk.sentiment import SentimentIntensityAnalyzer
            try:
                return SentimentIntensityAnalyzer()
            except Exception:  # noqa: BLE001
                nltk.download("vader_lexicon", quiet=True)
                return SentimentIntensityAnalyzer()
        except Exception:  # noqa: BLE001
            return None


def _score_headlines(texts) -> Tuple[pd.DataFrame, str]:
    """Score headlines into Sentiment/Positive/Negative/Neutral with a 3-tier
    fallback: FinBERT (if installed & small enough) → VADER → built-in lexicon.

    Returns a DataFrame with the four score columns and the engine name. All
    methods are local; no internet or external APIs are required.
    """
    texts = ["" if t is None else str(t) for t in texts]
    n = len(texts)

    # ---- Level 1: FinBERT (only when installed and the set is small enough) --
    if has_package("transformers") and n <= 800:
        try:
            from transformers import pipeline  # noqa: F401
            from utils.helper_functions import _finbert_pipeline  # reuse cached pipe
            pipe = _finbert_pipeline()
            if pipe is not None:
                pos, neg, neu, comp = [], [], [], []
                for t in texts:
                    snip = t[:512]
                    if not snip.strip():
                        pos.append(0.0); neg.append(0.0); neu.append(1.0); comp.append(0.0)
                        continue
                    res = pipe(snip, top_k=None)  # all label scores
                    d = {r["label"].lower(): float(r["score"]) for r in res}
                    p, ng, nu = d.get("positive", 0.0), d.get("negative", 0.0), d.get("neutral", 0.0)
                    pos.append(p); neg.append(ng); neu.append(nu); comp.append(p - ng)
                return pd.DataFrame({
                    "Sentiment_Score": comp, "Positive_Score": pos,
                    "Negative_Score": neg, "Neutral_Score": neu}), "FinBERT"
        except Exception:  # noqa: BLE001
            pass  # fall through

    # ---- Level 2: VADER ----------------------------------------------------
    vader = _get_vader()
    if vader is not None:
        try:
            rows = [vader.polarity_scores(t) for t in texts]
            return pd.DataFrame({
                "Sentiment_Score": [r["compound"] for r in rows],
                "Positive_Score":  [r["pos"] for r in rows],
                "Negative_Score":  [r["neg"] for r in rows],
                "Neutral_Score":   [r["neu"] for r in rows]}), "VADER"
        except Exception:  # noqa: BLE001
            pass

    # ---- Level 3: built-in financial lexicon (always available) ------------
    scores, _ = score_sentiment(texts, use_finbert=False)
    scores = np.asarray(scores, dtype=float)
    return pd.DataFrame({
        "Sentiment_Score": scores,
        "Positive_Score": np.clip(scores, 0, 1),
        "Negative_Score": np.clip(-scores, 0, 1),
        "Neutral_Score": 1.0 - np.abs(scores)}), "Lexicon (built-in)"


def _add_sentiment_features(sdf: pd.DataFrame) -> Tuple[pd.DataFrame, list, Optional[str]]:
    """Engineer per-headline + rolling sentiment features for one symbol's frame.

    Returns (sdf_with_features, sentiment_columns_added, engine_name). When no
    usable headlines exist, returns the frame unchanged with an empty column
    list and engine None, so the model falls back to technical-only.
    """
    if COLS.headlines not in sdf.columns:
        return sdf, [], None
    head = sdf[COLS.headlines].fillna("").astype(str)
    if head.str.strip().eq("").all():
        return sdf, [], None

    scores_df, engine = _score_headlines(head.tolist())
    scores_df.index = sdf.index
    for col in ["Sentiment_Score", "Positive_Score", "Negative_Score", "Neutral_Score"]:
        sdf[col] = scores_df[col].astype(float)

    # Rolling market-sentiment features (the frame is already date-sorted).
    s = sdf["Sentiment_Score"]
    sdf["Sentiment_3D_Mean"] = s.rolling(3, min_periods=1).mean()
    sdf["Sentiment_7D_Mean"] = s.rolling(7, min_periods=1).mean()
    sdf["Sentiment_14D_Mean"] = s.rolling(14, min_periods=1).mean()
    sdf["Sentiment_Momentum"] = sdf["Sentiment_3D_Mean"] - sdf["Sentiment_7D_Mean"]

    # Fill any rolling NaNs so the existing pipeline (incl. sklearn fallback)
    # never receives NaNs it can't handle.
    sent_cols = [c for c in SENTIMENT_FEATURES if c in sdf.columns]
    sdf[sent_cols] = sdf[sent_cols].fillna(0.0)
    return sdf, sent_cols, engine


@st.cache_data(show_spinner=False)
def _prepare(symbol: str) -> Tuple[pd.DataFrame, list, Optional[str]]:
    df = get_engineered_data()
    sdf = df[df[COLS.symbol] == symbol].sort_values(COLS.date).copy()
    sdf = sdf.dropna(subset=["Tomorrow_Close"])
    # Extend (never replace) the technical feature set with sentiment features.
    sdf, sent_cols, sent_engine = _add_sentiment_features(sdf)
    feats = available_features(sdf) + [c for c in sent_cols if c in sdf.columns]
    return sdf, feats, sent_engine


def run(df: Optional[pd.DataFrame] = None, df_full: Optional[pd.DataFrame] = None) -> None:
    page_header("Stock Direction Prediction",
                "Predict tomorrow's up/down move with gradient-boosted models", "📈")

    if df is None:
        df = get_engineered_data()
    if df.empty:
        st.warning("No data available.")
        return

    symbols = sorted(df[COLS.symbol].unique().tolist())
    c1, c2 = st.columns([2, 1])
    with c1:
        symbol = st.selectbox("Select stock", symbols, key="dir_symbol")
    with c2:
        algo = st.selectbox("Algorithm", ["XGBoost", "LightGBM"], key="dir_algo")

    sdf, feats, sent_engine = _prepare(symbol)
    if len(sdf) < 100 or not feats:
        st.warning("Not enough history for this stock to train reliably.")
        return

    X = sdf[feats].values
    y = sdf["Target_Up"].values
    cut = int(len(sdf) * 0.8)
    X_tr, X_te = X[:cut], X[cut:]
    y_tr, y_te = y[:cut], y[cut:]

    with st.spinner("Training model…"):
        model, used_name = _build_model(algo)
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        try:
            proba = model.predict_proba(X_te)[:, 1]
        except Exception:  # noqa: BLE001
            proba = y_pred.astype(float)

    if used_name.endswith("(fallback)"):
        st.info(f"ℹ️ `{algo}` not installed — using **{used_name}**. "
                f"Install with `pip install {algo.lower()}` for the full engine.")

    # ---- Tomorrow prediction ---------------------------------------------
    last_row = sdf[feats].iloc[[-1]].values
    next_dir = int(model.predict(last_row)[0])
    try:
        next_p = float(model.predict_proba(last_row)[0, 1])
    except Exception:  # noqa: BLE001
        next_p = float(next_dir)

    badge = "badge-buy" if next_dir == 1 else "badge-sell"
    call = "BUY ▲" if next_dir == 1 else "SELL ▼"
    st.markdown(
        f"<div class='glass'><h3>Next-Day Signal for {symbol}: "
        f"<span class='badge {badge}'>{call}</span></h3>"
        f"<p>Model confidence (probability of rise): "
        f"<b>{next_p*100:.1f}%</b></p></div>",
        unsafe_allow_html=True,
    )

    # ---- 📰 News Sentiment Impact (below the signal, above the metrics) ----
    _news_sentiment_section(sdf, feats, model, next_dir, next_p, sent_engine)

    # ---- Metrics ----------------------------------------------------------
    m = classification_metrics(y_te, y_pred)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Accuracy", f"{m['Accuracy']*100:.2f}%")
    k2.metric("Precision", f"{m['Precision']*100:.2f}%")
    k3.metric("Recall", f"{m['Recall']*100:.2f}%")
    k4.metric("F1 Score", f"{m['F1']*100:.2f}%")

    # ---- Confusion matrix + ROC ------------------------------------------
    from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

    g1, g2 = st.columns(2)
    with g1:
        cm = confusion_matrix(y_te, y_pred)
        st.plotly_chart(viz.confusion_matrix_fig(cm, ["Down", "Up"]),
                        use_container_width=True)
    with g2:
        if len(np.unique(y_te)) > 1:
            fpr, tpr, _ = roc_curve(y_te, proba)
            auc = roc_auc_score(y_te, proba)
            st.plotly_chart(viz.roc_curve_fig(fpr, tpr, auc),
                            use_container_width=True)
        else:
            st.info("ROC needs both classes in the test window.")

    # ---- Feature importance ----------------------------------------------
    importance = _get_importance(model, feats)
    if importance is not None:
        st.plotly_chart(
            viz.bar_chart(importance, "Importance", "Feature", orientation="h",
                          title="Feature Importance"),
            use_container_width=True,
        )

    # ---- Export -----------------------------------------------------------
    out = sdf.iloc[cut:][[COLS.date, COLS.stock_close]].copy()
    out["Actual_Up"] = y_te
    out["Predicted_Up"] = y_pred
    out["Prob_Up"] = proba
    st.markdown("#### Predictions")
    st.dataframe(out.tail(200), use_container_width=True, height=300)
    download_buttons(out, key="direction", label=f"{symbol}_direction_predictions")


def _get_importance(model, feats) -> Optional[pd.DataFrame]:
    imp = getattr(model, "feature_importances_", None)
    if imp is None:
        return None
    return (
        pd.DataFrame({"Feature": feats, "Importance": imp})
        .sort_values("Importance", ascending=True)
        .tail(15)
    )


def _sentiment_gauge(value: float) -> go.Figure:
    """Themed −1…+1 sentiment gauge (green = positive, red = negative)."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=float(value),
        number=dict(font=dict(size=34)),
        gauge=dict(
            axis=dict(range=[-1, 1], tickcolor=PALETTE.muted),
            bar=dict(color=PALETTE.accent),
            bgcolor=PALETTE.panel,
            steps=[
                {"range": [-1, -0.05], "color": PALETTE.down},
                {"range": [-0.05, 0.05], "color": PALETTE.warn},
                {"range": [0.05, 1], "color": PALETTE.up},
            ],
        ),
    ))
    fig.update_layout(template="plotly_dark", height=300,
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      title="Headline Sentiment (−1 to +1)")
    return fig


def _contribution_split(model, feats) -> Optional[Tuple[float, float]]:
    """Return (technical %, news %) from normalised feature importances."""
    imp = getattr(model, "feature_importances_", None)
    if imp is None:
        return None
    s = pd.Series(np.asarray(imp, dtype=float), index=feats)
    total = float(s.sum()) or 1.0
    news = float(s[[f for f in feats if f in SENTIMENT_FEATURES]].sum())
    news_pct = max(0.0, min(100.0, news / total * 100))
    return 100.0 - news_pct, news_pct


def _news_sentiment_section(sdf, feats, model, next_dir, next_p, engine) -> None:
    """Render '📰 News Sentiment Impact': headline, mood, gauge, contribution."""
    st.markdown("### 📰 News Sentiment Impact")

    sent_cols = [c for c in SENTIMENT_FEATURES if c in sdf.columns and c in feats]
    if COLS.headlines not in sdf.columns or not sent_cols:
        st.warning("No news headlines available. Prediction is based solely on "
                   "technical indicators.")
        return

    # Latest non-empty headline and its score.
    heads = sdf[COLS.headlines].fillna("").astype(str).tolist()
    latest_headline = next((h for h in reversed(heads) if h.strip()), "—")
    latest_score = float(sdf["Sentiment_Score"].iloc[-1])
    mood_val = float(sdf.get("Sentiment_7D_Mean", sdf["Sentiment_Score"]).iloc[-1])

    if latest_score > 0.05:
        label, emoji, mood = "Positive", "🟢", "Bullish News Environment"
    elif latest_score < -0.05:
        label, emoji, mood = "Negative", "🔴", "Bearish News Environment"
    else:
        label, emoji, mood = "Neutral", "🟡", "Neutral News Environment"

    left, right = st.columns([1.4, 1])
    with left:
        st.markdown(
            f"<div class='glass'>"
            f"<div style='font-size:0.8rem;color:{PALETTE.muted};text-transform:uppercase;"
            f"letter-spacing:.05em'>Latest headline</div>"
            f"<div style='font-size:1.02rem;margin:0.3rem 0 0.7rem 0'>“{latest_headline}”</div>"
            f"<div style='display:flex;gap:1.6rem;flex-wrap:wrap'>"
            f"<div><span style='color:{PALETTE.muted}'>Sentiment</span><br>"
            f"<b style='font-size:1.1rem'>{emoji} {label}</b></div>"
            f"<div><span style='color:{PALETTE.muted}'>Score</span><br>"
            f"<b style='font-size:1.1rem'>{latest_score:+.2f}</b></div>"
            f"<div><span style='color:{PALETTE.muted}'>Engine</span><br>"
            f"<b style='font-size:1.1rem'>{engine or '—'}</b></div>"
            f"</div>"
            f"<div style='margin-top:0.8rem;font-size:1.05rem'>{emoji} <b>{mood}</b> "
            f"<span style='color:{PALETTE.muted}'>(7-day mood {mood_val:+.2f})</span></div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    with right:
        st.plotly_chart(_sentiment_gauge(latest_score), use_container_width=True)

    # Contribution analysis + insight card.
    split = _contribution_split(model, feats)
    if split is None:
        return
    tech_pct, news_pct = split

    d1, d2 = st.columns([1, 1.4])
    with d1:
        donut = go.Figure(go.Pie(
            labels=["Technical", "News"], values=[tech_pct, news_pct], hole=0.6,
            marker=dict(colors=[PALETTE.accent, PALETTE.warn]),
            textinfo="label+percent", sort=False,
        ))
        donut.update_layout(template="plotly_dark", height=260,
                            paper_bgcolor="rgba(0,0,0,0)",
                            margin=dict(t=30, b=10, l=10, r=10),
                            title="Prediction Influence", showlegend=False)
        st.plotly_chart(donut, use_container_width=True)
    with d2:
        if news_pct < 10:
            insight = ("The prediction is primarily driven by technical indicators "
                       "because sentiment influence is weak.")
        elif next_dir == 1 and latest_score > 0.05:
            insight = "The model's BUY signal is strengthened by positive market sentiment."
        elif next_dir == 0 and latest_score < -0.05:
            insight = "The model's SELL signal is reinforced by negative news sentiment."
        elif next_dir == 1 and latest_score < -0.05:
            insight = ("The model leans BUY on the technicals, but current news sentiment "
                       "is negative — a point of caution.")
        elif next_dir == 0 and latest_score > 0.05:
            insight = ("The model leans SELL on the technicals, while news sentiment is "
                       "positive — signals are mixed.")
        else:
            insight = ("News and technical signals are broadly aligned for this "
                       "prediction.")
        st.markdown(
            f"<div class='glass'>"
            f"<div style='font-size:0.8rem;color:{PALETTE.muted};text-transform:uppercase;"
            f"letter-spacing:.05em'>Headline impact on prediction</div>"
            f"<p style='font-size:1.05rem;margin:0.5rem 0'>{insight}</p>"
            f"<div style='display:flex;gap:1.6rem;margin-top:0.4rem'>"
            f"<div><span style='color:{PALETTE.muted}'>Technical influence</span><br>"
            f"<b style='font-size:1.2rem;color:{PALETTE.accent}'>{tech_pct:.0f}%</b></div>"
            f"<div><span style='color:{PALETTE.muted}'>News influence</span><br>"
            f"<b style='font-size:1.2rem;color:{PALETTE.warn}'>{news_pct:.0f}%</b></div>"
            f"</div></div>",
            unsafe_allow_html=True,
        )