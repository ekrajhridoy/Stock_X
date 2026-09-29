"""
modules/sentiment_enhanced_prediction.py
-----------------------------------------
Model 15 — News-Headline Impact Classifier.

Research question
    "Given today's financial news headline(s), can we predict whether the stock
     price will move UP, DOWN, or stay NEUTRAL in the next trading session?"

Approach
    * Primary feature = the news headline(s) for the day.
    * Sentiment      = FinBERT when transformers+torch are installed, else a fast
                       built-in lexicon (both via `score_sentiment`).
    * Text embedding = FinBERT [CLS] embeddings when enabled & available, else a
                       TF-IDF + TruncatedSVD semantic vector (zero heavy deps).
    * Technical      = price/volume features (returns, rolling volatility, MA
                       ratios, RSI, volume ratios, ATR, day-of-week) — leakage-safe
                       (all computed from data up to & including today).
    * Labels (3-class) from next-session return vs a threshold that is either a
                       fixed % or a volatility-scaled (dynamic) band.
    * Classifier     = XGBoost with balanced sample weights (Random-Forest
                       fallback). Time-series (chronological) train/test split.
    * Output         = P(UP), P(DOWN), P(NEUTRAL) + final predicted impact, for
                       both the held-out test set and any headline you type in.

Keeps the project pattern: `run(df=None, df_full=None)`, project helpers,
graceful optional-dependency fallbacks, CSV/Excel export.
"""
from __future__ import annotations

import re
from typing import Callable, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from config.settings import APP, COLS, PALETTE
from utils import visualizations as viz
from utils.helper_functions import (
    classification_metrics,
    download_buttons,
    has_package,
    page_header,
    score_sentiment,
)
from utils.preprocessing import get_engineered_data

# 3-class label encoding (index == class id)
LABELS = ["DOWN", "NEUTRAL", "UP"]              # 0, 1, 2
IMPACT = {"UP": "POSITIVE", "DOWN": "NEGATIVE", "NEUTRAL": "NEUTRAL"}
BADGE = {"UP": "badge-buy", "DOWN": "badge-sell", "NEUTRAL": "badge-hold"}
ARROW = {"UP": "▲", "DOWN": "▼", "NEUTRAL": "■"}

MAX_TRAIN_ROWS = 4000        # cap for responsiveness
SVD_DIMS = 40                # semantic-embedding dimensionality
EXAMPLE = "Company reports record quarterly profits and raises guidance."

# Headlines can contain control characters (e.g. \x19) that openpyxl refuses to
# write to .xlsx. Strip the illegal ones before display / export.
_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def _excel_safe(text) -> str:
    return _CTRL.sub("", str(text))


# --------------------------------------------------------------------------- #
# Model + featurisation building blocks                                       #
# --------------------------------------------------------------------------- #
def _build_classifier():
    rs = APP.random_state
    if has_package("xgboost"):
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators=400, max_depth=4, learning_rate=0.03,
            min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
            gamma=0.5, reg_lambda=1.5, objective="multi:softprob",
            eval_metric="mlogloss", random_state=rs, n_jobs=-1,
        ), "XGBoost"
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(
        n_estimators=400, max_depth=12, min_samples_leaf=5,
        class_weight="balanced", random_state=rs, n_jobs=-1,
    ), "Random Forest (fallback)"


def _make_labels(next_close: np.ndarray, today_close: np.ndarray, thr: float) -> np.ndarray:
    """Fixed-threshold 3-class labels (kept for compatibility / fallback)."""
    ret = next_close / today_close - 1.0
    lab = np.ones(len(ret), dtype=int)           # NEUTRAL
    lab[ret > thr] = 2                            # UP
    lab[ret < -thr] = 0                           # DOWN
    return lab


def _dynamic_thresholds(today_close: np.ndarray, mult: float, window: int = 20) -> np.ndarray:
    """Per-day volatility-scaled threshold (trailing, leakage-safe)."""
    r1 = pd.Series(today_close).pct_change()
    vol = r1.rolling(window, min_periods=5).std()
    vol = vol.bfill().fillna(r1.std() if np.isfinite(r1.std()) else 0.01)
    return (mult * vol).clip(lower=0.002).to_numpy()   # floor at 0.2%


def _make_labels_dynamic(next_close: np.ndarray, today_close: np.ndarray,
                         thr_array: np.ndarray) -> np.ndarray:
    ret = next_close / today_close - 1.0
    lab = np.ones(len(ret), dtype=int)
    lab[ret > thr_array] = 2
    lab[ret < -thr_array] = 0
    return lab


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0).rolling(period).mean()
    down = (-delta.clip(upper=0)).rolling(period).mean()
    rs = up / down.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50.0)


def _tech_features(sdf: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    """Leakage-safe price/volume features (all use info up to today only)."""
    close = sdf[COLS.stock_close].astype(float)
    vol = (sdf[COLS.stock_volume].astype(float)
           if COLS.stock_volume in sdf.columns else pd.Series(0.0, index=sdf.index))
    f = pd.DataFrame(index=sdf.index)
    r1 = close.pct_change(1)
    f["ret_1"] = r1
    f["ret_3"] = close.pct_change(3)
    f["ret_5"] = close.pct_change(5)
    f["ret_10"] = close.pct_change(10)
    f["volat_5"] = r1.rolling(5).std()
    f["volat_10"] = r1.rolling(10).std()
    f["close_ma5"] = close / close.rolling(5).mean()
    f["close_ma10"] = close / close.rolling(10).mean()
    f["close_ma20"] = close / close.rolling(20).mean()
    f["rsi14"] = sdf["RSI"].astype(float) if "RSI" in sdf.columns else _rsi(close)
    f["vol_change"] = vol.pct_change()
    f["vol_ma_ratio"] = vol / vol.rolling(10).mean()
    if COLS.stock_high in sdf.columns and COLS.stock_low in sdf.columns:
        high = sdf[COLS.stock_high].astype(float)
        low = sdf[COLS.stock_low].astype(float)
        pc = close.shift(1)
        tr = pd.concat([(high - low), (high - pc).abs(), (low - pc).abs()],
                       axis=1).max(axis=1)
        f["atr_ratio"] = tr.rolling(14).mean() / close
    f["dow"] = pd.to_datetime(sdf[COLS.date]).dt.dayofweek
    f = f.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return f.to_numpy(dtype=float), list(f.columns)


@st.cache_data(show_spinner=False)
def _symbol_frame(symbol: str, use_finbert: bool) -> pd.DataFrame:
    """Per-symbol slice with sentiment scored once (cached, picklable)."""
    df = get_engineered_data()
    sdf = df[df[COLS.symbol] == symbol].sort_values(COLS.date).copy()
    sdf["Next_Close"] = sdf[COLS.stock_close].shift(-1)
    sdf = sdf.dropna(subset=["Next_Close"])
    if COLS.headlines in sdf.columns:
        texts = sdf[COLS.headlines].fillna("").astype(str).tolist()
        scores, _engine = score_sentiment(texts, use_finbert=use_finbert)
        sdf["Sentiment"] = scores
        sdf["Headline"] = texts
    else:
        sdf["Sentiment"] = 0.0
        sdf["Headline"] = ""
    if len(sdf) > MAX_TRAIN_ROWS:
        sdf = sdf.tail(MAX_TRAIN_ROWS)
    # Carry raw OHLCV + engineered RSI so technical features can be derived.
    keep = [COLS.date, COLS.stock_close, "Next_Close", "Sentiment", "Headline"]
    for extra in (COLS.stock_high, COLS.stock_low, COLS.stock_open,
                  COLS.stock_volume, "RSI"):
        if extra in sdf.columns and extra not in keep:
            keep.append(extra)
    return sdf[keep].reset_index(drop=True)


def _tfidf_embedder(train_texts: List[str]) -> Tuple[Callable[[List[str]], np.ndarray], int, str]:
    """Fit a TF-IDF + TruncatedSVD 'semantic embedding' and return an embed fn."""
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer

    vect = TfidfVectorizer(max_features=400, stop_words="english",
                           ngram_range=(1, 2), min_df=2)
    mat = vect.fit_transform(train_texts)
    k = int(max(2, min(SVD_DIMS, mat.shape[1] - 1)))
    svd = TruncatedSVD(n_components=k, random_state=APP.random_state).fit(mat)

    def embed(texts: List[str]) -> np.ndarray:
        return svd.transform(vect.transform(texts))

    return embed, k, "TF-IDF + SVD"


def _finbert_embedder() -> Tuple[Optional[Callable[[List[str]], np.ndarray]], Optional[str]]:
    """Optional FinBERT [CLS] embedder. Returns (embed_fn, name) or (None, None)."""
    if not (has_package("transformers") and has_package("torch")):
        return None, None
    try:  # pragma: no cover - exercised only when the heavy stack is present
        import torch
        from transformers import AutoModel, AutoTokenizer

        tok = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        mdl = AutoModel.from_pretrained("ProsusAI/finbert").eval()

        def embed(texts: List[str]) -> np.ndarray:
            out = []
            with torch.no_grad():
                for i in range(0, len(texts), 16):
                    enc = tok(texts[i:i + 16], padding=True, truncation=True,
                              max_length=64, return_tensors="pt")
                    cls = mdl(**enc).last_hidden_state[:, 0, :]
                    out.append(cls.cpu().numpy())
            return np.vstack(out) if out else np.zeros((0, mdl.config.hidden_size))

        return embed, "FinBERT [CLS]"
    except Exception:  # noqa: BLE001
        return None, None


def _meta(texts: List[str]) -> np.ndarray:
    """Light headline meta-features (counts/length)."""
    rows = []
    for t in texts:
        t = t or ""
        items = [p for p in t.split(" / ") if p.strip()]
        rows.append([len(t), len(t.split()), max(1, len(items))])
    return np.asarray(rows, dtype=float)


def _assemble(embed_fn, texts: List[str], sentiment: np.ndarray,
              tech: Optional[np.ndarray] = None) -> np.ndarray:
    """Stack embedding + sentiment + meta (+ optional technical) features.

    `tech` is backward-compatible: when omitted the output matches the original
    layout, so any existing caller keeps working.
    """
    emb = embed_fn(texts)
    sent = np.asarray(sentiment, dtype=float).reshape(-1, 1)
    meta = _meta(texts)
    parts = [emb, sent, meta]
    if tech is not None:
        tech = np.asarray(tech, dtype=float)
        if tech.ndim == 1:
            tech = tech.reshape(1, -1)
        parts.append(tech)
    return np.hstack(parts)


def _feature_names(n_emb: int, tech_names: List[str]) -> List[str]:
    return ([f"emb_{i}" for i in range(n_emb)]
            + ["Sentiment", "Headline_len", "Word_count", "Headline_items"]
            + list(tech_names))


# --------------------------------------------------------------------------- #
# Entry point                                                                 #
# --------------------------------------------------------------------------- #
def run(df: Optional[pd.DataFrame] = None, df_full: Optional[pd.DataFrame] = None) -> None:
    page_header("News Intelligence Engine",
                "Analyze financial news and predict its impact on future stock movements.")

    if df is None:
        df = get_engineered_data()
    if df.empty or COLS.headlines not in df.columns:
        st.warning("Headline data is required for this model but is unavailable.")
        return

    # ---- In-page controls (not the sidebar) ------------------------------
    symbols = sorted(df[COLS.symbol].unique().tolist())
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        symbol = st.selectbox("Select stock", symbols, key="news_symbol")
    with c2:
        use_finbert = st.toggle("FinBERT sentiment", value=False, key="news_finbert")
    with c3:
        thr_pct = st.slider("Neutral threshold (%)", 0.25, 3.0, 1.0, 0.25,
                            key="news_thr")

    c4, c5, c6 = st.columns([1.4, 1.3, 1.3])
    with c4:
        label_method = st.selectbox("Labeling method",
                                    ["Volatility-based (recommended)", "Fixed threshold"],
                                    key="news_labelmethod")
    with c5:
        vol_mult = st.slider("Volatility multiplier", 0.3, 1.5, 0.6, 0.1,
                             key="news_volmult",
                             help="UP/DOWN band = multiplier × trailing volatility.")
    with c6:
        calibrate = st.toggle("Calibrate probabilities", value=False,
                              key="news_calibrate",
                              help="Improves the reliability of the % confidences.")
    use_finbert_embed = st.toggle(
        "FinBERT embeddings (slower, needs transformers + torch)",
        value=False, key="news_embed")

    # FinBERT availability checks → graceful fallbacks
    finbert_ready = has_package("transformers") and has_package("torch")
    if use_finbert and not finbert_ready:
        st.info("ℹ️ FinBERT stack not installed — scoring sentiment with the fast "
                "built-in lexicon. Install `transformers` + `torch` for FinBERT.")
        use_finbert = False

    # Recommended-architecture note
    with st.expander("Which architecture is used (and why)"):
        st.markdown(
            "- **XGBoost + balanced sample weights — recommended & default.** The "
            "headline is turned into a fixed vector (sentiment + embedding + meta + "
            "technical), making this a tabular problem where gradient boosting is "
            "most accurate and trains in seconds.\n"
            "- **Random Forest** — automatic fallback when XGBoost is absent.\n"
            "- **FinBERT + LSTM / Transformer classifier** — operate on raw token "
            "sequences; they need much more labelled data and a GPU to beat XGBoost "
            "here, so they are not the production default.")

    sdf = _symbol_frame(symbol, use_finbert)
    if len(sdf) < 150:
        st.warning("Not enough history for this stock to train reliably.")
        return

    thr = thr_pct / 100.0
    today_close = sdf[COLS.stock_close].to_numpy()
    next_close = sdf["Next_Close"].to_numpy()
    if label_method.startswith("Volatility"):
        thr_array = _dynamic_thresholds(today_close, vol_mult)
        y = _make_labels_dynamic(next_close, today_close, thr_array)
    else:
        y = _make_labels(next_close, today_close, thr)
    texts = sdf["Headline"].tolist()
    sentiment = sdf["Sentiment"].values

    # ---- Build embedder (FinBERT optional, TF-IDF default) ---------------
    embed_fn, embed_name = (None, None)
    if use_finbert_embed:
        embed_fn, embed_name = _finbert_embedder()
        if embed_fn is None:
            st.info("ℹ️ FinBERT embeddings unavailable — using a TF-IDF semantic "
                    "vector instead.")
    if embed_fn is None:
        cut_for_fit = int(len(texts) * 0.8)
        embed_fn, _k, embed_name = _tfidf_embedder(texts[:cut_for_fit] or texts)

    # ---- Technical features (leakage-safe) -------------------------------
    tech_matrix, tech_names = _tech_features(sdf)
    last_tech = tech_matrix[-1:].copy()          # latest market state for live use

    # ---- Feature matrix + chronological split ----------------------------
    with st.spinner("Encoding headlines and training the classifier…"):
        X = _assemble(embed_fn, texts, sentiment, tech=tech_matrix)
        cut = int(len(X) * 0.8)                     # time-series split (no shuffle)
        X_tr, X_te = X[:cut], X[cut:]
        y_tr, y_te = y[:cut], y[cut:]

        from sklearn.preprocessing import LabelEncoder
        from sklearn.utils.class_weight import compute_sample_weight
        le = LabelEncoder().fit(y_tr)
        y_tr_enc = le.transform(y_tr)
        sample_w = compute_sample_weight(class_weight="balanced", y=y_tr_enc)

        model, used = _build_classifier()
        model.fit(X_tr, y_tr_enc, sample_weight=sample_w)

        # Optional probability calibration (chronological prefit holdout).
        shap_model = model                          # tree model for importance/SHAP
        if calibrate:
            try:
                from sklearn.calibration import CalibratedClassifierCV
                ccut = max(int(len(X_tr) * 0.8), len(X_tr) - 400)
                base, _ = _build_classifier()
                base.fit(X_tr[:ccut], y_tr_enc[:ccut],
                         sample_weight=sample_w[:ccut])
                cal = CalibratedClassifierCV(base, cv="prefit", method="sigmoid")
                cal.fit(X_tr[ccut:], y_tr_enc[ccut:])
                model, shap_model = cal, base
                used += " + calibrated"
            except Exception:  # noqa: BLE001
                pass

        pred_te = le.inverse_transform(model.predict(X_te))

    if used.startswith("Random Forest"):
        st.info("ℹ️ XGBoost not installed — training a Random Forest classifier.")
    st.caption(f"Sentiment: **{'FinBERT' if use_finbert else 'lexicon'}** · "
               f"Embedding: **{embed_name}** · Classifier: **{used}** · "
               f"Features: **{X.shape[1]}** (incl. {len(tech_names)} technical)")

    feature_names = _feature_names(X.shape[1] - 4 - len(tech_names), tech_names)

    # ---- Live single-headline prediction ---------------------------------
    st.markdown("### 📨 Try a headline")
    headline = st.text_area("Enter today's financial news headline(s)",
                            value=EXAMPLE, key="news_text", height=80)

    def _proba_for(text: str) -> np.ndarray:
        s, _ = score_sentiment([text], use_finbert=use_finbert)
        xv = _assemble(embed_fn, [text], s, tech=last_tech)
        enc = model.predict_proba(xv)[0]
        full = np.zeros(3)
        for j, orig in enumerate(le.classes_):
            full[int(orig)] = enc[j]
        return full

    if headline.strip():
        probs = _proba_for(headline)
        top = int(np.argmax(probs))
        call = LABELS[top]
        st.markdown(
            f"<div class='glass'><h3>Prediction: "
            f"<span class='badge {BADGE[call]}'>{call} {ARROW[call]}</span>"
            f" &nbsp; Expected market impact: <b>{IMPACT[call]}</b></h3></div>",
            unsafe_allow_html=True)
        p1, p2, p3 = st.columns(3)
        p3.metric("Probability UP", f"{probs[2]*100:.1f}%")
        p1.metric("Probability DOWN", f"{probs[0]*100:.1f}%")
        p2.metric("Probability NEUTRAL", f"{probs[1]*100:.1f}%")
        prob_df = pd.DataFrame({"Outcome": LABELS, "Probability": probs * 100})
        st.plotly_chart(
            viz.bar_chart(prob_df, "Outcome", "Probability", color="Outcome",
                          title="Predicted Probability by Outcome"),
            use_container_width=True)

    # ---- Test-set performance --------------------------------------------
    st.markdown("### 📊 Held-out test performance")
    m = classification_metrics(y_te, pred_te)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Accuracy", f"{m['Accuracy']*100:.2f}%")
    k2.metric("Precision", f"{m['Precision']*100:.2f}%")
    k3.metric("Recall", f"{m['Recall']*100:.2f}%")
    k4.metric("F1", f"{m['F1']*100:.2f}%")

    # Balanced accuracy + MCC (robust to class imbalance)
    from sklearn.metrics import (balanced_accuracy_score, matthews_corrcoef,
                                 precision_recall_fscore_support)
    bk1, bk2 = st.columns(2)
    bk1.metric("Balanced Accuracy", f"{balanced_accuracy_score(y_te, pred_te)*100:.2f}%")
    bk2.metric("Matthews Corr. (MCC)", f"{matthews_corrcoef(y_te, pred_te):+.3f}")

    g1, g2 = st.columns(2)
    with g1:
        dist = (pd.Series(y).map({0: "DOWN", 1: "NEUTRAL", 2: "UP"})
                .value_counts().reindex(LABELS).fillna(0).reset_index())
        dist.columns = ["Outcome", "Days"]
        st.plotly_chart(
            viz.bar_chart(dist, "Outcome", "Days", color="Outcome",
                          title="Label Distribution"),
            use_container_width=True)
    with g2:
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_te, pred_te, labels=[0, 1, 2])
        st.plotly_chart(viz.confusion_matrix_fig(cm, LABELS,
                        title="Confusion Matrix (test)"), use_container_width=True)

    # ---- Per-class classification report ---------------------------------
    prec, rec, f1c, sup = precision_recall_fscore_support(
        y_te, pred_te, labels=[0, 1, 2], zero_division=0)
    report = pd.DataFrame({
        "Class": LABELS, "Precision": np.round(prec, 3), "Recall": np.round(rec, 3),
        "F1": np.round(f1c, 3), "Support": sup.astype(int),
    })
    st.markdown("#### Classification report (per class)")
    st.dataframe(report, use_container_width=True)
    pr_long = report.melt(id_vars="Class", value_vars=["Precision", "Recall", "F1"],
                          var_name="Metric", value_name="Score")
    import plotly.express as px
    fig = px.bar(pr_long, x="Class", y="Score", color="Metric", barmode="group",
                 color_discrete_sequence=PALETTE.sequence)
    fig.update_layout(template="plotly_dark", height=380,
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      title="Precision / Recall / F1 by Class")
    st.plotly_chart(fig, use_container_width=True)

    # ---- Prediction distribution + Actual vs Predicted -------------------
    a1, a2 = st.columns(2)
    with a1:
        pdist = (pd.Series(pred_te).map({0: "DOWN", 1: "NEUTRAL", 2: "UP"})
                 .value_counts().reindex(LABELS).fillna(0).reset_index())
        pdist.columns = ["Outcome", "Count"]
        st.plotly_chart(viz.bar_chart(pdist, "Outcome", "Count", color="Outcome",
                        title="Prediction Distribution (test)"),
                        use_container_width=True)
    with a2:
        cmp = pd.DataFrame({
            "Class": LABELS * 2,
            "Kind": ["Actual"] * 3 + ["Predicted"] * 3,
            "Count": [int((y_te == i).sum()) for i in range(3)]
                     + [int((pred_te == i).sum()) for i in range(3)],
        })
        figc = px.bar(cmp, x="Class", y="Count", color="Kind", barmode="group",
                      color_discrete_sequence=[PALETTE.muted, PALETTE.accent])
        figc.update_layout(template="plotly_dark", height=380,
                           paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           title="Actual vs Predicted Class Counts")
        st.plotly_chart(figc, use_container_width=True)

    # ---- Rolling accuracy over time --------------------------------------
    test_dates = sdf[COLS.date].to_numpy()[cut:]
    correct = (pred_te == y_te).astype(float)
    roll = pd.DataFrame({
        COLS.date: test_dates,
        "Rolling_Accuracy": pd.Series(correct).rolling(20, min_periods=5).mean().to_numpy() * 100,
    }).dropna()
    if not roll.empty:
        st.plotly_chart(viz.line_chart(roll, COLS.date, ["Rolling_Accuracy"],
                        title="Rolling Model Accuracy (20-day window, %)"),
                        use_container_width=True)

    # ---- Cumulative strategy return --------------------------------------
    nxt_ret = (sdf["Next_Close"].to_numpy()[cut:] / sdf[COLS.stock_close].to_numpy()[cut:]) - 1.0
    position = np.where(pred_te == 2, 1.0, np.where(pred_te == 0, -1.0, 0.0))
    strat_curve = np.cumprod(1.0 + position * nxt_ret)
    bh_curve = np.cumprod(1.0 + nxt_ret)
    strat = pd.DataFrame({
        COLS.date: test_dates,
        "AI Strategy": strat_curve * 100,
        "Buy & Hold": bh_curve * 100,
    })
    st.plotly_chart(viz.line_chart(strat, COLS.date, ["AI Strategy", "Buy & Hold"],
                    title="Cumulative Return — Long UP / Short DOWN / Flat NEUTRAL (start=100)"),
                    use_container_width=True)
    st.caption("Strategy goes long on predicted UP days, short on DOWN days, and "
               "stays flat on NEUTRAL — a practical test of whether the signal pays.")

    # ---- Feature importance ----------------------------------------------
    imp = getattr(shap_model, "feature_importances_", None)
    if imp is not None:
        names = feature_names[:len(imp)]
        idf = (pd.DataFrame({"Feature": names, "Importance": imp})
               .sort_values("Importance", ascending=True).tail(15))
        st.plotly_chart(viz.bar_chart(idf, "Importance", "Feature", orientation="h",
                        title="What drives the prediction"), use_container_width=True)

    # ---- SHAP explainability (optional, graceful) ------------------------
    if has_package("shap") and hasattr(shap_model, "feature_importances_"):
        with st.expander("SHAP explainability (optional)"):
            try:
                import shap
                sample = X_te[:300] if len(X_te) else X_tr[:300]
                expl = shap.TreeExplainer(shap_model)
                sv = expl.shap_values(sample)
                if isinstance(sv, list):                 # list per class
                    mean_abs = np.mean([np.abs(s).mean(0) for s in sv], axis=0)
                    contrib_up = sv[2][0] if len(sv) > 2 else sv[-1][0]
                else:                                    # (n, feats, classes)
                    arr = np.asarray(sv)
                    mean_abs = np.abs(arr).mean(axis=(0, 2)) if arr.ndim == 3 else np.abs(arr).mean(0)
                    contrib_up = arr[0, :, -1] if arr.ndim == 3 else arr[0]
                names = feature_names[:len(mean_abs)]
                sdf_imp = (pd.DataFrame({"Feature": names, "Mean |SHAP|": mean_abs})
                           .sort_values("Mean |SHAP|", ascending=True).tail(15))
                st.plotly_chart(viz.bar_chart(sdf_imp, "Mean |SHAP|", "Feature",
                                orientation="h", title="SHAP Summary (mean |impact|)"),
                                use_container_width=True)
                contrib = (pd.DataFrame({"Feature": names[:len(contrib_up)],
                                         "Contribution": contrib_up}))
                pos = contrib.sort_values("Contribution", ascending=False).head(5)
                neg = contrib.sort_values("Contribution").head(5)
                cc1, cc2 = st.columns(2)
                cc1.markdown("**Top positive contributors (→ UP)**")
                cc1.dataframe(pos, use_container_width=True, hide_index=True)
                cc2.markdown("**Top negative contributors (→ DOWN)**")
                cc2.dataframe(neg, use_container_width=True, hide_index=True)
            except Exception as exc:  # noqa: BLE001
                st.info(f"SHAP could not run for this model ({type(exc).__name__}).")
    elif not has_package("shap"):
        st.caption("Install `shap` to enable SHAP explanations.")

    # ---- Sentiment vs realised next-day return ---------------------------
    sdf2 = sdf.assign(Next_Return=(sdf["Next_Close"] / sdf[COLS.stock_close] - 1) * 100)
    st.plotly_chart(
        viz.scatter_chart(sdf2, "Sentiment", "Next_Return",
                          title="Headline Sentiment vs Next-Day Return (%)"),
        use_container_width=True)

    # ---- Export test predictions -----------------------------------------
    out = sdf.iloc[cut:][[COLS.date, COLS.stock_close, "Sentiment", "Headline"]].copy()
    out["Headline"] = out["Headline"].map(_excel_safe)   # strip Excel-illegal chars
    out["Actual"] = [LABELS[i] for i in y_te]
    out["Predicted"] = [LABELS[i] for i in pred_te]
    st.markdown("#### Test predictions")
    st.dataframe(out.tail(200), use_container_width=True, height=320)
    download_buttons(out, key="news_impact", label=f"{symbol}_news_impact")

    with st.expander("More dashboard ideas to demonstrate news → price impact"):
        st.markdown(
            "- Price chart with **markers coloured by predicted impact** on each day.\n"
            "- **Rolling hit-rate** of the model over time (accuracy in a moving window).\n"
            "- **Event study**: average cumulative return in the days around the most "
            "positive / most negative headlines.\n"
            "- **Headline leaderboard**: the headlines that most strongly moved the "
            "stock, with their predicted vs actual impact.\n"
            "- **Sector view**: aggregate predicted impact across an industry to gauge "
            "broad news sentiment.")