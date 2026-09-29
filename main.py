"""
main.py
-------
Entry point for the AI Financial Intelligence Platform.

    streamlit run main.py

Architecture
------------
A single `FinancialIntelligenceApp` class owns:
  * page configuration & custom CSS (dark / light themes, glassmorphism)
  * cached data loading + feature engineering
  * the modern sidebar (streamlit-option-menu) with grouped navigation
  * global filters stored in session-state
  * routing to each feature module's `run()` function

Every feature lives in its own module under `modules/` and exposes `run()`,
so this file stays small and the project is fully modular.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Callable, Dict, List, Tuple

# Ensure the nested project directory is discoverable when running from workspace root.
ROOT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = ROOT_DIR / "ai_financial_intelligence_platform" / "project"
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import pandas as pd
import streamlit as st

from config.settings import APP, COLS
from utils.feature_engineering import add_technical_indicators, make_direction_target
from utils.helper_functions import LOGGER, deps
from utils.preprocessing import filter_dataframe, load_data

# streamlit-option-menu is a hard UI dependency; fall back to radio if missing.
try:
    from streamlit_option_menu import option_menu
    _HAS_OPTION_MENU = True
except Exception:  # noqa: BLE001
    _HAS_OPTION_MENU = False


# --------------------------------------------------------------------------- #
# Navigation registry: label -> (module_name, icon)                           #
# --------------------------------------------------------------------------- #
NAV_GROUPS: Dict[str, List[Tuple[str, str, str]]] = {
    "Analytics": [
        ("Dashboard", "_dashboard", "speedometer2"),
        ("EDA Analytics", "eda", "bar-chart-line"),
    ],
    "Machine Learning": [
        ("Stock Direction Prediction", "stock_direction_prediction", "graph-up-arrow"),
        ("Feature Importance", "feature_importance_analysis", "diagram-3"),
        ("Volatility Clustering", "volatility_clustering", "grid-3x3-gap"),
        ("Crash Detection", "market_crash_detection", "exclamation-triangle"),
        ("BUY / SELL / HOLD Signals", "buy_sell_hold_system", "signpost-split"),
    ],
    "Deep Learning": [
        ("LSTM Forecasting", "future_forecasting_lstm", "activity"),
        ("Sentiment Enhanced Prediction", "sentiment_enhanced_prediction", "cpu"),
        ("RL Trading Agent", "reinforcement_learning_trader", "robot"),
    ],
    "Forecasting": [
        ("Long-Term Investment Ranking", "long_term_investment_ranking", "trophy"),
        ("Forecasting Comparison", "forecasting_model_comparison", "clipboard-data"),
        ("Historical Pattern Matching", "historical_pattern_matching", "search"),
    ],
    "News Intelligence": [
        ("News Sentiment Impact", "news_sentiment_impact", "newspaper"),
        ("Company News Sensitivity", "company_news_reaction", "broadcast"),
    ],
    "Explainable AI": [
        ("SHAP / LIME Analysis", "explainable_ai", "lightbulb"),
        ("Sector vs Gold Analysis", "sector_gold_analysis", "bank"),
    ],
}


class FinancialIntelligenceApp:
    """Top-level Streamlit application controller."""

    def __init__(self) -> None:
        self._configure_page()
        self._init_state()

    # ----------------------------------------------------------------- setup
    @staticmethod
    def _configure_page() -> None:
        st.set_page_config(
            page_title=APP.app_name,
            page_icon="📈",
            layout="wide",
            initial_sidebar_state="expanded",
        )

    def _init_state(self) -> None:
        st.session_state.setdefault("theme", "Dark")
        st.session_state.setdefault("active_page", "Dashboard")

    # ----------------------------------------------------------------- styling
    def _inject_css(self) -> None:
        theme = st.session_state.get("theme", "Dark")
        if theme == "Dark":
            bg, panel, panel2, text, muted = (
                "#0b0f19", "#141a2a", "#1c2233", "#e6edf3", "#8b97a7")
        else:
            bg, panel, panel2, text, muted = (
                "#f4f6fb", "#ffffff", "#eef1f7", "#10141f", "#5b6677")

        st.markdown(
            f"""
            <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

            html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}
            .stApp {{
                background:
                  radial-gradient(1200px 600px at 80% -10%, rgba(124,92,255,0.12), transparent),
                  radial-gradient(900px 500px at -10% 10%, rgba(0,212,255,0.10), transparent),
                  {bg};
                color: {text};
            }}
            #MainMenu, footer, header {{ visibility: hidden; }}
            section[data-testid="stSidebar"] {{
                background: linear-gradient(180deg, {panel} 0%, {bg} 100%);
                border-right: 1px solid rgba(255,255,255,0.06);
            }}
            /* ---- Brand ---- */
            .brand {{
                display:flex; align-items:center; gap:12px; padding:8px 4px 16px 4px;
            }}
            .brand-logo {{
                width:44px; height:44px; border-radius:12px;
                background: linear-gradient(135deg, #00d4ff, #7c5cff);
                display:flex; align-items:center; justify-content:center;
                font-size:22px; box-shadow:0 6px 20px rgba(0,212,255,0.35);
            }}
            .brand-name {{ font-weight:800; font-size:15px; line-height:1.1; color:{text}; }}
            .brand-sub  {{ font-size:11px; color:{muted}; letter-spacing:.5px; }}

            /* ---- Page header ---- */
            .page-header {{
                display:flex; align-items:center; gap:16px; margin:4px 0 18px 0;
                padding:18px 22px; border-radius:18px;
                background: linear-gradient(135deg, rgba(0,212,255,0.10), rgba(124,92,255,0.10));
                border:1px solid rgba(255,255,255,0.08);
                backdrop-filter: blur(8px);
            }}
            .page-header-icon {{ font-size:30px; }}
            .page-header-title {{ font-size:24px; font-weight:800; color:{text}; }}
            .page-header-sub {{ font-size:13px; color:{muted}; }}

            /* ---- KPI cards ---- */
            .kpi-row {{ display:flex; flex-wrap:wrap; gap:16px; margin:6px 0 12px 0; }}
            .kpi-card {{
                flex:1 1 200px; min-width:180px;
                background: linear-gradient(160deg, {panel2} 0%, {panel} 100%);
                border:1px solid rgba(255,255,255,0.08);
                border-radius:18px; padding:18px 18px;
                display:flex; gap:14px; align-items:center;
                box-shadow: 0 10px 30px rgba(0,0,0,0.25);
                position:relative; overflow:hidden;
                transition: transform .18s ease, box-shadow .18s ease;
                animation: kpiIn .5s ease both;
            }}
            .kpi-card:hover {{ transform: translateY(-4px);
                box-shadow:0 16px 40px rgba(0,0,0,0.35); }}
            .kpi-card::before {{
                content:""; position:absolute; left:0; top:0; bottom:0; width:4px;
                background: var(--accent, #00d4ff);
            }}
            .kpi-icon {{ font-size:26px; }}
            .kpi-label {{ font-size:12px; color:{muted}; text-transform:uppercase;
                letter-spacing:.6px; }}
            .kpi-value {{ font-size:22px; font-weight:800; color:{text}; }}
            .kpi-delta {{ font-size:12px; color:{muted}; margin-top:2px; }}
            @keyframes kpiIn {{ from {{opacity:0; transform:translateY(10px);}}
                to {{opacity:1; transform:translateY(0);}} }}

            /* ---- Generic cards / tabs / buttons ---- */
            .glass {{
                background: {panel};
                border:1px solid rgba(255,255,255,0.08);
                border-radius:18px; padding:18px; margin-bottom:14px;
                box-shadow:0 8px 24px rgba(0,0,0,0.20);
            }}
            .stButton>button, .stDownloadButton>button {{
                border-radius:12px; border:1px solid rgba(255,255,255,0.12);
                background: linear-gradient(135deg, rgba(0,212,255,0.18), rgba(124,92,255,0.18));
                color:{text}; font-weight:600; transition: all .15s ease;
            }}
            .stButton>button:hover, .stDownloadButton>button:hover {{
                border-color:#00d4ff; box-shadow:0 0 0 2px rgba(0,212,255,0.25);
            }}
            div[data-testid="stMetric"] {{
                background:{panel}; border:1px solid rgba(255,255,255,0.08);
                border-radius:14px; padding:14px;
            }}
            .stTabs [data-baseweb="tab-list"] {{ gap:6px; }}
            .stTabs [data-baseweb="tab"] {{
                background:{panel}; border-radius:10px 10px 0 0; padding:8px 16px;
            }}
            .badge {{ display:inline-block; padding:4px 12px; border-radius:999px;
                font-size:12px; font-weight:700; }}
            .badge-buy {{ background:rgba(17,212,147,0.18); color:#11d493; }}
            .badge-sell {{ background:rgba(255,77,109,0.18); color:#ff4d6d; }}
            .badge-hold {{ background:rgba(255,176,32,0.18); color:#ffb020; }}
            </style>
            """,
            unsafe_allow_html=True,
        )

    # ----------------------------------------------------------------- data
    @staticmethod
    @st.cache_data(show_spinner="Loading & engineering market data…")
    def _load_engineered() -> pd.DataFrame:
        df = load_data()
        if df.empty:
            return df
        df = add_technical_indicators(df)
        df = make_direction_target(df)
        return df

    # ----------------------------------------------------------------- sidebar
    def _sidebar(self, df: pd.DataFrame) -> str:
        with st.sidebar:
            st.markdown(
                """
                <div class="brand">
                    <div class="brand-logo">📈</div>
                    <div>
                        <div class="brand-name">AI Financial<br>Intelligence Platform</div>
                        <div class="brand-sub">v{ver} • FINTECH SUITE</div>
                    </div>
                </div>
                """.format(ver=APP.version),
                unsafe_allow_html=True,
            )

            # Build flat option list grouped visually.
            labels: List[str] = []
            icons: List[str] = []
            for group, items in NAV_GROUPS.items():
                for label, _mod, icon in items:
                    labels.append(label)
                    icons.append(icon)

            if _HAS_OPTION_MENU:
                selected = option_menu(
                    menu_title=None,
                    options=labels,
                    icons=icons,
                    default_index=labels.index(st.session_state["active_page"])
                    if st.session_state["active_page"] in labels else 0,
                    styles={
                        "container": {"padding": "4px", "background-color": "transparent"},
                        "icon": {"color": "#00d4ff", "font-size": "15px"},
                        "nav-link": {
                            "font-size": "14px", "color": "#cdd6e3",
                            "border-radius": "10px", "margin": "2px 0",
                            "--hover-color": "rgba(0,212,255,0.12)",
                        },
                        "nav-link-selected": {
                            "background": "linear-gradient(135deg, rgba(0,212,255,0.25), rgba(124,92,255,0.25))",
                            "color": "#ffffff", "font-weight": "700",
                        },
                    },
                )
            else:
                selected = st.radio("Navigation", labels, label_visibility="collapsed")

            st.session_state["active_page"] = selected
            st.markdown("---")
            self._global_filters(df)
            st.markdown("---")
            self._theme_and_status()
            return selected

    def _global_filters(self, df: pd.DataFrame) -> None:
        st.markdown("#### 🔎 Global Filters")
        if df.empty:
            st.warning("Dataset not loaded.")
            return

        industries = sorted(df[COLS.industry].dropna().unique().tolist())
        sel_inds = st.multiselect("Industry", industries, default=[])

        pool = df[df[COLS.industry].isin(sel_inds)] if sel_inds else df
        companies = sorted(pool[COLS.company].dropna().unique().tolist())
        sel_comps = st.multiselect("Company", companies, default=[])

        dmin, dmax = df[COLS.date].min(), df[COLS.date].max()
        date_range = st.date_input(
            "Date range", value=(dmin, dmax), min_value=dmin, max_value=dmax
        )

        gmin, gmax = float(df[COLS.gold_close].min()), float(df[COLS.gold_close].max())
        gold_range = st.slider("Gold price", gmin, gmax, (gmin, gmax))

        vmax = float(df[COLS.stock_volume].quantile(0.99))
        vol_range = st.slider("Volume (≤ p99)", 0.0, vmax, (0.0, vmax))

        st.session_state["filters"] = {
            "industries": sel_inds,
            "companies": sel_comps,
            "date_range": date_range if isinstance(date_range, tuple) else (dmin, dmax),
            "gold_range": gold_range,
            "volume_range": vol_range,
        }

    def _theme_and_status(self) -> None:
        st.markdown("#### ⚙️ Settings")
        theme = st.radio(
            "Theme", ["Dark", "Light"],
            index=0 if st.session_state["theme"] == "Dark" else 1,
            horizontal=True,
        )
        if theme != st.session_state["theme"]:
            st.session_state["theme"] = theme
            st.rerun()

        with st.expander("Engine status"):
            for name, ok in deps().items():
                st.write(("🟢 " if ok else "⚪ ") + name)

    # ----------------------------------------------------------------- routing
    def _apply_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        f = st.session_state.get("filters", {})
        if not f:
            return df
        return filter_dataframe(
            df,
            companies=f.get("companies"),
            industries=f.get("industries"),
            date_range=f.get("date_range"),
            gold_range=f.get("gold_range"),
            volume_range=f.get("volume_range"),
        )

    def _route(self, page: str, df_full: pd.DataFrame, df_filtered: pd.DataFrame) -> None:
        # Resolve module name from registry.
        module_name = None
        for items in NAV_GROUPS.values():
            for label, mod, _icon in items:
                if label == page:
                    module_name = mod
                    break
        if module_name is None:
            st.error(f"Unknown page: {page}")
            return

        # Built-in dashboard handled locally.
        if module_name == "_dashboard":
            from modules import dashboard
            dashboard.run(df_full, df_filtered)
            return

        # EDA is user-provided; import defensively so the app never crashes if
        # the user has not created modules/eda.py yet.
        if module_name == "eda":
            try:
                eda = importlib.import_module("modules.eda")
            except Exception as exc:  # noqa: BLE001
                st.info(
                    "📋 **EDA module not found.** Create `modules/eda.py` with a "
                    "`run()` function and it will appear here automatically."
                )
                LOGGER.warning("eda import failed: %s", exc)
                return
            self._call_run(eda, df_full, df_filtered)
            return

        # All generated feature modules.
        try:
            module = importlib.import_module(f"modules.{module_name}")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not load module `{module_name}`.")
            st.exception(exc)
            LOGGER.exception("Module import error: %s", module_name)
            return
        self._call_run(module, df_full, df_filtered)

    @staticmethod
    def _call_run(module, df_full: pd.DataFrame, df_filtered: pd.DataFrame) -> None:
        """Call a module's run(); tolerate run() / run(df) / run(df, df2) signatures."""
        run: Callable = getattr(module, "run", None)
        if run is None:
            st.error(f"`{module.__name__}` has no run() function.")
            return
        try:
            try:
                run(df_filtered, df_full)            # preferred signature
            except TypeError:
                try:
                    run(df_filtered)                 # single-arg
                except TypeError:
                    run()                            # no-arg (e.g. eda)
        except Exception as exc:  # noqa: BLE001
            st.error("⚠️ This module hit an unexpected error but the app is still running.")
            with st.expander("Technical details"):
                st.exception(exc)
            LOGGER.exception("run() error in %s", module.__name__)

    # ----------------------------------------------------------------- main
    def run(self) -> None:
        self._inject_css()
        df_full = self._load_engineered()

        if df_full.empty:
            st.error(
                "Dataset could not be loaded. Ensure "
                "`data/cleaned_merged_data.csv` exists."
            )
            return

        page = self._sidebar(df_full)
        df_filtered = self._apply_filters(df_full)
        self._route(page, df_full, df_filtered)


def main() -> None:
    FinancialIntelligenceApp().run()


if __name__ == "__main__":
    main()
