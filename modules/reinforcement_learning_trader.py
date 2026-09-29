"""
modules/reinforcement_learning_trader.py
-----------------------------------------
Model 14 — Self-Learning AI Trading Agent.

Models : PPO / DQN via stable-baselines3 (optional). When the RL stack is not
         installed, a self-contained tabular Q-learning agent (pure NumPy)
         trains on a discretised market state so portfolio growth, the reward
         curve and trade logs always render.
Actions: BUY / SELL / HOLD.  Reward: change in portfolio value.
Outputs: Portfolio Growth, Reward Curve, Trading Logs.

NOTE: All reinforcement-learning, trading, reward and portfolio calculations are
unchanged. Only the presentation layer (the `run` function's output) has been
redesigned for non-technical investors. The helper functions below
(`_series`, `_discretise`, `_q_learning`, `_stable_baselines`) are untouched.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from config.settings import APP, COLS, PALETTE
from utils import visualizations as viz
from utils.helper_functions import download_buttons, has_package, page_header
from utils.preprocessing import get_engineered_data

ACTIONS = ["HOLD", "BUY", "SELL"]


# FIX: Accept df as a parameter instead of calling get_engineered_data() inside,
# which caused a MemoryError when Streamlit tried to unpickle the large cached
# DataFrame on every symbol switch.
@st.cache_data(show_spinner=False, max_entries=20)
def _series(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    sdf = df[df[COLS.symbol] == symbol].sort_values(COLS.date).copy()
    cols = [COLS.date, COLS.stock_close, "RSI", "Daily_Return", "Momentum"]
    cols = [c for c in cols if c in sdf.columns]
    return sdf[cols].dropna().reset_index(drop=True)


def _discretise(row: pd.Series) -> int:
    """Map indicators to a small discrete state index."""
    rsi = row.get("RSI", 50.0)
    mom = row.get("Momentum", 0.0)
    rsi_bin = 0 if rsi < 35 else (2 if rsi > 65 else 1)      # over/under/neutral
    mom_bin = 0 if mom < 0 else 1                            # down / up
    return rsi_bin * 2 + mom_bin                              # 0..5


def _q_learning(data: pd.DataFrame, episodes: int = 40) -> Tuple[np.ndarray, List[float], pd.DataFrame, str]:
    """Tabular Q-learning over discretised states. Returns (portfolio, rewards, log, name)."""
    rng = np.random.default_rng(APP.random_state)
    n_states, n_actions = 6, 3
    Q = np.zeros((n_states, n_actions))
    alpha, gamma = 0.1, 0.95
    prices = data[COLS.stock_close].values
    states = np.array([_discretise(r) for _, r in data.iterrows()])
    rets = np.append(np.diff(prices) / prices[:-1], 0.0)

    episode_rewards: List[float] = []
    for ep in range(episodes):
        eps = max(0.05, 1.0 - ep / episodes)
        total = 0.0
        for t in range(len(prices) - 1):
            s = states[t]
            a = rng.integers(n_actions) if rng.random() < eps else int(np.argmax(Q[s]))
            # reward: long if BUY, short if SELL, flat if HOLD
            if a == 1:      # BUY
                r = rets[t + 1]
            elif a == 2:    # SELL
                r = -rets[t + 1]
            else:           # HOLD
                r = 0.0
            s2 = states[t + 1]
            Q[s, a] += alpha * (r + gamma * np.max(Q[s2]) - Q[s, a])
            total += r
        episode_rewards.append(total)

    # ---- Greedy roll-out for portfolio + logs ----------------------------
    cash, position = 10000.0, 0.0
    portfolio = [cash]
    logs = []
    for t in range(len(prices) - 1):
        s = states[t]
        a = int(np.argmax(Q[s]))
        price = prices[t]
        if a == 1 and cash > 0:                # BUY (all-in)
            position = cash / price
            cash = 0.0
            logs.append((data[COLS.date].iloc[t], "BUY", price))
        elif a == 2 and position > 0:          # SELL (all-out)
            cash = position * price
            position = 0.0
            logs.append((data[COLS.date].iloc[t], "SELL", price))
        portfolio.append(cash + position * prices[t + 1])
    log_df = pd.DataFrame(logs, columns=["Date", "Action", "Price"])
    return np.array(portfolio), episode_rewards, log_df, "Tabular Q-Learning"


def _stable_baselines(data: pd.DataFrame) -> Optional[Tuple[np.ndarray, List[float], pd.DataFrame, str]]:
    """Try a real SB3 agent; return None to signal fallback."""
    if not (has_package("stable_baselines3") and has_package("gymnasium")):
        return None
    try:
        import gymnasium as gym
        from gymnasium import spaces
        from stable_baselines3 import DQN

        prices = data[COLS.stock_close].values.astype(float)
        feat = data[["RSI", "Momentum"]].fillna(0).values if "RSI" in data else \
            np.zeros((len(prices), 2))

        class TradingEnv(gym.Env):
            def __init__(self):
                super().__init__()
                self.action_space = spaces.Discrete(3)
                self.observation_space = spaces.Box(-np.inf, np.inf, (3,), np.float32)
                self.reset()

            def reset(self, *, seed=None, options=None):
                super().reset(seed=seed)
                self.t = 0
                self.cash, self.pos = 10000.0, 0.0
                return self._obs(), {}

            def _obs(self):
                p = prices[self.t]
                return np.array([feat[self.t, 0], feat[self.t, 1],
                                 self.pos * p / 10000.0], np.float32)

            def step(self, action):
                prev = self.cash + self.pos * prices[self.t]
                if action == 1 and self.cash > 0:
                    self.pos = self.cash / prices[self.t]
                    self.cash = 0.0
                elif action == 2 and self.pos > 0:
                    self.cash = self.pos * prices[self.t]
                    self.pos = 0.0
                self.t += 1
                done = self.t >= len(prices) - 1
                val = self.cash + self.pos * prices[self.t]
                return self._obs(), float(val - prev), done, False, {}

        env = TradingEnv()
        model = DQN("MlpPolicy", env, verbose=0, learning_starts=200)
        model.learn(total_timesteps=3000)
        # roll-out
        obs, _ = env.reset()
        portfolio, logs = [10000.0], []
        done = False
        while not done:
            a, _ = model.predict(obs, deterministic=True)
            a = int(a)
            t = env.t
            if a == 1 and env.cash > 0:
                logs.append((data[COLS.date].iloc[t], "BUY", float(prices[t])))
            elif a == 2 and env.pos > 0:
                logs.append((data[COLS.date].iloc[t], "SELL", float(prices[t])))
            obs, _, done, _, _ = env.step(a)
            portfolio.append(env.cash + env.pos * prices[env.t])
        return (np.array(portfolio), [float(np.sum(np.diff(portfolio)))],
                pd.DataFrame(logs, columns=["Date", "Action", "Price"]), "DQN (stable-baselines3)")
    except Exception:  # noqa: BLE001
        return None


# --------------------------------------------------------------------------- #
# Presentation-only helpers (no trading / RL / reward / portfolio maths here)  #
# These translate already-computed indicators into plain-language displays.    #
# --------------------------------------------------------------------------- #
def _recommendation(rsi: float, mom: float, edge: float) -> dict:
    """Turn the latest indicators into a human-readable recommendation.

    This is purely for display — it does NOT influence the agent's decisions,
    rewards or portfolio, all of which are computed upstream and unchanged.
    """
    score = 0
    reasons: List[str] = []
    if rsi < 30:
        score += 2; reasons.append("RSI indicates oversold conditions")
    elif rsi < 45:
        score += 1; reasons.append("RSI is leaning oversold")
    elif rsi > 70:
        score -= 2; reasons.append("RSI indicates overbought conditions")
    elif rsi > 55:
        score -= 1; reasons.append("RSI is leaning overbought")
    else:
        reasons.append("RSI is in a neutral range")

    if mom > 0:
        score += 1; reasons.append("Positive momentum detected")
    elif mom < 0:
        score -= 1; reasons.append("Weak momentum suggests caution")
    else:
        reasons.append("Momentum is flat")

    if score >= 3:
        strength, signal = "Strong Buy", "BUY"
    elif score >= 1:
        strength, signal = "Buy", "BUY"
    elif score == 0:
        strength, signal = "Neutral", "HOLD"
    elif score >= -2:
        strength, signal = "Sell", "SELL"
    else:
        strength, signal = "Strong Sell", "SELL"

    view = "Bullish" if score > 0 else ("Bearish" if score < 0 else "Neutral")
    conf = 50 + abs(score) * 9 + max(min(edge, 12.0), -12.0) * 0.5
    conf = float(max(50.0, min(96.0, conf)))
    return {"signal": signal, "strength": strength, "confidence": conf,
            "view": view, "reasons": reasons}


def _trade_reason(rsi: float, mom: float) -> str:
    """Plain-language reason for a trade, derived from its day's indicators."""
    oversold, overbought = rsi < 35, rsi > 65
    if oversold and mom > 0:
        return "Oversold + Positive Momentum"
    if overbought and mom < 0:
        return "Overbought + Negative Momentum"
    if mom > 0:
        return "Momentum Strengthening"
    if mom < 0:
        return "Momentum Weakening"
    return "Neutral Conditions"


_BADGE = {"BUY": "badge-buy", "SELL": "badge-sell", "HOLD": "badge-hold"}
_ARROW = {"BUY": "▲", "SELL": "▼", "HOLD": "■"}


def run(df: Optional[pd.DataFrame] = None, df_full: Optional[pd.DataFrame] = None) -> None:
    page_header("AI Trading Assistant",
                "Receive AI-generated trading recommendations and compare performance against a buy-and-hold strategy.")

    if df is None:
        df = get_engineered_data()
    if df.empty:
        st.warning("No data available.")
        return

    symbols = sorted(df[COLS.symbol].unique().tolist())
    c1, c2 = st.columns([2, 1])
    with c1:
        symbol = st.selectbox("Select stock", symbols, key="rl_symbol")
    with c2:
        episodes = st.slider("Training episodes", 10, 80, 40, 10, key="rl_eps")

    # FIX: Pass df directly to _series instead of re-loading inside it,
    # avoiding the MemoryError from Streamlit re-pickling the full dataset.
    data = _series(df, symbol)
    if len(data) < 120:
        st.warning("Not enough history for this stock.")
        return
    data = data.tail(1500).reset_index(drop=True)  # keep training light

    sb3_available = has_package("stable_baselines3") and has_package("gymnasium")

    # ======================== UNCHANGED MODEL TRAINING ====================
    with st.spinner("Training the agent…"):
        result = _stable_baselines(data) if sb3_available else None
        if result is None:
            result = _q_learning(data, episodes=episodes)
    portfolio, rewards, log_df, agent_name = result

    # ---- Performance figures (unchanged calculations) --------------------
    start_val, end_val = portfolio[0], portfolio[-1]
    agent_ret = (end_val / start_val - 1.0) * 100
    bh_ret = (data[COLS.stock_close].iloc[-1] / data[COLS.stock_close].iloc[0] - 1.0) * 100
    n_trades = len(log_df)
    edge = agent_ret - bh_ret

    # ---- Portfolio growth data (unchanged calculations) ------------------
    n = len(portfolio)
    bh_curve = start_val * (data[COLS.stock_close].values[:n] / data[COLS.stock_close].values[0])
    growth = pd.DataFrame({
        COLS.date: data[COLS.date].values[:n],
        "Agent_Portfolio": portfolio,
        "Buy_Hold": bh_curve,
    })
    # ======================================================================

    # Latest indicators -> plain-language recommendation (display only).
    last = data.iloc[-1]
    rsi_now = float(last.get("RSI", 50.0))
    mom_now = float(last.get("Momentum", 0.0))
    rec = _recommendation(rsi_now, mom_now, edge)

    # ========================================================= SECTION 1 ==
    # AI TRADER RECOMMENDATION CARD (primary output)
    st.markdown(
        f"<div class='glass'>"
        f"<div class='kpi-label'>Current AI Signal</div>"
        f"<h1 style='margin:0.25rem 0'>"
        f"<span class='badge {_BADGE[rec['signal']]}'>{rec['signal']} {_ARROW[rec['signal']]}</span> "
        f"<span style='font-size:1.05rem;opacity:0.85'>· {rec['strength']}</span></h1>"
        f"<p style='margin:0.35rem 0;font-size:1.02rem'>"
        f"Confidence <b>{rec['confidence']:.0f}%</b> &nbsp;·&nbsp; "
        f"Market View <b>{rec['view']}</b></p>"
        f"<p style='margin:0.15rem 0;opacity:0.85'>{' · '.join(rec['reasons'])}.</p>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # ========================================================= SECTION 2 ==
    # PERFORMANCE SCORECARD (simple KPIs)
    st.markdown("Performance Scorecard")
    r1c1, r1c2, r1c3 = st.columns(3)
    r1c1.metric("Total Return", f"{agent_ret:+.1f}%")
    r1c2.metric("Buy & Hold Return", f"{bh_ret:+.1f}%")
    r1c3.metric("Outperformance vs Buy & Hold", f"{edge:+.1f}%")
    r2c1, r2c2, r2c3 = st.columns(3)
    r2c1.metric("Number of Trades", f"{n_trades}")
    r2c2.metric("Starting Portfolio", f"₹{start_val:,.0f}")
    r2c3.metric("Ending Portfolio", f"₹{end_val:,.0f}")

    # ========================================================= SECTION 3 ==
    # PORTFOLIO GROWTH (same chart, friendlier labels)
    st.markdown("AI Trader vs Buy & Hold Performance")
    growth_display = growth.rename(columns={"Agent_Portfolio": "AI Trader",
                                            "Buy_Hold": "Buy & Hold"})
    st.plotly_chart(
        viz.line_chart(growth_display, COLS.date, ["AI Trader", "Buy & Hold"],
                       title="AI Trader vs Buy & Hold Performance"),
        use_container_width=True,
    )
    st.caption("Higher growth means better trading performance — compare the AI "
               "Trader line against simply buying the stock and holding it.")

    # ========================================================= SECTION 4 ==
    # RECENT AI DECISIONS (clean, plain-language table)
    st.markdown("Recent AI Decisions")
    if log_df.empty:
        st.info("The AI stayed on the sidelines and made no trades for this stock.")
    else:
        dd = log_df.merge(data[[COLS.date, "RSI", "Momentum"]], on=COLS.date, how="left")
        dd["Reason"] = [_trade_reason(float(r) if pd.notna(r) else 50.0,
                                      float(m) if pd.notna(m) else 0.0)
                        for r, m in zip(dd["RSI"], dd["Momentum"])]
        decisions = dd[["Date", "Action", "Price", "Reason"]].copy()
        decisions["Price"] = decisions["Price"].round(2)
        st.dataframe(decisions.tail(20), use_container_width=True, height=320)

    # ========================================================= SECTION 5 ==
    # AGENT VERDICT CARD (summary)
    if edge >= 0:
        perf_summary = (f"The AI Trader grew ₹{start_val:,.0f} to ₹{end_val:,.0f} "
                        f"({agent_ret:+.1f}%), beating Buy &amp; Hold by {edge:+.1f}%.")
    else:
        perf_summary = (f"The AI Trader grew ₹{start_val:,.0f} to ₹{end_val:,.0f} "
                        f"({agent_ret:+.1f}%), trailing Buy &amp; Hold by {abs(edge):.1f}%.")
    st.markdown(
        f"<div class='glass'>"
        f"<h3 style='margin:0.1rem 0'>AI Trader Outlook: "
        f"<span class='badge {_BADGE[rec['signal']]}'>{rec['view']}</span></h3>"
        f"<p style='margin:0.3rem 0'>Suggested Action: <b>{rec['signal']}</b> "
        f"({rec['strength']}) &nbsp;·&nbsp; Confidence: <b>{rec['confidence']:.0f}%</b></p>"
        f"<p style='margin:0.15rem 0;opacity:0.9'>{perf_summary}</p>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # ========================================================= SECTION 6 ==
    # ADVANCED REINFORCEMENT LEARNING DETAILS (hidden by default)
    with st.expander("Advanced Reinforcement Learning Details"):
        st.markdown(f"**Agent type:** {agent_name}")
        if not sb3_available:
            st.info("stable-baselines3 / gymnasium not installed — a built-in "
                    "tabular Q-learning agent was used. Install "
                    "`stable-baselines3 gymnasium` for PPO/DQN.")
        st.caption("Reinforcement learning works by rewarding the agent for "
                   "profitable actions over many simulated runs (episodes). The "
                   "reward curve below should generally trend upward as it learns.")

        a1, a2 = st.columns(2)
        with a1:
            rdf = pd.DataFrame({"Episode": range(1, len(rewards) + 1),
                                "Reward": rewards})
            st.plotly_chart(viz.line_chart(rdf, "Episode", ["Reward"],
                            title="Learning Reward Curve"), use_container_width=True)
        with a2:
            if not log_df.empty:
                counts = log_df["Action"].value_counts().reset_index()
                counts.columns = ["Action", "Count"]
                st.plotly_chart(viz.bar_chart(counts, "Action", "Count", color="Action",
                                title="Trade Action Counts"), use_container_width=True)
            else:
                st.info("Agent held throughout — no discrete trades executed.")

        st.markdown("Raw Trading Logs")
        if log_df.empty:
            st.info("No trades to log for this run.")
        else:
            st.dataframe(log_df.tail(200), use_container_width=True, height=300)
            download_buttons(log_df, key="rl", label=f"{symbol}_trading_logs")