"""
config/settings.py
-------------------
Centralised, type-hinted configuration for the AI Financial Intelligence Platform.

Keeping every tunable constant (paths, column names, colour palette, model
hyper-parameters) in a single place makes the rest of the codebase clean and
trivial to maintain.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

# --------------------------------------------------------------------------- #
# Filesystem paths                                                            #
# --------------------------------------------------------------------------- #
ROOT_DIR: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = ROOT_DIR / "data"
MODELS_DIR: Path = ROOT_DIR / "models" / "saved_models"
ASSETS_DIR: Path = ROOT_DIR / "assets"

DATA_FILE: Path = DATA_DIR / "cleaned_merged_data.csv"

# The shipped CSV contains non-UTF-8 bytes inside the news headlines, so we read
# it with a forgiving encoding. This is auto-detected at load time as well.
DATA_ENCODING: str = "latin-1"

# Ensure writable directories exist (safe to call repeatedly).
for _d in (MODELS_DIR, ASSETS_DIR / "animations"):
    _d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Column groups                                                               #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Columns:
    date: str = "Date"
    symbol: str = "Symbol"
    company: str = "Company"
    industry: str = "Industry"
    isin: str = "ISIN Code"
    headlines: str = "headlines"

    stock_prev_close: str = "Stock_Prev_Close"
    stock_open: str = "Stock_Open"
    stock_high: str = "Stock_High"
    stock_low: str = "Stock_Low"
    stock_last: str = "Stock_Last"
    stock_close: str = "Stock_Close"
    stock_vwap: str = "Stock_VWAP"
    stock_volume: str = "Stock_Volume"
    stock_turnover: str = "Stock_Turnover"
    stock_trades: str = "Stock_Trades"
    stock_deliverable_volume: str = "Stock_Deliverable_Volume"
    stock_deliverable_pct: str = "Stock_Deliverable_Percentage"

    gold_open: str = "Gold_Open"
    gold_high: str = "Gold_High"
    gold_low: str = "Gold_Low"
    gold_close: str = "Gold_Close"
    gold_volume: str = "Gold_Volume"

    @property
    def ohlc(self) -> List[str]:
        return [self.stock_open, self.stock_high, self.stock_low, self.stock_close]

    @property
    def numeric_core(self) -> List[str]:
        return [
            self.stock_open, self.stock_high, self.stock_low, self.stock_close,
            self.stock_vwap, self.stock_volume, self.gold_close,
        ]


COLS = Columns()


# --------------------------------------------------------------------------- #
# Visual palette (Bloomberg / TradingView inspired)                           #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Palette:
    bg: str = "#08090d"
    panel: str = "#11151d"
    panel_2: str = "#171c27"
    accent: str = "#22d3ee"
    accent_2: str = "#8b5cf6"
    up: str = "#2ec27e"
    down: str = "#f0563f"
    warn: str = "#f5b73d"
    text: str = "#e8edf4"
    muted: str = "#8a93a6"
    grid: str = "rgba(255,255,255,0.05)"

    sequence: List[str] = field(
        default_factory=lambda: [
            "#00d4ff", "#7c5cff", "#11d493", "#ffb020",
            "#ff4d6d", "#36cfc9", "#f759ab", "#9254de",
        ]
    )


PALETTE = Palette()


# --------------------------------------------------------------------------- #
# App-level settings                                                          #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AppConfig:
    app_name: str = "AI Financial Intelligence Platform"
    short_name: str = "AI-FIP"
    version: str = "1.0.0"
    # Hard cap on rows pulled into heavy model training to keep the UI snappy.
    max_train_rows: int = 60_000
    random_state: int = 42


APP = AppConfig()