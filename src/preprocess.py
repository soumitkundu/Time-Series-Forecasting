"""Load raw CSV, clean OHLCV data, and save cleaned DataFrame."""

import logging
from pathlib import Path

import pandas as pd

from .config import (
    RAW_DAILY_CSV,
    CLEANED_CSV,
    OHLCV_COLUMNS,
    DATE_COLUMN,
    ensure_dirs,
)

logger = logging.getLogger(__name__)


def load_raw(path: Path | str) -> pd.DataFrame:
    """Load raw stock CSV from path."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Raw data not found: {path}")
    df = pd.read_csv(path)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select Date + OHLCV, parse dates, set index.
    Drops Dividends and Stock Splits if present.
    """
    # Ensure we have a date-like column (yfinance may use 'Date' or 'Datetime')
    date_col = DATE_COLUMN if DATE_COLUMN in df.columns else "Datetime"
    if date_col not in df.columns:
        raise ValueError(f"Expected column '{date_col}' or 'Date' in data. Got: {list(df.columns)}")

    keep = [date_col] + [c for c in OHLCV_COLUMNS if c in df.columns]
    df = df[keep].copy()
    df[date_col] = pd.to_datetime(df[date_col], utc=True)
    # Strip timezone for consistency with notebook
    if df[date_col].dt.tz is not None:
        df[date_col] = df[date_col].dt.tz_localize(None)
    df = df.set_index(date_col)
    df = df.sort_index()
    df = df.dropna()
    return df


def get_close_series(df: pd.DataFrame) -> pd.Series:
    """Return Close series for LSTM (indexed by date)."""
    return df["Close"]


def run_preprocess(
    input_path: Path | str | None = None,
    output_path: Path | str | None = None,
) -> pd.DataFrame:
    """
    Load raw CSV, clean, save cleaned CSV. Returns cleaned DataFrame.
    Exposes Close via get_close_series(returned_df) and full DataFrame for XGBoost.
    """
    ensure_dirs()
    input_path = input_path or RAW_DAILY_CSV
    output_path = output_path or CLEANED_CSV

    df = load_raw(input_path)
    df = clean(df)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path)
    logger.info("Saved cleaned data to %s", output_path)
    return df
