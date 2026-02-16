"""Fetch stock data from yfinance and save raw CSV."""

import logging
from pathlib import Path

import pandas as pd
import yfinance as yf

from .config import (
    DATA_DIR,
    RAW_DAILY_CSV,
    TICKER,
    PERIOD_DAILY,
    INTERVAL_DAILY,
    ensure_dirs,
)

logger = logging.getLogger(__name__)


def fetch_daily(ticker: str = TICKER, period: str = PERIOD_DAILY) -> pd.DataFrame:
    """Download daily OHLCV data for the given ticker and period."""
    ensure_dirs()
    logger.info("Fetching daily data for %s, period=%s", ticker, period)
    obj = yf.Ticker(ticker)
    df = obj.history(period=period, interval=INTERVAL_DAILY)
    if df is None or df.empty:
        raise ValueError(f"No data returned for {ticker} (period={period})")
    df = df.reset_index()
    # yfinance may name the index column 'Date' or 'Datetime'
    if "Date" not in df.columns and "Datetime" in df.columns:
        df = df.rename(columns={"Datetime": "Date"})
    return df


def save_raw(df: pd.DataFrame, path: Path | str) -> None:
    """Save raw DataFrame to CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Saved raw data to %s", path)


def run_ingestion(
    ticker: str = TICKER,
    period: str = PERIOD_DAILY,
    output_path: Path | str | None = None,
) -> pd.DataFrame:
    """Download daily data and save to data directory. Returns the raw DataFrame."""
    output_path = output_path or RAW_DAILY_CSV
    df = fetch_daily(ticker=ticker, period=period)
    save_raw(df, output_path)
    return df
