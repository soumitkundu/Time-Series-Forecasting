"""Configuration for time-series forecasting pipeline."""

from pathlib import Path

# Project root (parent of src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_DAILY_CSV = DATA_DIR / "df_nvidia_stock_history.csv"
CLEANED_CSV = DATA_DIR / "df_nvidia_cleaned.csv"

# Model artifacts
MODELS_DIR = PROJECT_ROOT / "models"
LSTM_MODEL_PATH = MODELS_DIR / "model_lstm.keras"
SCALER_LSTM_PATH = MODELS_DIR / "scaler_lstm.joblib"
XGB_MODEL_PATH = MODELS_DIR / "model_xgb.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"

# Ingestion
TICKER = "NVDA"
PERIOD_DAILY = "5y"
INTERVAL_DAILY = "1d"

# Preprocess
OHLCV_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
DATE_COLUMN = "Date"

# LSTM
TIME_STEP = 100
TRAIN_RATIO = 0.7
LSTM_EPOCHS = 100
LSTM_BATCH_SIZE = 64
LSTM_UNITS = 50

# XGBoost
XGB_TEST_SIZE = 0.3


def ensure_dirs() -> None:
    """Create data and models directories if they do not exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
