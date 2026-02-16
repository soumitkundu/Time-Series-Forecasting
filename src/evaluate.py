"""Load trained models and compute RMSE/MAE; optionally save metrics."""

import json
import logging
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow import keras

from .config import (
    CLEANED_CSV,
    MODELS_DIR,
    LSTM_MODEL_PATH,
    SCALER_LSTM_PATH,
    XGB_MODEL_PATH,
    METRICS_PATH,
    TIME_STEP,
    TRAIN_RATIO,
    XGB_TEST_SIZE,
)
from .preprocess import get_close_series
from .features import create_target, train_test_split_xgb

logger = logging.getLogger(__name__)


def _xy_split(data: np.ndarray, time_step: int) -> tuple[np.ndarray, np.ndarray]:
    """Same as train.xy_split for evaluation."""
    df_x, df_y = [], []
    for i in range(len(data) - time_step - 1):
        df_x.append(data[i : (i + time_step)])
        df_y.append(data[i + time_step])
    return np.array(df_x), np.array(df_y)


def evaluate_lstm(
    model_path: Path | str | None = None,
    scaler_path: Path | str | None = None,
    cleaned_path: Path | str | None = None,
) -> dict:
    """
    Load LSTM and scaler, run on test segment, inverse_transform predictions, compute RMSE/MAE.
    Returns dict with lstm_rmse_test, lstm_mae_test (and optionally train metrics).
    """
    model_path = Path(model_path or LSTM_MODEL_PATH)
    scaler_path = Path(scaler_path or SCALER_LSTM_PATH)
    cleaned_path = Path(cleaned_path or CLEANED_CSV)

    if not model_path.exists() or not scaler_path.exists():
        logger.warning("LSTM model or scaler not found; skipping LSTM evaluation")
        return {}

    model = keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)

    df = pd.read_csv(cleaned_path, index_col=0, parse_dates=True)
    close = get_close_series(df)
    if isinstance(close, pd.DataFrame):
        close = close.squeeze()
    close_values = np.array(close).reshape(-1, 1)
    scaled = scaler.transform(close_values)

    n = int(len(scaled) * TRAIN_RATIO)
    train_scaled = scaled[:n]
    test_scaled = scaled[n:]

    X_train, y_train = _xy_split(train_scaled, TIME_STEP)
    X_test, y_test = _xy_split(test_scaled, TIME_STEP)

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    train_pred = model.predict(X_train, verbose=0)
    test_pred = model.predict(X_test, verbose=0)

    train_pred = scaler.inverse_transform(train_pred)
    test_pred = scaler.inverse_transform(test_pred)

    # y_train, y_test are scaled; inverse transform for metric
    y_train_orig = scaler.inverse_transform(y_train.reshape(-1, 1)).flatten()
    y_test_orig = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    train_rmse = math.sqrt(mean_squared_error(y_train_orig, train_pred.flatten()))
    test_rmse = math.sqrt(mean_squared_error(y_test_orig, test_pred.flatten()))
    train_mae = mean_absolute_error(y_train_orig, train_pred.flatten())
    test_mae = mean_absolute_error(y_test_orig, test_pred.flatten())

    metrics = {
        "lstm_rmse_train": float(train_rmse),
        "lstm_rmse_test": float(test_rmse),
        "lstm_mae_train": float(train_mae),
        "lstm_mae_test": float(test_mae),
    }
    logger.info("LSTM test RMSE: %.4f, test MAE: %.4f", test_rmse, test_mae)
    return metrics


def evaluate_xgb(
    model_path: Path | str | None = None,
    cleaned_path: Path | str | None = None,
) -> dict:
    """
    Load XGBoost, run on test set, compute RMSE and MAE.
    Returns dict with xgb_rmse_test, xgb_mae_test.
    """
    model_path = Path(model_path or XGB_MODEL_PATH)
    cleaned_path = Path(cleaned_path or CLEANED_CSV)

    if not model_path.exists():
        logger.warning("XGBoost model not found; skipping XGB evaluation")
        return {}

    model = joblib.load(model_path)
    df = pd.read_csv(cleaned_path, index_col=0, parse_dates=True)
    df_target = create_target(df)
    X_train, X_test, y_train, y_test = train_test_split_xgb(df_target, test_size=XGB_TEST_SIZE)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    metrics = {
        "xgb_rmse_train": float(math.sqrt(mean_squared_error(y_train, train_pred))),
        "xgb_rmse_test": float(math.sqrt(mean_squared_error(y_test, test_pred))),
        "xgb_mae_train": float(mean_absolute_error(y_train, train_pred)),
        "xgb_mae_test": float(mean_absolute_error(y_test, test_pred)),
    }
    logger.info("XGBoost test RMSE: %.4f, test MAE: %.4f", metrics["xgb_rmse_test"], metrics["xgb_mae_test"])
    return metrics


def run_evaluation(
    metrics_path: Path | str | None = None,
    cleaned_path: Path | str | None = None,
) -> dict:
    """
    Evaluate both models and merge metrics; optionally write to models/metrics.json.
    Returns combined metrics dict.
    """
    metrics_path = Path(metrics_path or METRICS_PATH)
    cleaned_path = cleaned_path or CLEANED_CSV

    lstm_metrics = evaluate_lstm(cleaned_path=cleaned_path)
    xgb_metrics = evaluate_xgb(cleaned_path=cleaned_path)
    combined = {**lstm_metrics, **xgb_metrics}

    if combined:
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, "w") as f:
            json.dump(combined, f, indent=2)
        logger.info("Saved metrics to %s", metrics_path)
    return combined
