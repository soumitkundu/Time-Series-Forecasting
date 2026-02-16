"""Train LSTM and XGBoost models; save models and scaler."""

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras import layers
from xgboost import XGBRegressor

from .config import (
    CLEANED_CSV,
    MODELS_DIR,
    LSTM_MODEL_PATH,
    SCALER_LSTM_PATH,
    XGB_MODEL_PATH,
    TIME_STEP,
    TRAIN_RATIO,
    LSTM_EPOCHS,
    LSTM_BATCH_SIZE,
    LSTM_UNITS,
    XGB_TEST_SIZE,
    ensure_dirs,
)
from .preprocess import load_raw, clean, get_close_series
from .features import create_target, train_test_split_xgb

logger = logging.getLogger(__name__)


def xy_split(data: np.ndarray, time_step: int) -> tuple[np.ndarray, np.ndarray]:
    """Create sequences for LSTM: X shape (n, time_step), y shape (n,)."""
    df_x, df_y = [], []
    for i in range(len(data) - time_step - 1):
        df_x.append(data[i : (i + time_step)])
        df_y.append(data[i + time_step])
    return np.array(df_x), np.array(df_y)


def train_lstm(
    cleaned_path: Path | str | None = None,
    model_path: Path | str | None = None,
    scaler_path: Path | str | None = None,
) -> tuple[keras.Model, MinMaxScaler]:
    """
    Load cleaned data, scale Close, split 70/30, build and fit LSTM, save model and scaler.
    Returns (model, scaler).
    """
    ensure_dirs()
    cleaned_path = Path(cleaned_path or CLEANED_CSV)
    model_path = Path(model_path or LSTM_MODEL_PATH)
    scaler_path = Path(scaler_path or SCALER_LSTM_PATH)

    df = pd.read_csv(cleaned_path, index_col=0, parse_dates=True)
    close = get_close_series(df)
    if isinstance(close, pd.DataFrame):
        close = close.squeeze()
    close_values = np.array(close).reshape(-1, 1)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(close_values)

    n = int(len(scaled) * TRAIN_RATIO)
    train_scaled = scaled[:n]
    test_scaled = scaled[n:]

    X_train, y_train = xy_split(train_scaled, TIME_STEP)
    X_test, y_test = xy_split(test_scaled, TIME_STEP)

    # Reshape for LSTM (samples, time_step, features)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    model = keras.Sequential([
        layers.LSTM(LSTM_UNITS, activation="relu", return_sequences=True, input_shape=(TIME_STEP, 1)),
        layers.LSTM(LSTM_UNITS, return_sequences=True),
        layers.LSTM(LSTM_UNITS),
        layers.Dense(1),
    ])
    model.compile(loss="mean_squared_error", optimizer="adam")

    logger.info("Training LSTM for %d epochs...", LSTM_EPOCHS)
    model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=LSTM_EPOCHS,
        batch_size=LSTM_BATCH_SIZE,
        verbose=1,
    )

    model.save(model_path)
    joblib.dump(scaler, scaler_path)
    logger.info("Saved LSTM model to %s and scaler to %s", model_path, scaler_path)
    return model, scaler


def train_xgb(
    cleaned_path: Path | str | None = None,
    model_path: Path | str | None = None,
) -> XGBRegressor:
    """
    Load cleaned data, create_target, time split, fit XGBRegressor, save model.
    Returns the fitted model.
    """
    ensure_dirs()
    cleaned_path = Path(cleaned_path or CLEANED_CSV)
    model_path = Path(model_path or XGB_MODEL_PATH)

    df = pd.read_csv(cleaned_path, index_col=0, parse_dates=True)
    df_target = create_target(df)

    X_train, X_test, y_train, y_test = train_test_split_xgb(df_target, test_size=XGB_TEST_SIZE)

    model = XGBRegressor()
    model.fit(X_train, y_train)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    logger.info("Saved XGBoost model to %s", model_path)
    return model


def run_training(
    cleaned_path: Path | str | None = None,
) -> tuple[keras.Model | None, MinMaxScaler | None, XGBRegressor | None]:
    """
    Train both LSTM and XGBoost; save all artifacts.
    Returns (lstm_model, scaler, xgb_model).
    """
    lstm_model, scaler = train_lstm(cleaned_path=cleaned_path)
    xgb_model = train_xgb(cleaned_path=cleaned_path)
    return lstm_model, scaler, xgb_model
