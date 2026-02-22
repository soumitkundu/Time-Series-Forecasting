"""
FastAPI app: load saved LSTM and XGBoost models and expose predict endpoint.
Run with: uvicorn api.main:app --reload (from project root).
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from tensorflow import keras

# Project root when running from repo (e.g. uvicorn api.main:app)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLEANED_CSV = PROJECT_ROOT / "data" / "df_nvidia_cleaned.csv"
LSTM_MODEL_PATH = PROJECT_ROOT / "models" / "model_lstm.keras"
SCALER_LSTM_PATH = PROJECT_ROOT / "models" / "scaler_lstm.joblib"
XGB_MODEL_PATH = PROJECT_ROOT / "models" / "model_xgb.joblib"
TIME_STEP = 100

app = FastAPI(title="Time-Series Forecasting API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy-loaded models
_lstm_model = None
_scaler = None
_xgb_model = None


def _load_lstm():
    global _lstm_model, _scaler
    if _lstm_model is None and LSTM_MODEL_PATH.exists():
        _lstm_model = keras.models.load_model(LSTM_MODEL_PATH)
        _scaler = joblib.load(SCALER_LSTM_PATH)
    return _lstm_model, _scaler


def _load_xgb():
    global _xgb_model
    if _xgb_model is None and XGB_MODEL_PATH.exists():
        _xgb_model = joblib.load(XGB_MODEL_PATH)
    return _xgb_model


def _get_last_close_values(n: int = TIME_STEP) -> np.ndarray:
    """Load cleaned CSV and return last n Close values (for LSTM)."""
    if not CLEANED_CSV.exists():
        raise FileNotFoundError("Cleaned data not found. Run the pipeline first.")
    df = pd.read_csv(CLEANED_CSV, index_col=0, parse_dates=True)
    close = df["Close"].values
    if len(close) < n:
        raise ValueError(f"Need at least {n} close values; got {len(close)}")
    return close[-n:].astype(np.float64)


class PredictResponse(BaseModel):
    model: str
    next_day_close: float


@app.get("/")
def home():
    """Home: service is running."""
    return {"status": "ok", "message": "Time-Series Forecasting API"}

@app.get("/health")
def health():
    """Liveness: service is running."""
    return {"status": "ok"}


@app.get("/ready")
def ready():
    """Readiness: models and data are loaded (for orchestrators)."""
    missing = []
    if not LSTM_MODEL_PATH.exists():
        missing.append("lstm_model")
    if not SCALER_LSTM_PATH.exists():
        missing.append("scaler_lstm")
    if not XGB_MODEL_PATH.exists():
        missing.append("xgb_model")
    if not CLEANED_CSV.exists():
        missing.append("cleaned_data")
    if missing:
        return Response(
            content={"detail": "Not ready", "missing": missing},
            status_code=503,
        )
    return {"status": "ready"}


@app.get("/predict/next_day", response_model=PredictResponse)
def predict_next_day(model: str = "xgboost"):
    """
    Predict next-day close. Uses last available data from cleaned CSV.
    model: 'lstm' | 'xgboost' (default: xgboost for single-value input).
    """
    if model == "lstm":
        lstm_model, scaler = _load_lstm()
        if lstm_model is None:
            raise HTTPException(status_code=503, detail="LSTM model not found. Run the pipeline first.")
        last = _get_last_close_values(TIME_STEP)
        scaled = scaler.transform(last.reshape(-1, 1))
        X = scaled.reshape(1, TIME_STEP, 1)
        pred_scaled = lstm_model.predict(X, verbose=0)
        next_close = float(scaler.inverse_transform(pred_scaled)[0, 0])
        return PredictResponse(model="lstm", next_day_close=next_close)

    if model == "xgboost":
        xgb_model = _load_xgb()
        if xgb_model is None:
            raise HTTPException(status_code=503, detail="XGBoost model not found. Run the pipeline first.")
        last = _get_last_close_values(1)
        X = last.reshape(1, -1)
        next_close = float(xgb_model.predict(X)[0])
        return PredictResponse(model="xgboost", next_day_close=next_close)

    raise HTTPException(status_code=400, detail="model must be 'lstm' or 'xgboost'")
