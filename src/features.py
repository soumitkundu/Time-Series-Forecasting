"""Feature and target construction for time-series models."""

import numpy as np
import pandas as pd


def create_target(df: pd.DataFrame, close_col: str = "Close") -> pd.DataFrame:
    """
    Add next-day Close as Target for XGBoost.
    Drops the last row (no target). Returns DataFrame with Close and Target.
    """
    out = df[[close_col]].copy()
    out["Target"] = out[close_col].shift(-1)
    out = out.dropna()
    return out


def train_test_split_xgb(
    data: pd.DataFrame, test_size: float = 0.2
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Time-based split for XGBoost. Expects DataFrame with features and last column = Target.
    Returns X_train, X_test, y_train, y_test as numpy arrays.
    """
    values = data.values
    n = int(len(values) * (1 - test_size))
    X_train = values[:n, :-1]
    X_test = values[n:, :-1]
    y_train = values[:n, -1]
    y_test = values[n:, -1]
    return X_train, X_test, y_train, y_test
