from __future__ import annotations

import numpy as np
import pandas as pd


def naive_monthly_forecast(series: pd.Series, horizon: int = 1) -> np.ndarray:
    if len(series) == 0:
        return np.zeros(horizon)
    return np.repeat(series.iloc[-1], horizon)


def moving_average_forecast(series: pd.Series, horizon: int = 1, window: int = 3) -> np.ndarray:
    if len(series) == 0:
        return np.zeros(horizon)
    mean = series.tail(min(window, len(series))).mean()
    return np.repeat(float(mean), horizon)


def evaluate_forecast(y_true, y_pred) -> dict:
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    nonzero = y_true != 0
    mape = float(np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100) if nonzero.any() else None
    return {"mae": round(float(mae), 2), "rmse": round(float(rmse), 2), "mape": None if mape is None else round(mape, 2)}
