from __future__ import annotations

import pandas as pd

from analytics.cashflow import forecast_cash_flow as _forecast_cash_flow
from ml.forecasting import naive_monthly_forecast, moving_average_forecast, evaluate_forecast


def forecast_cash_flow(df: pd.DataFrame, current_balance: float, horizon_days: int = 30) -> dict:
    """Forecast daily cash flow using the deterministic cash-flow engine."""
    return _forecast_cash_flow(df, current_balance=current_balance, horizon_days=horizon_days)


def compare_monthly_forecasts(actual: pd.Series, history: pd.Series) -> dict:
    """Return the two benchmark monthly forecasting strategies and their errors."""
    naive = naive_monthly_forecast(history)
    moving = moving_average_forecast(history)
    return {
        "naive": {"prediction": naive, "metrics": evaluate_forecast(actual, [naive] * len(actual))},
        "moving_average": {"prediction": moving, "metrics": evaluate_forecast(actual, [moving] * len(actual))},
    }
