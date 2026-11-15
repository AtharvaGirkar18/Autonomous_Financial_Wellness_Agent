from __future__ import annotations

import numpy as np
import pandas as pd
from datetime import timedelta


def _recurring_schedule(df: pd.DataFrame) -> dict[int, float]:
    recurring = df[df["is_recurring"].eq(True)].copy()
    if recurring.empty:
        return {}
    recurring["day"] = pd.to_datetime(recurring["date"]).dt.day
    recurring["signed"] = np.where(
        recurring["transaction_type"].eq("CREDIT"), recurring["amount"], -recurring["amount"]
    )
    return recurring.groupby("day")["signed"].median().to_dict()


def forecast_cash_flow(df: pd.DataFrame, current_balance: float, horizon_days: int = 30) -> dict:
    """Forecast daily balances using recent variable flow + learned recurring schedule."""
    x = df.copy()
    x["date"] = pd.to_datetime(x["date"], errors="coerce")
    x["amount"] = pd.to_numeric(x["amount"], errors="coerce").fillna(0.0)
    x["signed"] = np.where(x["transaction_type"].eq("CREDIT"), x["amount"], -x["amount"])
    x = x.dropna(subset=["date"])
    if x.empty:
        return {"forecast": [], "method": "insufficient data", "mae": None, "rmse": None, "mape": None}

    non_recurring = x[~x["is_recurring"].eq(True)]
    daily = non_recurring.groupby("date")["signed"].sum()
    baseline = float(daily.tail(60).median()) if not daily.empty else 0.0
    volatility = float(daily.tail(60).std(ddof=0) or 0.0) if len(daily) > 1 else 0.0
    schedule = _recurring_schedule(x)

    last_date = pd.Timestamp(x["date"].max())
    balance = float(current_balance)
    rows: list[dict] = []
    for i in range(1, horizon_days + 1):
        day = last_date + timedelta(days=i)
        recurring_flow = float(schedule.get(int(day.day), 0.0))
        balance += baseline + recurring_flow
        uncertainty = 1.28 * volatility * np.sqrt(i)
        rows.append({
            "date": day.date().isoformat(),
            "expected_balance": round(balance, 2),
            "lower": round(balance - uncertainty, 2),
            "upper": round(balance + uncertainty, 2),
            "recurring_net_flow": round(recurring_flow, 2),
        })

    min_row = min(rows, key=lambda row: row["expected_balance"])
    return {
        "forecast": rows,
        "method": "recent variable-flow median + learned recurring schedule",
        "baseline_daily_flow": round(baseline, 2),
        "volatility": round(volatility, 2),
        "lowest_projected_balance": min_row["expected_balance"],
        "lowest_projected_date": min_row["date"],
        "horizon_days": horizon_days,
    }
