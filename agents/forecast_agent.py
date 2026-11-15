import pandas as pd
from agents.base import run_agent
from analytics.cashflow import forecast_cash_flow


def forecast_node(state: dict) -> dict:
    def _run(s):
        df = pd.DataFrame(s["transactions"])
        current_balance = float(s.get("current_balance", s.get("starting_balance", 32000)))
        return {"forecast": forecast_cash_flow(df, current_balance, horizon_days=30)}
    return run_agent(state, "Cash Flow Forecasting Agent", _run, ["forecast_cash_flow"])
