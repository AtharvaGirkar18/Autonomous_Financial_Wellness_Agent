from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd


def get_transactions(df: pd.DataFrame, user_id: Optional[str] = None) -> pd.DataFrame:
    out = df.copy()
    if user_id:
        out = out[out["user_id"].eq(user_id)].copy()
    return out.sort_values("date").reset_index(drop=True)


def calculate_monthly_spending(df: pd.DataFrame) -> pd.DataFrame:
    x = df[df["transaction_type"].eq("DEBIT")].copy()
    x["month"] = pd.to_datetime(x["date"]).dt.to_period("M").astype(str)
    return x.groupby("month", as_index=False)["amount"].sum().rename(columns={"amount": "expenses"})


def calculate_category_spending(df: pd.DataFrame) -> pd.DataFrame:
    x = df[df["transaction_type"].eq("DEBIT")].copy()
    return x.groupby("category", as_index=False)["amount"].sum().sort_values("amount", ascending=False)


def calculate_savings_rate(df: pd.DataFrame) -> float:
    credits = df.loc[df["transaction_type"].eq("CREDIT"), "amount"].sum()
    debits = df.loc[df["transaction_type"].eq("DEBIT"), "amount"].sum()
    return 0.0 if credits <= 0 else max(0.0, (credits - debits) / credits)


def calculate_debt_ratio(df: pd.DataFrame) -> float:
    income = df.loc[df["transaction_type"].eq("CREDIT"), "amount"].sum()
    debt = df.loc[(df["transaction_type"].eq("DEBIT")) & (df["category"].eq("EMI / Debt")), "amount"].sum()
    return 0.0 if income <= 0 else debt / income


def calculate_budget_utilization(spent: float, allocated: float) -> float:
    return 0.0 if allocated <= 0 else spent / allocated


def calculate_emergency_fund(current_balance: float, essential_monthly_expense: float) -> float:
    return 0.0 if essential_monthly_expense <= 0 else current_balance / essential_monthly_expense


def simulate_emi(principal: float, annual_rate: float, months: int) -> float:
    if months <= 0:
        raise ValueError("months must be positive")
    r = annual_rate / 12 / 100
    if r == 0:
        return principal / months
    return principal * r * (1 + r) ** months / ((1 + r) ** months - 1)


def calculate_health_score(components: Dict[str, float]) -> float:
    # Deterministic weighted score in [0, 100].
    weights = {
        "savings_rate": 0.20,
        "expense_income": 0.15,
        "debt_health": 0.15,
        "budget_discipline": 0.15,
        "emergency_fund": 0.15,
        "spending_consistency": 0.10,
        "recurring_burden": 0.05,
        "cashflow_stability": 0.05,
    }
    bounded = {k: max(0.0, min(100.0, float(components.get(k, 0.0)))) for k in weights}
    return round(sum(bounded[k] * weights[k] for k in weights), 2)
