from __future__ import annotations

import numpy as np
import pandas as pd


ESSENTIAL_CATEGORIES = {
    "Groceries", "Healthcare", "Education", "Bills & Utilities",
    "Rent", "Fuel", "Transportation", "Insurance", "EMI / Debt"
}


def build_spending_analysis(df: pd.DataFrame) -> dict:
    x = df.copy()
    x["date"] = pd.to_datetime(x["date"])
    deb = x[x["transaction_type"].eq("DEBIT")].copy()
    cred = x[x["transaction_type"].eq("CREDIT")].copy()

    deb["month"] = deb["date"].dt.to_period("M").astype(str)
    cred["month"] = cred["date"].dt.to_period("M").astype(str)

    monthly_exp = deb.groupby("month")["amount"].sum().to_dict()
    monthly_inc = cred.groupby("month")["amount"].sum().to_dict()
    cat = deb.groupby("category")["amount"].sum().sort_values(ascending=False)

    cat_month = (
        deb.pivot_table(index="month", columns="category", values="amount", aggfunc="sum", fill_value=0)
        .sort_index()
    )
    trends = {}
    for c in cat_month.columns:
        vals = cat_month[c]
        if len(vals) >= 3:
            base = float(vals.iloc[-3])
            end = float(vals.iloc[-1])
            growth = 0.0 if base == 0 else (end - base) / base
            direction = "Increasing" if growth > 0.05 else "Decreasing" if growth < -0.05 else "Stable"
        else:
            growth = 0.0
            direction = "Insufficient history"
        trends[c] = {"growth_3m": round(growth, 4), "direction": direction}

    essential = deb[deb["category"].isin(ESSENTIAL_CATEGORIES)]["amount"].sum()
    discretionary = deb[~deb["category"].isin(ESSENTIAL_CATEGORIES)]["amount"].sum()
    income_total = cred["amount"].sum()
    savings = income_total - deb["amount"].sum()
    savings_rate = 0.0 if income_total <= 0 else savings / income_total

    return {
        "monthly_expenses": monthly_exp,
        "monthly_income": monthly_inc,
        "category_totals": {k: round(float(v), 2) for k, v in cat.to_dict().items()},
        "category_month": cat_month.to_dict(orient="index"),
        "trends": trends,
        "essential_total": round(float(essential), 2),
        "discretionary_total": round(float(discretionary), 2),
        "income_total": round(float(income_total), 2),
        "expense_total": round(float(deb["amount"].sum()), 2),
        "savings_total": round(float(savings), 2),
        "savings_rate": round(float(savings_rate), 4),
        "expense_volatility": round(float(deb.groupby("month")["amount"].sum().std() or 0), 2),
    }


def compute_monthly_balance(df: pd.DataFrame, starting_balance: float) -> pd.DataFrame:
    x = df.copy()
    x["date"] = pd.to_datetime(x["date"])
    x["signed"] = np.where(x["transaction_type"].eq("CREDIT"), x["amount"], -x["amount"])
    daily = x.groupby("date")["signed"].sum().sort_index().cumsum() + starting_balance
    return daily.rename("balance").reset_index()
