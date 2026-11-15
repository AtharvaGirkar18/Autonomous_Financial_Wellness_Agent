from __future__ import annotations

import pandas as pd

from tools.transaction_tools import calculate_health_score


def score_financial_health(
    df: pd.DataFrame,
    current_balance: float,
    budget_discipline_pct: float,
) -> dict:
    """Return a deterministic, transparent 0-100 financial health score."""
    x = df.copy()
    x["date"] = pd.to_datetime(x["date"], errors="coerce")
    x["amount"] = pd.to_numeric(x["amount"], errors="coerce").fillna(0.0)
    if "category" not in x.columns:
        x["category"] = x.get("predicted_category", "Other")

    income = float(x.loc[x["transaction_type"].eq("CREDIT"), "amount"].sum())
    expenses = float(x.loc[x["transaction_type"].eq("DEBIT"), "amount"].sum())
    debt = float(
        x.loc[
            x["transaction_type"].eq("DEBIT")
            & x["category"].eq("EMI / Debt"),
            "amount",
        ].sum()
    )
    essential = float(
        x.loc[
            x["transaction_type"].eq("DEBIT")
            & x["category"].isin(
                [
                    "Groceries",
                    "Healthcare",
                    "Education",
                    "Bills & Utilities",
                    "Rent",
                    "Fuel",
                    "Transportation",
                    "Insurance",
                    "EMI / Debt",
                ]
            ),
            "amount",
        ].sum()
    )

    savings_rate = 0.0 if income <= 0 else max(0.0, (income - expenses) / income)
    expense_ratio = 0.0 if income <= 0 else expenses / income
    debt_ratio = 0.0 if income <= 0 else debt / income

    # Convert the observed period into an approximate monthly essential burn rate.
    period_days = max(1, int((x["date"].max() - x["date"].min()).days) + 1)
    monthly_essential = essential / period_days * 30
    emergency_months = (
        0.0 if monthly_essential <= 0 else max(0.0, current_balance) / monthly_essential
    )

    monthly_savings = income - expenses

    # Each component is independently bounded to 0-100. This keeps the final score
    # deterministic and auditable instead of delegating arithmetic to an LLM.
    daily_net = x.assign(
        signed=x["amount"].where(x["transaction_type"].eq("CREDIT"), -x["amount"])
    ).groupby("date")["signed"].sum()
    mean_abs_net = max(float(daily_net.abs().mean()), 1.0) if len(daily_net) else 1.0
    net_volatility = float(daily_net.std(ddof=0) or 0.0)

    recurring = x.loc[x["is_recurring"].eq(True) & x["transaction_type"].eq("DEBIT"), "amount"].sum()
    spending_std = float(x.loc[x["transaction_type"].eq("DEBIT"), "amount"].std(ddof=0) or 0.0)
    spending_mean = max(float(x.loc[x["transaction_type"].eq("DEBIT"), "amount"].mean()), 1.0)

    components = {
        "savings_rate": min(100.0, savings_rate * 100.0),
        "expense_income": max(0.0, 100.0 - min(100.0, expense_ratio * 100.0)),
        "debt_health": max(0.0, 100.0 - min(100.0, debt_ratio * 200.0)),
        "budget_discipline": max(0.0, min(100.0, float(budget_discipline_pct))),
        "emergency_fund": min(100.0, emergency_months / 6.0 * 100.0),
        "spending_consistency": max(
            0.0, 100.0 - min(100.0, spending_std / spending_mean * 20.0)
        ),
        "recurring_burden": max(
            0.0, 100.0 - min(100.0, float(recurring) / max(income, 1.0) * 100.0)
        ),
        "cashflow_stability": max(
            0.0,
            min(100.0, 100.0 - (net_volatility / mean_abs_net * 20.0)),
        ),
    }

    score = calculate_health_score(components)
    observed_months = max(1, int(x["date"].dt.to_period("M").nunique())) if not x.empty else 1
    monthly_income = income / observed_months
    return {
        "score": score,
        "components": {k: round(float(v), 2) for k, v in components.items()},
        "monthly_income": round(float(monthly_income), 2),
        "monthly_expenses": round(float(expenses / observed_months), 2),
        "savings_rate": round(float(savings_rate * 100), 2),
        "debt_ratio": round(float(debt_ratio * 100), 2),
        "emergency_months": round(float(emergency_months), 2),
        "monthly_savings": round(float(monthly_savings), 2),
        "monthly_essential_expenses": round(float(monthly_essential), 2),
    }
