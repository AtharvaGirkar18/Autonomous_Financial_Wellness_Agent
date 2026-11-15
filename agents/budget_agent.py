import pandas as pd
from agents.base import run_agent
from tools.budget_tools import analyze_budget


DEFAULT_BUDGET = {
    "Food & Dining": 8000,
    "Groceries": 6500,
    "Transportation": 3500,
    "Fuel": 3000,
    "Shopping": 4500,
    "Healthcare": 2500,
    "Education": 2500,
    "Entertainment": 1800,
    "Bills & Utilities": 5000,
    "Rent": 18000,
    "Travel": 3000,
    "Insurance": 3000,
    "EMI / Debt": 10000,
    "Investments": 6000,
    "Subscriptions": 1500,
    "Other": 2500,
}


def budget_node(state: dict) -> dict:
    def _run(s):
        df = pd.DataFrame(s["transactions"])
        if "category" not in df.columns:
            df["category"] = df.get("predicted_category", "Other")
        df["month"] = pd.to_datetime(df["date"]).dt.to_period("M").astype(str)
        latest_month = str(df["month"].max())
        latest = df[(df["month"] == latest_month) & df["transaction_type"].eq("DEBIT")]
        category_totals = latest.groupby("category")["amount"].sum().to_dict()
        budgets = {**DEFAULT_BUDGET, **s.get("budget_overrides", {})}
        result = analyze_budget(category_totals, budgets)
        return {"budget": {"month": latest_month, "categories": budgets, **result}}
    return run_agent(state, "Budget Management Agent", _run, ["calculate_budget_utilization"])
