import pandas as pd
from agents.base import run_agent
from analytics.financial_health import score_financial_health


def health_node(state: dict) -> dict:
    def _run(s):
        df = pd.DataFrame(s["transactions"])
        current_balance = float(s.get("current_balance", s.get("starting_balance", 32000)))
        budget_rows = s.get("budget", {}).get("rows", [])
        discipline = 100.0 - min(100.0, sum(1 for r in budget_rows if r["risk"] == "High") * 12.5)
        result = score_financial_health(df, current_balance, discipline)
        months = max(1, len(result.get("components", {})))
        # Convert period savings into monthly savings for goals and what-if tools.
        observed_months = pd.to_datetime(df["date"]).dt.to_period("M").nunique() if not df.empty else 1
        result["monthly_savings"] = round(float(result["monthly_savings"]) / max(1, int(observed_months)), 2)
        return {"health": result}
    return run_agent(state, "Financial Health Scoring Agent", _run, ["calculate_savings_rate", "calculate_debt_ratio", "calculate_emergency_fund", "calculate_financial_health_score"])
