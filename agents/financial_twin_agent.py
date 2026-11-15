from __future__ import annotations

from datetime import date

from agents.base import run_agent
from tools.goal_tools import goal_status


def financial_twin_node(state: dict) -> dict:
    def _run(s):
        health = s.get("health", {})
        analytics = s.get("analytics", {})
        monthly_capacity = float(health.get("monthly_savings", 0.0))
        goals = []
        for goal in s.get("goals", []):
            try:
                target_date = date.fromisoformat(str(goal["target_date"]))
                goals.append({
                    **goal,
                    **goal_status(
                        float(goal.get("target_amount", 0)),
                        float(goal.get("current_amount", 0)),
                        target_date,
                        monthly_capacity,
                    ),
                })
            except (KeyError, TypeError, ValueError):
                continue

        twin = {
            "current_balance": float(s.get("current_balance", 32000)),
            "income": float(analytics.get("income_total", 0)),
            "expenses": float(analytics.get("expense_total", 0)),
            "savings": float(analytics.get("savings_total", 0)),
            "monthly_savings_capacity": monthly_capacity,
            "debt_ratio": float(health.get("debt_ratio", 0)),
            "emergency_months": float(health.get("emergency_months", 0)),
            "health_score": float(health.get("score", 0)),
            "goals": goals,
        }
        return {"financial_twin": twin, "goals": goals}

    return run_agent(
        state,
        "Financial Twin Agent",
        _run,
        ["retrieve_financial_state", "simulate_goal", "apply_scenario"],
    )
