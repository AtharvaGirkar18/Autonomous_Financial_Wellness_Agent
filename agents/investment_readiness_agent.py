from __future__ import annotations

from agents.base import run_agent


def investment_readiness_node(state: dict) -> dict:
    def _run(s):
        h = s.get("health", {})
        score = float(h.get("score", 0) or 0)
        emergency_months = float(h.get("emergency_months", 0) or 0)
        debt_ratio = float(h.get("debt_ratio", 0) or 0)
        if emergency_months < 3 or debt_ratio > 20:
            level = "Low"
        elif score >= 75:
            level = "High"
        else:
            level = "Moderate"
        return {
            "investment_readiness": {
                "level": level,
                "reason": (
                    "Readiness is based on emergency-fund coverage, debt burden, "
                    "savings consistency and cash-flow stability."
                ),
                "disclaimer": "No individualized securities recommendation is provided.",
            }
        }

    return run_agent(
        state,
        "Investment Readiness",
        _run,
        ["calculate_financial_health_score", "calculate_emergency_fund", "calculate_debt_ratio"],
    )
