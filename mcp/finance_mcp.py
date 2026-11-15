"""Finance MCP Backend — spending, budget, goal, and simulation tools.

Wraps existing deterministic finance tools and registers them into the
MCP registry so agents can call them by name through the registry broker.
All imports are lazy (inside _register) to handle missing optional deps gracefully.
"""
from __future__ import annotations

from mcp.registry import mcp_registry


def _register() -> None:
    from tools.transaction_tools import (
        calculate_budget_utilization,
        calculate_category_spending,
        calculate_savings_rate,
        get_transactions,
    )
    from tools.simulation_tools import simulate_car, simulate_emi, simulate_salary_change
    from tools.subscription_tools import detect_subscriptions
    from tools.forecasting_tools import forecast_cash_flow

    tools: dict = {
        # ── Spending ──────────────────────────────────────────────────────────
        "calculate_category_spending": {
            "fn": calculate_category_spending,
            "description": "Aggregate transaction amounts by category for a user.",
            "tags": ["spending", "analytics"],
        },
        "calculate_savings_rate": {
            "fn": calculate_savings_rate,
            "description": "Compute monthly savings rate from income and expenditure.",
            "tags": ["health", "savings"],
        },
        "calculate_budget_utilization": {
            "fn": calculate_budget_utilization,
            "description": "Compare actual spending versus budget targets per category.",
            "tags": ["budget", "analytics"],
        },
        # ── Simulations ───────────────────────────────────────────────────────
        "simulate_emi": {
            "fn": simulate_emi,
            "description": "Calculate EMI and affordability for a loan.",
            "tags": ["simulation", "loan"],
        },
        "simulate_car": {
            "fn": simulate_car,
            "description": "Full vehicle affordability simulation including EMI, down-payment impact.",
            "tags": ["simulation", "affordability"],
        },
        "simulate_salary_change": {
            "fn": simulate_salary_change,
            "description": "Model the financial impact of a salary increase or decrease.",
            "tags": ["simulation", "scenario"],
        },
        # ── Subscriptions ─────────────────────────────────────────────────────
        "detect_subscriptions": {
            "fn": detect_subscriptions,
            "description": "Identify recurring subscription charges from transactions.",
            "tags": ["subscriptions"],
        },
        # ── Transactions ──────────────────────────────────────────────────────
        "get_transactions": {
            "fn": get_transactions,
            "description": "Retrieve raw transaction records for a user.",
            "tags": ["data", "transactions"],
        },
        # ── Cash Flow ─────────────────────────────────────────────────────────
        "forecast_cash_flow": {
            "fn": forecast_cash_flow,
            "description": "30-day deterministic cash flow forecast from spending patterns.",
            "tags": ["forecast", "cash_flow"],
        },
    }

    # Optional goal tools (needs python-dateutil)
    try:
        from tools.goal_tools import evaluate_goal_feasibility, simulate_goal
        tools["simulate_goal"] = {
            "fn": simulate_goal,
            "description": "Simulate progress toward a savings/investment goal.",
            "tags": ["goals", "simulation"],
        }
        tools["evaluate_goal_feasibility"] = {
            "fn": evaluate_goal_feasibility,
            "description": "Evaluate if a financial goal is reachable given current savings rate.",
            "tags": ["goals"],
        }
    except Exception:
        pass

    mcp_registry.register_backend("finance_mcp", tools)


_register()
