from __future__ import annotations

from typing import Dict

from tools.transaction_tools import simulate_emi


def simulate_car(
    current_monthly_savings: float,
    price: float,
    down_payment: float,
    annual_rate: float,
    tenure_months: int,
    current_balance: float,
    current_debt_ratio: float,
    monthly_income: float = 0.0,
) -> Dict[str, float]:
    loan = max(0.0, price - down_payment)
    emi = simulate_emi(loan, annual_rate, tenure_months)
    new_savings = current_monthly_savings - emi
    projected_12m_savings = new_savings * 12
    incremental_debt_ratio = emi / max(monthly_income, 1.0)
    debt_load_pct = (current_debt_ratio + incremental_debt_ratio) * 100
    post_purchase_buffer = current_balance - max(0.0, down_payment)
    return {
        "loan": round(loan, 2),
        "emi": round(emi, 2),
        "monthly_savings_after_emi": round(new_savings, 2),
        "projected_12m_savings_after_emi": round(projected_12m_savings, 2),
        "estimated_debt_burden_pct": round(debt_load_pct, 2),
        "post_down_payment_cash_buffer": round(post_purchase_buffer, 2),
        "current_balance": round(current_balance, 2),
    }


def simulate_salary_change(monthly_savings: float, old_salary: float, new_salary: float) -> dict:
    delta = new_salary - old_salary
    return {
        "salary_change": round(delta, 2),
        "new_savings_capacity": round(monthly_savings + delta, 2),
    }


def simulate_job_loss(
    essential_monthly_expenses: float,
    current_balance: float,
    months: int = 6,
) -> dict:
    runway = 0 if essential_monthly_expenses <= 0 else current_balance / essential_monthly_expenses
    lowest_balance = current_balance - essential_monthly_expenses * months
    return {
        "runway_months": round(runway, 2),
        "balance_after_scenario": round(lowest_balance, 2),
    }
