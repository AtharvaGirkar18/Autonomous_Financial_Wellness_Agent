from __future__ import annotations
from datetime import date
from dateutil.relativedelta import relativedelta


def goal_status(target_amount: float, current_amount: float, target_date: date, monthly_capacity: float) -> dict:
    remaining = max(0.0, target_amount - current_amount)
    months = max(1, (target_date.year - date.today().year) * 12 + target_date.month - date.today().month)
    required = remaining / months
    gap = required - monthly_capacity
    return {
        "remaining": round(remaining,2),
        "months_remaining": months,
        "required_monthly_contribution": round(required,2),
        "current_capacity": round(monthly_capacity,2),
        "gap": round(max(0, gap),2),
        "status": "On Track" if gap <= 0 else "At Risk",
    }
