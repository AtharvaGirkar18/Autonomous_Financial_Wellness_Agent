from __future__ import annotations
from typing import Dict, Any


def analyze_budget(category_spend: Dict[str, float], budget: Dict[str, float]) -> Dict[str, Any]:
    rows = []
    for category, allocated in budget.items():
        spent = float(category_spend.get(category, 0))
        util = spent / allocated if allocated > 0 else 0
        risk = "High" if util >= 0.90 else "Medium" if util >= 0.75 else "Low"
        rows.append({
            "category": category,
            "allocated": round(allocated,2),
            "spent": round(spent,2),
            "remaining": round(max(0, allocated-spent),2),
            "utilization": round(util,4),
            "risk": risk,
        })
    return {"rows": rows}
