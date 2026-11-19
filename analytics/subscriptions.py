from __future__ import annotations

import pandas as pd


KNOWN_SUBSCRIPTIONS = {
    "NETFLIX": 499,
    "SPOTIFY": 119,
    "AMAZON PRIME": 299,
    "YOUTUBE PREMIUM": 129,
    "CHATGPT": 1800,
    "ADOBE": 1675,
    "GOOGLE ONE": 130,
    "HOTSTAR": 299,
}


def detect_subscriptions(df: pd.DataFrame) -> list[dict]:
    """Detect likely subscriptions without mislabeling rent/EMIs as subscriptions."""
    x = df[df["transaction_type"].eq("DEBIT")].copy()
    if x.empty:
        return []
    x["date"] = pd.to_datetime(x["date"], errors="coerce")
    x["merchant_norm"] = x["merchant"].astype(str).str.upper().str.strip()
    x["description_norm"] = x["description"].astype(str).str.upper()
    out: list[dict] = []

    for merchant, group in x.groupby("merchant_norm"):
        g = group.sort_values("date")
        if len(g) < 3:
            continue
        known = merchant in KNOWN_SUBSCRIPTIONS
        category_subscription = bool((g["category"] == "Subscriptions").any())
        recurring = bool(g.get("is_subscription", False).astype(bool).any()) if "is_subscription" in g.columns else False
        if not (known or category_subscription or recurring):
            continue

        gaps = g["date"].diff().dt.days.dropna()
        monthly_share = float(gaps.between(25, 35).mean()) if len(gaps) else 0.0
        yearly_share = float(gaps.between(350, 380).mean()) if len(gaps) else 0.0
        detected_recurring = monthly_share >= 0.5 or yearly_share >= 0.5
        if not detected_recurring:
            continue

        monthly_charge = float(g["amount"].median())
        frequency = "monthly" if monthly_share >= yearly_share else "yearly"
        monthly_cost = monthly_charge if frequency == "monthly" else monthly_charge / 12.0
        evidence = (
            "Recurring subscription charge detected from the transaction pattern. "
            "Actual product usage data is unavailable."
        )
        out.append({
            "merchant": merchant,
            "monthly_cost": round(monthly_cost, 2),
            "yearly_cost": round(monthly_cost * 12, 2),
            "billing_frequency": frequency,
            "observed_months": int(g["date"].dt.to_period("M").nunique()),
            "usage_evidence": "unavailable",
            "potentially_unused": False,
            "evidence": evidence,
        })
    return sorted(out, key=lambda row: row["monthly_cost"], reverse=True)
