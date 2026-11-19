from __future__ import annotations
from datetime import datetime, timezone
from agents.base import run_agent


def notification_node(state: dict) -> dict:
    def _run(s):
        alerts = []
        rows = s.get("budget", {}).get("rows", [])
        for r in rows:
            if r["risk"] == "High":
                alerts.append({
                    "severity": "HIGH",
                    "title": f"{r['category']} budget at high risk",
                    "explanation": f"{r['category']} is at {r['utilization']*100:.0f}% of its allocated budget.",
                    "recommended_action": f"Review remaining {r['category']} spending.",
                    "timestamp": datetime.now(timezone.utc).replace(tzinfo=None).isoformat(),
                })
        for a in s.get("anomalies", [])[:5]:
            alerts.append({
                "severity": a["severity"],
                "title": "Unusual transaction detected",
                "explanation": f"{a['merchant']} for ₹{a['amount']:,.2f} was flagged by anomaly rules/model.",
                "recommended_action": "Verify the transaction and mark it as expected if legitimate.",
                "timestamp": datetime.now(timezone.utc).replace(tzinfo=None).isoformat(),
            })
        if s.get("duplicate_candidates"):
            alerts.append({
                "severity": "MEDIUM",
                "title": "Duplicate charge candidates",
                "explanation": f"{len(s['duplicate_candidates'])} transactions share the same user/date/merchant/amount pattern.",
                "recommended_action": "Verify the affected charges before treating them as final expenses.",
                "timestamp": datetime.now(timezone.utc).replace(tzinfo=None).isoformat(),
            })
        if s.get("subscriptions"):
            monthly = sum(float(x.get("monthly_cost", 0)) for x in s["subscriptions"])
            alerts.append({
                "severity": "MEDIUM",
                "title": "Subscription review opportunity",
                "explanation": f"Recurring subscriptions total approximately ₹{monthly:,.0f} per month. Check your usage reviews on the Subscriptions tab.",
                "recommended_action": "Review recurring services and cancel only those you no longer need.",
                "timestamp": datetime.now(timezone.utc).replace(tzinfo=None).isoformat(),
            })
        if float(s.get("health", {}).get("savings_rate", 100)) < 10:
            alerts.append({
                "severity": "MEDIUM",
                "title": "Savings rate below target",
                "explanation": f"Observed savings rate is approximately {s['health'].get('savings_rate', 0):.1f}%.",
                "recommended_action": "Review discretionary categories and protect planned savings contributions.",
                "timestamp": datetime.now(timezone.utc).replace(tzinfo=None).isoformat(),
            })
        for goal in s.get("goals", []):
            if goal.get("status") == "At Risk":
                alerts.append({
                    "severity": "HIGH",
                    "title": f"Goal at risk: {goal.get('name', 'Goal')}",
                    "explanation": f"Required monthly contribution is approximately ₹{goal.get('required_monthly_contribution', 0):,.0f}, exceeding current modeled capacity by ₹{goal.get('gap', 0):,.0f}.",
                    "recommended_action": "Increase the contribution, extend the timeline, or reduce the goal amount.",
                    "timestamp": datetime.now(timezone.utc).replace(tzinfo=None).isoformat(),
                })
        if s.get("forecast", {}).get("lowest_projected_balance", 10**9) < 5000:
            alerts.append({
                "severity": "HIGH",
                "title": "Cash-flow risk",
                "explanation": f"Projected balance may reach ₹{s['forecast']['lowest_projected_balance']:,.0f} on {s['forecast']['lowest_projected_date']}.",
                "recommended_action": "Reduce discretionary spending or increase near-term cash buffer.",
                "timestamp": datetime.now(timezone.utc).replace(tzinfo=None).isoformat(),
            })
        return {"alerts": alerts}
    return run_agent(state, "Alert & Notification Agent", _run, ["generate_alert"])
