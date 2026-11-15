from __future__ import annotations

from agents.base import run_agent


def reviewer_node(state: dict) -> dict:
    def _run(s):
        issues: list[str] = []
        required_by_agent = {
            "Spending": "analytics",
            "Budget": "budget",
            "Health": "health",
            "Anomaly": "anomalies",
            "Subscription": "subscriptions",
            "Forecast": "forecast",
            "Financial Twin": "financial_twin",
        }
        plan_agents = s.get("plan", {}).get("agents", [])
        for agent, key in required_by_agent.items():
            if agent in plan_agents and key not in s:
                issues.append(f"Missing output required by {agent}: {key}")

        unsupported: list[str] = []
        for rec in s.get("recommendations", []):
            if not rec.get("evidence"):
                unsupported.append(str(rec.get("text", "")))

        if unsupported:
            issues.append(f"{len(unsupported)} recommendation(s) lack evidence metadata.")

        # Deterministic repair: drop unsupported recommendations rather than
        # allowing the UI to present an ungrounded financial claim.
        grounded_recs = [r for r in s.get("recommendations", []) if r.get("evidence")]
        repaired = bool(unsupported)
        if repaired:
            s["recommendations"] = grounded_recs[:5]

        passed = not issues or (not [x for x in issues if "Missing output" in x] and not unsupported)
        status = "PASSED" if passed else "REJECTED"
        return {
            "reviewer": {
                "status": status,
                "issues": issues,
                "repaired": repaired,
                "grounded_recommendations": len(grounded_recs),
            },
            "recommendations": s.get("recommendations", grounded_recs[:5]),
        }

    return run_agent(state, "Reviewer", _run, ["validate_calculations", "validate_grounding", "repair_response"])
