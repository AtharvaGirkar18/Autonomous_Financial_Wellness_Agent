"""Governance Agent — Agents of Governance.

The GovernanceAgent acts as the system supervisor and policy authority:

  1. **Policy Enforcement**: Validates agent plans against governance rules
     (allowed agents, tool access control, output format compliance).

  2. **Agent Lifecycle Control**: Can disable/enable agents and MCP tools
     at runtime based on risk signals (anomalies, repeated errors, etc.).

  3. **Audit Authority**: Generates governance reports summarizing security
     events, blocked queries, rate-limit hits, and agent error rates.

  4. **Compliance Checks**: Ensures the final output meets regulatory-grade
     requirements (no securities picks, no ungrounded numbers, PII-clean).

Architecture:
  - Runs BEFORE the planner (pre-flight policy check).
  - Runs AFTER the reviewer (post-flight compliance sign-off).
  - Has veto power: can abort a workflow and return a safe fallback.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from agents.base import run_agent
from guardrails.audit import audit_event, recent_events


# ── Governance Policy ──────────────────────────────────────────────────────────
ALLOWED_AGENT_TRANSITIONS: dict[str, list[str]] = {
    # Each agent can only be followed by agents in its allowed successor set.
    "Planner": ["Ingestion", "Categorization", "Spending", "Budget", "Health",
                 "Anomaly", "Subscription", "Forecast", "Financial Twin",
                 "Advisor", "Reviewer", "Investment Readiness"],
    "Ingestion": ["Categorization", "Spending", "Advisor", "Reviewer"],
    "Categorization": ["Spending", "Budget", "Anomaly", "Advisor", "Reviewer"],
    "Spending": ["Budget", "Health", "Advisor", "Reviewer"],
    "Budget": ["Health", "Financial Twin", "Advisor", "Reviewer"],
    "Health": ["Financial Twin", "Investment Readiness", "Advisor", "Reviewer"],
    "Anomaly": ["Advisor", "Reviewer"],
    "Subscription": ["Advisor", "Reviewer"],
    "Forecast": ["Health", "Financial Twin", "Advisor", "Reviewer"],
    "Financial Twin": ["Investment Readiness", "Advisor", "Reviewer"],
    "Investment Readiness": ["Advisor", "Reviewer"],
    "Advisor": ["Reviewer"],
    "Reviewer": [],  # Terminal node
}

BLOCKED_TOOL_PATTERNS = [
    "sql_inject", "exec_", "system_call", "file_write",
]

# Maximum fraction of recent events that can be errors before governance intervenes
ERROR_RATE_THRESHOLD = 0.4
MAX_RECENT_EVENTS = 20


@dataclass
class GovernanceVerdict:
    approved: bool = True
    violations: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    disabled_tools: list[str] = field(default_factory=list)
    disabled_agents: list[str] = field(default_factory=list)
    governance_message: str = ""

    def as_dict(self) -> dict:
        return {
            "approved": self.approved,
            "violations": self.violations,
            "warnings": self.warnings,
            "disabled_tools": self.disabled_tools,
            "disabled_agents": self.disabled_agents,
            "governance_message": self.governance_message,
        }


class GovernancePolicy:
    """Stateless policy evaluator."""

    def validate_plan(self, plan: dict) -> GovernanceVerdict:
        """Pre-flight: validate the agent execution plan."""
        verdict = GovernanceVerdict()
        agents = plan.get("agents", [])

        # 1. Validate agent names and membership. The coordinator owns
        # dependency ordering and topologically schedules the selected agents;
        # the planner's list is a selection, not a serial execution chain.
        known_agents = set(ALLOWED_AGENT_TRANSITIONS) - {"Planner"}
        for agent in agents:
            if agent not in known_agents:
                verdict.violations.append(f"Unknown agent '{agent}' in plan.")
                verdict.approved = False

        # 2. Reviewer must always be last
        if agents and agents[-1] != "Reviewer":
            verdict.violations.append("Reviewer must be the final agent in every plan.")
            verdict.approved = False

        # 3. Tool access control: no dangerous tool patterns
        for tool in plan.get("tools", []):
            if any(pat in tool.lower() for pat in BLOCKED_TOOL_PATTERNS):
                verdict.violations.append(f"Tool '{tool}' is blocked by governance policy.")
                verdict.approved = False

        # 4. Health check: recent error rate
        events = recent_events(limit=MAX_RECENT_EVENTS)
        error_count = sum(1 for e in events if e.get("level") in ("ERROR", "WARN"))
        if events and error_count / len(events) > ERROR_RATE_THRESHOLD:
            verdict.warnings.append(
                f"High error rate detected ({error_count}/{len(events)} recent events). "
                "Consider investigating before proceeding."
            )

        if verdict.approved:
            verdict.governance_message = (
                f"✅ Plan approved by Governance Agent. "
                f"Agents: {' → '.join(agents)}"
            )
        else:
            verdict.governance_message = (
                f"❌ Plan REJECTED by Governance Agent. Violations: "
                + "; ".join(verdict.violations)
            )

        return verdict

    def post_flight_check(self, state: dict) -> GovernanceVerdict:
        """Post-flight: sign off on the completed workflow."""
        verdict = GovernanceVerdict()
        recommendations = state.get("recommendations", [])

        # Check output guardrail compliance
        from guardrails.output_guard import check_output
        for rec in recommendations[:3]:
            text = rec.get("text", "")
            check = check_output(text)
            if not check.ok:
                verdict.warnings.append(
                    f"Recommendation failed output check: {check.violations}"
                )

        # Check that blocked requests were not processed
        if state.get("blocked") and state.get("plan"):
            verdict.violations.append("A blocked request reached the planner — policy breach.")
            verdict.approved = False

        verdict.governance_message = (
            "✅ Post-flight governance check passed."
            if verdict.approved
            else f"⚠️ Post-flight governance warnings: {verdict.warnings}"
        )
        return verdict

    def generate_report(self, n: int = 50) -> dict:
        """Generate a governance audit report from recent audit events."""
        events = recent_events(limit=n)
        by_level: dict[str, int] = {}
        by_action: dict[str, int] = {}
        for e in events:
            lvl = e.get("level", "INFO")
            act = e.get("action", "UNKNOWN")
            by_level[lvl] = by_level.get(lvl, 0) + 1
            by_action[act] = by_action.get(act, 0) + 1

        return {
            "total_events": len(events),
            "by_level": by_level,
            "by_action": by_action,
            "recent_errors": [e for e in events if e.get("level") == "ERROR"][:5],
            "recent_blocks": [e for e in events if e.get("action") == "BLOCK"][:5],
            "mcp_calls": by_action.get("TOOL_OK", 0),
            "llm_calls": by_action.get("OK", 0),
            "report_time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }


# ── Singleton policy instance ──────────────────────────────────────────────────
governance_policy = GovernancePolicy()


# ── Agent node (LangGraph-compatible) ─────────────────────────────────────────
def governance_node(state: dict) -> dict:
    """Pre-flight governance check node — runs right after the planner."""
    def _run(s: dict) -> dict:
        plan = s.get("plan", {})
        verdict = governance_policy.validate_plan(plan)
        audit_event(
            "governance",
            "APPROVED" if verdict.approved else "REJECTED",
            "plan_validation",
            verdict.governance_message,
        )

        if not verdict.approved:
            # Governance veto: replace plan with a safe minimal plan
            safe_plan = {
                "intent": plan.get("intent", "general"),
                "agents": ["Health", "Advisor", "Reviewer"],
                "tools": ["calculate_financial_health_score"],
                "order": ["Health", "Advisor", "Reviewer"],
                "reason": f"Plan overridden by Governance Agent: {'; '.join(verdict.violations)}",
            }
            return {
                "plan": safe_plan,
                "governance": verdict.as_dict(),
                "governance_override": True,
            }

        return {"governance": verdict.as_dict(), "governance_override": False}

    return run_agent(
        state,
        "Governance Agent",
        _run,
        ["validate_plan", "check_agent_transitions", "audit_tool_access", "check_error_rate"],
    )


def governance_post_node(state: dict) -> dict:
    """Post-flight governance sign-off — runs after the reviewer."""
    def _run(s: dict) -> dict:
        verdict = governance_policy.post_flight_check(s)
        audit_event(
            "governance",
            "SIGNED_OFF" if verdict.approved else "POST_FLIGHT_WARNING",
            "post_flight",
            verdict.governance_message,
        )
        report = governance_policy.generate_report(n=30)
        return {
            "governance_post": verdict.as_dict(),
            "governance_report": report,
        }

    return run_agent(
        state,
        "Governance Agent (Post-Flight)",
        _run,
        ["check_output_compliance", "generate_audit_report", "sign_off_workflow"],
    )
