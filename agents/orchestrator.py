"""Orchestrator — Guard → Governance → Planner → Distributed Coordinator → Reviewer → Governance Post-Flight → Alerts.

Architecture:
  1. GuardrailBus  — unified input guardrail pipeline
  2. GovernanceAgent (pre-flight) — validates plan, enforces policy, can veto
  3. Planner (Qwen 2.5 7B via Ollama) — intent detection → agent selection
  4. DistributedCoordinator — parallel/sequential agent scheduling with event bus
  5. Reviewer — quality check on recommendations
  6. GovernanceAgent (post-flight) — compliance sign-off + audit report
  7. NotificationAgent — proactive alerts
  8. Persistence — state saved to SQLite via memory tools

MCP Registry is bootstrapped once at import time; all agents access tools
through `mcp_registry.invoke(tool_name, **kwargs)` instead of fixed imports.
"""
from __future__ import annotations

import time
from functools import lru_cache

from agents.advisor_agent import advisor_node
from agents.anomaly_agent import anomaly_node
from agents.budget_agent import budget_node
from agents.categorization_agent import categorization_node
from agents.coordinator import DistributedCoordinator
from agents.financial_twin_agent import financial_twin_node
from agents.forecast_agent import forecast_node
from agents.governance_agent import governance_node, governance_post_node, governance_policy
from agents.health_agent import health_node
from agents.ingestion_agent import ingestion_node
from agents.investment_readiness_agent import investment_readiness_node
from agents.notification_agent import notification_node
from agents.planner import plan_node
from agents.reviewer import reviewer_node
from agents.spending_agent import spending_node
from agents.subscription_agent import subscription_node
from guardrails.audit import audit_event
from guardrails.guardrail_bus import guardrail_bus
from models.database import init_db
from models.schemas import FinanceState
from tools.memory_tools import persist_financial_state, retrieve_memory

# ── Bootstrap MCP backends (import triggers auto-registration) ─────────────────
try:
    import mcp.finance_mcp    # noqa: F401  registers finance tools
    import mcp.analytics_mcp  # noqa: F401  registers ML/analytics tools
    import mcp.data_mcp       # noqa: F401  registers data/memory tools
except Exception as _mcp_err:
    audit_event("orchestrator", "WARN", "mcp_bootstrap", str(_mcp_err)[:200])


AGENT_MAP = {
    "Ingestion": ingestion_node,
    "Categorization": categorization_node,
    "Spending": spending_node,
    "Budget": budget_node,
    "Health": health_node,
    "Anomaly": anomaly_node,
    "Subscription": subscription_node,
    "Forecast": forecast_node,
    "Financial Twin": financial_twin_node,
    "Advisor": advisor_node,
    "Reviewer": reviewer_node,
    "Investment Readiness": investment_readiness_node,
}

LANGGRAPH_NAME_MAP = {
    "Ingestion": "ingestion",
    "Categorization": "categorization",
    "Spending": "spending",
    "Budget": "budget",
    "Health": "health",
    "Anomaly": "anomaly",
    "Subscription": "subscription",
    "Forecast": "forecast",
    "Financial Twin": "financial_twin",
    "Investment Readiness": "investment_readiness",
    "Advisor": "advisor",
    "Reviewer": "reviewer",
}

# Singleton distributed coordinator
_coordinator = DistributedCoordinator(max_workers=4)


@lru_cache(maxsize=1)
def _build_langgraph():
    try:
        from langgraph.graph import END, StateGraph
    except ImportError:
        return None

    graph = StateGraph(FinanceState)
    graph.add_node("planner", plan_node)
    graph.add_node("governance", governance_node)
    graph.add_node("dispatch", lambda state: state)
    graph.add_node("ingestion", ingestion_node)
    graph.add_node("categorization", categorization_node)
    graph.add_node("spending", spending_node)
    graph.add_node("budget", budget_node)
    graph.add_node("health", health_node)
    graph.add_node("anomaly", anomaly_node)
    graph.add_node("subscription", subscription_node)
    graph.add_node("forecast", forecast_node)
    graph.add_node("financial_twin", financial_twin_node)
    graph.add_node("investment_readiness", investment_readiness_node)
    graph.add_node("advisor", advisor_node)
    graph.add_node("reviewer", reviewer_node)
    graph.add_node("governance_post", governance_post_node)
    graph.add_node("alerts", notification_node)

    graph.set_entry_point("planner")
    graph.add_edge("planner", "governance")   # governance validates plan first
    graph.add_edge("governance", "dispatch")

    routes = list(LANGGRAPH_NAME_MAP.values())
    graph.add_conditional_edges(
        "dispatch",
        lambda state: state.get("plan", {}).get("agents", ["Reviewer"])[0],
        {name: node for name, node in LANGGRAPH_NAME_MAP.items()},
    )

    def next_stage(current_agent: str):
        def route(state):
            agents = state.get("plan", {}).get("agents", [])
            try:
                idx = agents.index(current_agent)
            except ValueError:
                return "reviewer"
            next_agent = agents[idx + 1] if idx + 1 < len(agents) else "Reviewer"
            return LANGGRAPH_NAME_MAP.get(next_agent, "reviewer")
        return route

    for agent_name, node_name in LANGGRAPH_NAME_MAP.items():
        if agent_name == "Reviewer":
            continue
        graph.add_conditional_edges(
            node_name,
            next_stage(agent_name),
            {node: node for node in routes},
        )

    graph.add_edge("reviewer", "governance_post")
    graph.add_edge("governance_post", "alerts")
    graph.add_edge("alerts", END)
    return graph.compile()


def _run_distributed_fallback(state: dict) -> dict:
    """Fallback coordinator path when LangGraph is unavailable.

    Uses the DistributedCoordinator for parallel agent execution.
    """
    # Step 1: Plan
    plan_result = plan_node(state)
    state.update(plan_result)

    # Step 2: Governance pre-flight
    gov_result = governance_node(state)
    state.update(gov_result)

    # Step 3: Get agent list (governance may have overridden the plan)
    planned_agents = state.get("plan", {}).get("agents", [])
    specialist_agents = [a for a in planned_agents if a not in {"Reviewer"}]

    # Step 4: Run specialists via DistributedCoordinator (parallel where safe)
    state = _coordinator.run(AGENT_MAP, specialist_agents, state)

    # Step 5: Reviewer
    state.update(reviewer_node(state))

    # Step 6: Governance post-flight
    state.update(governance_post_node(state))

    # Step 7: Coordinator event summary (for Agent Trace tab)
    state["coordinator_events"] = _coordinator.event_summary()

    # Step 8: Alerts
    state.update(notification_node(state))

    return state


def run_workflow(state: dict) -> dict:
    """GuardrailBus → Governance → Planner → Coordinator → Reviewer → Governance → Alerts → Persist."""
    init_db()
    state = dict(state)
    state.setdefault("trace", [])

    # ── 1. GuardrailBus input check ───────────────────────────────────────────
    request = str(state.get("request", "") or "")
    bus_result = guardrail_bus.check_input(request)
    audit_event("input", bus_result.decision, bus_result.category, request, severity=bus_result.severity)
    state["guardrail"] = bus_result.as_dict()

    if not bus_result.allowed:
        state["blocked"] = True
        state["final_answer"] = bus_result.user_message
        state["ai_answer"] = {
            "text": bus_result.user_message,
            "source": "guardrail_bus",
            "category": bus_result.category,
        }
        state["recommendations"] = [{
            "text": bus_result.user_message,
            "evidence": {
                "source": "GuardrailBus",
                "layers_failed": bus_result.layers_failed,
                "decision": bus_result.decision,
                "category": bus_result.category,
            },
        }]
        state["reviewer"] = {"status": "PASSED", "grounded_recommendations": 1, "issues": []}
        state["trace"] = [{
            "agent": "GuardrailBus",
            "status": "BLOCKED" if bus_result.decision == "BLOCK" else "REDIRECTED",
            "start_time": time.strftime("%H:%M:%S"),
            "execution_time_ms": 0.0,
            "input_summary": f"category={bus_result.category} severity={bus_result.severity}",
            "output_summary": bus_result.user_message,
            "tool_calls": bus_result.layers_failed,
        }]
        return state

    state["blocked"] = False
    state["request"] = bus_result.sanitized_text or request

    state.setdefault("memory", retrieve_memory(str(state.get("user_id", ""))))
    state.setdefault("profile", {})

    # ── 2. Try LangGraph (full orchestration graph) ───────────────────────────
    graph = _build_langgraph()
    if graph is not None:
        try:
            result = graph.invoke(state)
        except Exception as exc:
            audit_event("orchestrator", "LANGGRAPH_FALLBACK", "error", str(exc)[:200])
            result = _run_distributed_fallback(state)
    else:
        result = _run_distributed_fallback(state)

    # ── 3. Assemble final answer ──────────────────────────────────────────────
    recommendations = result.get("recommendations", [])
    if recommendations:
        result["final_answer"] = "\n\n".join(r["text"] for r in recommendations[:5])
    result["last_run_status"] = "REVIEWED"

    # ── 4. Output guardrail bus check ─────────────────────────────────────────
    final_text = result.get("final_answer", "")
    if final_text:
        out_check = guardrail_bus.check_output(final_text)
        if not out_check.allowed:
            audit_event("guardrail_bus", "OUTPUT_BLOCKED_FINAL", "final_answer", str(out_check.metadata))
            # Fall back to deterministic recommendations only
            result["final_answer"] = "\n\n".join(
                r["text"] for r in recommendations[1:6]  # skip the LLM rec, use deterministic
            )
        else:
            result["final_answer"] = out_check.sanitized_text

    # ── 5. Persist ────────────────────────────────────────────────────────────
    try:
        persist_financial_state(result)
    except Exception as exc:
        result["persistence_warning"] = str(exc)

    return result
