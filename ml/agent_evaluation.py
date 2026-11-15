from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from agents.orchestrator import run_workflow
from agents.reviewer import reviewer_node

CASES = [
    {
        "name": "complete_analysis",
        "request": "Run my complete financial wellness analysis and proactive monitoring cycle.",
        "expected_agents": {"Ingestion", "Categorization", "Spending", "Budget", "Anomaly", "Subscription", "Forecast", "Health", "Financial Twin", "Advisor", "Reviewer"},
        "expected_keywords": ["upward", "transaction"],
        "expected_tools": {"calculate_monthly_spending", "calculate_category_spending", "detect_anomalies", "detect_subscriptions", "forecast_cash_flow"},
    },
    {
        "name": "car_affordability",
        "request": "Can I afford a ₹12 lakh car?",
        "expected_agents": {"Health", "Forecast", "Financial Twin", "Advisor", "Reviewer"},
        "expected_keywords": ["EMI", "car", "savings"],
        "expected_tools": {"simulate_emi", "forecast_cash_flow", "retrieve_financial_state"},
    },
    {
        "name": "save_more",
        "request": "How can I save ₹5,000 more per month?",
        "expected_agents": {"Spending", "Budget", "Health", "Financial Twin", "Advisor", "Reviewer"},
        "expected_keywords": ["₹5,000", "discretionary"],
        "expected_tools": {"calculate_category_spending", "calculate_budget_utilization", "calculate_savings_rate"},
    },
    {
        "name": "anomaly_question",
        "request": "Show me suspicious or unusual transactions.",
        "expected_agents": {"Anomaly", "Advisor", "Reviewer"},
        "expected_keywords": ["transaction", "flagged"],
        "expected_tools": {"detect_anomalies", "retrieve_financial_state"},
    },
    {
        "name": "forecast_question",
        "request": "What could my balance look like next month?",
        "expected_agents": {"Forecast", "Health", "Financial Twin", "Advisor", "Reviewer"},
        "expected_keywords": ["forecast", "balance"],
        "expected_tools": {"forecast_cash_flow", "retrieve_financial_state"},
    },
    {
        "name": "salary_change",
        "request": "What if my salary increases by ₹15,000?",
        "expected_agents": {"Health", "Financial Twin", "Advisor", "Reviewer"},
        "expected_keywords": ["salary", "savings"],
        "expected_tools": {"apply_scenario", "retrieve_financial_state"},
    },
    {
        "name": "investment_readiness",
        "request": "Am I investment ready?",
        "expected_agents": {"Health", "Investment Readiness", "Financial Twin", "Advisor", "Reviewer"},
        "expected_keywords": ["investment readiness"],
        "expected_tools": {"retrieve_financial_state", "calculate_financial_health_score"},
    },
]


def _state_for(df: pd.DataFrame, request: str) -> dict:
    profile = {"monthly_income": 75_000.0, "starting_balance": 32_000.0, "name": "Demo User"}
    goals = [{
        "name": "Car", "target_amount": 1_200_000.0, "target_date": "2028-06-01",
        "current_amount": 120_000.0, "monthly_contribution": 20_000.0,
    }]
    return {
        "user_id": "USER001",
        "profile": profile,
        "request": request,
        "transactions": df[df.user_id.eq("USER001")].to_dict(orient="records"),
        "current_balance": 32_000.0,
        "starting_balance": 32_000.0,
        "goals": goals,
        "trace": [],
    }


def _flatten_tool_calls(trace: list[dict]) -> list[str]:
    return [tool for item in trace for tool in item.get("tool_calls", [])]


def _run_case(df: pd.DataFrame, case: dict) -> dict:
    state = run_workflow(_state_for(df, case["request"]))
    planned = list(state.get("plan", {}).get("agents", []))
    actual = set(planned)
    expected = case["expected_agents"]
    planning_success = expected.issubset(actual) and bool(planned) and planned[-1] == "Reviewer"
    tools = set(_flatten_tool_calls(state.get("trace", [])))
    expected_tools = set(case.get("expected_tools", set()))
    tool_selection_success = expected_tools.issubset(tools) and all(
        item.get("status") == "SUCCESS" for item in state.get("trace", [])
    )
    recs = state.get("recommendations", [])
    grounded = sum(bool(r.get("evidence")) for r in recs)
    grounding_rate = grounded / len(recs) if recs else 1.0
    answer = str(state.get("final_answer", "")).lower()
    keyword_hit = sum(k.lower() in answer for k in case["expected_keywords"])
    correct = keyword_hit == len(case["expected_keywords"])
    context_relevant = bool(recs) and grounded == len(recs)
    hallucination = grounded < len(recs)
    return {
        "case": case["name"],
        "planned_agents": planned,
        "planning_success": bool(planning_success),
        "tool_selection_success": bool(tool_selection_success),
        "recommendation_grounded": bool(context_relevant),
        "grounding_rate": round(grounding_rate, 4),
        "correct": bool(correct),
        "context_relevant": bool(context_relevant),
        "hallucination": bool(hallucination),
        "reviewer_status": state.get("reviewer", {}).get("status"),
        "recommendation_count": len(recs),
    }


def _reviewer_rejection_cases() -> dict:
    cases = []

    missing_output = {
        "request": "test missing output",
        "plan": {"agents": ["Health", "Advisor", "Reviewer"]},
        # health intentionally omitted so the reviewer has a genuine missing required output.
        "trace": [],
        "recommendations": [],
    }
    result = reviewer_node(missing_output)
    cases.append({"name": "missing_required_output", "rejected": result["reviewer"]["status"] == "REJECTED"})

    unsupported_recommendation = {
        "request": "test unsupported recommendation",
        "plan": {"agents": ["Advisor", "Reviewer"]},
        "recommendations": [{"text": "Buy this product immediately", "evidence": {}}],
        "trace": [],
    }
    result = reviewer_node(unsupported_recommendation)
    cases.append({"name": "unsupported_recommendation", "rejected": result["reviewer"]["status"] == "REJECTED" and result["reviewer"]["repaired"]})

    rejected = sum(x["rejected"] for x in cases)
    return {
        "seeded_invalid_cases": len(cases),
        "rejections": rejected,
        "rejection_rate": round(rejected / len(cases), 4),
        "cases": cases,
    }


def run_agent_evaluation(csv_path: str, output_dir: str | None = "artifacts/evaluation") -> dict:
    df = pd.read_csv(csv_path)
    case_results = [_run_case(df, case) for case in CASES]
    reviewer_eval = _reviewer_rejection_cases()

    result = {
        "cases": case_results,
        "summary": {
            "planning_success_rate": round(sum(x["planning_success"] for x in case_results) / len(case_results), 4),
            "tool_selection_success_rate": round(sum(x["tool_selection_success"] for x in case_results) / len(case_results), 4),
            "grounded_recommendation_rate": round(sum(x["grounding_rate"] for x in case_results) / len(case_results), 4),
            "hallucination_rate_proxy": round(sum(x["hallucination"] for x in case_results) / len(case_results), 4),
            "correct_response_rate": round(sum(x["correct"] for x in case_results) / len(case_results), 4),
            "context_relevant_rate": round(sum(x["context_relevant"] for x in case_results) / len(case_results), 4),
        },
        "reviewer_evaluation": reviewer_eval,
        "interpretation": "Hallucination rate is a deterministic-mode proxy equal to the share of recommendations lacking evidence metadata; LLM semantic hallucination requires an LLM-backed evaluation set.",
    }

    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "agent_evaluation.json").write_text(json.dumps(result, indent=2))
        pd.DataFrame(case_results).to_csv(out / "agent_case_evaluation.csv", index=False)
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/synthetic_transactions.csv")
    parser.add_argument("--output-dir", default="artifacts/evaluation")
    args = parser.parse_args()
    print(json.dumps(run_agent_evaluation(args.csv, args.output_dir), indent=2))
