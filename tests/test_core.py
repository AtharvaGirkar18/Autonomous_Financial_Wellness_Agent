from pathlib import Path

import pandas as pd

from agents.planner import _fallback_plan
from analytics.cashflow import forecast_cash_flow
from analytics.financial_health import score_financial_health
from analytics.subscriptions import detect_subscriptions
from ml.anomaly_detection import detect_anomalies
from tools.pdf_tools import extract_transactions_from_pdf
from tools.goal_tools import goal_status
from tools.simulation_tools import simulate_car
from tools.memory_tools import retrieve_memory, persist_financial_state


def test_pdf_ingestion():
    pdf = Path("data/sample_pdfs/HDFC_statement.pdf").read_bytes()
    df = extract_transactions_from_pdf(pdf, "TEST")
    assert len(df) == 28
    assert {"transaction_id", "date", "amount", "transaction_type"}.issubset(df.columns)


def test_subscription_detector_only_subscriptions():
    df = pd.DataFrame([
        {"transaction_type": "DEBIT", "date": "2026-01-12", "merchant": "NETFLIX", "description": "NETFLIX", "amount": 499, "category": "Subscriptions", "is_subscription": True},
        {"transaction_type": "DEBIT", "date": "2026-02-12", "merchant": "NETFLIX", "description": "NETFLIX", "amount": 499, "category": "Subscriptions", "is_subscription": True},
        {"transaction_type": "DEBIT", "date": "2026-03-12", "merchant": "NETFLIX", "description": "NETFLIX", "amount": 499, "category": "Subscriptions", "is_subscription": True},
        {"transaction_type": "DEBIT", "date": "2026-01-02", "merchant": "RENT", "description": "MONTHLY RENT", "amount": 18000, "category": "Rent", "is_subscription": False},
        {"transaction_type": "DEBIT", "date": "2026-02-02", "merchant": "RENT", "description": "MONTHLY RENT", "amount": 18000, "category": "Rent", "is_subscription": False},
        {"transaction_type": "DEBIT", "date": "2026-03-02", "merchant": "RENT", "description": "MONTHLY RENT", "amount": 18000, "category": "Rent", "is_subscription": False},
    ])
    out = detect_subscriptions(df)
    assert [x["merchant"] for x in out] == ["NETFLIX"]


def test_anomaly_detector_finds_ground_truth():
    df = pd.read_csv("data/synthetic_transactions.csv")
    out = detect_anomalies(df)
    assert out.loc[df["is_anomaly"], "is_detected_anomaly"].all()


def test_forecast_and_goal_and_car_tools():
    df = pd.read_csv("data/synthetic_transactions.csv").query("user_id == 'USER001'")
    forecast = forecast_cash_flow(df, 32000, 30)
    assert len(forecast["forecast"]) == 30
    assert "lowest_projected_balance" in forecast

    goal = goal_status(1200000, 120000, pd.Timestamp("2028-06-01").date(), 10000)
    assert goal["status"] == "At Risk"

    car = simulate_car(10000, 1200000, 240000, 9, 60, 32000, 0.13, 75000)
    assert car["emi"] > 0
    assert car["estimated_debt_burden_pct"] > 13

def test_planner_fallback_is_agentic():
    intent, plan = _fallback_plan("Can I afford a ₹12 lakh car?")
    assert intent == "affordability"
    assert plan["agents"] == ["Health", "Forecast", "Financial Twin", "Advisor", "Reviewer"]


def test_investment_readiness_plan():
    intent, plan = _fallback_plan("Am I investment ready?")
    assert intent == "investment_readiness"
    assert "Investment Readiness" in plan["agents"]
    assert plan["agents"][-1] == "Reviewer"


def test_memory_persists_recommendation():
    persist_financial_state({
        "user_id": "MEMORY_TEST",
        "profile": {"name": "Memory Test", "monthly_income": 50000, "starting_balance": 10000},
        "starting_balance": 10000,
        "transactions": [],
        "recommendations": [{"text": "Review discretionary spending.", "evidence": {"test": True}}],
        "alerts": [],
        "trace": [],
    })
    memory = retrieve_memory("MEMORY_TEST", limit=3)
    assert any(item["text"] == "Review discretionary spending." for item in memory)
