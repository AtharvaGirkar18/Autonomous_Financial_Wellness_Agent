import pandas as pd
from tools.transaction_tools import simulate_emi, calculate_health_score, calculate_savings_rate
from tools.budget_tools import analyze_budget


def test_emi_zero_rate():
    assert round(simulate_emi(120000, 0, 12),2) == 10000.00


def test_health_score_range():
    score = calculate_health_score({k:80 for k in ["savings_rate","expense_income","debt_health","budget_discipline","emergency_fund","spending_consistency","recurring_burden","cashflow_stability"]})
    assert 0 <= score <= 100


def test_budget_risk():
    out = analyze_budget({"Food & Dining":7200},{"Food & Dining":8000})
    assert out["rows"][0]["risk"] == "High"


def test_savings_rate():
    df = pd.DataFrame({"transaction_type":["CREDIT","DEBIT"],"amount":[100,25]})
    assert calculate_savings_rate(df) == 0.75
