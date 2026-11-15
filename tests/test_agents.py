import pandas as pd
from agents.orchestrator import run_workflow


def test_end_to_end():
    df = pd.DataFrame([
        {"transaction_id":"1","user_id":"U1","date":"2026-01-01","description":"SALARY","merchant":"SALARY","amount":50000,"transaction_type":"CREDIT","account_type":"BANK","payment_mode":"NEFT","category":"Salary / Income","subcategory":"Salary","is_recurring":True,"is_anomaly":False,"is_duplicate":False,"is_subscription":False},
        {"transaction_id":"2","user_id":"U1","date":"2026-01-05","description":"SWIGGY","merchant":"Swiggy","amount":1000,"transaction_type":"DEBIT","account_type":"BANK","payment_mode":"UPI","category":"Food & Dining","subcategory":"Food Delivery","is_recurring":False,"is_anomaly":False,"is_duplicate":False,"is_subscription":False},
        {"transaction_id":"3","user_id":"U1","date":"2026-01-10","description":"RENT","merchant":"Rent","amount":12000,"transaction_type":"DEBIT","account_type":"BANK","payment_mode":"NEFT","category":"Rent","subcategory":"House Rent","is_recurring":True,"is_anomaly":False,"is_duplicate":False,"is_subscription":False},
    ])
    state=run_workflow({"user_id":"U1","request":"Run complete analysis","transactions":df.to_dict("records"),"starting_balance":10000,"trace":[]})
    assert "health" in state
    assert "recommendations" in state
    assert "reviewer" in state
