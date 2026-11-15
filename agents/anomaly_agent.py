import pandas as pd
from agents.base import run_agent
from tools.anomaly_tools import detect_transaction_anomalies


def anomaly_node(state: dict) -> dict:
    def _run(s):
        out = detect_transaction_anomalies(pd.DataFrame(s["transactions"]))
        rows = out[out["is_detected_anomaly"]].copy()
        duplicate_rows = out[out["duplicate_candidate"]].copy()
        anomalies = []
        for _, r in rows.iterrows():
            severity = "HIGH" if r.get("rule_anomaly", False) or float(r.get("amount",0)) >= 10000 else "MEDIUM"
            anomalies.append({
                "transaction_id": r["transaction_id"],
                "merchant": r["merchant"],
                "amount": float(r["amount"]),
                "date": str(r["date"]),
                "anomaly_score": float(r["anomaly_score"]),
                "severity": severity,
                "duplicate_candidate": bool(r.get("duplicate_candidate", False)),
                "reason": "Large amount, model anomaly, or duplicate pattern detected."
            })
        duplicates = [
            {
                "transaction_id": r["transaction_id"],
                "merchant": r["merchant"],
                "amount": float(r["amount"]),
                "date": str(r["date"]),
            }
            for _, r in duplicate_rows.iterrows()
        ]
        return {
            "transactions": out.to_dict(orient="records"),
            "anomalies": anomalies,
            "duplicate_candidates": duplicates,
        }
    return run_agent(state, "Anomaly Detection Agent", _run, ["detect_anomalies", "detect_duplicate_candidates"])
