from __future__ import annotations
import pandas as pd
from agents.base import run_agent


REQUIRED = ["transaction_id", "user_id", "date", "description", "merchant", "amount", "transaction_type"]


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x.columns = [str(c).strip().lower().replace(" ", "_") for c in x.columns]
    for c in REQUIRED:
        if c not in x.columns:
            if c == "transaction_id":
                x[c] = [f"UPLOADED_{i:06d}" for i in range(len(x))]
            elif c == "user_id":
                x[c] = "USER001"
            elif c == "description" and "details" in x.columns:
                x[c] = x["details"]
            elif c == "merchant":
                x[c] = x.get("description", "Unknown")
            elif c == "amount":
                x[c] = pd.to_numeric(x.get("debit", x.get("credit")), errors="coerce").abs()
            elif c == "transaction_type":
                if "credit" in x.columns and "debit" in x.columns:
                    x[c] = ["CREDIT" if float(cr or 0) > 0 else "DEBIT" for cr in x["credit"].fillna(0)]
                else:
                    x[c] = "DEBIT"
            elif c not in x.columns:
                raise ValueError(f"Missing required field: {c}")
    x["date"] = pd.to_datetime(x["date"], errors="coerce")
    x["amount"] = pd.to_numeric(x["amount"], errors="coerce").abs()
    x["merchant"] = x["merchant"].fillna(x["description"]).astype(str).str.upper().str.replace(r"\s+", " ", regex=True).str.strip()
    x["description"] = x["description"].fillna("").astype(str)
    x["transaction_type"] = x["transaction_type"].astype(str).str.upper().replace({"DEBITED":"DEBIT", "CREDITED":"CREDIT"})
    x = x.dropna(subset=["date", "amount"])
    x = x.drop_duplicates(subset=["transaction_id"])
    for col, default in [
        ("account_type", "BANK"), ("payment_mode", "UPI"), ("category", "Other"),
        ("subcategory", ""), ("is_recurring", False), ("is_anomaly", False),
        ("is_duplicate", False), ("is_subscription", False)
    ]:
        if col not in x.columns:
            x[col] = default
    return x


def ingestion_node(state: dict) -> dict:
    def _run(s):
        df = pd.DataFrame(s.get("transactions", []))
        if df.empty:
            raise ValueError("No transactions available")
        x = normalize(df)
        return {
            "transactions": x.to_dict(orient="records"),
            "data_quality": {
                "records_processed": len(df),
                "records_after_normalization": len(x),
                "missing_values": int(x.isna().sum().sum()),
            },
        }
    return run_agent(state, "Ingestion Agent", _run, ["get_transactions", "normalize_transactions", "validate_schema"])
