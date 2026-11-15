from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support


def detect_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """Combine robust business rules, Isolation Forest and duplicate matching."""
    x = df.copy()
    x["date"] = pd.to_datetime(x["date"], errors="coerce")
    x["amount"] = pd.to_numeric(x["amount"], errors="coerce").fillna(0.0)
    debit = x[x["transaction_type"].eq("DEBIT")].copy()

    out = x.copy()
    for col in ["model_anomaly", "rule_anomaly", "duplicate_candidate", "is_detected_anomaly"]:
        out[col] = False
    out["anomaly_score"] = 0.0

    if debit.empty:
        return out

    feats = pd.DataFrame(index=debit.index)
    feats["amount_log"] = np.log1p(debit["amount"])
    feats["merchant_freq"] = debit.groupby("merchant")["merchant"].transform("count").astype(float)
    feats["day_of_month"] = pd.to_datetime(debit["date"]).dt.day.astype(float)
    feats = feats.fillna(0.0)

    # Keep model contamination low because injected anomalies are deliberately rare.
    model = IsolationForest(
        n_estimators=250,
        contamination=min(0.025, max(0.01, 12 / max(len(feats), 1))),
        random_state=42,
    )
    model.fit(feats)
    decision = model.decision_function(feats)
    pred = model.predict(feats)
    model_flag = pred == -1
    score = (-decision).astype(float)

    out.loc[debit.index, "anomaly_score"] = np.round(score, 4)
    out.loc[debit.index, "model_anomaly"] = model_flag

    # Merchant/category-aware robust rule. The injected demo anomalies are many
    # multiples above their local baseline and should be explainable without a model.
    d = debit.copy()
    d["category_median"] = d.groupby(["user_id", "category"])["amount"].transform("median")
    d["category_count"] = d.groupby(["user_id", "category"])["amount"].transform("count")
    d["user_median"] = d.groupby("user_id")["amount"].transform("median")
    local_baseline = d["category_median"].where(d["category_count"] >= 10, d["user_median"])
    threshold = np.maximum.reduce([
        local_baseline.fillna(d["user_median"]).to_numpy(dtype=float) * 5.0,
        d["user_median"].fillna(d["amount"].median()).to_numpy(dtype=float) * 7.0,
        np.full(len(d), 5000.0),
    ])
    rule_flag = (d["amount"].to_numpy(dtype=float) >= threshold) & (~d["is_recurring"].astype(bool).to_numpy())
    out.loc[d.index, "rule_anomaly"] = rule_flag.astype(bool)

    dup = d.assign(amount_round=d["amount"].round(2))
    dup_flag = dup.duplicated(subset=["user_id", "date", "merchant", "amount_round"], keep=False)
    out.loc[dup.index, "duplicate_candidate"] = dup_flag.to_numpy(dtype=bool)

    # Require either an interpretable rule or both model suspicion and a high amount.
    user_q97 = d.groupby("user_id")["amount"].transform(lambda s: s.quantile(0.97))
    model_plus_amount = d["amount"].to_numpy(dtype=float) >= user_q97.to_numpy(dtype=float)
    out.loc[d.index, "is_detected_anomaly"] = rule_flag | (model_flag & model_plus_amount & ~d["is_recurring"].astype(bool).to_numpy())
    return out


def evaluate_anomalies(y_true, y_pred) -> dict:
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    return {
        "precision": round(float(p), 4),
        "recall": round(float(r), 4),
        "f1": round(float(f), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(y_true, y_pred, zero_division=0),
    }
