from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import delete, select

from models.database import (
    AgentRun,
    Alert,
    Budget,
    FinancialHealthHistory,
    Goal,
    Recommendation,
    Subscription,
    TransactionRow,
    User,
    get_session,
    init_db,
)


def persist_financial_state(state: dict[str, Any]) -> None:
    """Persist the latest financial twin and observability data in SQLite."""
    init_db()
    user_id = str(state.get("user_id", ""))
    if not user_id:
        return

    session = get_session()
    try:
        profile = state.get("profile", {})
        user = session.get(User, user_id)
        if user is None:
            user = User(
                user_id=user_id,
                name=str(profile.get("name", user_id)),
                monthly_income=float(profile.get("monthly_income", 0) or 0),
                starting_balance=float(state.get("starting_balance", profile.get("starting_balance", 0)) or 0),
            )
            session.add(user)
        else:
            user.name = str(profile.get("name", user.name))
            user.monthly_income = float(profile.get("monthly_income", user.monthly_income) or 0)
            user.starting_balance = float(state.get("starting_balance", user.starting_balance) or 0)

        for raw in state.get("transactions", []):
            try:
                tx_id = str(raw["transaction_id"])
                row = session.get(TransactionRow, tx_id)
                values = {
                    "user_id": user_id,
                    "date": datetime.fromisoformat(str(raw["date"]).replace("Z", "")).date(),
                    "description": str(raw.get("description", "")),
                    "merchant": str(raw.get("merchant", "Unknown")),
                    "amount": float(raw.get("amount", 0) or 0),
                    "transaction_type": str(raw.get("transaction_type", "DEBIT")),
                    "account_type": str(raw.get("account_type", "BANK")),
                    "payment_mode": str(raw.get("payment_mode", "UPI")),
                    "category": str(raw.get("category", "Other")),
                    "subcategory": str(raw.get("subcategory", "")),
                    "is_recurring": bool(raw.get("is_recurring", False)),
                    "is_anomaly": bool(raw.get("is_anomaly", False)),
                    "is_duplicate": bool(raw.get("is_duplicate", False)),
                    "is_subscription": bool(raw.get("is_subscription", False)),
                }
                if row is None:
                    session.add(TransactionRow(transaction_id=tx_id, **values))
                else:
                    for key, value in values.items():
                        setattr(row, key, value)
            except (KeyError, ValueError, TypeError):
                continue

        # Replace the current-month budget snapshot for this user.
        budget = state.get("budget", {})
        month = str(budget.get("month") or state.get("latest_month") or "latest")[:7]
        session.execute(delete(Budget).where(Budget.user_id == user_id, Budget.month == month))
        for category, allocated in budget.get("categories", {}).items():
            session.add(Budget(user_id=user_id, month=month, category=str(category), allocated=float(allocated)))

        session.execute(delete(Goal).where(Goal.user_id == user_id))
        for g in state.get("goals", []):
            try:
                session.add(
                    Goal(
                        user_id=user_id,
                        name=str(g.get("name", "Custom Goal")),
                        target_amount=float(g.get("target_amount", 0)),
                        target_date=datetime.fromisoformat(str(g.get("target_date"))).date(),
                        current_amount=float(g.get("current_amount", 0)),
                        monthly_contribution=float(g.get("monthly_contribution", 0)),
                    )
                )
            except (TypeError, ValueError):
                continue

        session.execute(delete(Subscription).where(Subscription.user_id == user_id))
        for sub in state.get("subscriptions", []):
            session.add(
                Subscription(
                    user_id=user_id,
                    merchant=str(sub.get("merchant", "Unknown")),
                    monthly_cost=float(sub.get("monthly_cost", 0)),
                    billing_frequency=str(sub.get("billing_frequency", "monthly")),
                    observed_months=int(sub.get("observed_months", 0)),
                    usage_evidence=str(sub.get("usage_evidence", "unavailable")),
                )
            )

        for rec in state.get("recommendations", []):
            session.add(
                Recommendation(
                    user_id=user_id,
                    text=str(rec.get("text", "")),
                    evidence=json.dumps(rec.get("evidence", {}), default=str),
                )
            )
        for alert in state.get("alerts", []):
            timestamp = datetime.now(timezone.utc).replace(tzinfo=None)
            if alert.get("timestamp"):
                try:
                    timestamp = datetime.fromisoformat(str(alert["timestamp"]).replace("Z", "+00:00")).replace(tzinfo=None)
                except ValueError:
                    pass
            session.add(
                Alert(
                    user_id=user_id,
                    severity=str(alert.get("severity", "INFO")),
                    title=str(alert.get("title", "Financial alert")),
                    explanation=str(alert.get("explanation", "")),
                    recommended_action=str(alert.get("recommended_action", "Review the alert.")),
                    timestamp=timestamp,
                )
            )

        health = state.get("health", {})
        if health:
            month = str(state.get("latest_month", datetime.now(timezone.utc).replace(tzinfo=None).strftime("%Y-%m")))[:7]
            session.add(
                FinancialHealthHistory(
                    user_id=user_id,
                    month=month,
                    score=float(health.get("score", 0)),
                    components_json=json.dumps(health.get("components", {}), default=str),
                )
            )

        for item in state.get("trace", []):
            session.add(
                AgentRun(
                    user_id=user_id,
                    agent_name=str(item.get("agent", "")),
                    status=str(item.get("status", "UNKNOWN")),
                    input_summary=str(item.get("input_summary", "")),
                    tool_calls=json.dumps(item.get("tool_calls", []), default=str),
                    execution_time_ms=float(item.get("execution_time_ms", 0) or 0),
                    output_summary=str(item.get("output_summary", "")),
                    error=item.get("error"),
                )
            )
        session.commit()
    finally:
        session.close()


def store_memory(
    user_id: str,
    recommendations: list[dict],
    health: dict,
    alerts: list[dict],
    trace: list[dict],
) -> None:
    """Backward-compatible wrapper used by the orchestrator."""
    persist_financial_state(
        {
            "user_id": user_id,
            "recommendations": recommendations,
            "health": health,
            "alerts": alerts,
            "trace": trace,
        }
    )


def retrieve_memory(user_id: str, limit: int = 10) -> list[dict]:
    init_db()
    if not user_id:
        return []
    session = get_session()
    try:
        rows = session.execute(
            select(Recommendation)
            .where(Recommendation.user_id == user_id)
            .order_by(Recommendation.created_at.desc())
            .limit(limit)
        ).scalars().all()
        return [
            {
                "text": r.text,
                "evidence": json.loads(r.evidence or "{}"),
                "created_at": r.created_at.isoformat(),
            }
            for r in rows
        ]
    finally:
        session.close()
