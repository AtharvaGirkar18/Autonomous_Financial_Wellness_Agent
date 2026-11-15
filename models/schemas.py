from __future__ import annotations

from datetime import datetime, date
from typing import Any, Dict, List, Optional, TypedDict

from pydantic import BaseModel, Field


class Transaction(BaseModel):
    transaction_id: str
    user_id: str
    date: date
    description: str
    merchant: str
    amount: float
    transaction_type: str
    account_type: str
    payment_mode: str
    category: str = "Other"
    subcategory: str = ""
    is_recurring: bool = False
    is_anomaly: bool = False
    is_duplicate: bool = False
    is_subscription: bool = False


class AgentRunRecord(BaseModel):
    user_id: str
    agent_name: str
    status: str
    input_summary: str = ""
    tool_calls: List[str] = Field(default_factory=list)
    execution_time_ms: float = 0.0
    output_summary: str = ""
    error: Optional[str] = None
    timestamp: datetime


class FinanceState(TypedDict, total=False):
    user_id: str
    request: str
    intent: str
    plan: Dict[str, Any]
    profile: Dict[str, Any]
    current_balance: float
    starting_balance: float
    data_quality: Dict[str, Any]
    transactions: List[Dict[str, Any]]
    analytics: Dict[str, Any]
    budget: Dict[str, Any]
    health: Dict[str, Any]
    investment_readiness: Dict[str, Any]
    financial_twin: Dict[str, Any]
    anomalies: List[Dict[str, Any]]
    duplicate_candidates: List[Dict[str, Any]]
    subscriptions: List[Dict[str, Any]]
    forecast: Dict[str, Any]
    goals: List[Dict[str, Any]]
    simulations: List[Dict[str, Any]]
    recommendations: List[Dict[str, Any]]
    alerts: List[Dict[str, Any]]
    reviewer: Dict[str, Any]
    memory: List[Dict[str, Any]]
    trace: List[Dict[str, Any]]
    final_answer: str
    # Keys below were previously undeclared; LangGraph drops undeclared keys, which
    # silently discarded budget overrides and would have discarded guardrail metadata.
    budget_overrides: Dict[str, float]
    latest_month: str
    categorization_summary: Dict[str, Any]
    guardrail: Dict[str, Any]
    ai_answer: Dict[str, Any]
    upload_report: Dict[str, Any]
    blocked: bool
    persistence_warning: str
    last_run_status: str
