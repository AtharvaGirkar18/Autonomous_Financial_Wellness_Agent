"""Data MCP Backend — database, memory, and file-ingestion tools."""
from __future__ import annotations

from mcp.registry import mcp_registry


def _register() -> None:
    tools: dict = {}

    # ── Memory / Persistence ──────────────────────────────────────────────────
    try:
        from tools.memory_tools import persist_financial_state, retrieve_memory
        tools["retrieve_memory"] = {
            "fn": retrieve_memory,
            "description": "Retrieve historical financial state and recommendations from memory.",
            "tags": ["memory", "persistence"],
        }
        tools["persist_financial_state"] = {
            "fn": persist_financial_state,
            "description": "Persist the current financial state snapshot to the database.",
            "tags": ["memory", "persistence"],
        }
    except Exception:
        pass

    # ── File Ingestion ────────────────────────────────────────────────────────
    try:
        from tools.pdf_tools import extract_transactions_from_pdf
        tools["extract_transactions_from_pdf"] = {
            "fn": extract_transactions_from_pdf,
            "description": "Extract structured transactions from a PDF bank statement.",
            "tags": ["ingestion", "pdf"],
        }
    except Exception:
        pass

    # ── Health Scoring ────────────────────────────────────────────────────────
    try:
        from tools.transaction_tools import (
            calculate_debt_ratio,
            calculate_emergency_fund,
            calculate_health_score,
        )
        tools["calculate_financial_health_score"] = {
            "fn": calculate_health_score,
            "description": "Compute composite financial health score (0-100) from transactions and profile.",
            "tags": ["health", "scoring"],
        }
        tools["calculate_emergency_fund"] = {
            "fn": calculate_emergency_fund,
            "description": "Assess emergency fund adequacy (months of runway).",
            "tags": ["health", "emergency_fund"],
        }
        tools["calculate_debt_ratio"] = {
            "fn": calculate_debt_ratio,
            "description": "Calculate debt-to-income ratio from transaction data.",
            "tags": ["health", "debt"],
        }
    except Exception:
        pass

    if tools:
        mcp_registry.register_backend("data_mcp", tools)


_register()
