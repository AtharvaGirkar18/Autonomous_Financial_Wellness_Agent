"""Analytics MCP Backend — ML, anomaly detection, forecasting, and health scoring tools.

All ML-heavy imports are wrapped in try/except since sklearn/scipy may be unavailable
in some environments. The registry degrades gracefully to deterministic alternatives.
"""
from __future__ import annotations

from mcp.registry import mcp_registry


def _register() -> None:
    tools: dict = {}

    # ── ML Anomaly Detection (optional — needs sklearn) ────────────────────────
    try:
        from ml.anomaly_detection import detect_anomalies, detect_duplicate_candidates
        tools["detect_anomalies"] = {
            "fn": detect_anomalies,
            "description": "ML-based detection of unusual transactions using IsolationForest.",
            "tags": ["anomaly", "ml", "security"],
        }
        tools["detect_duplicate_candidates"] = {
            "fn": detect_duplicate_candidates,
            "description": "Find potential duplicate charge candidates in transactions.",
            "tags": ["anomaly", "duplicates"],
        }
    except Exception:
        pass

    # ── ML Categorization (optional) ──────────────────────────────────────────
    try:
        from ml.categorization import categorize_transactions
        tools["categorize_transactions"] = {
            "fn": categorize_transactions,
            "description": "ML-assisted transaction categorization by merchant/description.",
            "tags": ["categorization", "ml"],
        }
    except Exception:
        pass

    # ── Anomaly tools (pure Python, always available) ─────────────────────────
    try:
        from tools.anomaly_tools import flag_high_value_transactions
        tools["flag_high_value_transactions"] = {
            "fn": flag_high_value_transactions,
            "description": "Flag transactions above a threshold amount for review.",
            "tags": ["anomaly", "review"],
        }
    except Exception:
        pass

    # ── Trend Analysis ────────────────────────────────────────────────────────
    try:
        from tools.forecasting_tools import calculate_monthly_trends
        tools["calculate_monthly_trends"] = {
            "fn": calculate_monthly_trends,
            "description": "Calculate month-over-month spending trends per category.",
            "tags": ["trends", "analytics"],
        }
    except Exception:
        pass

    if tools:
        mcp_registry.register_backend("analytics_mcp", tools)


_register()
