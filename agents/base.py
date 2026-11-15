from __future__ import annotations

import json
import time
from typing import Any, Callable, Iterable


def _summary(value: Any, limit: int = 240) -> str:
    try:
        if isinstance(value, dict):
            text = json.dumps({k: value[k] for k in list(value)[:8]}, default=str)
        elif isinstance(value, list):
            text = json.dumps(value[:3], default=str)
        else:
            text = str(value)
    except Exception:
        text = repr(value)
    return text[:limit]


def run_agent(
    state: dict,
    agent_name: str,
    fn: Callable[[dict], dict],
    tool_calls: Iterable[str] | None = None,
) -> dict:
    """Execute an agent and record an evaluator-friendly trace entry."""
    started = time.perf_counter()
    trace = state.setdefault("trace", [])
    item = {
        "agent": agent_name,
        "status": "RUNNING",
        "start_time": time.strftime("%H:%M:%S"),
        "input_summary": _summary({
            "request": state.get("request", ""),
            "transaction_count": len(state.get("transactions", [])),
            "intent": state.get("intent", ""),
        }),
        "tool_calls": list(tool_calls or []),
    }
    trace.append(item)
    try:
        result = fn(state) or {}
        elapsed = round((time.perf_counter() - started) * 1000, 2)
        item.update({
            "status": "SUCCESS",
            "execution_time_ms": elapsed,
            "output_summary": _summary(result),
        })
        return result
    except Exception as exc:
        elapsed = round((time.perf_counter() - started) * 1000, 2)
        item.update({
            "status": "ERROR",
            "execution_time_ms": elapsed,
            "error": str(exc),
            "output_summary": "",
        })
        raise
