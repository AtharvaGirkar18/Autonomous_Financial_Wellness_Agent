"""Guardrail audit trail.

Only *redacted* snippets and a SHA-256 of the original text are stored, so the
audit log itself can never leak secrets or PII.
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import deque
from pathlib import Path

from config.settings import ROOT_DIR

AUDIT_PATH = ROOT_DIR / "artifacts" / "guardrail_audit.jsonl"
_MAX_BYTES = 1_000_000
_lock = threading.Lock()
_ring: deque[dict] = deque(maxlen=200)


def audit_event(layer: str, decision: str, category: str, text: str = "", **extra) -> dict:
    from guardrails.pii import redact_pii  # local import avoids a cycle

    redacted, _ = redact_pii(text or "")
    event = {
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "layer": layer,
        "decision": decision,
        "category": category,
        "sha256": hashlib.sha256((text or "").encode("utf-8", "ignore")).hexdigest()[:16],
        "snippet": redacted[:120],
        **extra,
    }
    _ring.append(event)
    try:
        with _lock:
            AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
            if AUDIT_PATH.exists() and AUDIT_PATH.stat().st_size > _MAX_BYTES:
                AUDIT_PATH.replace(AUDIT_PATH.with_suffix(".jsonl.1"))
            with AUDIT_PATH.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(event, ensure_ascii=False) + "\n")
    except OSError:
        pass  # auditing must never break the product
    return event


def recent_events(limit: int = 50) -> list[dict]:
    return list(_ring)[-limit:][::-1]
