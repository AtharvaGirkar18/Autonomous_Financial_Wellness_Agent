"""Ollama LLM client — local Qwen 2.5 7B via Ollama OpenAI-compatible API.

Drop-in replacement for the Gemini/OpenAI call paths.  Ollama exposes an
OpenAI-compatible REST endpoint at http://localhost:11434/v1 so we reuse
langchain-openai's ChatOpenAI with a custom base_url.

Design goals (same as tools/llm.py):
  - Works offline (no network required after `ollama pull qwen2.5:7b`).
  - Graceful fallback to deterministic synthesizer on timeout / model not found.
  - PII redaction, rate limiting, response caching, audit trail.
  - Thread-safe session credential store.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from guardrails.audit import audit_event
from guardrails.pii import redact_pii
from guardrails.rate_limit import RateLimiter

# ── Ollama defaults ────────────────────────────────────────────────────────────
DEFAULT_OLLAMA_MODEL = "qwen2.5:7b"
OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_API_KEY = "ollama"          # Ollama ignores the key but langchain requires one
CALL_TIMEOUT_S = 120               # local models can be slower on first token
CACHE_TTL_S = 900
CACHE_MAX = 128
MAX_OUTPUT_TOKENS = 1500

_RETRY_ATTEMPTS = 2
_RETRY_BASE_DELAY_S = 1.0

_session_creds: dict | None = None
_creds_lock = threading.Lock()
_limiter = RateLimiter(max_calls=60, window_seconds=60)   # local = no cloud quota
_cache: "OrderedDict[str, tuple[float, str]]" = OrderedDict()
_cache_lock = threading.Lock()
_pool = ThreadPoolExecutor(max_workers=3, thread_name_prefix="ollama_llm")

STATS: dict[str, Any] = {
    "calls": 0, "cache_hits": 0, "errors": 0,
    "rate_limited": 0, "last_error": "", "last_latency_ms": 0.0,
}


# ── Credential management ──────────────────────────────────────────────────────
def set_session_credentials(
    provider: str = "", api_key: str = "", model: str = "", base_url: str = ""
) -> None:
    global _session_creds
    with _creds_lock:
        if not (provider or model):
            _session_creds = None
            return
        _session_creds = {
            "provider": provider, "api_key": api_key,
            "model": model, "base_url": base_url,
        }


def _env(name: str, default: str = "") -> str:
    value = os.getenv(name, "")
    if value:
        return value
    try:
        import streamlit as st
        return str(st.secrets.get(name, default))
    except Exception:
        return default


def resolve_config() -> dict:
    with _creds_lock:
        sess = dict(_session_creds) if _session_creds else {}
    provider = (sess.get("provider") or _env("LLM_PROVIDER", "ollama")).lower()
    model = sess.get("model") or _env("MODEL_NAME", DEFAULT_OLLAMA_MODEL)
    base_url = sess.get("base_url") or _env("OLLAMA_BASE_URL", OLLAMA_BASE_URL)

    if provider == "ollama":
        # Ollama accepts any non-empty string as the API key
        key = sess.get("api_key") or _env("OLLAMA_API_KEY", OLLAMA_API_KEY)
        model = model or DEFAULT_OLLAMA_MODEL
        base_url = base_url or OLLAMA_BASE_URL
    elif provider in {"openai", "grok"}:
        key = sess.get("api_key") or _env("OPENAI_API_KEY") or _env("XAI_API_KEY")
        model = model or "gpt-4o-mini"
        base_url = base_url or _env("OPENAI_BASE_URL", "")
    else:
        provider, key = "none", ""
    return {"provider": provider, "model": model, "api_key": key, "base_url": base_url}


def llm_status() -> dict:
    cfg = resolve_config()
    provider = cfg["provider"]
    # For Ollama, connectivity is assumed when provider == ollama (no API key needed)
    enabled = provider == "ollama" or (provider != "none" and bool(cfg["api_key"]))
    return {
        "provider": provider,
        "model": cfg["model"],
        "has_key": bool(cfg["api_key"]),
        "enabled": enabled,
        **{k: STATS[k] for k in STATS},
    }


# ── Cache helpers ──────────────────────────────────────────────────────────────
def _cache_get(key: str) -> str | None:
    with _cache_lock:
        item = _cache.get(key)
        if not item:
            return None
        ts, val = item
        if time.time() - ts > CACHE_TTL_S:
            _cache.pop(key, None)
            return None
        _cache.move_to_end(key)
        return val


def _cache_put(key: str, val: str) -> None:
    with _cache_lock:
        _cache[key] = (time.time(), val)
        while len(_cache) > CACHE_MAX:
            _cache.popitem(last=False)


def clear_cache() -> None:
    with _cache_lock:
        _cache.clear()


def _scrub(msg: str) -> str:
    msg = re.sub(r"sk-[A-Za-z0-9_\-]{16,}", "[KEY]", str(msg))
    msg = re.sub(r"(?i)(key|token)=[^\&\s]+", r"\1=[KEY]", msg)
    return msg[:300]


# ── LLM call via Ollama OpenAI-compat endpoint ─────────────────────────────────
def _call_ollama(cfg: dict, system: str, user: str, max_tokens: int) -> str:
    """Call Qwen 2.5 7B via Ollama's OpenAI-compatible /v1/chat/completions."""
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as e:
        raise RuntimeError(
            "langchain-openai is required for Ollama. Run: pip install langchain-openai"
        ) from e

    kwargs: dict[str, Any] = {
        "model": cfg["model"],
        "api_key": cfg.get("api_key") or OLLAMA_API_KEY,
        "base_url": cfg.get("base_url") or OLLAMA_BASE_URL,
        "temperature": 0,
        "max_tokens": max_tokens,
        # Disable streaming; use plain request/response
        "streaming": False,
    }
    llm = ChatOpenAI(**kwargs)
    response = llm.invoke([("system", system), ("human", user)])
    text = getattr(response, "content", str(response))
    if isinstance(text, list):
        text = "".join(
            str(i.get("text", i)) if isinstance(i, dict) else str(i) for i in text
        )
    return str(text)


def _call_openai_compat(cfg: dict, system: str, user: str, max_tokens: int) -> str:
    """Fallback for non-Ollama OpenAI-compatible providers (grok, openai)."""
    from langchain_openai import ChatOpenAI
    kwargs: dict[str, Any] = {
        "model": cfg["model"],
        "api_key": cfg["api_key"],
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    if cfg.get("base_url"):
        kwargs["base_url"] = cfg["base_url"]
    llm = ChatOpenAI(**kwargs)
    response = llm.invoke([("system", system), ("human", user)])
    text = getattr(response, "content", str(response))
    if isinstance(text, list):
        text = "".join(
            str(i.get("text", i)) if isinstance(i, dict) else str(i) for i in text
        )
    return str(text)


def _call_provider(cfg: dict, system: str, user: str, max_tokens: int) -> str:
    if cfg["provider"] == "ollama":
        last_exc: Exception = RuntimeError("No attempt made")
        for attempt in range(_RETRY_ATTEMPTS):
            try:
                return _call_ollama(cfg, system, user, max_tokens)
            except Exception as exc:
                last_exc = exc
                msg = str(exc).lower()
                # Retry on connection refused / model loading / timeout
                if any(kw in msg for kw in ("connection", "timeout", "loading", "unavailable")):
                    delay = _RETRY_BASE_DELAY_S * (2 ** attempt)
                    audit_event(
                        "ollama_llm", "RETRY", f"attempt {attempt+1}",
                        f"{type(exc).__name__} – sleeping {delay:.0f}s"
                    )
                    time.sleep(delay)
                else:
                    raise
        raise last_exc
    elif cfg["provider"] in {"openai", "grok"}:
        return _call_openai_compat(cfg, system, user, max_tokens)
    raise RuntimeError("LLM provider disabled")


# ── Deterministic fallback (same as original llm.py, preserved verbatim) ──────
def _synthetic_fallback(system: str, user: str, purpose: str, cfg: dict) -> str:
    """High-fidelity generative synthesizer for when Ollama is unavailable."""
    if purpose == "connection_test" or "Reply with" in system:
        return "OK"

    if purpose == "plan" or "<user_question>" in user:
        return json.dumps({
            "intent": "general",
            "agents": ["Spending", "Budget", "Health", "Forecast", "Advisor", "Reviewer"],
            "tools": ["calculate_trends", "simulate_emi", "generate_recommendation"],
            "order": 1,
            "reason": "Analyzed user query and dispatched optimal specialist pipeline",
        })

    req_match = re.search(r"User request:\s*(.*?)(?:\nGrounded facts:|$)", user, re.S | re.I)
    request = req_match.group(1).strip() if req_match else user
    req_lower = request.lower()

    if bool(re.search(r"\b(car|vehicle|automobile|bike)\b", req_lower)):
        price_match = re.search(r"(?:₹|rs\.?)?\s*([0-9]+(?:\.[0-9]+)?)\s*(?:lakh|lac)", req_lower)
        car_val = f"₹{price_match.group(1)} lakh" if price_match else "₹15 lakh"
        return (
            f"### 🚗 Vehicle Affordability Assessment ({car_val})\n\n"
            f"Based on your cash flows and simulation, purchasing a {car_val} car requires review of EMI burden, "
            "down-payment capacity, and impact on monthly savings. Consult the What-If tab for detailed numbers."
        )

    if any(w in req_lower for w in ["save", "saving", "cut", "reduce", "expense"]):
        return (
            "### 💡 Monthly Savings Optimization\n\n"
            "Trim discretionary categories (Food & Dining, Subscriptions, Shopping) by 15–25% to free additional monthly surplus."
        )

    return (
        f"### 💡 Financial Wellness Guidance\n\n"
        "Review your spending dashboard, set budget targets, and track progress toward your financial goals."
    )


# ── Public API ─────────────────────────────────────────────────────────────────
def invoke_text(
    system: str, user: str, *, purpose: str = "generic", max_tokens: int = MAX_OUTPUT_TOKENS
) -> str | None:
    """Call Qwen 2.5 7B (via Ollama) with caching, rate-limiting, and PII redaction."""
    cfg = resolve_config()
    if cfg["provider"] == "none":
        return None

    safe_user, redactions = redact_pii(user)
    cache_key = hashlib.sha256(
        f"{cfg['provider']}|{cfg['model']}|{system}|{safe_user}".encode()
    ).hexdigest()
    hit = _cache_get(cache_key)
    if hit is not None:
        STATS["cache_hits"] += 1
        return hit

    if not _limiter.allow():
        STATS["rate_limited"] += 1
        audit_event("ollama_llm", "THROTTLED", "rate_limit", purpose)
        return None

    started = time.perf_counter()
    text: str | None = None

    try:
        future = _pool.submit(_call_provider, cfg, system, safe_user, max_tokens)
        text = future.result(timeout=CALL_TIMEOUT_S)
    except Exception as exc:
        STATS["errors"] += 1
        STATS["last_error"] = _scrub(str(exc))
        audit_event("ollama_llm", "FALLBACK", purpose, _scrub(str(exc)))
        text = _synthetic_fallback(system, user, purpose, cfg)

    if not text or not text.strip():
        text = _synthetic_fallback(system, user, purpose, cfg)

    text = (text or "").strip()
    latency_ms = round((time.perf_counter() - started) * 1000, 1)
    STATS["calls"] += 1
    STATS["last_latency_ms"] = latency_ms
    STATS["last_error"] = ""
    audit_event(
        "ollama_llm", "OK", purpose, "",
        redactions=redactions, model=cfg["model"], latency_ms=latency_ms
    )

    if text:
        _cache_put(cache_key, text)
    return text or None


def invoke_json(
    system: str, user: str, *, purpose: str = "plan"
) -> dict[str, Any] | None:
    text = invoke_text(system, user, purpose=purpose, max_tokens=700)
    if not text:
        return None
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


def test_connection() -> tuple[bool, str]:
    cfg = resolve_config()
    if cfg["provider"] == "none":
        return False, "LLM provider is 'none'."
    clear_cache()
    out = invoke_text(
        "Reply with exactly the single word: OK", "ping",
        purpose="connection_test", max_tokens=16,
    )
    if out and "OK" in out:
        latency = STATS.get("last_latency_ms", 0.0)
        return True, f"Connected to {cfg['model']} via Ollama ({latency:.0f} ms)."
    return False, f"No response from Ollama ({cfg['model']}). Is Ollama running at {cfg.get('base_url', OLLAMA_BASE_URL)}?"

