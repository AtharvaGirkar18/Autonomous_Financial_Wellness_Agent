"""PII / secret detection and redaction.

Used before anything leaves the process (LLM prompts), before anything is
persisted (audit log) and on every generated answer.
"""
from __future__ import annotations

import re

_CARD_RE = re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)")
_AADHAAR_RE = re.compile(r"(?<!\d)[2-9]\d{3}[ -]?\d{4}[ -]?\d{4}(?!\d)")
_PAN_RE = re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b")
_PHONE_RE = re.compile(r"(?<!\d)(?:\+?91[\s-]?)?[6-9]\d{9}(?!\d)")
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_ACCOUNT_RE = re.compile(r"(?<!\d)\d{9,18}(?!\d)")
_CREDENTIAL_RE = re.compile(
    r"(?i)\b(otp|cvv|cvc|upi[ -]?pin|atm[ -]?pin|password|passcode)\b\s*(?:is|:|=)\s*([\w@#$!*]{3,})"
)
_API_KEY_RE = re.compile(r"\b(AIza[0-9A-Za-z_\-]{30,}|sk-[A-Za-z0-9_\-]{20,}|ghp_[A-Za-z0-9]{30,})\b")


def _luhn_ok(digits: str) -> bool:
    total, alt = 0, False
    for ch in reversed(digits):
        d = int(ch)
        if alt:
            d *= 2
            if d > 9:
                d -= 9
        total += d
        alt = not alt
    return total % 10 == 0


def redact_pii(text: str) -> tuple[str, dict[str, int]]:
    """Mask PII/secrets. Returns (redacted_text, {kind: count})."""
    counts: dict[str, int] = {}

    def bump(kind: str) -> None:
        counts[kind] = counts.get(kind, 0) + 1

    out = str(text or "")

    def _api(m):
        bump("api_key")
        return "[REDACTED_API_KEY]"

    out = _API_KEY_RE.sub(_api, out)

    def _cred(m):
        bump("credential")
        return f"{m.group(1)}: [REDACTED_CREDENTIAL]"

    out = _CREDENTIAL_RE.sub(_cred, out)

    def _card(m):
        digits = re.sub(r"\D", "", m.group(0))
        if 13 <= len(digits) <= 19 and _luhn_ok(digits):
            bump("card_number")
            return "[REDACTED_CARD]"
        return m.group(0)

    out = _CARD_RE.sub(_card, out)

    def _sub(pattern, label, kind):
        nonlocal out

        def repl(_m):
            bump(kind)
            return label

        out = pattern.sub(repl, out)

    _sub(_AADHAAR_RE, "[REDACTED_AADHAAR]", "aadhaar")
    _sub(_PAN_RE, "[REDACTED_PAN]", "pan")
    _sub(_EMAIL_RE, "[REDACTED_EMAIL]", "email")
    _sub(_PHONE_RE, "[REDACTED_PHONE]", "phone")
    _sub(_ACCOUNT_RE, "[REDACTED_ACCOUNT]", "account_number")
    return out, counts


def contains_pii(text: str) -> bool:
    return bool(redact_pii(text)[1])
