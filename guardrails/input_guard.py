"""Input guardrail for natural-language queries.

Defence-in-depth, deterministic and LLM-free (so it cannot itself be prompt
injected, and works with no API key). Layers, evaluated in order:

 1. Structural   – empty / oversized / gibberish / control characters
 2. Normalisation – NFKC, zero-width removal, leetspeak folding (matching only)
 3. Attack classes – prompt injection, secret exfiltration, SQL injection,
                     script/XSS, command / path injection
 4. Harmful finance – laundering, tax evasion, forged documents, account takeover
 5. Policy redirect – individual securities picks, guaranteed-return requests,
                      clearly off-topic requests
 6. PII/credential redaction (the text is still answered, but masked)

Decisions: ALLOW | BLOCK | REDIRECT.  BLOCK/REDIRECT never reach the planner,
the agents or any LLM.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from guardrails.pii import redact_pii

MAX_QUERY_CHARS = 500

_ZERO_WIDTH = dict.fromkeys(map(ord, "\u200b\u200c\u200d\u2060\ufeff\u202a\u202b\u202c\u202d\u202e"), None)
_LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s"})

# ───────────────────────────── attack patterns ─────────────────────────────
PROMPT_INJECTION = [
    r"\b(ignore|disregard|forget|override|bypass|skip|drop)\b.{0,40}\b(previous|prior|above|earlier|all|any|your|the|these|those)\b.{0,30}\b(instructions?|rules?|prompts?|guidelines?|guardrails?|policy|policies|restrictions?|safety|filters?)\b",
    r"\b(ignore|skip|bypass|disable|turn off)\b.{0,25}\b(reviewer|guardrails?|safety|filters?|validation|checks?)\b",
    r"\byou are (now|no longer)\b",
    r"\b(act|behave|respond) as (if you (are|were)|an?|the)\b.{0,40}\b(unrestricted|unfiltered|jailbroken|dan|developer|admin|root|evil)\b",
    r"\b(developer|god|admin|debug|sudo|jailbreak|dan)\s*mode\b",
    r"\bdo anything now\b",
    r"\bpretend (you are|to be|that you)\b",
    r"\b(reveal|show|print|display|repeat|leak|tell me|output)\b.{0,30}\b(system|hidden|initial|original|secret)\b.{0,15}\b(prompt|instructions?|message|rules)\b",
    r"\bwhat (is|are|were) your (system |initial |original )?(instructions?|prompt|rules)\b",
    r"<\|?(im_start|im_end|system|endoftext)\|?>",
    r"\[/?(inst|sys)\]",
    r"^\s*#{2,}\s*(system|instruction)",
    r"\bnew (instructions?|rules?|persona)\b\s*:",
    r"\b(without|no) (any )?(restrictions?|limits?|filters?|guardrails?|censorship)\b",
    r"\bfrom now on\b.{0,40}\b(you will|you must|always|only)\b",
]
SECRET_EXFIL = [
    r"\b(api|secret|access)[ _-]?keys?\b",
    r"\bgoogle_api_key\b|\bopenai_api_key\b|\.env\b|\benvironment variables?\b|\bsecrets?\.toml\b",
    r"\b(show|reveal|print|give|tell|dump|leak|send)\b.{0,25}\b(password|credentials?|token|passphrase)\b",
    r"\b(show|dump|list|export|print|read)\b.{0,25}\b(database|db|sqlite|all users|other users|everyone'?s)\b",
    r"\bother (users?|customers?|people'?s?)\b.{0,25}\b(data|transactions?|balance|accounts?)\b",
]
SQL_INJECTION = [
    r"('|\")\s*(or|and)\s*('|\")?\s*\w+\s*('|\")?\s*=\s*('|\")?\s*\w+",
    r"\bor\s+1\s*=\s*1\b|\band\s+1\s*=\s*1\b|\bor\s+'1'\s*=\s*'1'",
    r";\s*(drop|delete|insert|update|alter|truncate|create|exec|attach)\b",
    r"\bunion\s+(all\s+)?select\b",
    r"\b(drop|truncate|alter)\s+(table|database|schema)\b",
    r"\bdelete\s+from\s+\w+|\binsert\s+into\s+\w+",
    r"\bselect\s+(\*|[\w.]+\s*(,\s*[\w.]+\s*)*)\s+from\s+[\w.\"`\[\]]+\s*(where\b|;|--|limit\b|order\b|group\b|join\b|$)",
    r"\b(information_schema|sqlite_master|sqlite_schema|pg_catalog|sysobjects)\b",
    r"\bxp_cmdshell\b|\bpragma\s+\w+|\bwaitfor\s+delay\b|\bsleep\s*\(\s*\d+",
    r"--\s*$|/\*.*?\*/|#\s*$",
]
SCRIPT_XSS = [
    r"<\s*script\b", r"javascript\s*:", r"\bon(error|load|click|mouseover|focus)\s*=",
    r"<\s*(iframe|object|embed|svg|img|link|meta|style)\b[^>]*>", r"data\s*:\s*text/html",
    r"\b(alert|prompt|confirm)\s*\(", r"document\.(cookie|location|write)",
]
COMMAND_PATH = [
    r"\.\./|\.\.\\|%2e%2e", r"(^|[;&|`])\s*(rm|curl|wget|powershell|cmd(\.exe)?|bash|sh|nc|chmod|sudo)\b\s",
    r"\$\(|`[^`]+`|\|\||&&", r"\b(os\.system|subprocess|__import__|eval\s*\(|exec\s*\(|open\s*\()",
    r"(/etc/passwd|/etc/shadow|c:\\windows|boot\.ini|win\.ini)",
    r"\{\{.*?\}\}|\{%.*?%\}|\$\{.*?\}",  # template injection
]
HARMFUL_FINANCE = [
    r"\b(launder|laundering)\b|\bmoney mule\b|\bhawala\b|\bblack money\b|\bshell compan(y|ies)\b",
    r"\b(evade|evading|dodge|dodging|avoid)\b.{0,15}\b(taxes|tax|gst|itr)\b.{0,30}\b(illegal|hide|without (getting )?caught|undetected)\b|\btax evasion\b|\bhide (my )?(income|money|cash)\b.{0,30}\b(tax|government|irs|authorit)",
    r"\b(fake|forge|forged|fabricate|falsify|tamper|doctor)\b.{0,25}\b(invoice|bills?|bank statements?|salary slips?|payslips?|documents?|itr|form ?16|income proof|kyc|receipts?)\b",
    r"\b(steal|hack|crack|clone|skim|phish|spoof|sim[- ]?swap)\w*\b.{0,30}\b(account|card|upi|bank|otp|password|credentials?|wallet|money)\b",
    r"\b(bypass|get around|beat|trick)\b.{0,20}\b(kyc|otp|2fa|two[- ]factor|aml|fraud (detection|check)|verification)\b",
    r"\b(insider trading|pump and dump|ponzi|pyramid scheme|front[- ]?running|market manipulation)\b",
    r"\b(scam|defraud|cheat|con)\b.{0,20}\b(people|someone|investors?|customers?|elderly|bank|insurance)\b",
    r"\bfalse (insurance )?claim\b|\bfake (loan|insurance) claim\b",
    r"\bwithdraw\b.{0,30}\bfrom (someone|another person)('s)?\b.{0,15}\b(account|card)\b",
]
CREDENTIAL_REQUEST = [
    r"\b(my|your|the)\b.{0,10}\b(otp|cvv|upi ?pin|atm ?pin|net ?banking password)\b.{0,15}\b(is|=|:)",
]
REGULATED_ADVICE = [
    r"\bwhich (stocks?|shares?|crypto(currency)?|coins?|mutual funds?|etfs?|penny stocks?)\b.{0,30}\b(buy|invest|pick|best|purchase|good)\b",
    r"\b(best|top|hot|multibagger|undervalued)\b.{0,15}\b(stocks?|shares?|cryptos?|coins?|etfs?)\b.{0,15}\b(to buy|to invest|right now|today|this (week|month)|for (intraday|returns))\b",
    r"\bshould i (buy|sell|short)\b.{0,25}\b(bitcoin|btc|ethereum|eth|dogecoin|tesla|reliance|tcs|infosys|nifty|sensex|adani|[a-z]{2,10} (stock|shares?))\b",
    r"\b(guaranteed|assured|risk[- ]?free|sure[- ]?shot|100%)\b.{0,20}\b(returns?|profits?|gains?|income)\b",
    r"\bintraday (tips?|calls?)\b|\boptions? trading tips?\b|\bgive me (a )?(stock|crypto) tips?\b",
    r"\bdouble my money\b|\bget rich quick\b",
]
OFF_TOPIC = [
    r"\b(recipe|poem|poetry|song lyrics|write (me )?(a |an )?(story|essay|code|script|program|joke)|tell me a joke|riddle)\b",
    r"\b(weather|cricket score|football score|movie review|celebrity|horoscope|who won)\b",
    r"\b(translate|summari[sz]e) (this|the following)\b",
    r"\b(homework|assignment|solve this equation|python (code|script)|javascript code|sql query for)\b",
    r"\b(medical advice|diagnos(e|is)|legal advice|lawyer)\b",
]
FINANCE_CUES = re.compile(
    r"\b(spend\w*|spent|expense\w*|income|salary|budget\w*|saving\w*|save|savings|bills?|emi|loan|credit|debit|"
    r"balance|cash|forecast|subscription\w*|netflix|spotify|invest\w*|fund|goal\w*|afford\w*|car|house|rent|"
    r"swiggy|zomato|uber|food|fuel|groceries|transaction\w*|merchant|payment\w*|upi|bank|health score|"
    r"anomal\w*|duplicate|fraud|unusual|alert\w*|insurance|tax|₹|rs\.?|rupees?|lakh|money|finance\w*|financial\w*|"
    r"wallet|score|emergency|runway|retire\w*|debt|interest|cut|reduce|trend|compare|month\w*|week\w*|year\w*|"
    r"vacation|trip|travel|holiday|wedding|purchase|buy|cost|price|fee|charge|cibil|sip|fd|rd|asset|wealth|"
    r"networth|analysis|analytics|profile|twin|portfolio|review|rate|statement|audit)\b",
    re.I,
)

_COMPILED = {
    "prompt_injection": [re.compile(p, re.I | re.S) for p in PROMPT_INJECTION],
    "secret_exfiltration": [re.compile(p, re.I) for p in SECRET_EXFIL],
    "sql_injection": [re.compile(p, re.I | re.S) for p in SQL_INJECTION],
    "script_injection": [re.compile(p, re.I) for p in SCRIPT_XSS],
    "command_or_path_injection": [re.compile(p, re.I | re.S) for p in COMMAND_PATH],
    "harmful_financial_request": [re.compile(p, re.I) for p in HARMFUL_FINANCE],
    "credential_sharing": [re.compile(p, re.I) for p in CREDENTIAL_REQUEST],
}
_REDIRECT = {
    "regulated_advice": [re.compile(p, re.I) for p in REGULATED_ADVICE],
    "off_topic": [re.compile(p, re.I) for p in OFF_TOPIC],
}

SEVERITY = {
    "prompt_injection": "HIGH", "secret_exfiltration": "HIGH", "sql_injection": "HIGH",
    "script_injection": "HIGH", "command_or_path_injection": "HIGH",
    "harmful_financial_request": "HIGH", "credential_sharing": "MEDIUM",
    "regulated_advice": "LOW", "off_topic": "LOW", "oversized_input": "MEDIUM",
    "empty_input": "LOW", "gibberish": "LOW",
}

MESSAGES = {
    "prompt_injection": "I can't follow instructions that try to change my rules or reveal internal configuration. Ask me about your spending, budget, forecast or goals instead.",
    "secret_exfiltration": "I can't share keys, credentials, other users' data or system internals. I can only discuss your own financial data.",
    "sql_injection": "That looks like a database command. I never run raw queries from chat — try a plain-language question like “How much did I spend on food last month?”.",
    "script_injection": "Markup or script content isn't accepted. Please ask a plain-text question about your finances.",
    "command_or_path_injection": "System commands, file paths and template syntax aren't accepted. Please ask a plain-text question about your finances.",
    "harmful_financial_request": "I can't help with anything that involves fraud, laundering, tax evasion, forged documents or account takeover. I can help you budget, save, and stay financially healthy within the law.",
    "credential_sharing": "Please never share OTPs, PINs, CVVs or passwords — not even with me. I've discarded that message.",
    "regulated_advice": "I'm an educational wellness assistant and can't recommend specific stocks, coins or funds, or promise returns. I can assess your **investment readiness** (emergency fund, debt, savings rate) — try “Am I ready to invest?”.",
    "off_topic": "⚠️ Query not related to financial advice. I am an Autonomous Financial Wellness Agent specialized exclusively in personal finance, budgeting, spending analysis, savings goals, loans, and investment readiness. Please ask a finance-related question.",
    "oversized_input": f"Please keep questions under {MAX_QUERY_CHARS} characters.",
    "empty_input": "Please type a question about your finances.",
    "gibberish": "I couldn't understand that. Try a plain-language question about your spending, budget or goals.",
}


@dataclass
class GuardResult:
    decision: str = "ALLOW"            # ALLOW | BLOCK | REDIRECT
    category: str = "clean"
    severity: str = "NONE"
    reasons: list[str] = field(default_factory=list)
    sanitized_text: str = ""
    redactions: dict[str, int] = field(default_factory=dict)
    user_message: str = ""

    @property
    def allowed(self) -> bool:
        return self.decision == "ALLOW"

    def as_dict(self) -> dict:
        return {
            "decision": self.decision, "category": self.category, "severity": self.severity,
            "reasons": self.reasons, "redactions": self.redactions, "user_message": self.user_message,
        }


def normalize(text: str) -> str:
    """Canonicalise text so trivial obfuscation does not evade the patterns."""
    t = unicodedata.normalize("NFKC", str(text or ""))
    t = t.translate(_ZERO_WIDTH)
    t = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _variants(norm: str) -> list[str]:
    # Match against the raw normalised form, a leetspeak-folded form, and a
    # separator-collapsed form (i.g.n.o.r.e) so cheap obfuscation is caught.
    folded = norm.translate(_LEET)
    collapsed = re.sub(r"(?<=\w)[.\-_*]+(?=\w)", "", norm)
    return [norm, folded, collapsed]


def scan_untrusted_text(text: str) -> list[str]:
    """Return injection categories found in *data* (CSV cells, PDF text)."""
    norm = normalize(text)
    hits: list[str] = []
    for variant in _variants(norm):
        for cat in ("prompt_injection", "script_injection", "command_or_path_injection"):
            if cat not in hits and any(p.search(variant) for p in _COMPILED[cat]):
                hits.append(cat)
    return hits


def check_query(text: str) -> GuardResult:
    raw = "" if text is None else str(text)
    norm = normalize(raw)

    def block(cat: str, reason: str, decision: str = "BLOCK") -> GuardResult:
        return GuardResult(
            decision=decision, category=cat, severity=SEVERITY.get(cat, "MEDIUM"),
            reasons=[reason], sanitized_text="", user_message=MESSAGES[cat],
        )

    if not norm:
        return block("empty_input", "Empty or whitespace-only input.")
    if len(raw) > MAX_QUERY_CHARS * 4 or len(norm) > MAX_QUERY_CHARS:
        return block("oversized_input", f"Input length {len(norm)} exceeds {MAX_QUERY_CHARS} characters.")
    letters = sum(ch.isalpha() for ch in norm)
    if len(norm) >= 8 and letters / len(norm) < 0.35 and not re.search(r"\d", norm):
        return block("gibberish", "Input is mostly symbols; likely an attack string or noise.")

    # Order matters: the most dangerous classes are checked first.
    order = [
        "prompt_injection", "secret_exfiltration", "sql_injection", "script_injection",
        "command_or_path_injection", "harmful_financial_request", "credential_sharing",
    ]
    for cat in order:
        patterns = _COMPILED[cat]
        # SQL/script/command patterns must see punctuation, so skip leetspeak folding for them.
        variants = _variants(norm) if cat in {"prompt_injection", "harmful_financial_request", "secret_exfiltration"} else [norm]
        for variant in variants:
            for p in patterns:
                m = p.search(variant)
                if m:
                    return block(cat, f"Matched {cat.replace('_', ' ')} pattern: “{m.group(0)[:60]}”.")

    for cat, patterns in _REDIRECT.items():
        for p in patterns:
            m = p.search(norm)
            if m:
                if cat == "off_topic" and FINANCE_CUES.search(norm):
                    continue  # e.g. "cricket tickets spending" is still a finance question
                return block(cat, f"Matched {cat.replace('_', ' ')} policy: “{m.group(0)[:60]}”.", decision="REDIRECT")

    # Domain Guardrail: Queries without financial or economic concepts are strictly off-topic
    is_batch_cmd = norm.lower().startswith(("run ", "update my budget", "reassess my goals"))
    if not is_batch_cmd and not FINANCE_CUES.search(norm):
        return block("off_topic", "Query not related to financial advice.", decision="REDIRECT")

    sanitized, redactions = redact_pii(norm)
    reasons = [f"Redacted {n} × {k.replace('_', ' ')} before processing." for k, n in redactions.items()]
    return GuardResult(
        decision="ALLOW",
        category="pii_redacted" if redactions else "clean",
        severity="LOW" if redactions else "NONE",
        reasons=reasons,
        sanitized_text=sanitized,
        redactions=redactions,
    )
